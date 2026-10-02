"""Five-fold cross-validation of the MRI segmenter, then a five-model ensemble on test.

The single model was selected on one fixed set of 20 validation patients, so its
validation score depends on which 20 they were. Cross-validation rotates that
set: every one of the 100 source-training patients is used for validation
exactly once, and the spread across folds says how much the score depends on
the split. Each fold keeps four patients of every diagnosis for validation.

The five fold models are then combined on the 50 held-out test patients by
averaging their class probabilities. Each variant below is scored the same way
as `evaluate_mri.py`, so the numbers are directly comparable:

* the existing single model, as it is today;
* the single model with largest-connected-component cleanup;
* the five-fold ensemble;
* the five-fold ensemble with cleanup.

A fold that has finished is skipped on a re-run, so training can be stopped and
resumed (a closed laptop loses at most the fold in progress).

Usage:
    python scripts/train_mri_cv.py                           # full run
    python scripts/train_mri_cv.py --epochs 1 --limit-patients 10 --folds 2   # quick check
    python scripts/train_mri_cv.py --evaluate-only           # re-score finished folds
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np
import torch
from scipy import ndimage

from cardiac_nexus import mri_data
from cardiac_nexus.models import select_device
from cardiac_nexus.mri_data import PATHOLOGIES, PHASES, STRUCTURES
from cardiac_nexus.mri_dataset import ACDCSlices, normalize_volumes
from cardiac_nexus.mri_training import MRITrainConfig, score_patients, train_segmenter
from evaluate_mri import agreement, bootstrap_mean, hausdorff_mm, load_segmenter


def fold_assignment(metadata, folds: int, seed: int = 42) -> list[np.ndarray]:
    """Validation patients for each fold, stratified by diagnosis.

    Each diagnosis has 20 source-training patients; shuffling them and dealing
    them round-robin gives every fold the same number of each.
    """
    rng = np.random.default_rng(seed)
    pool = metadata[metadata["source_split"] == "train"]
    buckets: list[list[int]] = [[] for _ in range(folds)]
    for pathology in PATHOLOGIES:
        members = rng.permutation(pool.index[pool["pathology"] == pathology].to_numpy())
        for position, patient in enumerate(members):
            buckets[position % folds].append(int(patient))
    return [np.sort(np.asarray(bucket)) for bucket in buckets]


@torch.no_grad()
def predict_ensemble(models, images, masks, patient, phase, spacing, patients, device,
                     batch_size: int = 32) -> dict:
    """Like `predict_volumes`, but averages the softmax of several models."""
    dataset = ACDCSlices(images, masks, patient, phase, patients, augment=False)
    chunks = []
    for start in range(0, len(dataset), batch_size):
        stop = min(start + batch_size, len(dataset))
        batch = torch.stack([dataset[i][0] for i in range(start, stop)]).to(device)
        probabilities = sum(model(batch).softmax(dim=1) for model in models)
        chunks.append(probabilities.argmax(dim=1).cpu().numpy().astype(np.uint8))
    predictions = np.concatenate(chunks)

    rows = dataset.rows
    keys = patient[rows].astype(np.int64) * 2 + phase[rows]
    volumes = {}
    for key in np.unique(keys):
        selected = keys == key
        volumes[(int(key // 2), int(key % 2))] = (
            predictions[selected], masks[rows][selected], tuple(float(s) for s in spacing[rows][selected][0])
        )
    return volumes


def keep_largest_component(volume: np.ndarray) -> np.ndarray:
    """Keep only the largest 3D connected region of each structure.

    A heart has one LV cavity, one myocardium and one RV. Small islands of a
    label elsewhere in the volume are always errors; they barely move Dice but
    dominate the Hausdorff distance.
    """
    cleaned = volume.copy()
    for label in STRUCTURES:
        components, count = ndimage.label(volume == label)
        if count > 1:
            largest = np.argmax(np.bincount(components.ravel())[1:]) + 1
            cleaned[(components != largest) & (volume == label)] = 0
    return cleaned


def cleaned(volumes: dict) -> dict:
    return {key: (keep_largest_component(predicted), actual, voxel)
            for key, (predicted, actual, voxel) in volumes.items()}


def summarize(volumes: dict) -> dict:
    """Dice, Hausdorff and ejection-fraction agreement, as in evaluate_mri.py."""
    rows = {row["patient"]: row for row in score_patients(volumes)}
    for (patient_id, phase_id), (predicted, actual, voxel) in volumes.items():
        for label, name in STRUCTURES.items():
            rows[patient_id][f"hausdorff_{name}_{PHASES[phase_id]}"] = hausdorff_mm(predicted, actual, label, voxel)
    patients = list(rows.values())
    report = {"dice": {}, "hausdorff_mm": {}, "derived_measures": {}}
    for name in STRUCTURES.values():
        for phase in PHASES:
            report["dice"][f"{name}_{phase}"] = bootstrap_mean([p[f"dice_{name}_{phase}"] for p in patients])
            report["hausdorff_mm"][f"{name}_{phase}"] = bootstrap_mean(
                [p[f"hausdorff_{name}_{phase}"] for p in patients])
    for name in ("LV", "RV"):
        report["derived_measures"][f"{name}_ejection_fraction_percent"] = agreement(
            [p[f"ef_{name}_pred"] for p in patients], [p[f"ef_{name}_true"] for p in patients])
    return report


def print_comparison(reports: dict) -> None:
    print(f"\n{'variant':30}" + "".join(f"{f'{n} {ph.upper()}':>9}" for n in ("LV", "RV", "MYO") for ph in PHASES)
          + f"{'HD MYO ED':>11}{'LV EF r':>9}{'LV EF MAE':>11}")
    for variant, report in reports.items():
        dice = "".join(f"{report['dice'][f'{n}_{ph}']['estimate']:>9.3f}" for n in ("LV", "RV", "MYO") for ph in PHASES)
        lv_ef = report["derived_measures"]["LV_ejection_fraction_percent"]
        print(f"{variant:30}{dice}{report['hausdorff_mm']['MYO_ed']['estimate']:>11.1f}"
              f"{lv_ef['correlation']:>9.3f}{lv_ef['mean_absolute_error']:>11.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--myo-weight", type=float, default=1.5,
                        help="weight of the myocardium Dice term relative to LV and RV")
    parser.add_argument("--base-width", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit-patients", type=int, default=None,
                        help="train each fold on only this many patients, for a quick pipeline check")
    parser.add_argument("--evaluate-only", action="store_true", help="skip training; score finished folds")
    parser.add_argument("--single", type=Path, default=REPO_ROOT / "results" / "mri_segmentation" / "mri_segmenter.pt",
                        help="existing single model to compare against")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "results" / "mri_segmentation" / "cv")
    args = parser.parse_args()

    metadata, cache, splits = mri_data.prepare()
    validation_folds = fold_assignment(metadata, args.folds)
    source_train = metadata.index[metadata["source_split"] == "train"].to_numpy()
    args.output.mkdir(parents=True, exist_ok=True)

    if not args.evaluate_only:
        for fold, validation in enumerate(validation_folds):
            fold_dir = args.output / f"fold{fold}"
            if (fold_dir / "finished.json").exists():
                print(f"Fold {fold}: already finished, skipping")
                continue
            train = np.setdiff1d(source_train, validation)
            if args.limit_patients:
                train, validation = train[: args.limit_patients], validation[: max(2, args.limit_patients // 4)]
            config = MRITrainConfig(epochs=args.epochs, base_width=args.base_width, batch_size=args.batch_size,
                                    learning_rate=args.learning_rate, seed=42 + fold, device=args.device,
                                    myo_weight=args.myo_weight)
            print(f"\n=== Fold {fold + 1} of {args.folds}: {len(train)} train, {len(validation)} validation patients ===")
            started = time.perf_counter()
            result = train_segmenter(config, cache, {"train": train, "val": validation, "test": splits["test"]},
                                     fold_dir)
            metrics = json.loads((fold_dir / "mri_training_metrics.json").read_text())
            best = metrics["history"][result["best_epoch"] - 1]
            (fold_dir / "finished.json").write_text(json.dumps({
                "fold": fold, "best_epoch": result["best_epoch"], "val_dice": best["val_dice"],
                "val_mean_dice": best["val_mean_dice"], "hours": (time.perf_counter() - started) / 3600,
                "validation_patients": [str(metadata.loc[i, "pid"]) for i in validation],
                "config": asdict(config),
            }, indent=2))

    finished = sorted(args.output.glob("fold*/finished.json"))
    if not finished:
        sys.exit("No finished folds to evaluate.")
    folds = [json.loads(path.read_text()) for path in finished]
    cross_validation = {name: {"mean": float(np.mean([f["val_dice"][name] for f in folds])),
                               "std": float(np.std([f["val_dice"][name] for f in folds], ddof=1)) if len(folds) > 1 else 0.0}
                        for name in STRUCTURES.values()}
    print(f"\nCross-validation over {len(folds)} folds (validation Dice, mean +/- std):")
    for name, value in cross_validation.items():
        print(f"  {name:4} {value['mean']:.3f} +/- {value['std']:.3f}")

    device = select_device(args.device)
    images = normalize_volumes(cache["images"], cache["patient"], cache["phase"])
    inputs = (images, cache["masks"], cache["patient"], cache["phase"], cache["spacing"], splits["test"], device)
    fold_models = [load_segmenter(path.parent / "mri_segmenter.pt", device)[0] for path in finished]

    reports = {}
    if args.single.exists():
        single = predict_ensemble([load_segmenter(args.single, device)[0]], *inputs)
        reports["single model"] = summarize(single)
        reports["single model + cleanup"] = summarize(cleaned(single))
    ensemble = predict_ensemble(fold_models, *inputs)
    reports[f"{len(fold_models)}-fold ensemble"] = summarize(ensemble)
    reports[f"{len(fold_models)}-fold ensemble + cleanup"] = summarize(cleaned(ensemble))

    print(f"\nACDC test set, {len(splits['test'])} patients, Dice per structure and phase:")
    print_comparison(reports)
    (args.output / "cv_test_comparison.json").write_text(json.dumps(
        {"folds": folds, "cross_validation_dice": cross_validation, "test": reports}, indent=2, default=str))
    print(f"\nWrote {args.output / 'cv_test_comparison.json'}")


if __name__ == "__main__":
    main()
