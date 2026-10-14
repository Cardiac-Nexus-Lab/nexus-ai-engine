"""Train the MRI segmenter on ACDC, M&Ms and M&Ms-2 together, then test on each.

The ACDC-only model has only ever seen one hospital's scanners. This script
pools the three public cohorts that share ACDC's labels (830 patients against
150), trains one segmenter on their training patients, and reports Dice,
Hausdorff distance and ejection-fraction agreement on each cohort's held-out
test patients separately. The ACDC-only model is scored on the same patients,
so the comparison answers two questions at once: how well the old model
transfers to other centres and vendors, and whether pooled training helps.

By default training starts from the ACDC-only model's weights rather than from
random ones, which matters on a CPU where a full from-scratch schedule is out of
reach. That model never saw an ACDC test patient, so its weights carry no test
information.

Before training, every cohort's labels are checked against cardiac anatomy, and
patients duplicated across cohorts are removed from training and validation.

Usage:
    python scripts/train_mri_pooled.py --epochs 12
    python scripts/train_mri_pooled.py --epochs 1 --limit-patients 20     # quick check
    python scripts/train_mri_pooled.py --evaluate-only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np

from cardiac_nexus import mri_cohorts
from cardiac_nexus.models import select_device
from cardiac_nexus.mri_data import PHASES, STRUCTURES
from cardiac_nexus.mri_dataset import normalize_volumes
from cardiac_nexus.mri_training import MRITrainConfig, train_segmenter
from evaluate_mri import bootstrap_mean, load_segmenter
from train_mri_cv import cleaned, predict_ensemble, print_comparison, summarize

ACDC_MODEL = REPO_ROOT / "results" / "mri_segmentation" / "mri_segmenter.pt"


def check_labels(metadata, cache) -> dict:
    """Stated against mask-derived end-diastolic volumes, and missing structures, per cohort."""
    table = mri_cohorts.verify_against_reference(metadata, cache)
    report = {}
    for cohort, group in table.groupby("cohort"):
        stated = group.dropna(subset=["lv_edv_stated", "rv_edv_stated"])
        entry = {"patients": int(len(group)), "with_stated_volumes": int(len(stated)),
                 "missing_structure_at_ed": int(((group[["lv_edv_mask", "rv_edv_mask", "myo_ed_ml"]] <= 0)
                                                 .any(axis=1)).sum())}
        if len(stated):
            entry["lv_edv_max_abs_difference_ml"] = float((stated.lv_edv_mask - stated.lv_edv_stated).abs().max())
            entry["rv_edv_max_abs_difference_ml"] = float((stated.rv_edv_mask - stated.rv_edv_stated).abs().max())
        report[cohort] = entry
    return report


def per_vendor_dice(volumes: dict, metadata) -> dict:
    """Mean Dice over structures and phases, grouped by scanner vendor."""
    from cardiac_nexus.mri_training import score_patients
    groups: dict[str, list[float]] = {}
    for row in score_patients(volumes):
        vendor = str(metadata.loc[row["patient"], "vendor"])
        groups.setdefault(vendor, []).append(
            float(np.mean([row[f"dice_{n}_{p}"] for n in STRUCTURES.values() for p in PHASES])))
    return {vendor: bootstrap_mean(values) for vendor, values in sorted(groups.items())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--myo-weight", type=float, default=1.0)
    parser.add_argument("--from-scratch", action="store_true", help="do not start from the ACDC-only model")
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit-patients", type=int, default=None,
                        help="train on only this many patients, for a quick pipeline check")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "results" / "mri_segmentation_pooled")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    metadata, cache, splits = mri_cohorts.prepare()
    labels = check_labels(metadata, cache)
    anatomy = mri_cohorts.label_anatomy(metadata, cache)
    for cohort in labels:
        labels[cohort]["anatomy"] = anatomy[cohort]
        if anatomy[cohort]["LV_border_myocardium"] < 0.9 or anatomy[cohort]["RV_border_myocardium"] > 0.6:
            sys.exit(f"{cohort}: LV/RV labels do not match ACDC's convention: {anatomy[cohort]}")
    print("Label check:")
    for cohort, entry in labels.items():
        print(f"  {cohort:6} {entry}")

    duplicates = mri_cohorts.find_duplicates(metadata, cache)
    held_out = set(int(i) for i in splits["test"])
    # Of each duplicate pair, a copy in a test set stays there and the other copy
    # leaves training and validation; with neither in test, the second copy goes.
    removed = set()
    for first, second, _ in duplicates:
        if first in held_out and second in held_out:
            continue
        removed.add(first if second in held_out else second)
    splits = mri_cohorts.exclude(splits, removed)
    print(f"Cross-cohort duplicates found: {len(duplicates)}; removed from training/validation: {len(removed)}")
    print({name: len(ids) for name, ids in splits.items() if name != "test_by_cohort"},
          {name: len(ids) for name, ids in splits["test_by_cohort"].items()})

    checkpoint = args.output / "mri_segmenter.pt"
    if not args.evaluate_only:
        train_splits = {"train": splits["train"], "val": splits["val"], "test": splits["test"]}
        if args.limit_patients:
            rng = np.random.default_rng(0)
            train_splits["train"] = np.sort(rng.choice(splits["train"], args.limit_patients, replace=False))
            train_splits["val"] = np.sort(rng.choice(splits["val"], max(4, args.limit_patients // 4), replace=False))
        config = MRITrainConfig(epochs=args.epochs, learning_rate=args.learning_rate, batch_size=args.batch_size,
                                device=args.device, myo_weight=args.myo_weight,
                                init_checkpoint=None if args.from_scratch else str(ACDC_MODEL))
        started = time.perf_counter()
        result = train_segmenter(config, cache, train_splits, args.output)
        print(f"Best epoch {result['best_epoch']}: validation mean Dice {result['best_val_mean_dice']:.3f} "
              f"({(time.perf_counter() - started) / 3600:.2f} h)")

    device = select_device(args.device)
    images = normalize_volumes(cache["images"], cache["patient"], cache["phase"])
    models = {"ACDC-only model": load_segmenter(ACDC_MODEL, device)[0],
              "pooled model": load_segmenter(checkpoint, device)[0]}

    report = {"label_check": labels, "duplicates": [[metadata.loc[a, "cohort"] + ":" + str(metadata.loc[a, "pid"]),
                                                     metadata.loc[b, "cohort"] + ":" + str(metadata.loc[b, "pid"]), s]
                                                    for a, b, s in duplicates],
              "patients": {"train": int(len(splits["train"])), "val": int(len(splits["val"])),
                           **{f"test_{c}": int(len(ids)) for c, ids in splits["test_by_cohort"].items()}},
              "test": {}}
    for cohort, patients in splits["test_by_cohort"].items():
        reports = {}
        for name, model in models.items():
            volumes = predict_ensemble([model], images, cache["masks"], cache["patient"], cache["phase"],
                                       cache["spacing"], patients, device)
            reports[name] = summarize(volumes)
            reports[name + " + cleanup"] = summarize(cleaned(volumes))
            if cohort != "acdc":
                reports[name]["dice_by_vendor"] = per_vendor_dice(volumes, metadata)
        report["test"][cohort] = reports
        print(f"\n{cohort} test set, {len(patients)} patients:")
        print_comparison(reports)

    (args.output / "pooled_test_comparison.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"\nWrote {args.output / 'pooled_test_comparison.json'}")


if __name__ == "__main__":
    main()
