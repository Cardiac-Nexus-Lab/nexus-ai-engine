"""Evaluate the MRI segmenter once, on the held-out ACDC test patients.

Reports what a reader of a cardiac MRI actually uses:

* Dice and Hausdorff distance per structure at end-diastole and end-systole,
  with 95% bootstrap intervals over patients;
* ejection fraction and end-diastolic volume derived from the predicted masks,
  compared with the same quantities from the expert masks (correlation, bias and
  Bland-Altman limits of agreement);
* the same Dice broken down by diagnosis, because a model that fails on one
  disease group can still average well.

Usage:
    python scripts/evaluate_mri.py --checkpoint results/mri_segmentation/mri_segmenter.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import ndimage

from cardiac_nexus import mri_data
from cardiac_nexus.models import select_device
from cardiac_nexus.mri_data import PHASES, STRUCTURES
from cardiac_nexus.mri_dataset import normalize_volumes
from cardiac_nexus.mri_models import MRISegmenter
from cardiac_nexus.mri_training import predict_volumes, score_patients

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_segmenter(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = MRISegmenter(in_channels=checkpoint["in_channels"], num_classes=checkpoint["num_classes"],
                         widths=tuple(checkpoint["widths"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval(), checkpoint


def _surface(mask: np.ndarray) -> np.ndarray:
    return mask & ~ndimage.binary_erosion(mask)


def hausdorff_mm(prediction: np.ndarray, truth: np.ndarray, label: int,
                 spacing: tuple[float, float, float]) -> float:
    """Symmetric Hausdorff distance between the two surfaces, in millimetres.

    Dice rewards bulk overlap; Hausdorff catches the stray blob far from the heart
    that Dice barely notices. Volumes are stored [slices, x, y] while spacing is
    (x, y, z), so the sampling order is reordered to match the array.
    """
    predicted, actual = prediction == label, truth == label
    if not predicted.any() and not actual.any():
        return 0.0
    if not predicted.any() or not actual.any():
        return float("nan")
    sampling = (spacing[2], spacing[0], spacing[1])
    predicted_surface, actual_surface = _surface(predicted), _surface(actual)
    to_actual = ndimage.distance_transform_edt(~actual_surface, sampling=sampling)
    to_predicted = ndimage.distance_transform_edt(~predicted_surface, sampling=sampling)
    return float(max(to_actual[predicted_surface].max(), to_predicted[actual_surface].max()))


def bootstrap_mean(values, resamples: int = 2000, seed: int = 42) -> dict:
    """Mean over patients with a 95% percentile interval from resampling patients."""
    finite = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    rng = np.random.default_rng(seed)
    means = finite[rng.integers(0, len(finite), (resamples, len(finite)))].mean(axis=1)
    return {"estimate": float(finite.mean()), "low": float(np.percentile(means, 2.5)),
            "high": float(np.percentile(means, 97.5)), "n": int(len(finite)),
            "excluded_undefined": int(len(values) - len(finite))}


def agreement(predicted, actual) -> dict:
    """Bland-Altman agreement between a derived measure and its reference."""
    predicted, actual = np.asarray(predicted, float), np.asarray(actual, float)
    difference = predicted - actual
    bias, spread = difference.mean(), difference.std(ddof=1)
    return {"correlation": float(np.corrcoef(predicted, actual)[0, 1]), "bias": float(bias),
            "limits_of_agreement": [float(bias - 1.96 * spread), float(bias + 1.96 * spread)],
            "mean_absolute_error": float(np.abs(difference).mean()), "n": int(len(difference))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=REPO_ROOT / "results" / "mri_segmentation" / "mri_segmenter.pt")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    output = args.output or args.checkpoint.parent
    output.mkdir(parents=True, exist_ok=True)

    device = select_device(args.device)
    model, checkpoint = load_segmenter(args.checkpoint, device)
    metadata, cache, _ = mri_data.prepare()
    # The checkpoint records the split it was trained under; evaluating against a
    # freshly computed one would silently break if the split logic ever changed.
    test = np.asarray(checkpoint["splits"]["test"])
    images = normalize_volumes(cache["images"], cache["patient"], cache["phase"])
    volumes = predict_volumes(model, images, cache["masks"], cache["patient"], cache["phase"],
                              cache["spacing"], test, device)

    rows = {row["patient"]: row for row in score_patients(volumes)}
    for (patient_id, phase_id), (predicted, actual, voxel) in volumes.items():
        for label, name in STRUCTURES.items():
            rows[patient_id][f"hausdorff_{name}_{PHASES[phase_id]}"] = hausdorff_mm(predicted, actual, label, voxel)
    for patient_id, row in rows.items():
        row["pid"] = metadata.loc[patient_id, "pid"]
        row["pathology"] = metadata.loc[patient_id, "pathology"]
    patients = list(rows.values())

    report = {
        "checkpoint": str(args.checkpoint), "selected_epoch": checkpoint["epoch"],
        "validation_dice_at_selection": checkpoint["val_dice"], "test_patients": len(patients),
        "dice": {}, "hausdorff_mm": {}, "derived_measures": {}, "dice_by_pathology": {},
    }
    for name in STRUCTURES.values():
        for phase in PHASES:
            report["dice"][f"{name}_{phase}"] = bootstrap_mean([p[f"dice_{name}_{phase}"] for p in patients])
            report["hausdorff_mm"][f"{name}_{phase}"] = bootstrap_mean([p[f"hausdorff_{name}_{phase}"] for p in patients])
    for name in ("LV", "RV"):
        report["derived_measures"][f"{name}_ejection_fraction_percent"] = agreement(
            [p[f"ef_{name}_pred"] for p in patients], [p[f"ef_{name}_true"] for p in patients])
        report["derived_measures"][f"{name}_end_diastolic_volume_ml"] = agreement(
            [p[f"volume_{name}_ed_pred"] for p in patients], [p[f"volume_{name}_ed_true"] for p in patients])
    for pathology in mri_data.PATHOLOGIES:
        group = [p for p in patients if p["pathology"] == pathology]
        report["dice_by_pathology"][pathology] = bootstrap_mean(
            [np.mean([p[f"dice_{name}_{phase}"] for name in STRUCTURES.values() for phase in PHASES]) for p in group])
    report["per_patient"] = [{k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in p.items()}
                             for p in patients]
    (output / "test_evaluation.json").write_text(json.dumps(report, indent=2, default=str))

    print(f"\nACDC test set: {len(patients)} patients, checkpoint from epoch {checkpoint['epoch']}\n")
    print(f"{'structure':10}{'Dice ED (95% CI)':>26}{'Dice ES (95% CI)':>26}{'HD ED mm':>11}{'HD ES mm':>11}")
    for name in ("LV", "RV", "MYO"):
        ed, es = report["dice"][f"{name}_ed"], report["dice"][f"{name}_es"]
        print(f"{name:10}{ed['estimate']:>9.3f} ({ed['low']:.3f}-{ed['high']:.3f}){es['estimate']:>9.3f} "
              f"({es['low']:.3f}-{es['high']:.3f}){report['hausdorff_mm'][f'{name}_ed']['estimate']:>11.1f}"
              f"{report['hausdorff_mm'][f'{name}_es']['estimate']:>11.1f}")
    print()
    for key, value in report["derived_measures"].items():
        low, high = value["limits_of_agreement"]
        print(f"{key:34} r={value['correlation']:.3f}  bias {value['bias']:+.2f}  "
              f"limits of agreement [{low:+.2f}, {high:+.2f}]  MAE {value['mean_absolute_error']:.2f}")
    print()
    for pathology, value in report["dice_by_pathology"].items():
        print(f"  {pathology:5} mean Dice {value['estimate']:.3f} ({value['low']:.3f}-{value['high']:.3f})  n={value['n']}")

    _plot_agreement(patients, output / "test_ef_agreement.png")
    _plot_examples(patients, volumes, images, cache, output / "test_examples.png")
    print(f"\nWrote {output / 'test_evaluation.json'} and figures")


def _plot_agreement(patients: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for axis, name in zip(axes, ("LV", "RV")):
        predicted = np.array([p[f"ef_{name}_pred"] for p in patients])
        actual = np.array([p[f"ef_{name}_true"] for p in patients])
        mean, difference = (predicted + actual) / 2, predicted - actual
        bias, spread = difference.mean(), difference.std(ddof=1)
        axis.scatter(mean, difference, s=18, color="#1f5f8b")
        for level, style in ((bias, "-"), (bias + 1.96 * spread, "--"), (bias - 1.96 * spread, "--")):
            axis.axhline(level, color="#8a8a8a", linestyle=style, linewidth=1)
        axis.set_title(f"{name} ejection fraction: model minus expert")
        axis.set_xlabel("mean of model and expert EF (%)")
        axis.set_ylabel("difference (percentage points)")
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _plot_examples(patients, volumes, images, cache, path: Path) -> None:
    """Best, median and worst test patient by mean Dice, mid-ventricular ED slice."""
    ranked = sorted(patients, key=lambda p: np.mean([p[f"dice_{n}_{ph}"] for n in STRUCTURES.values() for ph in PHASES]))
    chosen = [ranked[-1], ranked[len(ranked) // 2], ranked[0]]
    colours = {1: "#e4572e", 2: "#29335c", 3: "#17bebb"}
    fig, axes = plt.subplots(2, 3, figsize=(11, 7.4))
    for column, patient in enumerate(chosen):
        predicted, actual, _ = volumes[(patient["patient"], 0)]
        rows = np.flatnonzero((cache["patient"] == patient["patient"]) & (cache["phase"] == 0))
        middle = len(rows) // 2
        for row_index, (mask, title) in enumerate(((actual, "expert"), (predicted, "model"))):
            axis = axes[row_index, column]
            axis.imshow(images[rows[middle]], cmap="gray")
            for label, colour in colours.items():
                if (mask[middle] == label).any():
                    axis.contour(mask[middle] == label, levels=[0.5], colors=colour, linewidths=1.2)
            score = np.mean([patient[f"dice_{n}_ed"] for n in STRUCTURES.values()])
            axis.set_title(f"{patient['pid']} ({patient['pathology']}) {title}"
                           + (f"\nED mean Dice {score:.3f}" if title == "model" else ""), fontsize=9)
            axis.axis("off")
    fig.suptitle("Held-out test patients: best, median, worst. Red RV, dark blue MYO, teal LV", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
