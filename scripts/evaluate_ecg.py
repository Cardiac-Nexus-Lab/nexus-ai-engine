"""Report a trained model's test performance with uncertainty and calibration.

Fits temperature on the validation split, then reports per-class AUROC and
average precision with bootstrap intervals, plus calibration error before and
after scaling.

Usage:
    python scripts/evaluate_ecg.py --checkpoint results/local/ecg_multilabel.pt
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
from torch.utils.data import DataLoader

from cardiac_nexus import data
from cardiac_nexus.evaluate import (
    TemperatureScaler,
    VectorScaler,
    evaluate_with_uncertainty,
    expected_calibration_error,
)
from cardiac_nexus.training import ECGDataset, build_model

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_checkpoint(path: Path, device: torch.device):
    import numpy

    with torch.serialization.safe_globals(
        [numpy._core.multiarray.scalar, numpy.dtype, numpy.dtypes.Float64DType, numpy.dtypes.Int64DType]
    ):
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    classes = checkpoint["classes"]
    model = build_model(checkpoint.get("architecture", "cnn"), len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval(), classes, checkpoint


@torch.no_grad()
def collect_logits(model, signals, labels, indices, device, batch_size=128):
    loader = DataLoader(ECGDataset(signals, labels, indices), batch_size=batch_size, shuffle=False)
    logits, targets = [], []
    for x, y in loader:
        logits.append(model(x.to(device)).cpu().numpy())
        targets.append(y.numpy())
    return np.concatenate(logits), np.concatenate(targets)


def plot_reliability(rows_before, rows_after, classes, destination: Path) -> None:
    fig, axes = plt.subplots(1, len(classes), figsize=(3.1 * len(classes), 3.4), sharey=True)
    for axis, name, before, after in zip(axes, classes, rows_before, rows_after):
        axis.plot([0, 1], [0, 1], "--", color="#999", linewidth=1, label="perfect")
        for bins, colour, label in ((before["calibration"]["bins"], "#d1495b", "raw"),
                                    (after["calibration"]["bins"], "#1f6fd1", "scaled")):
            points = [(b["confidence"], b["observed"]) for b in bins if b["count"] > 0]
            if points:
                xs, ys = zip(*points)
                axis.plot(xs, ys, "o-", color=colour, markersize=4, linewidth=1.4, label=label)
        axis.set_title(f"{name}\nECE {before['calibration']['ece']:.3f} → {after['calibration']['ece']:.3f}",
                       fontsize=9)
        axis.set_xlabel("predicted")
        axis.set_xlim(0, 1); axis.set_ylim(0, 1)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
    axes[0].set_ylabel("observed frequency")
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(destination, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "results" / "local" / "ecg_multilabel.pt")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resamples", type=int, default=2000)
    args = parser.parse_args()

    output = args.output or args.checkpoint.parent
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    model, classes, checkpoint = load_checkpoint(args.checkpoint, device)
    metadata, signals, splits = data.prepare()
    labels = metadata[data.SUPERCLASSES].to_numpy(dtype=np.float32)
    signals = np.asarray(signals)

    val_logits, val_targets = collect_logits(model, signals, labels, splits["val"], device)
    test_logits, test_targets = collect_logits(model, signals, labels, splits["test"], device)

    raw_probabilities = 1.0 / (1.0 + np.exp(-test_logits))

    # Fit both, report the one that actually calibrates better on validation.
    temperature_scaler = TemperatureScaler()
    temperature = temperature_scaler.fit(val_logits, val_targets)
    vector_scaler = VectorScaler(len(classes))
    vector_parameters = vector_scaler.fit(val_logits, val_targets)

    val_raw = 1.0 / (1.0 + np.exp(-val_logits))
    def mean_ece(probabilities):
        return float(np.mean([
            expected_calibration_error(val_targets[:, i], probabilities[:, i])["ece"]
            for i in range(len(classes))
        ]))

    candidates = {
        "none": (val_raw, lambda z: 1.0 / (1.0 + np.exp(-z))),
        "temperature": (temperature_scaler.apply(val_logits), temperature_scaler.apply),
        "vector": (vector_scaler.apply(val_logits), vector_scaler.apply),
    }
    chosen = min(candidates, key=lambda k: mean_ece(candidates[k][0]))
    print("Validation mean ECE:  " + "  ".join(
        f"{name}={mean_ece(probs):.3f}" for name, (probs, _) in candidates.items()
    ) + f"   -> using '{chosen}'")
    scaled_probabilities = candidates[chosen][1](test_logits)

    before = evaluate_with_uncertainty(test_targets, raw_probabilities, classes, args.resamples)
    after = evaluate_with_uncertainty(test_targets, scaled_probabilities, classes, args.resamples)

    plot_reliability(before, after, classes, output / "calibration.png")

    print(f"Checkpoint: {args.checkpoint.name}  ({checkpoint.get('architecture','cnn')})")
    print(f"Temperature {temperature:.3f} | per-class scales "
          f"{np.round(vector_parameters['scales'], 2).tolist()}\n")
    header = f"{'class':>6} {'AUROC (95% CI)':>24} {'AP (95% CI)':>24} {'ECE raw':>8} {'ECE cal':>8} {'n':>5}"
    print(header)
    print("-" * len(header))
    for b, a in zip(before, after):
        auroc = f"{b['auroc']['estimate']:.3f} ({b['auroc']['low']:.3f}-{b['auroc']['high']:.3f})"
        ap = f"{b['average_precision']['estimate']:.3f} ({b['average_precision']['low']:.3f}-{b['average_precision']['high']:.3f})"
        print(f"{b['class']:>6} {auroc:>24} {ap:>24} "
              f"{b['calibration']['ece']:>8.3f} {a['calibration']['ece']:>8.3f} {b['support']:>5d}")

    macro = float(np.mean([b["auroc"]["estimate"] for b in before]))
    mean_ece_raw = float(np.mean([b["calibration"]["ece"] for b in before]))
    mean_ece_cal = float(np.mean([a["calibration"]["ece"] for a in after]))
    print(f"\nMacro AUROC {macro:.4f} | mean ECE {mean_ece_raw:.3f} raw, {mean_ece_cal:.3f} after scaling")

    report = {
        "checkpoint": args.checkpoint.name,
        "architecture": checkpoint.get("architecture", "cnn"),
        "temperature": temperature,
        "vector_scaling": vector_parameters,
        "calibrator_used": chosen,
        "macro_auroc": macro,
        "mean_ece_raw": mean_ece_raw,
        "mean_ece_calibrated": mean_ece_cal,
        "per_class_raw": before,
        "per_class_calibrated": after,
    }
    (output / "evaluation_report.json").write_text(json.dumps(report, indent=2))
    print(f"Wrote evaluation_report.json and calibration.png to {output}")


if __name__ == "__main__":
    main()
