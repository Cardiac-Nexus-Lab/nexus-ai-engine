"""Attribute a trained ECG model's predictions and validate the attributions.

Produces, for a sample of held-out recordings: Integrated Gradients and
SmoothGrad maps, their agreement, and the model-randomization sanity check.

Usage:
    python scripts/explain_ecg.py --checkpoint results/local/ecg_multilabel.pt
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

from cardiac_nexus import data
from cardiac_nexus.explain import (
    attribute,
    model_randomization_test,
    similarity,
    smoothgrad_attribute,
)
from cardiac_nexus.training import build_model

REPO_ROOT = Path(__file__).resolve().parents[1]
ACCENT_POSITIVE = "#d1495b"
ACCENT_NEGATIVE = "#1f6fd1"


def load_checkpoint(path: Path, device: torch.device):
    import numpy  # noqa: F401 - referenced by the allowlist below

    with torch.serialization.safe_globals(
        [numpy._core.multiarray.scalar, numpy.dtype, numpy.dtypes.Float64DType, numpy.dtypes.Int64DType]
    ):
        checkpoint = torch.load(path, map_location=device, weights_only=True)

    classes = checkpoint["classes"]
    model = build_model(checkpoint.get("architecture", "cnn"), len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval(), classes, checkpoint


def plot_attribution(signal: np.ndarray, values: np.ndarray, title: str, destination: Path) -> None:
    fig, axes = plt.subplots(12, 1, figsize=(11, 15), sharex=True)
    fig.suptitle(title, y=0.995, fontsize=11)
    time_axis = np.arange(signal.shape[1]) / 100.0
    scale = np.abs(values).max() + 1e-9

    for index, lead in enumerate(data.LEAD_NAMES):
        axis = axes[index]
        axis.plot(time_axis, signal[index], color="black", linewidth=0.7)
        normalized = values[index] / scale * signal[index].std() * 3
        axis.fill_between(time_axis, 0, normalized, where=normalized >= 0,
                          color=ACCENT_POSITIVE, alpha=0.45, linewidth=0)
        axis.fill_between(time_axis, 0, normalized, where=normalized < 0,
                          color=ACCENT_NEGATIVE, alpha=0.40, linewidth=0)
        axis.set_ylabel(lead, rotation=0, ha="right", va="center", fontsize=9)
        axis.set_yticks([])
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(destination, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "results" / "local" / "ecg_multilabel.pt")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "results" / "explainability_local")
    parser.add_argument("--examples", type=int, default=6)
    parser.add_argument("--device", default="cpu", help="cpu keeps this off an in-use GPU")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(args.device)
    model, classes, checkpoint = load_checkpoint(args.checkpoint, device)
    print(f"Loaded {args.checkpoint.name}: {checkpoint.get('architecture', 'cnn')}, classes {classes}")

    metadata, signals, splits = data.prepare()
    labels = metadata[data.SUPERCLASSES].to_numpy(dtype=np.float32)
    test_positions = splits["test"]

    rng = np.random.default_rng(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)

    report: list[dict] = []
    for class_index, class_name in enumerate(classes):
        # Attribute each class on recordings that genuinely carry that label.
        candidates = test_positions[labels[test_positions, class_index] == 1]
        if len(candidates) == 0:
            continue
        chosen = rng.choice(candidates, size=min(args.examples // len(classes) + 1, len(candidates)), replace=False)

        for position in chosen:
            signal = np.asarray(signals[position])
            ig = attribute(model, signal, class_index, device)
            smooth = smoothgrad_attribute(model, signal, class_index, device)
            agreement = similarity(ig.values, smooth.values)
            sanity = model_randomization_test(model, signal, class_index, device)

            record_id = int(metadata.index[position])
            name = f"{class_name}_{record_id}_p{ig.probability:.2f}"
            plot_attribution(
                signal,
                ig.values,
                f"{class_name}  |  ecg_id {record_id}  |  P({class_name})={ig.probability:.3f}  "
                f"|  Integrated Gradients",
                args.output / f"{name}.png",
            )

            report.append(
                {
                    "ecg_id": record_id,
                    "class": class_name,
                    "probability": ig.probability,
                    "convergence_delta": ig.convergence_delta,
                    "ig_vs_smoothgrad": agreement,
                    "sanity_check": sanity,
                }
            )
            print(
                f"  {class_name:<5} ecg_id {record_id:<6} P={ig.probability:.3f}  "
                f"delta={ig.convergence_delta:+.2e}  "
                f"IG~SmoothGrad r={agreement['spearman']:.3f}  "
                f"vs-random r={sanity['spearman_vs_random_mean']:+.3f}"
            )

    summary = {
        "checkpoint": str(args.checkpoint.relative_to(REPO_ROOT)),
        "architecture": checkpoint.get("architecture", "cnn"),
        "examples": report,
        "aggregate": {
            "mean_ig_smoothgrad_spearman": float(np.mean([r["ig_vs_smoothgrad"]["spearman"] for r in report])),
            "mean_spearman_vs_random": float(np.mean([r["sanity_check"]["spearman_vs_random_mean"] for r in report])),
            "max_spearman_vs_random": float(np.max([r["sanity_check"]["spearman_vs_random_max"] for r in report])),
            "mean_abs_convergence_delta": float(np.mean([abs(r["convergence_delta"]) for r in report])),
        },
    }
    (args.output / "attribution_report.json").write_text(json.dumps(summary, indent=2))

    aggregate = summary["aggregate"]
    print("\nAggregate over", len(report), "examples:")
    print(f"  IG vs SmoothGrad agreement (Spearman):     {aggregate['mean_ig_smoothgrad_spearman']:+.3f}")
    print(f"  Trained vs randomised model (Spearman):    {aggregate['mean_spearman_vs_random']:+.3f} "
          f"(max {aggregate['max_spearman_vs_random']:+.3f})")
    print(f"  Mean |convergence delta|:                  {aggregate['mean_abs_convergence_delta']:.2e}")
    print(f"\nWrote {len(report)} figures and attribution_report.json to {args.output}")


if __name__ == "__main__":
    main()
