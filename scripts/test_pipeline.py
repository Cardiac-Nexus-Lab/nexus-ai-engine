"""End-to-end check: signal -> printout -> photograph -> digitized signal -> prediction.

Because the source signal is known, every stage can be scored against ground
truth rather than inspected by eye. Two questions matter and they are different:

* how faithfully is the waveform recovered (correlation and error per lead);
* does the classifier reach the same conclusion from the recovered signal as
  from the original. A digitization can look poor and still preserve the
  diagnosis, or look close and lose it, so neither answers the other.

Usage:
    python scripts/test_pipeline.py --records 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from cardiac_nexus import data
from cardiac_nexus.ecg_image.dataset import StripDataset  # noqa: F401 - shared conventions
from cardiac_nexus.ecg_image.digitize import STRIP_HEIGHT, STRIP_WIDTH
from cardiac_nexus.ecg_image.distort import DistortionConfig, distort
from cardiac_nexus.ecg_image.pipeline import digitize_page, load_localizer
from cardiac_nexus.ecg_image.render import PaperSpec, render
from cardiac_nexus.training import build_model

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_classifier(path: Path, device: torch.device):
    import numpy

    with torch.serialization.safe_globals(
        [numpy._core.multiarray.scalar, numpy.dtype, numpy.dtypes.Float64DType, numpy.dtypes.Int64DType]
    ):
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    classes = checkpoint["classes"]
    model = build_model(checkpoint.get("architecture", "cnn"), len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval(), classes


@torch.no_grad()
def predict(model, signal: np.ndarray, device: torch.device) -> np.ndarray:
    tensor = torch.tensor(np.ascontiguousarray(signal), dtype=torch.float32, device=device).unsqueeze(0)
    return torch.sigmoid(model(tensor)).cpu().numpy()[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=int, default=20)
    parser.add_argument("--localizer", type=Path,
                        default=REPO_ROOT / "results" / "digitizer" / "trace_localizer.pt")
    parser.add_argument("--classifier", type=Path,
                        default=REPO_ROOT / "results" / "local_xresnet18_full" / "ecg_multilabel.pt")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "results" / "pipeline")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    device = torch.device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)

    localizer = load_localizer(args.localizer, device)
    classifier, classes = load_classifier(args.classifier, device)

    metadata, signals, splits = data.prepare()
    rng = np.random.default_rng(args.seed)
    chosen = rng.choice(splits["test"], size=args.records, replace=False)

    # The page geometry is known here, so strip extraction uses it directly. On a
    # real photograph it would have to be detected, which is a separate problem and
    # a separate source of error; keeping it exact isolates what is being measured.
    spec = PaperSpec(layout="12x1", pixels_per_mm=4.0, row_height_mm=16.0, margin_mm=4.0)
    margin_fraction = spec.mm(spec.margin_mm) / (12 * spec.mm(spec.row_height_mm) + 2 * spec.mm(spec.margin_mm))

    rows = []
    for position in chosen:
        truth = np.asarray(signals[position], dtype=np.float32)
        page = render(truth, spec, amplitude_mv_per_unit=0.3)
        photo, _ = distort(cv2.cvtColor(page.array, cv2.COLOR_RGB2BGR),
                           np.random.default_rng(int(position)), DistortionConfig())

        result = digitize_page(photo, localizer, device)
        recovered = result.signal

        # Per-lead correlation against the true waveform.
        correlations = []
        for lead in range(12):
            a, b = truth[lead], recovered[lead][: truth.shape[1]]
            if len(b) < len(a):
                b = np.pad(b, (0, len(a) - len(b)))
            correlations.append(float(np.corrcoef(a, b)[0, 1]) if b.std() > 1e-6 else 0.0)

        aligned = recovered[:, : truth.shape[1]]
        if aligned.shape[1] < truth.shape[1]:
            aligned = np.pad(aligned, ((0, 0), (0, truth.shape[1] - aligned.shape[1])))

        original_prediction = predict(classifier, truth, device)
        recovered_prediction = predict(classifier, aligned, device)

        rows.append(
            {
                "ecg_id": int(metadata.index[position]),
                "mean_correlation": float(np.mean(correlations)),
                "min_correlation": float(np.min(correlations)),
                "mae": float(np.abs(truth - aligned).mean()),
                "confidence": result.mean_confidence,
                "prediction_delta": float(np.abs(original_prediction - recovered_prediction).max()),
                "same_top_class": bool(original_prediction.argmax() == recovered_prediction.argmax()),
                "warnings": result.warnings,
            }
        )

    summary = {
        "records": len(rows),
        "mean_correlation": float(np.mean([r["mean_correlation"] for r in rows])),
        "median_correlation": float(np.median([r["mean_correlation"] for r in rows])),
        "mean_mae": float(np.mean([r["mae"] for r in rows])),
        "mean_confidence": float(np.mean([r["confidence"] for r in rows])),
        "top_class_agreement": float(np.mean([r["same_top_class"] for r in rows])),
        "mean_prediction_delta": float(np.mean([r["prediction_delta"] for r in rows])),
        "per_record": rows,
    }
    (args.output / "pipeline_report.json").write_text(json.dumps(summary, indent=2))

    print(f"\nEnd-to-end over {len(rows)} held-out recordings")
    print(f"  waveform correlation      {summary['mean_correlation']:+.3f} mean, "
          f"{summary['median_correlation']:+.3f} median")
    print(f"  mean absolute error       {summary['mean_mae']:.3f} (standardized units)")
    print(f"  digitizer confidence      {summary['mean_confidence']:.3f}")
    print(f"  top-class agreement       {100 * summary['top_class_agreement']:.0f}%")
    print(f"  max probability shift     {summary['mean_prediction_delta']:.3f} mean")

    # One worked example, so the numbers can be checked against the waveform.
    example = chosen[0]
    truth = np.asarray(signals[example], dtype=np.float32)
    page = render(truth, spec, amplitude_mv_per_unit=0.3)
    photo, _ = distort(cv2.cvtColor(page.array, cv2.COLOR_RGB2BGR),
                       np.random.default_rng(int(example)), DistortionConfig())
    recovered = digitize_page(photo, localizer, device).signal

    fig, axes = plt.subplots(4, 1, figsize=(11, 7), sharex=True)
    for axis, lead in zip(axes, [0, 1, 6, 10]):
        length = min(truth.shape[1], recovered.shape[1])
        axis.plot(truth[lead, :length], color="black", linewidth=0.9, label="original")
        axis.plot(recovered[lead, :length], color="#d1495b", linewidth=0.9, alpha=0.8, label="digitized")
        axis.set_ylabel(f"lead {lead}", fontsize=9)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
    axes[0].legend(fontsize=8, frameon=False, ncol=2)
    axes[-1].set_xlabel("sample")
    fig.suptitle("Original vs digitized from a simulated photograph", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.output / "digitization_example.png", dpi=120, bbox_inches="tight")
    cv2.imwrite(str(args.output / "example_photo.png"), photo)
    print(f"\nWrote report and figures to {args.output}")


if __name__ == "__main__":
    main()
