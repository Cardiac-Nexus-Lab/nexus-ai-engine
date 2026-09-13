"""Explain the MRI segmenter and check whether the explanations mean anything.

Usage:
    python scripts/explain_mri.py --split test
    python scripts/explain_mri.py --split val --patients 4 --device cpu   # quick check
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from cardiac_nexus import mri_data
from cardiac_nexus.models import select_device
from cardiac_nexus.mri_dataset import ACDCSlices, normalize_volumes
from cardiac_nexus.mri_explain import (
    heart_region,
    localisation,
    map_similarity,
    randomized_copy,
    seg_grad_cam,
    uncertainty_checks,
    uncertainty_for_patients,
)
from evaluate_mri import load_segmenter

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=REPO_ROOT / "results" / "mri_segmentation" / "mri_segmenter.pt")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--patients", type=int, default=None, help="limit, for a quick check")
    parser.add_argument("--device", default=None)
    parser.add_argument("--threads", type=int, default=None, help="cap CPU threads")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    output = args.output or args.checkpoint.parent / "explainability"
    output.mkdir(parents=True, exist_ok=True)

    device = select_device(args.device)
    model, checkpoint = load_segmenter(args.checkpoint, device)
    metadata, cache, _ = mri_data.prepare()
    patients = np.asarray(checkpoint["splits"][args.split])[: args.patients]
    images = normalize_volumes(cache["images"], cache["patient"], cache["phase"])
    masks, patient, phase = cache["masks"], cache["patient"], cache["phase"]

    rows, predictions, entropy = uncertainty_for_patients(model, images, masks, patient, phase, patients, device)
    uncertainty = uncertainty_checks(rows, predictions, entropy, masks, patient, phase)

    random_model = randomized_copy(model).to(device)
    layer, random_layer = model.encoder.stages[-1], random_model.encoder.stages[-1]
    dataset = ACDCSlices(images, masks, patient, phase, patients, augment=False)
    similarities, inside, baseline, examples = [], [], [], []
    for index in patients:
        slices = np.flatnonzero((patient[dataset.rows] == index) & (phase[dataset.rows] == 0))
        item = slices[len(slices) // 2]
        inputs = dataset[item][0][None].to(device)
        row = dataset.rows[item]
        heart = heart_region(masks[row][None], masks[row][None], margin=5)[0]
        cams = {}
        for label, name in mri_data.STRUCTURES.items():
            cam = seg_grad_cam(model, inputs, label, layer)
            random_cam = seg_grad_cam(random_model, inputs, label, random_layer)
            similarities.append(map_similarity(cam, random_cam))
            placement = localisation(cam, heart)
            inside.append(placement["mass_inside_heart"])
            baseline.append(placement["uniform_map_baseline"])
            cams[name] = cam
        if len(examples) < 3:
            examples.append((index, row, cams))

    finite = [s for s in similarities if np.isfinite(s)]
    report = {
        "checkpoint": str(args.checkpoint), "checkpoint_epoch": checkpoint["epoch"],
        "split": args.split, "patients": int(len(patients)),
        "uncertainty": uncertainty,
        "seg_grad_cam": {
            "layer": "encoder bottleneck",
            "maps": len(similarities),
            "spearman_trained_vs_randomized_mean": float(np.mean(finite)) if finite else float("nan"),
            "spearman_trained_vs_randomized_max": float(np.max(finite)) if finite else float("nan"),
            "mass_inside_heart_mean": float(np.nanmean(inside)),
            "uniform_map_baseline_mean": float(np.mean(baseline)),
        },
        "reading": {
            "pixel_error_detection_auroc": "0.5 means uncertainty is unrelated to error; higher is better.",
            "spearman_trained_vs_randomized": "Near zero passes: the map depends on learned weights.",
            "mass_inside_heart": "Should clearly exceed the uniform baseline.",
        },
    }
    (output / f"explainability_{args.split}.json").write_text(json.dumps(report, indent=2))

    u, g = report["uncertainty"], report["seg_grad_cam"]
    print(f"\nExplainability on {len(patients)} {args.split} patients (checkpoint epoch {checkpoint['epoch']})")
    print(f"  uncertainty -> error, pixel AUROC          {u['pixel_error_detection_auroc']:.3f}")
    print(f"  volume uncertainty vs Dice error, Spearman {u['volume_uncertainty_vs_dice_error_spearman']:+.3f} "
          f"(p={u['volume_uncertainty_vs_dice_error_p']:.3g}, {u['volumes']} volumes)")
    print(f"  Grad-CAM trained vs randomized, Spearman   {g['spearman_trained_vs_randomized_mean']:+.3f} "
          f"(max {g['spearman_trained_vs_randomized_max']:+.3f})")
    print(f"  Grad-CAM mass inside heart                 {g['mass_inside_heart_mean']:.3f} "
          f"(uniform map would give {g['uniform_map_baseline_mean']:.3f})")

    _figure(examples, images, masks, rows, predictions, entropy, metadata, output / f"explainability_{args.split}.png")
    print(f"Wrote {output}")


def _figure(examples, images, masks, rows, predictions, entropy, metadata, path: Path) -> None:
    colours = {1: "#e4572e", 2: "#29335c", 3: "#17bebb"}
    fig, axes = plt.subplots(len(examples), 4, figsize=(12.5, 3.3 * len(examples)), squeeze=False)
    position = {int(r): i for i, r in enumerate(rows)}
    for line, (index, row, cams) in enumerate(examples):
        image = images[row]
        panels = [("MRI slice", None), ("model segmentation", predictions[position[row]]),
                  ("uncertainty (TTA entropy)", entropy[position[row]]), ("Seg-Grad-CAM, LV", cams["LV"])]
        for column, (title, overlay) in enumerate(panels):
            axis = axes[line, column]
            axis.imshow(image, cmap="gray")
            if column == 1:
                for label, colour in colours.items():
                    if (masks[row] == label).any():
                        axis.contour(masks[row] == label, levels=[0.5], colors="white", linewidths=0.8)
                    if (overlay == label).any():
                        axis.contour(overlay == label, levels=[0.5], colors=colour, linewidths=1.3)
            elif overlay is not None:
                axis.imshow(overlay, cmap="magma", alpha=0.55)
            axis.set_title(f"{metadata.loc[index, 'pid']} ({metadata.loc[index, 'pathology']}): {title}"
                           if column == 0 else title, fontsize=9)
            axis.axis("off")
    fig.suptitle("White: expert outline. Colour: model. Bright: where the model is unsure or looking.", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
