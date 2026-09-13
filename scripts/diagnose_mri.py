"""Classify ACDC diagnosis from cardiac measurements.

The classifier is trained on measurements from expert masks of the 100 training
patients, choosing between a few low-capacity models by repeated cross-validation.
Test patients are touched only with --evaluate-test, and then scored twice:

* measurements from expert masks: the ceiling, what perfect segmentation allows;
* measurements from the segmenter's own masks: the real pipeline.

Usage:
    python scripts/diagnose_mri.py                      # cross-validation only
    python scripts/diagnose_mri.py --evaluate-test      # one look at the test set
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix

from cardiac_nexus import mri_data
from cardiac_nexus.models import select_device
from cardiac_nexus.mri_diagnosis import (
    FEATURES,
    explain_patient,
    feature_table,
    reference_ranges,
    select_model,
    wilson_interval,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def expert_masks(cache):
    def masks_for(index, phase):
        selected = (cache["patient"] == index) & (cache["phase"] == phase)
        return cache["masks"][selected], tuple(float(s) for s in cache["spacing"][selected][0])
    return masks_for


def score(model, table: pd.DataFrame, ranges: dict) -> dict:
    predicted = model.predict(table[list(FEATURES)])
    correct = int((predicted == table["pathology"].to_numpy()).sum())
    low, high = wilson_interval(correct, len(table))
    classes = mri_data.PATHOLOGIES
    matrix = confusion_matrix(table["pathology"], predicted, labels=classes)
    return {
        "accuracy": correct / len(table), "correct": correct, "n": len(table),
        "accuracy_95ci": [low, high],
        "confusion_matrix": {"labels": classes, "rows_true_columns_predicted": matrix.tolist()},
        "per_patient": [
            {"pid": row["pid"], "true": row["pathology"], "predicted": str(p),
             "explanation": explain_patient(row, ranges)}
            for (_, row), p in zip(table.iterrows(), predicted)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--checkpoint", type=Path,
                        default=REPO_ROOT / "results" / "mri_segmentation" / "mri_segmenter.pt")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "results" / "mri_diagnosis")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    metadata, cache, splits = mri_data.prepare()
    required = metadata[["height", "weight"]]
    assert required.notna().all().all(), "height and weight are needed for body surface area"

    pool = np.concatenate([splits["train"], splits["val"]])
    train_table = feature_table(metadata, pool, expert_masks(cache))
    name, model, results = select_model(train_table[list(FEATURES)], train_table["pathology"], args.seed)
    ranges = reference_ranges(train_table)

    print(f"Cross-validation on {len(train_table)} training patients (5 folds x 10 repeats):")
    for candidate, value in results.items():
        marker = "  <- selected" if candidate == name else ""
        print(f"  {candidate:15} accuracy {value['mean_accuracy']:.3f} ± {value['sd_accuracy']:.3f}{marker}")

    report = {"selected_model": name, "cross_validation": results, "normal_reference_ranges": ranges,
              "features": {k: {"label": v[0], "unit": v[1]} for k, v in FEATURES.items()}}

    if args.evaluate_test:
        test = splits["test"]
        report["test_expert_masks"] = score(model, feature_table(metadata, test, expert_masks(cache)), ranges)
        if args.checkpoint.exists():
            from cardiac_nexus.mri_dataset import normalize_volumes
            from cardiac_nexus.mri_training import predict_volumes
            sys.path.insert(0, str(REPO_ROOT / "scripts"))
            from evaluate_mri import load_segmenter

            device = select_device(None)
            segmenter, checkpoint = load_segmenter(args.checkpoint, device)
            assert sorted(checkpoint["splits"]["test"]) == sorted(int(i) for i in test), "split mismatch"
            images = normalize_volumes(cache["images"], cache["patient"], cache["phase"])
            volumes = predict_volumes(segmenter, images, cache["masks"], cache["patient"], cache["phase"],
                                      cache["spacing"], test, device)
            predicted_masks = lambda index, phase: (volumes[(int(index), phase)][0], volumes[(int(index), phase)][2])
            report["test_model_masks"] = score(model, feature_table(metadata, test, predicted_masks), ranges)
            report["segmenter_checkpoint_epoch"] = checkpoint["epoch"]

        for key in ("test_expert_masks", "test_model_masks"):
            if key in report:
                r = report[key]
                print(f"\n{key}: {r['correct']}/{r['n']} = {r['accuracy']:.1%} "
                      f"(95% CI {r['accuracy_95ci'][0]:.1%}-{r['accuracy_95ci'][1]:.1%})")
                print(pd.DataFrame(r["confusion_matrix"]["rows_true_columns_predicted"],
                                   index=mri_data.PATHOLOGIES, columns=mri_data.PATHOLOGIES))

    args.output.mkdir(parents=True, exist_ok=True)
    name_suffix = "report" if args.evaluate_test else "cross_validation"
    (args.output / f"diagnosis_{name_suffix}.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"\nWrote {args.output / f'diagnosis_{name_suffix}.json'}")


if __name__ == "__main__":
    main()
