"""Likelihood ratios of the MRI diagnosis classifier, for combining with the ECG.

No public dataset pairs ECG and MRI for the same patients, so no fusion model can
be trained or tested. What can be estimated is how much an MRI finding should
shift belief in a diagnosis the ECG also speaks to, under the assumption that,
given the true diagnosis, the ECG and MRI findings are independent. That
assumption is not testable here and is stated wherever the result is shown.

Only diagnoses with a counterpart in both label sets are used:

* ECG MI (myocardial infarction) and MRI MINF (previous infarction);
* ECG HYP (hypertrophy) and MRI HCM (hypertrophic cardiomyopathy). This pairing
  is loose: ECG hypertrophy also arises from hypertension and valve disease, so
  an MRI that does not call HCM says little about hypertrophy in general. For
  this pair only the positive direction is applied (`update_when_not_called`
  false): an MRI HCM call raises the ECG probability, and its absence leaves it
  unchanged.

For each MRI class the classifier's out-of-fold predictions on the ACDC training
patients (measurements from expert masks, five stratified folds) give its
sensitivity and specificity, and from them

    LR+ = sensitivity / (1 - specificity)    (MRI calls the class)
    LR- = (1 - sensitivity) / specificity    (MRI calls another class)

with 0.5 added to every cell of the 2x2 table (Haldane's correction), because
with 20 patients per class a perfect cell otherwise gives an infinite ratio.
95% intervals use the usual log method. The held-out test patients are not used.

Usage:
    python scripts/fusion_likelihood_ratios.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from cardiac_nexus import mri_data
from cardiac_nexus.mri_diagnosis import FEATURES, candidate_models, feature_table
from diagnose_mri import expert_masks

PAIRS = {"MINF": ("MI", True), "HCM": ("HYP", False)}


def likelihood_ratios(true_positive, false_negative, false_positive, true_negative) -> dict:
    tp, fn, fp, tn = (x + 0.5 for x in (true_positive, false_negative, false_positive, true_negative))
    sensitivity, specificity = tp / (tp + fn), tn / (tn + fp)
    positive, negative = sensitivity / (1 - specificity), (1 - sensitivity) / specificity
    se_positive = np.sqrt(1 / tp - 1 / (tp + fn) + 1 / fp - 1 / (fp + tn))
    se_negative = np.sqrt(1 / fn - 1 / (tp + fn) + 1 / tn - 1 / (fp + tn))
    return {
        "counts": {"tp": int(true_positive), "fn": int(false_negative),
                   "fp": int(false_positive), "tn": int(true_negative)},
        "sensitivity": float(sensitivity), "specificity": float(specificity),
        "lr_positive": float(positive),
        "lr_positive_95ci": [float(positive * np.exp(-1.96 * se_positive)),
                             float(positive * np.exp(1.96 * se_positive))],
        "lr_negative": float(negative),
        "lr_negative_95ci": [float(negative * np.exp(-1.96 * se_negative)),
                             float(negative * np.exp(1.96 * se_negative))],
    }


def main() -> None:
    metadata, cache, splits = mri_data.prepare()
    pool = np.concatenate([splits["train"], splits["val"]])
    table = feature_table(metadata, pool, expert_masks(cache))
    model = candidate_models(42)["logistic_C10"]
    predicted = cross_val_predict(model, table[list(FEATURES)], table["pathology"],
                                  cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42))
    truth = table["pathology"].to_numpy()

    report = {
        "assumption": "ECG and MRI findings are independent given the true diagnosis; untested, "
                      "because no public dataset pairs ECG and MRI for the same patients.",
        "source": f"out-of-fold predictions, logistic_C10, {len(table)} ACDC training patients, expert masks",
        "pairs": {},
    }
    for mri_class, (ecg_class, both_directions) in PAIRS.items():
        called, actual = predicted == mri_class, truth == mri_class
        report["pairs"][mri_class] = {"ecg_class": ecg_class, "update_when_not_called": both_directions,
                                      **likelihood_ratios(
            (called & actual).sum(), (~called & actual).sum(), (called & ~actual).sum(), (~called & ~actual).sum())}
        entry = report["pairs"][mri_class]
        print(f"{mri_class} (pairs with ECG {ecg_class}): sens {entry['sensitivity']:.2f} spec {entry['specificity']:.2f}"
              f"  LR+ {entry['lr_positive']:.1f} {entry['lr_positive_95ci']}  LR- {entry['lr_negative']:.2f}"
              f" {entry['lr_negative_95ci']}")

    output = REPO_ROOT / "results" / "fusion"
    output.mkdir(parents=True, exist_ok=True)
    (output / "likelihood_ratios.json").write_text(json.dumps(report, indent=2))
    print(f"Wrote {output / 'likelihood_ratios.json'}")


if __name__ == "__main__":
    main()
