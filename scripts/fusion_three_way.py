"""Settings for combining the health-record, ECG and MRI models into one probability.

No public dataset holds health records, an ECG and a cardiac MRI for the same
patients, so no fusion model can be trained or tested. Each model is validated on
its own data; this combines their outputs by Bayes' rule for one target they all
speak to, ischaemic heart disease:

    prior          = health-record model's probability of coronary artery disease
    ECG evidence   = likelihood ratio of the ECG's myocardial-infarction probability
    MRI evidence   = likelihood ratio of the MRI classifier calling previous infarction
    posterior odds = prior odds x ECG ratio x MRI ratio

The ECG ratio comes from its calibrated probability p: calibration was fitted on
PTB-XL fold 9, so p is a posterior at that fold's MI rate r, and
LR = [p / (1 - p)] / [r / (1 - r)]. The MRI ratios are the out-of-fold ones from
scripts/fusion_likelihood_ratios.py.

Two assumptions, neither testable here and both shown wherever the result is:
the three findings are independent given the true state, and evidence about
infarction (ECG, MRI) carries over to coronary disease as the health-record model
defines it (angiographic narrowing). Probabilities are clipped to 2-98% before
conversion, so one confident model cannot alone push the result to certainty.

Usage:
    python scripts/fusion_three_way.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cardiac_nexus import data

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = REPO_ROOT / "results" / "fusion" / "three_way.json"


def main() -> None:
    metadata = data.fetch_metadata()
    calibration_fold = metadata[metadata["strat_fold"] == 9]
    rate = float(calibration_fold["MI"].mean())
    ratios = json.loads((REPO_ROOT / "results" / "fusion" / "likelihood_ratios.json").read_text())
    minf = ratios["pairs"]["MINF"]
    settings = {
        "target": "ischaemic heart disease (coronary artery disease or previous infarction)",
        "prior": {"model": "results/ehr/ehr_model.joblib", "outcome": "angiographic coronary artery disease"},
        "ecg": {"class": "MI", "calibration_fold": 9, "calibration_records": int(len(calibration_fold)),
                "calibration_rate": rate},
        "mri": {"class": "MINF", "lr_positive": minf["lr_positive"], "lr_positive_95ci": minf["lr_positive_95ci"],
                "lr_negative": minf["lr_negative"], "lr_negative_95ci": minf["lr_negative_95ci"],
                "source": ratios["source"]},
        "probability_clip": [0.02, 0.98],
        "assumptions": [
            "The health-record, ECG and MRI findings are independent given the true state.",
            "ECG and MRI evidence of infarction applies to coronary artery disease as the health-record model "
            "defines it (angiographic narrowing).",
            "No dataset with all three for the same patients exists publicly, so the combined probability is "
            "not validated.",
        ],
    }
    OUTPUT.write_text(json.dumps(settings, indent=2))
    print(f"MI rate in calibration fold 9: {rate:.3f} of {len(calibration_fold)} records")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
