"""Train and test the health-record model: coronary artery disease from the web form's findings.

Two tests, both on patients the model never trained on:

* internal: a stratified 20% of all four centres, held out before model selection;
* external: leave one centre out. A model is selected and trained on three centres
  and scored on the fourth, once per centre. This is the honest estimate for a
  hospital the model has not seen, and the four centres differ widely (Switzerland
  has 93% disease, Cleveland 46%).

The deployed model is the one trained on the internal training split, so the
internal test numbers describe exactly the model the web portal serves.

Usage:
    python scripts/train_ehr.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import numpy as np
from sklearn.model_selection import train_test_split

from cardiac_nexus import ehr

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = REPO_ROOT / "results" / "ehr"
SEED = 42


def calibration_bins(labels, probabilities, bins: int = 5) -> list[dict]:
    """Observed disease rate in fifths of predicted probability."""
    order = np.argsort(probabilities)
    return [{"mean_predicted": float(probabilities[part].mean()), "observed_rate": float(labels[part].mean()),
             "patients": int(len(part))} for part in np.array_split(order, bins)]


def deployed_probabilities(model, patients, reference) -> np.ndarray:
    """What the portal computes: blanks averaged over the training patients' values."""
    return np.array([ehr.predict(model, patients.iloc[[i]][ehr.FEATURES], reference) for i in range(len(patients))])


def blank_form_check(model, train) -> list[dict]:
    """Only age and sex entered: the result should sit near the training rate for that group,
    not above it, because a blank field carries no information."""
    rows = []
    for sex, label in ((1, "Male"), (0, "Female")):
        for low, high in ((30, 45), (45, 60), (60, 80)):
            group = train[(train["sex"] == sex) & (train["age"] >= low) & (train["age"] < high)]
            row = ehr.form_row((low + high) / 2, label, None, None, None, None, None)
            rows.append({"sex": label, "age": f"{low}-{high - 1}", "patients": int(len(group)),
                         "observed_rate": float(group["cad"].mean()) if len(group) else None,
                         "predicted": ehr.predict(model, row, train[ehr.FEATURES])})
    return rows


def main() -> None:
    table = ehr.fetch()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    print(f"{len(table)} patients; disease rate {table['cad'].mean():.0%}")

    cleaning = {
        site: {"patients": int(len(part)), "disease_rate": float(part["cad"].mean()),
               "missing": {f: int(part[f].isna().sum()) for f in ehr.FEATURES if part[f].isna().any()}}
        for site, part in table.groupby("site")
    }

    strata = table["site"] + "_" + table["cad"].astype(str)
    train, test = train_test_split(table, test_size=0.2, stratify=strata, random_state=SEED)
    name, model, cross_validation = ehr.select_model(train[ehr.FEATURES], train["cad"], SEED)
    test_probability = deployed_probabilities(model, test, train[ehr.FEATURES])
    internal = ehr.scores(test["cad"].to_numpy(), test_probability, SEED)
    internal["calibration"] = calibration_bins(test["cad"].to_numpy(), test_probability)
    print(f"selected {name}; internal test AUROC {internal['auroc']:.3f} on {internal['patients']} patients")

    external = {}
    for site in ehr.SITES.values():
        seen, unseen = table[table["site"] != site], table[table["site"] == site]
        chosen, site_model, _ = ehr.select_model(seen[ehr.FEATURES], seen["cad"], SEED)
        probability = deployed_probabilities(site_model, unseen, seen[ehr.FEATURES])
        external[site] = {"selected_model": chosen, **ehr.scores(unseen["cad"].to_numpy(), probability, SEED)}
        auroc = external[site]["auroc"]
        print(f"held-out centre {site}: AUROC {auroc:.3f}" if auroc is not None else f"held-out centre {site}: one class")

    # The reference is the training patients' seven findings (public, de-identified), needed to
    # average over blank fields at prediction time.
    joblib.dump({"model": model, "features": ehr.FEATURES, "selected_model": name,
                 "reference": train[ehr.FEATURES].reset_index(drop=True),
                 "training_disease_rate": float(train["cad"].mean())}, OUTPUT / "ehr_model.joblib")
    blank_check = blank_form_check(model, train)
    for item in blank_check:
        print(f"  only age+sex {item['sex']} {item['age']}: predicted {item['predicted']:.2f}, "
              f"observed {item['observed_rate']:.2f} (n={item['patients']})")
    report = {
        "dataset": "UCI Heart Disease (Cleveland, Hungary, Switzerland, VA Long Beach); CC BY 4.0",
        "outcome": "angiographic coronary artery disease (>50% narrowing of a major vessel)",
        "features": ehr.FEATURES,
        "cleaning": cleaning,
        "split": {"train": int(len(train)), "test": int(len(test)), "seed": SEED},
        "selected_model": name,
        "cross_validation": cross_validation,
        "internal_test": internal,
        "leave_one_centre_out": external,
        "blank_form_check": blank_check,
    }
    (OUTPUT / "ehr_report.json").write_text(json.dumps(report, indent=2))
    print(f"Wrote {OUTPUT / 'ehr_model.joblib'} and ehr_report.json")


if __name__ == "__main__":
    main()
