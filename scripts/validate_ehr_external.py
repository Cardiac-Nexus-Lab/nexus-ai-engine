"""External test of the health-record model at a fifth hospital: Z-Alizadeh Sani (Tehran).

Data: UCI dataset 412 (CC BY 4.0), 303 patients at the Shahid Rajaei Cardiovascular
Center who had coronary angiography; outcome "Cad" when a vessel is narrowed by 50%
or more, the same definition as UCI Heart Disease. No patient is shared with the
four training centres.

The deployed model (results/ehr/ehr_model.joblib) is used unchanged. Only four of
its seven inputs exist here: age, sex, resting blood pressure and chest-pain type
(three yes/no columns; none of them means asymptomatic). Cholesterol is recorded
only as LDL/HDL/triglycerides, there is no exercise test (resting pulse is not the
maximum heart rate on exercise) and "exertional chest pain" is "N" for every
patient, so those three are left blank and averaged over, as the portal does.

To separate the effect of a new hospital from the effect of having fewer fields,
the UCI internal test patients are also scored with the same four fields only.

Usage:
    python scripts/validate_ehr_external.py
"""

from __future__ import annotations

import io
import json
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from cardiac_nexus import ehr

URL = "https://archive.ics.uci.edu/static/public/412/z+alizadeh+sani.zip"
DATA = REPO_ROOT / "data" / "z_alizadeh_sani"
OUTPUT = REPO_ROOT / "results" / "ehr_external"
MODEL = REPO_ROOT / "results" / "ehr" / "ehr_model.joblib"
USED = ["age", "sex", "trestbps", "cp"]
SEED = 42


def read_sheet(data: bytes) -> pd.DataFrame:
    """The first worksheet of an .xlsx file, read from its XML (no spreadsheet library needed)."""
    book = zipfile.ZipFile(io.BytesIO(data))
    strings = [re.sub(r"<[^>]+>", "", s) for s in
               re.findall(r"<si>(.*?)</si>", book.read("xl/sharedStrings.xml").decode(), re.S)]
    sheet = book.read("xl/worksheets/sheet1.xml").decode()
    rows = []
    for row in re.findall(r"<row [^>]*>(.*?)</row>", sheet, re.S):
        cells = {}
        for column, attrs, value in re.findall(r'<c r="([A-Z]+)\d+"([^>]*)>(?:<v>(.*?)</v>)?', row):
            cells[column] = strings[int(value)] if 't="s"' in attrs and value else value
        rows.append(cells)
    header = rows[0]
    return pd.DataFrame([{header[c]: r.get(c) for c in header} for r in rows[1:]])


def fetch() -> pd.DataFrame:
    DATA.mkdir(parents=True, exist_ok=True)
    archive = DATA / "z_alizadeh_sani.zip"
    if not archive.exists():
        urllib.request.urlretrieve(URL, archive)
    with zipfile.ZipFile(archive) as outer:
        name = next(n for n in outer.namelist() if n.endswith(".xlsx"))
        return read_sheet(outer.read(name))


def to_model_inputs(table: pd.DataFrame) -> pd.DataFrame:
    def chest_pain(row) -> float:
        if row["Typical Chest Pain"] == "1":
            return 1.0
        if row["Atypical"] == "Y":
            return 2.0
        if row["Nonanginal"] == "Y":
            return 3.0
        return 4.0
    return pd.DataFrame({
        "age": table["Age"].astype(float),
        "trestbps": table["BP"].astype(float),
        "chol": np.nan,
        "thalach": np.nan,
        "sex": table["Sex"].str.strip().str.lower().map({"male": 1.0, "fmale": 0.0, "female": 0.0}),
        "cp": table.apply(chest_pain, axis=1),
        "exang": np.nan,
    })[ehr.FEATURES]


def deployed(bundle, patients: pd.DataFrame) -> np.ndarray:
    return np.array([ehr.predict(bundle["model"], patients.iloc[[i]], bundle["reference"])
                     for i in range(len(patients))])


def calibration_bins(labels, probabilities, bins: int = 5) -> list[dict]:
    order = np.argsort(probabilities)
    return [{"mean_predicted": float(probabilities[part].mean()), "observed_rate": float(labels[part].mean()),
             "patients": int(len(part))} for part in np.array_split(order, bins)]


def main() -> None:
    bundle = joblib.load(MODEL)  # our own artefact, written by scripts/train_ehr.py
    table = fetch()
    patients = to_model_inputs(table)
    labels = (table["Cath"].str.strip().str.lower() == "cad").astype(int).to_numpy()
    probability = deployed(bundle, patients)
    external = {**ehr.scores(labels, probability, SEED), "calibration": calibration_bins(labels, probability),
                "chest_pain_counts": patients["cp"].value_counts().sort_index().to_dict()}

    # The UCI internal test patients exactly as train_ehr.py split them, with only the same four fields.
    uci = ehr.fetch()
    strata = uci["site"] + "_" + uci["cad"].astype(str)
    _, test = train_test_split(uci, test_size=0.2, stratify=strata, random_state=SEED)
    four_only = test[ehr.FEATURES].copy()
    four_only[[f for f in ehr.FEATURES if f not in USED]] = np.nan
    internal_all = ehr.scores(test["cad"].to_numpy(), deployed(bundle, test[ehr.FEATURES]), SEED)
    internal_four = ehr.scores(test["cad"].to_numpy(), deployed(bundle, four_only), SEED)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    report = {
        "dataset": "Z-Alizadeh Sani, UCI 412 (CC BY 4.0); 303 angiography patients, Tehran",
        "outcome": "angiographic coronary artery disease (>=50% narrowing), as in UCI Heart Disease",
        "fields_used": USED, "fields_missing_averaged_over": [f for f in ehr.FEATURES if f not in USED],
        "external_z_alizadeh_sani": external,
        "uci_internal_test_all_seven_fields": internal_all,
        "uci_internal_test_same_four_fields": internal_four,
    }
    (OUTPUT / "ehr_external_report.json").write_text(json.dumps(report, indent=2))
    for name, result in (("Z-Alizadeh Sani (4 fields)", external), ("UCI test, 7 fields", internal_all),
                         ("UCI test, same 4 fields", internal_four)):
        print(f"{name}: AUROC {result['auroc']:.3f} ({result['auroc_95ci'][0]:.3f}-{result['auroc_95ci'][1]:.3f}), "
              f"Brier {result['brier']:.3f}, observed {result['observed_rate']:.2f} vs predicted {result['mean_predicted']:.2f}")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
