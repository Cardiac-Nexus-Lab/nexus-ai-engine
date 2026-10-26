"""Coronary artery disease from routine clinical findings: the health-record branch.

Data: the UCI Heart Disease collection (Janosi, Steinbrunn, Pfisterer and Detrano,
1988; CC BY 4.0), 920 patients referred for coronary angiography at four centres:
Cleveland Clinic, the Hungarian Institute of Cardiology in Budapest, University
Hospitals of Zurich and Basel, and the VA Medical Center in Long Beach. The outcome
is angiographic disease: more than 50% narrowing of any major coronary artery.

Only the findings the web form asks for are used (age, sex, chest-pain type,
resting blood pressure, cholesterol, maximum heart rate on exercise, exercise
angina), so the model sees in testing exactly what it will see in use. Because
every patient was referred for angiography, about half have disease; a predicted
probability is a probability among such referred patients, not in the general
population.

Four centres allow the test that matters for a deployed model: train on three and
test on the fourth, which it never saw.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "uci_heart"
UCI_BASE = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease"
SITES = {"cleveland": "Cleveland", "hungarian": "Hungary", "switzerland": "Switzerland", "va": "VA Long Beach"}
COLUMNS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope",
           "ca", "thal", "num"]

# The form's fields, in UCI coding. cp: 1 typical angina, 2 atypical angina,
# 3 non-anginal pain, 4 asymptomatic. sex and exang: 1 male / yes.
NUMERIC = ["age", "trestbps", "chol", "thalach"]
CATEGORICAL = ["sex", "cp", "exang"]
FEATURES = NUMERIC + CATEGORICAL
LABELS = {
    "age": "Age", "trestbps": "Resting blood pressure", "chol": "Cholesterol",
    "thalach": "Maximum heart rate on exercise", "sex": "Sex", "cp": "Chest-pain type", "exang": "Exercise angina",
}
CHEST_PAIN = {"typical angina": 1, "atypical angina": 2, "non-anginal pain": 3, "asymptomatic": 4}


def fetch(root: Path = DATA_ROOT) -> pd.DataFrame:
    """Download the four centres' processed files and return one cleaned table."""
    root.mkdir(parents=True, exist_ok=True)
    frames = []
    for site, name in SITES.items():
        path = root / f"processed.{site}.data"
        if not path.exists():
            urllib.request.urlretrieve(f"{UCI_BASE}/processed.{site}.data", path)
        frame = pd.read_csv(path, header=None, names=COLUMNS, na_values="?")
        frame["site"] = name
        frames.append(frame)
    return clean(pd.concat(frames, ignore_index=True))


def clean(table: pd.DataFrame) -> pd.DataFrame:
    """Mark the dataset's coded-as-zero missing values and derive the outcome.

    Switzerland recorded every cholesterol, and VA many, as 0; one Hungarian
    resting blood pressure is 0. Neither is a measurement, so both become missing
    rather than being learned as "very low cholesterol means disease".
    """
    table = table.copy()
    table.loc[table["chol"] <= 0, "chol"] = np.nan
    table.loc[table["trestbps"] <= 0, "trestbps"] = np.nan
    table["cad"] = (table["num"] > 0).astype(int)
    return table.dropna(subset=["age", "sex", "num"])


def form_row(age, gender, blood_pressure, cholesterol, max_heart_rate, chest_pain, exercise_angina) -> pd.DataFrame:
    """One web-form submission in the model's coding; blanks stay missing."""
    pain = CHEST_PAIN.get((chest_pain or "").strip().lower())
    if pain is None and "asymptomatic" in (chest_pain or "").lower():
        pain = 4
    angina = {"yes": 1, "no": 0}.get((exercise_angina or "").strip().lower())
    sex = {"male": 1, "female": 0}.get((gender or "").strip().lower())
    value = lambda v: np.nan if v is None else float(v)
    return pd.DataFrame([{
        "age": float(age), "trestbps": value(blood_pressure), "chol": value(cholesterol),
        "thalach": value(max_heart_rate), "sex": value(sex), "cp": value(pain), "exang": value(angina),
    }])[FEATURES]


def candidate_models(seed: int) -> dict:
    """Low-capacity logistic models.

    Missingness is not given to the model: in this collection it marks the centre
    (Switzerland recorded no cholesterol and has 93% disease), so a missing-value
    flag, or a tree model that splits on missing, learns "blank means disease".
    Training gaps are filled with the median; blanks at prediction time are
    averaged over instead (see predict).
    """
    def logistic(c):
        prepare = ColumnTransformer([
            ("numeric", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUMERIC),
            ("categorical", make_pipeline(SimpleImputer(strategy="most_frequent"),
                                          OneHotEncoder(handle_unknown="ignore")), CATEGORICAL),
        ])
        return make_pipeline(prepare, LogisticRegression(C=c, max_iter=5000))
    return {"logistic_C0.1": logistic(0.1), "logistic_C1": logistic(1.0), "logistic_C10": logistic(10.0)}


def predict(model, row: pd.DataFrame, reference: pd.DataFrame) -> float:
    """Probability given only the entered findings.

    Each blank field is filled in turn with the values real training patients had,
    and the predictions averaged: the expected probability over what is unknown,
    instead of one invented value. Missing training values are dropped from the
    reference, so only measured values are averaged over.
    """
    blank = [f for f in FEATURES if pd.isna(row[f].iloc[0])]
    if not blank:
        return float(model.predict_proba(row)[0, 1])
    fills = reference[blank].dropna()
    filled = pd.concat([row.drop(columns=blank)] * len(fills), ignore_index=True)
    filled[blank] = fills.to_numpy()
    return float(model.predict_proba(filled[FEATURES])[:, 1].mean())


def select_model(features: pd.DataFrame, labels: pd.Series, seed: int = 42):
    """Choose by repeated stratified cross-validation on training patients only."""
    splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=seed)
    results = {}
    for name, model in candidate_models(seed).items():
        scores = cross_val_score(model, features, labels, cv=splitter, scoring="roc_auc")
        results[name] = {"mean_auroc": float(scores.mean()), "sd_auroc": float(scores.std())}
    best = max(results, key=lambda name: results[name]["mean_auroc"])
    return best, candidate_models(seed)[best].fit(features, labels), results


def scores(labels: np.ndarray, probabilities: np.ndarray, seed: int = 42, resamples: int = 2000) -> dict:
    """AUROC and Brier score with 95% bootstrap intervals over patients, plus calibration in the large."""
    rng = np.random.default_rng(seed)
    labels, probabilities = np.asarray(labels), np.asarray(probabilities)
    aurocs, briers = [], []
    for _ in range(resamples):
        pick = rng.integers(0, len(labels), len(labels))
        if labels[pick].min() == labels[pick].max():
            continue
        aurocs.append(roc_auc_score(labels[pick], probabilities[pick]))
        briers.append(brier_score_loss(labels[pick], probabilities[pick]))
    single_class = labels.min() == labels.max()
    return {
        "patients": int(len(labels)),
        "with_disease": int(labels.sum()),
        "auroc": None if single_class else float(roc_auc_score(labels, probabilities)),
        "auroc_95ci": None if single_class else [float(np.percentile(aurocs, 2.5)), float(np.percentile(aurocs, 97.5))],
        "brier": float(brier_score_loss(labels, probabilities)),
        "brier_95ci": [float(np.percentile(briers, 2.5)), float(np.percentile(briers, 97.5))] if briers else None,
        "observed_rate": float(labels.mean()),
        "mean_predicted": float(probabilities.mean()),
    }


def explain(model, row: pd.DataFrame, reference: pd.DataFrame, top: int = 3) -> list[str]:
    """Which entered findings moved this patient's probability most, by leaving each out.

    Each entered field in turn is treated as unknown (averaged over, as in predict),
    and the change in probability is that field's contribution: how much knowing
    it changed this patient's result compared with not knowing it.
    """
    base = predict(model, row, reference)
    effects = []
    for name in FEATURES:
        if pd.isna(row[name].iloc[0]):
            continue
        without = row.copy()
        without[name] = np.nan
        change = base - predict(model, without, reference)
        effects.append((abs(change), f"{LABELS[name]}: {'raises' if change > 0 else 'lowers'} the probability by "
                                     f"{abs(change) * 100:.0f} points"))
    return [text for size, text in sorted(effects, reverse=True)[:top] if size >= 0.005]
