"""Diagnosis from cardiac measurements rather than from raw pixels.

Follows up the ACDC diagnosis experiment in notebooks/MRI.ipynb by Sahana NS,
which trained a ResNet-18 directly on the scans. It reached 98.6% on its 70
training patients and 7 of 15 on validation: an 11-million-parameter network
given 70 examples memorises them.

The teams that did best at ACDC diagnosis segmented the heart first and
classified on the resulting measurements (Khened et al., Medical Image Analysis
2019). A few physiologically meaningful numbers leave a low-capacity classifier
little to memorise, and every prediction can be explained in the units a
cardiologist reads: millilitres, grams and percent.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .mri_data import structure_volumes_ml

MYOCARDIAL_DENSITY_G_PER_ML = 1.05

# Name -> (readable label, unit). Volumes and mass are indexed to body surface
# area, as in clinical reporting, so a large patient is not read as dilated.
FEATURES = {
    "lv_edv_index": ("LV end-diastolic volume index", "mL/m²"),
    "lv_esv_index": ("LV end-systolic volume index", "mL/m²"),
    "lv_ef": ("LV ejection fraction", "%"),
    "rv_edv_index": ("RV end-diastolic volume index", "mL/m²"),
    "rv_esv_index": ("RV end-systolic volume index", "mL/m²"),
    "rv_ef": ("RV ejection fraction", "%"),
    "myo_mass_index": ("LV myocardial mass index", "g/m²"),
    "mass_to_volume": ("LV mass-to-volume ratio", "g/mL"),
    "rv_to_lv_edv": ("RV/LV end-diastolic volume ratio", ""),
    "lv_ef_minus_rv_ef": ("LV EF minus RV EF", "points"),
}


def body_surface_area_m2(height_m: float, weight_kg: float) -> float:
    """Mosteller formula."""
    return math.sqrt(height_m * 100.0 * weight_kg / 3600.0)


def measurement_features(ed_mask: np.ndarray, es_mask: np.ndarray, spacing: tuple[float, float, float],
                         height_m: float, weight_kg: float) -> dict[str, float]:
    """Clinical measurements for one patient from end-diastolic and end-systolic masks."""
    ed = structure_volumes_ml(ed_mask, spacing)
    es = structure_volumes_ml(es_mask, spacing)
    bsa = body_surface_area_m2(height_m, weight_kg)

    def ratio(numerator: float, denominator: float) -> float:
        return numerator / denominator if denominator > 0 else float("nan")

    lv_ef = 100.0 * (1.0 - ratio(es["LV"], ed["LV"]))
    rv_ef = 100.0 * (1.0 - ratio(es["RV"], ed["RV"]))
    mass = ed["MYO"] * MYOCARDIAL_DENSITY_G_PER_ML
    return {
        "lv_edv_index": ed["LV"] / bsa,
        "lv_esv_index": es["LV"] / bsa,
        "lv_ef": lv_ef,
        "rv_edv_index": ed["RV"] / bsa,
        "rv_esv_index": es["RV"] / bsa,
        "rv_ef": rv_ef,
        "myo_mass_index": mass / bsa,
        "mass_to_volume": ratio(mass, ed["LV"]),
        "rv_to_lv_edv": ratio(ed["RV"], ed["LV"]),
        "lv_ef_minus_rv_ef": lv_ef - rv_ef,
    }


def feature_table(metadata: pd.DataFrame, patients, masks_for) -> pd.DataFrame:
    """One row of measurements per patient; `masks_for(index, phase)` returns (mask, spacing)."""
    rows = []
    for index in patients:
        row = metadata.loc[index]
        ed_mask, spacing = masks_for(index, 0)
        es_mask, _ = masks_for(index, 1)
        features = measurement_features(ed_mask, es_mask, spacing, float(row["height"]), float(row["weight"]))
        rows.append({"index": int(index), "pid": row["pid"], "pathology": row["pathology"], **features})
    return pd.DataFrame(rows).set_index("index")


def candidate_models(seed: int) -> dict:
    """Deliberately low-capacity models: 100 training patients cannot support more."""
    return {
        "logistic_C0.1": make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000)),
        "logistic_C1": make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)),
        "logistic_C10": make_pipeline(StandardScaler(), LogisticRegression(C=10.0, max_iter=5000)),
        "random_forest": RandomForestClassifier(n_estimators=500, min_samples_leaf=2, random_state=seed),
    }


def select_model(features: pd.DataFrame, labels: pd.Series, seed: int = 42):
    """Choose by repeated stratified cross-validation on training patients only.

    Ten repeats of five folds, because with 100 patients a single fold split moves
    accuracy by several points from the shuffle alone.
    """
    splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=seed)
    results = {}
    for name, model in candidate_models(seed).items():
        scores = cross_val_score(model, features, labels, cv=splitter, scoring="accuracy")
        results[name] = {"mean_accuracy": float(scores.mean()), "sd_accuracy": float(scores.std())}
    best = max(results, key=lambda name: results[name]["mean_accuracy"])
    model = candidate_models(seed)[best].fit(features, labels)
    return best, model, results


def reference_ranges(table: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Mean and standard deviation of each measurement in the training NOR group."""
    normal = table[table["pathology"] == "NOR"]
    return {name: {"mean": float(normal[name].mean()), "sd": float(normal[name].std(ddof=1))} for name in FEATURES}


def explain_patient(row: pd.Series, ranges: dict, top: int = 3) -> list[str]:
    """The measurements furthest from the normal group, in clinical units.

    This is faithful by construction: these numbers are the classifier's actual
    inputs, not a post-hoc approximation of what it attended to.
    """
    deviations = []
    for name, (label, unit) in FEATURES.items():
        reference = ranges[name]
        if not np.isfinite(row[name]) or reference["sd"] <= 0:
            continue
        z = (row[name] - reference["mean"]) / reference["sd"]
        deviations.append((abs(z), f"{label} {row[name]:.1f} {unit} (normal group {reference['mean']:.1f} ± "
                                    f"{reference['sd']:.1f}; {z:+.1f} SD)".replace("  ", " ")))
    return [text for _, text in sorted(deviations, reverse=True)[:top]]


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% interval for a proportion that stays sensible at small n and near 0 or 1."""
    if total == 0:
        return float("nan"), float("nan")
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return centre - half, centre + half
