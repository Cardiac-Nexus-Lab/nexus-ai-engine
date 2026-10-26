"""MRI diagnosis with body-size indexing, trained on ACDC plus the M&Ms patients who record height and weight.

diagnose_mri_pooled.py showed that dropping body-size indexing to pool all three
cohorts costs more than the extra patients bring (ACDC test 37/50 from the
segmenter's masks, against 43/50 for the deployed ACDC-only model). This keeps
the deployed model's ten BSA-indexed measurements and adds the M&Ms patients
with height and weight (about 40% of M&Ms; M&Ms-2 records neither), mapped to
ACDC's classes as in COHORT_LABELS.

Both this model and the deployed one are scored on the same held-out patients
(ACDC test, and the M&Ms test patients with height and weight), from expert
masks and from the pooled segmenter's masks, each with an uncertainty threshold
set out of fold on its own training patients.

Usage:
    python scripts/diagnose_mri_indexed.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from cardiac_nexus import mri_cohorts
from cardiac_nexus.models import select_device
from cardiac_nexus.mri_dataset import normalize_volumes
from cardiac_nexus.mri_diagnosis import FEATURES, candidate_models, measurement_features, select_model
from cardiac_nexus.mri_diagnosis import uncertainty_threshold
from cardiac_nexus.mri_training import predict_volumes
from diagnose_mri_pooled import SEGMENTER, expert_masks, mapped_labels, model_masks, score
from evaluate_mri import load_segmenter

OUTPUT = REPO_ROOT / "results" / "mri_diagnosis_indexed"
DEPLOYED = REPO_ROOT / "results" / "mri_diagnosis" / "diagnosis_model.joblib"
COLUMNS = list(FEATURES)
SEED = 42


def body_size(metadata: pd.DataFrame) -> pd.DataFrame:
    """Height in metres and weight in kg per patient; ACDC states metres, M&Ms centimetres."""
    rows = []
    for cohort, folder, to_metres in (("acdc", "acdc", 1.0), ("mnms", "mnms", 0.01)):
        splits = ("train", "test") if cohort == "acdc" else ("train", "val", "test")
        table = pd.concat([pd.read_csv(REPO_ROOT / "data" / folder / f"{s}.csv") for s in splits])
        table["pid"] = table["pid"].astype(str)
        rows.append(pd.DataFrame({"cohort": cohort, "pid": table["pid"], "height_m": table["height"] * to_metres,
                                  "weight_kg": table["weight"]}))
    sizes = pd.concat(rows)
    merged = metadata[["cohort", "pid"]].reset_index().merge(sizes, on=["cohort", "pid"], how="left")
    return merged.set_index("index")[["height_m", "weight_kg"]]


def indexed_rows(patients, masks_for, sizes) -> pd.DataFrame:
    rows = {}
    for index in patients:
        ed, spacing = masks_for(int(index), 0)
        es, _ = masks_for(int(index), 1)
        rows[int(index)] = measurement_features(ed, es, spacing, float(sizes.loc[index, "height_m"]),
                                                float(sizes.loc[index, "weight_kg"]))
    return pd.DataFrame.from_dict(rows, orient="index")[COLUMNS]


def train(features: pd.DataFrame, labels: pd.Series) -> dict:
    medians = features.median()
    X = features.fillna(medians)
    name, model, cross_validation = select_model(X, labels, SEED)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    out_of_fold = cross_val_predict(clone(candidate_models(SEED)[name]), X, labels, cv=folds, method="predict_proba")
    correct = np.array(sorted(labels.unique()))[out_of_fold.argmax(axis=1)] == labels.to_numpy()
    return {"name": name, "model": model, "medians": medians, "cross_validation": cross_validation,
            "uncertainty": uncertainty_threshold(out_of_fold, correct)}


def deployed_as_trained(features: pd.DataFrame, labels: pd.Series) -> dict:
    """The deployed model unchanged, with an uncertainty threshold from its own training patients."""
    bundle = joblib.load(DEPLOYED)  # our own artefact, written by scripts/diagnose_mri.py
    X = features.fillna(features.median())[bundle["features"]]
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    out_of_fold = cross_val_predict(clone(bundle["model"]), X, labels, cv=folds, method="predict_proba")
    correct = np.array(sorted(labels.unique()))[out_of_fold.argmax(axis=1)] == labels.to_numpy()
    return {"name": "deployed (ACDC only)", "model": bundle["model"], "medians": features.median(),
            "uncertainty": uncertainty_threshold(out_of_fold, correct)}


def main() -> None:
    started = time.perf_counter()
    metadata, cache, splits = mri_cohorts.prepare()
    labels = mapped_labels(metadata)
    sizes = body_size(metadata)
    usable = labels.notna() & sizes.notna().all(axis=1) & metadata["cohort"].isin(["acdc", "mnms"])
    pool = np.union1d(splits["train"], splits["val"])
    train_ids = [i for i in pool if usable[i]]
    acdc_train = [i for i in train_ids if metadata.loc[i, "cohort"] == "acdc"]
    expert = expert_masks(cache)
    features = indexed_rows(train_ids, expert, sizes)
    print(f"training on {len(train_ids)} patients ({len(acdc_train)} ACDC, {len(train_ids) - len(acdc_train)} M&Ms)")

    models = {"ACDC + M&Ms, indexed": train(features, labels[train_ids]),
              "deployed (ACDC only)": deployed_as_trained(features.loc[acdc_train], labels[acdc_train])}
    device = select_device(None)
    segmenter, _ = load_segmenter(SEGMENTER, device)
    images = normalize_volumes(cache["images"], cache["patient"], cache["phase"])
    report = {"features": COLUMNS, "training_patients": {"acdc": len(acdc_train), "mnms": len(train_ids) - len(acdc_train)},
              "selected_model": models["ACDC + M&Ms, indexed"]["name"],
              "uncertainty": {k: v["uncertainty"] for k, v in models.items()}, "test": {}}
    for cohort in ("acdc", "mnms"):
        ids = [i for i in splits["test_by_cohort"][cohort] if usable[i]]
        volumes = predict_volumes(segmenter, images, cache["masks"], cache["patient"], cache["phase"],
                                  cache["spacing"], np.asarray(ids), device)
        sources = {"expert_masks": indexed_rows(ids, expert, sizes),
                   "model_masks": indexed_rows(ids, model_masks(volumes), sizes)}
        report["test"][cohort] = {name: {source: score(m, table[COLUMNS] if name != "deployed (ACDC only)"
                                                      else table[joblib.load(DEPLOYED)["features"]], labels[ids])
                                         for source, table in sources.items()}
                                  for name, m in models.items()}
        for name in models:
            e, s = report["test"][cohort][name]["expert_masks"], report["test"][cohort][name]["model_masks"]
            w = s["with_uncertain"]
            print(f"{cohort} test, {name}: {e['correct']}/{e['patients']} expert, {s['correct']}/{s['patients']} model masks; "
                  f"called {w['called']} at {w['accuracy_when_called']}")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    chosen = models["ACDC + M&Ms, indexed"]
    joblib.dump({"model": chosen["model"], "features": COLUMNS, "classes": list(chosen["model"].classes_),
                 "uncertainty": chosen["uncertainty"], "selected_model": chosen["name"]},
                OUTPUT / "diagnosis_model.joblib")
    report["minutes"] = (time.perf_counter() - started) / 60
    (OUTPUT / "diagnosis_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"Wrote {OUTPUT} in {report['minutes']:.1f} min")


if __name__ == "__main__":
    main()
