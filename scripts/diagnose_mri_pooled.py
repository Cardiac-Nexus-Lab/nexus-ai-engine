"""MRI diagnosis trained on ACDC, M&Ms and M&Ms-2, with an "uncertain" result.

The deployed classifier (diagnose_mri.py) uses measurements indexed to body
surface area, so it can only be trained on ACDC's 100 patients: M&Ms-2 records no
height or weight and M&Ms only about 40%. This one uses the eight measurements in
POOLED_FEATURES, which need neither, and trains on every patient of the three
cohorts whose diagnosis maps to ACDC's five classes (COHORT_LABELS). Patients with
other conditions are kept out of training and used to check the uncertain result.

Measurements come from expert masks for training. Tests, all on patients the
model never trained on:

* each cohort's held-out test set, from expert masks and from the pooled
  segmenter's own masks (the web portal's pipeline);
* leave one dataset out: trained on two cohorts, tested on all of the third;
* the deployed BSA-indexed model on the same ACDC test patients, for comparison.

The uncertainty threshold is chosen on out-of-fold predictions of training
patients only, to make the cases still called 90% accurate.

Usage:
    python scripts/diagnose_mri_pooled.py
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
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from cardiac_nexus import mri_cohorts
from cardiac_nexus.models import select_device
from cardiac_nexus.mri_dataset import normalize_volumes
from cardiac_nexus.mri_diagnosis import (
    COHORT_LABELS,
    POOLED_FEATURES,
    candidate_models,
    measurement_features,
    pooled_features,
    select_model,
    uncertainty_threshold,
    wilson_interval,
)
from cardiac_nexus.mri_training import predict_volumes
from evaluate_mri import load_segmenter

OUTPUT = REPO_ROOT / "results" / "mri_diagnosis_pooled"
SEGMENTER = REPO_ROOT / "results" / "mri_segmentation_pooled" / "mri_segmenter.pt"
DEPLOYED = REPO_ROOT / "results" / "mri_diagnosis" / "diagnosis_model.joblib"
COLUMNS = list(POOLED_FEATURES)
SEED = 42


def mapped_labels(metadata: pd.DataFrame) -> pd.Series:
    return pd.Series([COHORT_LABELS[c].get(p) if isinstance(p, str) else None
                      for c, p in zip(metadata["cohort"], metadata["pathology"])], index=metadata.index)


def expert_masks(cache):
    def masks_for(index, phase):
        selected = (cache["patient"] == index) & (cache["phase"] == phase)
        return cache["masks"][selected], tuple(float(s) for s in cache["spacing"][selected][0])
    return masks_for


def model_masks(volumes):
    return lambda index, phase: (volumes[(index, phase)][0], volumes[(index, phase)][2])


def feature_rows(patients, masks_for) -> pd.DataFrame:
    rows = {}
    for index in patients:
        ed, spacing = masks_for(int(index), 0)
        es, _ = masks_for(int(index), 1)
        rows[int(index)] = pooled_features(ed, es, spacing)
    return pd.DataFrame.from_dict(rows, orient="index")[COLUMNS]


def train(features: pd.DataFrame, labels: pd.Series) -> dict:
    """Select a model by repeated cross-validation, then set the uncertainty threshold out of fold."""
    medians = features.median()
    X = features.fillna(medians)
    name, model, cross_validation = select_model(X, labels, SEED)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    out_of_fold = cross_val_predict(clone(candidate_models(SEED)[name]), X, labels, cv=folds, method="predict_proba")
    classes = np.array(sorted(labels.unique()))
    correct = classes[out_of_fold.argmax(axis=1)] == labels.to_numpy()
    return {"name": name, "model": model, "medians": medians, "cross_validation": cross_validation,
            "uncertainty": uncertainty_threshold(out_of_fold, correct)}


def score(trained: dict, features: pd.DataFrame, labels: pd.Series) -> dict:
    model = trained["model"]
    probabilities = model.predict_proba(features.fillna(trained["medians"]))
    classes = np.array(model.classes_)
    calls, truth = classes[probabilities.argmax(axis=1)], labels.to_numpy()
    called = probabilities.max(axis=1) >= trained["uncertainty"]["threshold"]
    correct = calls == truth
    low, high = wilson_interval(int(correct.sum()), len(truth))
    present = sorted(set(truth) | set(calls))
    return {
        "patients": int(len(truth)),
        "correct": int(correct.sum()),
        "accuracy": float(correct.mean()),
        "accuracy_95ci": [low, high],
        "balanced_accuracy": float(balanced_accuracy_score(truth, calls)),
        "recall_by_class": {c: {"patients": int((truth == c).sum()), "correct": int(((truth == c) & correct).sum())}
                            for c in sorted(set(truth))},
        "with_uncertain": {"called": int(called.sum()), "coverage": float(called.mean()),
                           "correct_when_called": int((correct & called).sum()),
                           "accuracy_when_called": float(correct[called].mean()) if called.any() else None},
        "confusion_matrix": {"labels": present, "rows_true_columns_predicted":
                             confusion_matrix(truth, calls, labels=present).tolist()},
    }


def out_of_scope(trained: dict, features: pd.DataFrame, conditions: pd.Series) -> dict:
    """Patients with conditions outside the five classes: how often the model says uncertain."""
    probabilities = trained["model"].predict_proba(features.fillna(trained["medians"]))
    uncertain = probabilities.max(axis=1) < trained["uncertainty"]["threshold"]
    calls = np.array(trained["model"].classes_)[probabilities.argmax(axis=1)]
    return {condition: {"patients": int((conditions == condition).sum()),
                        "uncertain": int(uncertain[(conditions == condition).to_numpy()].sum()),
                        "called_as": pd.Series(calls[(conditions == condition).to_numpy()]).value_counts().to_dict()}
            for condition in sorted(conditions.unique())}


def deployed_on_acdc(metadata, patients, masks_by_source) -> dict:
    """The current BSA-indexed classifier on the same ACDC test patients (ACDC records height and weight)."""
    bundle = joblib.load(DEPLOYED)  # our own artefact, written by scripts/diagnose_mri.py; never load joblib from elsewhere
    acdc = pd.read_csv(REPO_ROOT / "data" / "acdc" / "test.csv").set_index("pid")
    report = {}
    for source, masks_for in masks_by_source.items():
        correct = 0
        for index in patients:
            row = acdc.loc[metadata.loc[index, "pid"]]
            ed, spacing = masks_for(int(index), 0)
            es, _ = masks_for(int(index), 1)
            features = measurement_features(ed, es, spacing, float(row["height"]), float(row["weight"]))
            call = bundle["model"].predict(pd.DataFrame([features])[bundle["features"]])[0]
            correct += call == row["pathology"]
        report[source] = {"patients": len(patients), "correct": int(correct)}
    return report


def main() -> None:
    started = time.perf_counter()
    metadata, cache, splits = mri_cohorts.prepare()
    labels = mapped_labels(metadata)
    expert = expert_masks(cache)
    all_features = feature_rows(metadata.index, expert)
    in_scope = labels.notna()
    print(f"{len(metadata)} patients; {int(in_scope.sum())} with a diagnosis in the five classes")

    pool = np.union1d(splits["train"], splits["val"])
    train_ids = [i for i in pool if in_scope[i]]
    trained = train(all_features.loc[train_ids], labels[train_ids])
    print(f"selected {trained['name']} on {len(train_ids)} patients; uncertainty threshold "
          f"{trained['uncertainty']['threshold']:.2f} (training coverage {trained['uncertainty']['training_coverage']:.0%})")

    device = select_device(None)
    segmenter, _ = load_segmenter(SEGMENTER, device)
    images = normalize_volumes(cache["images"], cache["patient"], cache["phase"])
    test_report, scope_report = {}, {}
    for cohort, patients in splits["test_by_cohort"].items():
        volumes = predict_volumes(segmenter, images, cache["masks"], cache["patient"], cache["phase"],
                                  cache["spacing"], patients, device)
        predicted = feature_rows(patients, model_masks(volumes))
        ids = [i for i in patients if in_scope[i]]
        test_report[cohort] = {"expert_masks": score(trained, all_features.loc[ids], labels[ids]),
                               "model_masks": score(trained, predicted.loc[ids], labels[ids])}
        other = [i for i in patients if not in_scope[i] and isinstance(metadata.loc[i, "pathology"], str)]
        if other:
            scope_report[cohort] = out_of_scope(trained, predicted.loc[other], metadata.loc[other, "pathology"])
        if cohort == "acdc":
            baseline = deployed_on_acdc(metadata, patients, {"expert_masks": expert, "model_masks": model_masks(volumes)})
        e, m = test_report[cohort]["expert_masks"], test_report[cohort]["model_masks"]
        print(f"test {cohort}: {e['correct']}/{e['patients']} expert masks, {m['correct']}/{m['patients']} model masks")

    leave_one_out = {}
    for held_out in mri_cohorts.COHORTS:
        seen = [i for i in metadata.index if in_scope[i] and metadata.loc[i, "cohort"] != held_out]
        unseen = [i for i in metadata.index if in_scope[i] and metadata.loc[i, "cohort"] == held_out]
        fold = train(all_features.loc[seen], labels[seen])
        leave_one_out[held_out] = {"trained_on": len(seen), "selected_model": fold["name"],
                                   "uncertainty": fold["uncertainty"],
                                   **score(fold, all_features.loc[unseen], labels[unseen])}
        r = leave_one_out[held_out]
        print(f"held-out dataset {held_out}: {r['correct']}/{r['patients']} (balanced {r['balanced_accuracy']:.2f})")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": trained["model"], "features": COLUMNS, "classes": list(trained["model"].classes_),
                 "medians": trained["medians"].to_dict(), "uncertainty": trained["uncertainty"],
                 "label_map": COHORT_LABELS, "selected_model": trained["name"]}, OUTPUT / "diagnosis_model.joblib")
    counts = pd.crosstab(metadata["cohort"], labels.fillna("out of scope / unlabelled")).to_dict()
    report = {
        "features": COLUMNS, "label_map": COHORT_LABELS, "patients_by_cohort_and_class": counts,
        "training_patients": len(train_ids), "selected_model": trained["name"],
        "cross_validation": trained["cross_validation"], "uncertainty": trained["uncertainty"],
        "test": test_report, "deployed_model_on_acdc_test": baseline, "out_of_scope_test_patients": scope_report,
        "leave_one_dataset_out": leave_one_out, "minutes": (time.perf_counter() - started) / 60,
    }
    (OUTPUT / "diagnosis_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"Wrote {OUTPUT} in {report['minutes']:.1f} min")


if __name__ == "__main__":
    main()
