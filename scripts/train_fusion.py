"""Train and test the ECG + MRI + health-record fusion network (cardiac_nexus.fusion).

Virtual patients. No public dataset has an ECG, an MRI and a health record for the
same people, so each training example joins an ECG, an MRI and a health record of
three different patients who share the label (heart disease present or not). Each
source keeps its own patient-wise split, so no test patient of any modality is
seen in training:

* ECG: PTB-XL fold 9, 80% train / 20% validation by patient (the ECG network never
  trained on fold 9); fold 10 test (PTB-XL's folds never share a patient);
* MRI: the pooled segmenter's patient-wise train / validation / test splits of
  ACDC, M&Ms and M&Ms-2;
* health record: UCI Heart Disease, the same 184 test patients as the stand-alone
  health-record model; its 736 training patients 80% train / 20% validation.

Training draws fresh virtual patients every epoch, drops each whole modality with
probability 0.3 (keeping at least one) and blanks each health-record field with
probability 0.25, so the network learns to work with whatever a user provides.
Validation and test use fixed sets of virtual patients, with every combination of
modalities scored. The model is selected on validation AUROC averaged over the
seven combinations, calibrated with one slope and intercept on validation, and a
decision threshold is set for each combination on validation (Youden's J) before
the test set is read.

Run history: the first configuration (no input dropout, weight decay 1e-4, one
threshold for every combination) over-trusted the MRI, whose training pool is a
few hundred patients, and called everyone ill without an MRI because the one
threshold did not fit them; its report is kept in results/fusion_dl/run1. The
second (input dropout 0.2/0.5/0.1, weight decay 1e-3, a threshold per combination)
improved with every modality added but stayed below the mean of the three single
readings (results/fusion_dl/run2). Since the modalities of a virtual patient are
independent given the label, summed log-odds are the right base, so the network's
output became that sum plus the attention layers' correction (FusionNet.forward),
with a lower learning rate, the first epoch having been the best.

A simple baseline is reported alongside: the mean of the three single-modality
probabilities that the same network gives with the other two switched off.

Results are on virtual patients and say how well the network combines evidence
from three independent sources; they are not a clinical validation.

Usage:
    python scripts/train_fusion.py
"""

from __future__ import annotations

import copy
import itertools
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from torch import nn

from cardiac_nexus import ehr
from cardiac_nexus.evaluate import VectorScaler, bootstrap_metric, expected_calibration_error
from cardiac_nexus.fusion import (CLINICAL_FIELDS, FIELD_COLUMNS, CLINICAL_COLUMNS, MODALITIES, MRI_FEATURES,
                                  FusionNet, clinical_stats, encode_clinical, mri_vector)

INPUTS = REPO_ROOT / "data" / "fusion" / "branch_inputs.npz"
OUTPUT = REPO_ROOT / "results" / "fusion_dl"
SEED = 42
MODALITY_DROP, FIELD_DROP = 0.3, 0.25
SINGLE_FRAME = 0.35   # share of training MRIs reduced to one frame (either phase), as most uploads are
INPUT_DROPOUT = {"ecg": 0.2, "mri": 0.5, "clinical": 0.1}
WEIGHT_DECAY = 1e-3
LEARNING_RATE = 3e-4
SUBSETS = [tuple(m for m, keep in zip(MODALITIES, mask) if keep)
           for mask in itertools.product([1, 0], repeat=3) if any(mask)]


def pools() -> tuple[dict, dict]:
    """Per split, per modality: inputs [n, d] and labels [n]; plus the statistics used to encode them."""
    z = np.load(INPUTS, allow_pickle=False)
    rng = np.random.default_rng(SEED)
    # Fold 9 is split by patient, not by recording: 2,146 recordings come from 1,917 patients,
    # and one patient's recordings must not sit in both training and validation.
    patient_of = pd.read_csv(REPO_ROOT / "data" / "ptbxl" / "ptbxl_database.csv", index_col=0)["patient_id"]
    patients = patient_of.loc[z["ecg_record"]].to_numpy()
    fold9_patients = np.unique(patients[z["ecg_split"] == "fold9"])
    val_patients = set(rng.choice(fold9_patients, size=len(fold9_patients) // 5, replace=False).tolist())
    ecg_split = np.array(["test" if s == "fold10" else ("val" if patients[i] in val_patients else "train")
                          for i, s in enumerate(z["ecg_split"])])

    measures = z["mri_measurements"]
    train_m = z["mri_split"] == "train"
    median = np.nanmedian(measures[train_m], axis=0)
    filled = np.where(np.isfinite(measures), measures, median)
    mean, std = filled[train_m].mean(axis=0), filled[train_m].std(axis=0) + 1e-6
    mri_stats = {"measure_median": median.tolist(), "measure_mean": mean.tolist(), "measure_std": std.tolist()}
    names = list(z["mri_measurement_names"])
    ed, es = z["mri_features"][:, :256], z["mri_features"][:, 256:]

    def vectors(first, second, table):
        return np.stack([mri_vector(first[i], None if second is None else second[i], dict(zip(names, table[i])),
                                    mri_stats) for i in range(len(first))]).astype(np.float32)
    mri = vectors(ed, es, measures)
    mri_ed_only = vectors(ed, None, z["mri_measurements_ed_only"])
    mri_es_only = vectors(es, None, z["mri_measurements_es_only"])

    table = ehr.fetch()
    strata = table["site"] + "_" + table["cad"].astype(str)
    train_val, test = train_test_split(table, test_size=0.2, stratify=strata, random_state=SEED)
    strata = train_val["site"] + "_" + train_val["cad"].astype(str)
    train, val = train_test_split(train_val, test_size=0.2, stratify=strata, random_state=SEED)
    stats = clinical_stats(train)

    def encoded(part):
        return np.stack([encode_clinical(r, stats)[0] for r in part[CLINICAL_FIELDS].to_dict("records")])

    out = {}
    for split in ("train", "val", "test"):
        ehr_part = {"train": train, "val": val, "test": test}[split]
        out[split] = {
            "ecg": (z["ecg_embedding"][ecg_split == split], z["ecg_label"][ecg_split == split].astype(int)),
            "mri": (mri[z["mri_split"] == split], z["mri_label"][z["mri_split"] == split].astype(int)),
            "mri_single": (mri_ed_only[z["mri_split"] == split], mri_es_only[z["mri_split"] == split]),
            "clinical": (encoded(ehr_part), ehr_part["cad"].to_numpy().astype(int)),
        }
    branches = {"ecg": {"path": str(z["ecg_checkpoint"]), "sha256": str(z["ecg_sha256"])},
                "mri": {"path": str(z["mri_checkpoint"]), "sha256": str(z["mri_sha256"])}}
    return out, {"clinical": stats, "mri": mri_stats, "branches": branches}


def virtual_patients(pool: dict, n: int, rng: np.random.Generator, single_frame: float = 0.0) -> tuple[dict, np.ndarray]:
    """n virtual patients, half with heart disease: for each, one record of each modality with that label.

    A `single_frame` share of the MRIs is replaced by one of that patient's two frames alone."""
    y = rng.permutation(np.arange(n) % 2)
    batch, picks = {}, {}
    for m in MODALITIES:
        x, labels = pool[m]
        by_label = [np.flatnonzero(labels == k) for k in (0, 1)]
        picks[m] = np.array([rng.choice(by_label[k]) for k in y])
        batch[m] = x[picks[m]].copy()
    if single_frame:  # drawn after every pick, so the same seed gives the same patients either way
        ed_only, es_only = pool["mri_single"]
        one = rng.random(n) < single_frame
        phase = rng.random(n) < 0.5
        batch["mri"][one & phase] = ed_only[picks["mri"][one & phase]]
        batch["mri"][one & ~phase] = es_only[picks["mri"][one & ~phase]]
    return batch, y


def masks(n: int, subset: tuple | None, rng: np.random.Generator) -> np.ndarray:
    """[n, 3] presence: a fixed subset, or random modality dropout keeping at least one."""
    if subset is not None:
        return np.tile([m in subset for m in MODALITIES], (n, 1))
    present = rng.random((n, 3)) >= MODALITY_DROP
    empty = ~present.any(axis=1)
    present[empty, rng.integers(0, 3, empty.sum())] = True
    return present


def blank_random_fields(x: np.ndarray, stats: dict, rng: np.random.Generator) -> np.ndarray:
    x = x.copy()
    for field in CLINICAL_FIELDS:
        rows = rng.random(len(x)) < FIELD_DROP
        for column in FIELD_COLUMNS[field]:
            i = CLINICAL_COLUMNS.index(column)
            x[rows, i] = stats["column_mean"][i]
    return x


def tensors(batch: dict, present: np.ndarray):
    return (torch.from_numpy(batch["ecg"]), torch.from_numpy(batch["mri"]),
            torch.from_numpy(batch["clinical"].astype(np.float32)), torch.from_numpy(present))


@torch.no_grad()
def logits_for(model, batch, present, chunk: int = 1024) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    out = [[], [], []]
    for s in range(0, len(present), chunk):
        part = {m: batch[m][s:s + chunk] for m in MODALITIES}
        logit, weights, solo = model(*tensors(part, present[s:s + chunk]))
        for k, v in enumerate((logit, weights, solo)):
            out[k].append(v.numpy())
    return tuple(np.concatenate(o) for o in out)


def metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    tn, fp, fn, tp = confusion_matrix(y, p >= threshold, labels=[0, 1]).ravel()
    return {"auroc": bootstrap_metric(y, p, roc_auc_score, 1000, seed=SEED),
            "auprc": bootstrap_metric(y, p, average_precision_score, 1000, seed=SEED),
            "brier": float(brier_score_loss(y, p)), "ece": expected_calibration_error(y, p)["ece"],
            "threshold": threshold, "confusion": {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)},
            "accuracy": float((tp + tn) / len(y)), "precision": float(tp / max(tp + fp, 1)),
            "recall": float(tp / max(tp + fn, 1)), "specificity": float(tn / max(tn + fp, 1)),
            "f1": float(2 * tp / max(2 * tp + fp + fn, 1))}


def main() -> None:
    started = time.perf_counter()
    torch.manual_seed(SEED)
    data, stats = pools()
    for split, part in data.items():
        print(split, {m: f"{len(y)} ({y.mean():.0%} disease)" for m, (_, y) in part.items() if m in MODALITIES},
              flush=True)

    val_batch, val_y = virtual_patients(data["val"], 3000, np.random.default_rng(1))
    test_batch, test_y = virtual_patients(data["test"], 5000, np.random.default_rng(2))

    model_args = {"ecg_dim": int(data["train"]["ecg"][0].shape[1]), "mri_dim": int(data["train"]["mri"][0].shape[1]),
                  "clinical_dim": len(CLINICAL_COLUMNS), "input_dropout": INPUT_DROPOUT}
    model = FusionNet(**model_args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    rng = np.random.default_rng(SEED)
    epochs, patience = 60, 8
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    def val_score() -> dict:
        return {"+".join(s): float(roc_auc_score(val_y, logits_for(model, val_batch, masks(len(val_y), s, rng))[0]))
                for s in SUBSETS}

    history, best, best_epoch, best_state, waited = [], -1.0, 0, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        batch, y = virtual_patients(data["train"], 8192, rng, single_frame=SINGLE_FRAME)
        batch["clinical"] = blank_random_fields(batch["clinical"], stats["clinical"], rng)
        present = masks(len(y), None, rng)
        order, losses = rng.permutation(len(y)), []
        for s in range(0, len(y), 64):
            idx = order[s:s + 64]
            part = {m: batch[m][idx] for m in MODALITIES}
            target = torch.from_numpy(y[idx].astype(np.float32))
            logit, _, solo = model(*tensors(part, present[idx]))
            solo_loss = (criterion(solo, target[:, None].expand_as(solo)) * torch.from_numpy(present[idx])).sum() \
                / max(int(present[idx].sum()), 1)
            loss = criterion(logit, target).mean() + 0.3 * solo_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()
        scores = val_score()
        mean = float(np.mean(list(scores.values())))
        solo_val = [logits_for(model, val_batch, masks(len(val_y), (m,), rng))[0] for m in MODALITIES]
        scores["baseline: sum of single readings"] = float(roc_auc_score(val_y, np.sum(solo_val, axis=0)))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_auroc": scores, "val_mean": mean})
        marker = ""
        if mean > best:
            best, best_epoch, best_state, waited, marker = mean, epoch, copy.deepcopy(model.state_dict()), 0, "  <- best"
        else:
            waited += 1
        print(f"Epoch {epoch:02d}: loss {np.mean(losses):.4f} | val AUROC mean {mean:.4f}, all three "
              f"{scores['ecg+mri+clinical']:.4f}, baseline {scores['baseline: sum of single readings']:.4f}{marker}",
              flush=True)
        if waited >= patience:
            break
    model.load_state_dict(best_state)

    # Calibration and threshold, on validation only.
    cal_rng = np.random.default_rng(3)
    z_val = np.concatenate([logits_for(model, val_batch, masks(len(val_y), s, cal_rng))[0] for s in SUBSETS])
    scaler = VectorScaler(1)
    scaler.fit(z_val[:, None], np.tile(val_y, len(SUBSETS))[:, None].astype(np.float32))
    calibration = {"scale": float(scaler.scales[0]), "bias": float(scaler.biases[0])}

    def calibrated(z):
        return 1.0 / (1.0 + np.exp(-(z * calibration["scale"] + calibration["bias"])))

    thresholds = {}
    for subset in SUBSETS:
        p_val = calibrated(logits_for(model, val_batch, masks(len(val_y), subset, cal_rng))[0])
        fpr, tpr, cut = roc_curve(val_y, p_val)
        thresholds["+".join(subset)] = float(cut[np.argmax(tpr - fpr)])
    # One-frame MRIs give the network less to go on, so their probabilities sit differently:
    # their own thresholds, from the same validation patients with each MRI reduced to one frame.
    val_single, _ = virtual_patients(data["val"], 3000, np.random.default_rng(1), single_frame=1.0)
    for subset in SUBSETS:
        if "mri" in subset:
            p_val = calibrated(logits_for(model, val_single, masks(len(val_y), subset, cal_rng))[0])
            fpr, tpr, cut = roc_curve(val_y, p_val)
            thresholds["+".join(subset) + " (MRI: one frame only)"] = float(cut[np.argmax(tpr - fpr)])

    report = {"target": "heart disease present (ECG not normal; MRI not NOR; UCI num > 0)",
              "training_data": {split: {m: {"records": int(len(y)), "with_disease": int(y.sum())}
                                        for m, (_, y) in part.items() if m in MODALITIES}
                                for split, part in data.items()},
              "virtual_patients": {"train_per_epoch": 8192, "validation": 3000, "test": 5000},
              "settings": {"optimizer": "AdamW", "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
                           "input_dropout": INPUT_DROPOUT, "batch": 64,
                           "schedule": "cosine", "max_epochs": epochs, "patience": patience,
                           "modality_dropout": MODALITY_DROP, "field_dropout": FIELD_DROP,
                           "single_frame_mri": SINGLE_FRAME, "solo_loss_weight": 0.3,
                           "parameters": int(sum(p.numel() for p in model.parameters()))},
              "selected_epoch": best_epoch, "calibration": calibration, "thresholds": thresholds, "test": {},
              "caveat": "Virtual patients join three different people with the same label; results measure how "
                        "the network combines independent evidence, not clinical performance."}
    test_rng = np.random.default_rng(4)
    solo_probs = {}
    for subset in SUBSETS:
        present = masks(len(test_y), subset, test_rng)
        z, weights, _ = logits_for(model, test_batch, present)
        p = calibrated(z)
        if len(subset) == 1:
            solo_probs[subset[0]] = p
        entry = metrics(test_y, p, thresholds["+".join(subset)])
        entry["mean_gate_weight"] = {m: float(weights[:, i].mean()) for i, m in enumerate(MODALITIES) if m in subset}
        report["test"]["+".join(subset)] = entry
        print(f"test {'+'.join(subset):22} AUROC {entry['auroc']['estimate']:.3f} "
              f"({entry['auroc']['low']:.3f}-{entry['auroc']['high']:.3f})  F1 {entry['f1']:.3f}", flush=True)
    # The commonest upload: one MRI frame. Same test patients, each MRI reduced to one of its frames.
    single_batch, single_y = virtual_patients(data["test"], 5000, np.random.default_rng(2), single_frame=1.0)
    assert np.array_equal(single_y, test_y) and np.array_equal(single_batch["ecg"], test_batch["ecg"])
    for subset in (("ecg", "mri", "clinical"), ("mri",)):
        z, _, _ = logits_for(model, single_batch, masks(len(test_y), subset, test_rng))
        key = "+".join(subset) + " (MRI: one frame only)"
        entry = metrics(test_y, calibrated(z), thresholds[key])
        report["test"][key] = entry
        print(f"test {'+'.join(subset) + ' one MRI frame':22} AUROC {entry['auroc']['estimate']:.3f}  F1 {entry['f1']:.3f}",
              flush=True)
    baseline = np.mean([solo_probs[m] for m in MODALITIES], axis=0)
    fpr, tpr, cut = roc_curve(val_y, np.mean([calibrated(logits_for(model, val_batch, masks(len(val_y), (m,), cal_rng))[0])
                                              for m in MODALITIES], axis=0))
    report["test"]["baseline: mean of single-modality probabilities"] = metrics(test_y, baseline,
                                                                                float(cut[np.argmax(tpr - fpr)]))
    print(f"baseline mean of three AUROC {report['test']['baseline: mean of single-modality probabilities']['auroc']['estimate']:.3f}")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "model_args": model_args, "calibration": calibration,
                "thresholds": thresholds, "clinical_stats": stats["clinical"], "mri_stats": stats["mri"],
                "mri_features": MRI_FEATURES, "selected_epoch": best_epoch,
                "branch_checkpoints": stats["branches"]}, OUTPUT / "fusion_model.pt")
    report["history"] = history
    report["minutes"] = (time.perf_counter() - started) / 60
    (OUTPUT / "fusion_report.json").write_text(json.dumps(report, indent=2))
    print(f"Wrote {OUTPUT} in {report['minutes']:.1f} min")


if __name__ == "__main__":
    main()
