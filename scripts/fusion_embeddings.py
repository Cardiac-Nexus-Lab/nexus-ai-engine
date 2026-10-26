"""Branch inputs for the fusion network: ECG embeddings and MRI features, from the deployed models.

ECG: the deployed xresnet1d18's 128-number embedding of PTB-XL folds 9 and 10.
Folds 1-8 trained that network, so their embeddings would be more confident than
any new patient's, and a fusion network trained on them would over-trust the ECG.
Fold 9 (which only calibrated it) trains and validates the fusion network; fold
10 tests it. Label: heart disease present = any finding other than normal.

MRI: the deployed pooled segmenter, run on every ACDC, M&Ms and M&Ms-2 patient
with a diagnosis in ACDC's classes, gives the deepest feature map averaged over
the heart's slices (end-diastole and end-systole) and the eight measurements of
its own outlines, exactly as the portal computes them. Splits are the
segmenter's own patient-wise splits. Label: anything other than NOR.

The health record needs no network to precompute; train_fusion.py reads UCI.

Writes data/fusion/branch_inputs.npz (data/ is not committed).

Usage:
    python scripts/fusion_embeddings.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np
import torch

from cardiac_nexus import data, mri_cohorts
from cardiac_nexus.fusion import MRI_FEATURES, ecg_embedding, mri_branch_input
from diagnose_mri_pooled import SEGMENTER, mapped_labels
from evaluate_ecg import load_checkpoint
from evaluate_mri import load_segmenter

ECG_CHECKPOINT = REPO_ROOT / "results" / "local_xresnet18_full" / "ecg_multilabel.pt"
OUTPUT = REPO_ROOT / "data" / "fusion" / "branch_inputs.npz"


def main() -> None:
    started = time.perf_counter()
    device = torch.device("cpu")

    classifier, classes, _ = load_checkpoint(ECG_CHECKPOINT, device)
    metadata, signals, splits = data.prepare()
    labels = metadata[data.SUPERCLASSES].to_numpy(dtype=np.float32)
    ecg_rows = np.concatenate([splits["val"], splits["test"]])
    ecg_split = np.array(["fold9"] * len(splits["val"]) + ["fold10"] * len(splits["test"]))
    ecg_emb, ecg_logits = ecg_embedding(classifier, np.asarray(signals[ecg_rows], dtype=np.float32))
    normal = classes.index("NORM")
    abnormal = labels[ecg_rows][:, [i for i in range(len(classes)) if i != normal]].max(axis=1)
    print(f"ECG: {len(ecg_rows)} recordings, {abnormal.mean():.0%} with a finding other than normal", flush=True)

    mri_meta, cache, mri_splits = mri_cohorts.prepare()
    mri_labels = mapped_labels(mri_meta)
    segmenter, _ = load_segmenter(SEGMENTER, device)
    split_of = {int(i): "train" for i in mri_splits["train"]} | {int(i): "val" for i in mri_splits["val"]}
    for cohort, ids in mri_splits["test_by_cohort"].items():
        split_of |= {int(i): "test" for i in ids}
    patients = [i for i in mri_meta.index if isinstance(mri_labels[i], str) and int(i) in split_of]
    mri_feat, mri_measure, mri_class, mri_split, mri_cohort = [], [], [], [], []
    for n, index in enumerate(patients, 1):
        volumes = {}
        for phase in (0, 1):
            selected = (cache["patient"] == index) & (cache["phase"] == phase)
            order = np.argsort(cache["position"][selected]) if "position" in cache else slice(None)
            volumes[phase] = cache["images"][selected][order]
            spacing = tuple(float(s) for s in cache["spacing"][selected][0])
        features, measurements, _, _ = mri_branch_input(segmenter, volumes[0], volumes[1], spacing, device)
        mri_feat.append(features)
        mri_measure.append([measurements[k] for k in MRI_FEATURES])
        mri_class.append(mri_labels[index])
        mri_split.append(split_of[int(index)])
        mri_cohort.append(mri_meta.loc[index, "cohort"])
        if n % 50 == 0:
            print(f"  MRI {n}/{len(patients)}", flush=True)
    mri_class = np.array(mri_class)
    print(f"MRI: {len(patients)} patients, {np.mean(mri_class != 'NOR'):.0%} not normal", flush=True)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT,
        ecg_embedding=ecg_emb.astype(np.float32), ecg_logits=ecg_logits.astype(np.float32),
        ecg_label=abnormal.astype(np.int8), ecg_split=ecg_split, ecg_classes=np.array(classes),
        ecg_record=metadata.index.to_numpy()[ecg_rows],
        mri_features=np.asarray(mri_feat, np.float32), mri_measurements=np.asarray(mri_measure, np.float32),
        mri_class=mri_class, mri_label=(mri_class != "NOR").astype(np.int8), mri_split=np.array(mri_split),
        mri_cohort=np.array(mri_cohort), mri_patient=np.asarray(patients),
        mri_measurement_names=np.array(MRI_FEATURES),
    )
    print(f"Wrote {OUTPUT} in {(time.perf_counter() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
