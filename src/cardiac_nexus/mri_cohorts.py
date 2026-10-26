"""Pooled short-axis cine MRI from three public cohorts: ACDC, M&Ms and M&Ms-2.

ACDC alone is 150 patients from one hospital. M&Ms ("Multi-Centre, Multi-Vendor
and Multi-Disease Cardiac Segmentation: The M&Ms Challenge") adds 320 labelled
patients from several hospitals and four scanner vendors; M&Ms-2 ("Deep Learning
Segmentation of the Right Ventricle in Cardiac MRI: The M&Ms Challenge") adds 360
more, enriched for right-ventricular disease. All three label the same structures, so one model can
learn from all of them, and each cohort's own held-out patients test whether it
generalises beyond the scanners ACDC was acquired on.

The files come from the same third-party Hugging Face mirrors as `mri_data`,
which resample every cohort to 1 mm in-plane. Other releases of these cohorts do
not all number the structures as ACDC does, so the mirror's labelling is checked
here rather than trusted. The volumes the
mirror states agree with its own masks to rounding error, which shows they were
computed from those masks and therefore cannot catch a swap. `label_anatomy`
checks anatomy instead: the LV cavity is enclosed by myocardium and the RV is
not, so whichever label is ringed by label 2 must be the LV.

Splits follow each mirror's published train/validation/test sets, with the test
patients held out untouched. ACDC publishes no validation set, so it keeps the
exact split of `mri_data` and its test score stays comparable with the
ACDC-only model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from . import mri_data
from .mri_data import IMAGE_SIZE, PHASES, _download, _fit, load_volume, structure_volumes_ml

MIRROR = "https://huggingface.co/datasets/viennh2012/cardiac_cine_{name}/resolve/main"
DATA_ROOT = mri_data.DATA_ROOT.parent
COHORTS = ("acdc", "mnms", "mnms2")


def fetch_cohort(name: str, root: Path = DATA_ROOT) -> pd.DataFrame:
    """Download one cohort's tables and its ED/ES images and masks."""
    if name == "acdc":
        table = mri_data.fetch_acdc(root / "acdc")
        table["cohort"] = "acdc"
        return table
    base, folder = MIRROR.format(name=name), root / name
    tables = []
    for split in ("train", "val", "test"):
        _download(f"{base}/{split}.csv", folder / f"{split}.csv")
        table = pd.read_csv(folder / f"{split}.csv")
        table["source_split"] = split
        tables.append(table)
    table = pd.concat(tables, ignore_index=True)
    table["pid"] = table["pid"].astype(str)
    table["cohort"] = name

    columns = [f"sax_{phase}{suffix}" for phase in PHASES for suffix in ("", "_gt")]
    missing = [row[c] for _, row in table.iterrows() for c in columns if not (folder / row[c]).exists()]
    if missing:
        mri_data.download_all([(f"{base}/{relative}", folder / relative) for relative in missing],
                              f"{name} ED/ES files")
    return table


def fetch_all(root: Path = DATA_ROOT) -> pd.DataFrame:
    """All three cohorts in one table, indexed 0..N-1 for use as patient ids."""
    keep = ["cohort", "pid", "source_split", "pathology", "vendor",
            "sax_ed", "sax_ed_gt", "sax_es", "sax_es_gt", "lv_edv", "rv_edv", "lv_ef", "rv_ef"]
    tables = []
    for name in COHORTS:
        table = fetch_cohort(name, root)
        if "vendor" not in table:
            table["vendor"] = np.nan
        table["folder"] = str(root / name)
        tables.append(table[keep + ["folder"]])
    return pd.concat(tables, ignore_index=True)


def build_cache(metadata: pd.DataFrame, root: Path = DATA_ROOT) -> dict:
    """Decode every ED/ES volume once, in the same layout as `mri_data.build_cache`."""
    cache_path = root / f"pooled_edes_{IMAGE_SIZE}.npz"
    if cache_path.exists():
        print(f"Reusing pooled cache: {cache_path}")
        with np.load(cache_path) as stored:
            return {key: stored[key] for key in stored.files}

    images, masks, patient, phase, position, spacing = [], [], [], [], [], []
    for index, row in tqdm(metadata.iterrows(), total=len(metadata), desc="decode cohorts"):
        folder = Path(row["folder"])
        for phase_id, name in enumerate(PHASES):
            volume, voxel = load_volume(folder / row[f"sax_{name}"])
            label, _ = load_volume(folder / row[f"sax_{name}_gt"])
            if volume.shape != label.shape:
                raise ValueError(f"{row['cohort']} {row['pid']} {name}: image {volume.shape} vs mask {label.shape}")
            images.append(_fit(volume.astype(np.float32)))
            masks.append(_fit(label.astype(np.uint8)))
            slices = volume.shape[0]
            patient += [index] * slices
            phase += [phase_id] * slices
            position += list(range(slices))
            spacing += [voxel] * slices

    arrays = {
        "images": np.concatenate(images).astype(np.float32),
        "masks": np.concatenate(masks),
        "patient": np.asarray(patient, dtype=np.int16),
        "phase": np.asarray(phase, dtype=np.int8),
        "position": np.asarray(position, dtype=np.int16),
        "spacing": np.asarray(spacing, dtype=np.float32),
    }
    np.savez(cache_path, **arrays)
    print(f"Cached {len(arrays['images']):,} slices to {cache_path} ({cache_path.stat().st_size / 1e6:.0f} MB)")
    return arrays


def verify_against_reference(metadata: pd.DataFrame, cache: dict) -> pd.DataFrame:
    """End-diastolic LV and RV volume from the expert masks against the stated values.

    Where a cohort states no volumes the stated columns are empty, and only the
    presence of every structure at end-diastole is checked.
    """
    rows = []
    for index, row in metadata.iterrows():
        selected = (cache["patient"] == index) & (cache["phase"] == 0)
        volumes = structure_volumes_ml(cache["masks"][selected], tuple(cache["spacing"][selected][0]))
        rows.append({"cohort": row["cohort"], "pid": row["pid"],
                     "lv_edv_mask": volumes["LV"], "lv_edv_stated": row["lv_edv"],
                     "rv_edv_mask": volumes["RV"], "rv_edv_stated": row["rv_edv"],
                     "myo_ed_ml": volumes["MYO"]})
    return pd.DataFrame(rows)


def label_anatomy(metadata: pd.DataFrame, cache: dict, every: int = 3) -> dict:
    """Fraction of each cavity label's in-plane border that is myocardium, per cohort.

    In ACDC's convention label 3 (LV) should be close to 1.0 and label 1 (RV)
    well below it; a cohort with the two swapped shows the reverse.
    """
    from scipy import ndimage

    report = {}
    for cohort in metadata["cohort"].unique():
        fractions = {1: [], 3: []}
        for index in metadata.index[metadata["cohort"] == cohort][::every]:
            volume = cache["masks"][(cache["patient"] == index) & (cache["phase"] == 0)]
            for label in fractions:
                region = volume == label
                ring = ndimage.binary_dilation(region, structure=np.ones((1, 3, 3), bool)) & ~region
                if ring.any():
                    fractions[label].append(float((volume[ring] == 2).mean()))
        report[cohort] = {"RV_border_myocardium": float(np.mean(fractions[1])),
                          "LV_border_myocardium": float(np.mean(fractions[3])),
                          "patients_checked": len(fractions[3])}
    return report


def find_duplicates(metadata: pd.DataFrame, cache: dict, threshold: float = 0.99) -> list[tuple[int, int, float]]:
    """Pairs of patients in different cohorts whose ED volumes are near-identical.

    M&Ms and M&Ms-2 come from the same organisers; a patient present in
    both would let a test scan be seen during training. Volumes are compared by
    the correlation of a 32x32 thumbnail of every slice, which survives
    re-encoding but not a genuinely different scan.
    """
    fingerprints = {}
    for index in metadata.index:
        selected = (cache["patient"] == index) & (cache["phase"] == 0)
        volume = cache["images"][selected]
        thumbnail = volume.reshape(len(volume), 32, IMAGE_SIZE // 32, 32, IMAGE_SIZE // 32).mean(axis=(2, 4)).ravel()
        thumbnail = (thumbnail - thumbnail.mean()) / (thumbnail.std() + 1e-6)
        fingerprints[index] = thumbnail
    by_length: dict[int, list[int]] = {}
    for index, vector in fingerprints.items():
        by_length.setdefault(len(vector), []).append(index)
    pairs = []
    for members in by_length.values():
        if len(members) < 2:
            continue
        matrix = np.stack([fingerprints[i] for i in members])
        similarity = matrix @ matrix.T / matrix.shape[1]
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                first, second = members[a], members[b]
                if similarity[a, b] > threshold and metadata.loc[first, "cohort"] != metadata.loc[second, "cohort"]:
                    pairs.append((first, second, float(similarity[a, b])))
    return pairs


def cohort_splits(metadata: pd.DataFrame, seed: int = 42) -> dict:
    """Patient-wise train/validation/test indices, plus each cohort's test set."""
    train, validation, test = [], [], {}

    acdc = metadata[metadata["cohort"] == "acdc"]
    acdc_local = acdc.reset_index(drop=True)
    acdc_splits = mri_data.patient_splits(acdc_local, seed=seed)
    to_global = acdc.index.to_numpy()
    train += list(to_global[acdc_splits["train"]])
    validation += list(to_global[acdc_splits["val"]])
    test["acdc"] = to_global[acdc_splits["test"]]

    mnms = metadata[metadata["cohort"] == "mnms"]
    train += list(mnms.index[mnms["source_split"] == "train"])
    validation += list(mnms.index[mnms["source_split"] == "val"])
    test["mnms"] = mnms.index[mnms["source_split"] == "test"].to_numpy()

    mnms2 = metadata[metadata["cohort"] == "mnms2"]
    train += list(mnms2.index[mnms2["source_split"] == "train"])
    validation += list(mnms2.index[mnms2["source_split"] == "val"])
    test["mnms2"] = mnms2.index[mnms2["source_split"] == "test"].to_numpy()

    return {"train": np.sort(np.asarray(train)), "val": np.sort(np.asarray(validation)),
            "test": np.sort(np.concatenate(list(test.values()))), "test_by_cohort": test}


def exclude(splits: dict, patients) -> dict:
    """Drop patients (such as cross-cohort duplicates) from training and validation."""
    patients = np.asarray(list(patients), dtype=np.int64)
    out = dict(splits)
    out["train"] = np.setdiff1d(splits["train"], patients)
    out["val"] = np.setdiff1d(splits["val"], patients)
    return out


def prepare(root: Path = DATA_ROOT) -> tuple[pd.DataFrame, dict, dict]:
    """Fetch, cache and split all three cohorts in one call."""
    metadata = fetch_all(root)
    cache = build_cache(metadata, root)
    return metadata, cache, cohort_splits(metadata)
