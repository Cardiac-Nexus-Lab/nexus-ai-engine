"""ACDC cardiac MRI acquisition, caching and patient-wise splits.

ACDC (Bernard et al., IEEE TMI 2018) holds 150 patients with short-axis cine MRI,
expert masks of the right ventricle, myocardium and left ventricle at end-diastole
(ED) and end-systole (ES), and one of five diagnoses per patient.

Only the ED and ES frames and their masks are fetched, about 78 MB. The full cine
series is 93% of the dataset's bytes and no published ACDC result uses it.

The files come from a public Hugging Face mirror that was resampled to 1 mm
in-plane by a third party. `verify_against_reference` checks that preprocessing
did not corrupt the anatomy, by recomputing each patient's ejection fraction from
the expert masks and comparing it with the value the mirror states.
"""

from __future__ import annotations

from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlretrieve

import nibabel as nib
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

ACDC_BASE = "https://huggingface.co/datasets/viennh2012/cardiac_cine_acdc/resolve/main"
DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "acdc"

PATHOLOGIES = ["NOR", "MINF", "DCM", "HCM", "RV"]
# ACDC label convention: 0 background, 1 right ventricle, 2 myocardium, 3 left ventricle.
STRUCTURES = {1: "RV", 2: "MYO", 3: "LV"}
PHASES = ("ed", "es")
IMAGE_SIZE = 192


def _download(url: str, destination: Path, attempts: int = 4) -> None:
    """Fetch one file once; re-running skips anything already present."""
    if destination.exists() and destination.stat().st_size > 0:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    last_error: Exception | None = None
    for _ in range(attempts):
        try:
            urlretrieve(url, temporary)
            temporary.replace(destination)
            return
        except (HTTPError, URLError, OSError) as error:
            last_error = error
    temporary.unlink(missing_ok=True)
    raise RuntimeError(f"failed to download {url} after {attempts} attempts") from last_error


def fetch_acdc(root: Path = DATA_ROOT) -> pd.DataFrame:
    """Download the patient tables and the ED/ES images and masks."""
    tables = []
    for split in ("train", "test"):
        _download(f"{ACDC_BASE}/{split}.csv", root / f"{split}.csv")
        table = pd.read_csv(root / f"{split}.csv")
        table["source_split"] = split
        tables.append(table)
    metadata = pd.concat(tables, ignore_index=True)

    columns = [f"sax_{phase}{suffix}" for phase in PHASES for suffix in ("", "_gt")]
    wanted = [(row[column], root / row[column]) for _, row in metadata.iterrows() for column in columns]
    missing = [(relative, path) for relative, path in wanted if not path.exists()]
    if missing:
        for relative, path in tqdm(missing, desc="ACDC ED/ES files"):
            _download(f"{ACDC_BASE}/{relative}", path)
    else:
        print(f"All {len(wanted)} ACDC files already present.")
    return metadata


def load_volume(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Return a volume as [slices, H, W] and its voxel spacing in mm (x, y, z)."""
    image = nib.load(str(path))
    array = np.asarray(image.dataobj)
    spacing = tuple(float(s) for s in image.header.get_zooms()[:3])
    return np.transpose(array, (2, 0, 1)), spacing


def _fit(array: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    """Centre-crop or zero-pad the last two axes to size x size."""
    out = np.zeros(array.shape[:-2] + (size, size), dtype=array.dtype)
    height, width = array.shape[-2:]
    src_y, dst_y = max(0, (height - size) // 2), max(0, (size - height) // 2)
    src_x, dst_x = max(0, (width - size) // 2), max(0, (size - width) // 2)
    h, w = min(size, height), min(size, width)
    out[..., dst_y:dst_y + h, dst_x:dst_x + w] = array[..., src_y:src_y + h, src_x:src_x + w]
    return out


def build_cache(metadata: pd.DataFrame, root: Path = DATA_ROOT) -> dict:
    """Decode every ED/ES volume once into flat slice arrays held in one file.

    Slices are stored with a table recording which patient, phase and position
    each came from, so a dataset can gather neighbours for 2.5D input and results
    can be reassembled into volumes for per-patient evaluation.
    """
    cache = root / f"acdc_edes_{IMAGE_SIZE}.npz"
    if cache.exists():
        print(f"Reusing ACDC cache: {cache}")
        with np.load(cache) as stored:
            return {key: stored[key] for key in stored.files}

    images, masks, patient, phase, position, spacing = [], [], [], [], [], []
    for index, row in tqdm(metadata.iterrows(), total=len(metadata), desc="decode ACDC"):
        for phase_id, name in enumerate(PHASES):
            volume, voxel = load_volume(root / row[f"sax_{name}"])
            label, _ = load_volume(root / row[f"sax_{name}_gt"])
            if volume.shape != label.shape:
                raise ValueError(f"{row['pid']} {name}: image {volume.shape} vs mask {label.shape}")
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
        "position": np.asarray(position, dtype=np.int8),
        "spacing": np.asarray(spacing, dtype=np.float32),
    }
    np.savez(cache, **arrays)
    print(f"Cached {len(arrays['images']):,} slices to {cache} ({cache.stat().st_size / 1e6:.0f} MB)")
    return arrays


def structure_volumes_ml(masks: np.ndarray, spacing: tuple[float, float, float]) -> dict[str, float]:
    """Volume of each labelled structure in millilitres, from a [slices, H, W] mask."""
    voxel_ml = float(np.prod(spacing)) / 1000.0
    return {name: float((masks == label).sum()) * voxel_ml for label, name in STRUCTURES.items()}


def verify_against_reference(metadata: pd.DataFrame, cache: dict) -> pd.DataFrame:
    """Recompute LV and RV ejection fraction from the expert masks.

    If the mirror's resampling had damaged the masks, volumes computed from them
    would disagree with the values it states. Agreement is evidence the anatomy
    survived preprocessing; it is not evidence about any model.
    """
    rows = []
    for index, row in metadata.iterrows():
        volumes = {}
        for phase_id, name in enumerate(PHASES):
            selected = (cache["patient"] == index) & (cache["phase"] == phase_id)
            volumes[name] = structure_volumes_ml(cache["masks"][selected], tuple(cache["spacing"][selected][0]))
        rows.append({
            "pid": row["pid"],
            "lv_ef_mask": 100 * (1 - volumes["es"]["LV"] / volumes["ed"]["LV"]),
            "lv_ef_stated": row["lv_ef"],
            "lv_edv_mask": volumes["ed"]["LV"],
            "lv_edv_stated": row["lv_edv"],
            "rv_ef_mask": 100 * (1 - volumes["es"]["RV"] / volumes["ed"]["RV"]),
            "rv_ef_stated": row["rv_ef"],
        })
    return pd.DataFrame(rows)


def patient_splits(metadata: pd.DataFrame, validation_per_class: int = 4, seed: int = 42) -> dict:
    """Patient-wise train/validation/test indices.

    The 50 source test patients are held out untouched. Validation takes the same
    number of patients from each diagnosis within the source training set, so
    model selection is not steered by whichever class happens to dominate it.
    Every slice of a patient stays on one side of each boundary.
    """
    rng = np.random.default_rng(seed)
    train_pool = metadata[metadata["source_split"] == "train"]
    validation = []
    for pathology in PATHOLOGIES:
        members = train_pool.index[train_pool["pathology"] == pathology].to_numpy()
        validation += list(rng.choice(members, size=validation_per_class, replace=False))
    validation = np.sort(np.asarray(validation))
    train = np.setdiff1d(train_pool.index.to_numpy(), validation)
    test = metadata.index[metadata["source_split"] == "test"].to_numpy()
    return {"train": train, "val": validation, "test": test}


def prepare(root: Path = DATA_ROOT) -> tuple[pd.DataFrame, dict, dict]:
    """Fetch, cache and split in one call."""
    metadata = fetch_acdc(root)
    cache = build_cache(metadata, root)
    return metadata, cache, patient_splits(metadata)
