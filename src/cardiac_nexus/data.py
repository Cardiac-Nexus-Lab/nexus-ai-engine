"""PTB-XL acquisition and caching.

The published PTB-XL archive is 1.71 GB because it ships both the 100 Hz and the
500 Hz copy of every recording. Only the 100 Hz copy is used here, so the records
are fetched individually rather than by downloading the full archive: roughly
0.52 GB instead of 1.71 GB before extraction.

Downloads resume. Re-running skips anything already on disk.
"""

from __future__ import annotations

import ast
import shutil
import zipfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen, urlretrieve

import numpy as np
import pandas as pd
import wfdb
from tqdm.auto import tqdm

PTBXL_BASE = "https://physionet.org/files/ptb-xl/1.0.3"
PTBXL_ARCHIVE_URL = (
    "https://physionet.org/static/published-projects/ptb-xl/"
    "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip"
)
SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
SIGNAL_LENGTH = 1000  # 10 seconds at 100 Hz
NUM_LEADS = 12

DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "ptbxl"


def _download(url: str, destination: Path, attempts: int = 4) -> None:
    """Fetch one file, tolerating the transient 5xx responses PhysioNet returns under load."""
    if destination.exists() and destination.stat().st_size > 0:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            urlretrieve(url, temporary)
            temporary.replace(destination)
            return
        except (HTTPError, URLError, OSError) as error:
            last_error = error
    temporary.unlink(missing_ok=True)
    raise RuntimeError(f"failed to download {url} after {attempts} attempts") from last_error


def fetch_metadata(root: Path = DATA_ROOT) -> pd.DataFrame:
    """Download the two metadata tables and return the labelled record index."""
    root.mkdir(parents=True, exist_ok=True)
    for name in ("ptbxl_database.csv", "scp_statements.csv"):
        _download(f"{PTBXL_BASE}/{name}", root / name)

    metadata = pd.read_csv(root / "ptbxl_database.csv", index_col=0)
    scp = pd.read_csv(root / "scp_statements.csv", index_col=0)
    scp = scp[scp["diagnostic"] == 1]
    diagnostic_map = scp["diagnostic_class"].dropna().to_dict()

    def superclasses(raw_codes: str) -> set[str]:
        codes = ast.literal_eval(raw_codes) if isinstance(raw_codes, str) else raw_codes
        return {diagnostic_map[code] for code in codes if code in diagnostic_map}

    classes = metadata["scp_codes"].apply(superclasses)
    for name in SUPERCLASSES:
        metadata[name] = classes.apply(lambda found, name=name: int(name in found))

    # A recording with no diagnostic superclass carries no usable label for this task.
    metadata = metadata[metadata[SUPERCLASSES].sum(axis=1) > 0].copy()
    return metadata


def _download_archive(destination: Path, chunk_bytes: int = 1 << 20) -> None:
    """Stream the published archive to disk with progress."""
    if destination.exists() and destination.stat().st_size > 0:
        print(f"Archive already downloaded: {destination.name}")
        return

    temporary = destination.with_suffix(".part")
    with urlopen(PTBXL_ARCHIVE_URL) as response:  # noqa: S310 - fixed, known https URL
        total = int(response.headers.get("content-length", 0))
        with open(temporary, "wb") as handle, tqdm(
            total=total, unit="B", unit_scale=True, unit_divisor=1024, desc="archive"
        ) as progress:
            while chunk := response.read(chunk_bytes):
                handle.write(chunk)
                progress.update(len(chunk))
    temporary.replace(destination)


def download_records(metadata: pd.DataFrame, root: Path = DATA_ROOT, keep_archive: bool = False) -> None:
    """Obtain every 100 Hz record, via the published archive.

    Fetching the ~42,800 record files individually is not viable: PhysioNet
    throttles sustained per-file requests, degrading from roughly 11 files/second
    to under one, which would take over 17 hours. The archive is a single request.

    It contains both the 100 Hz and 500 Hz copies, but only the 100 Hz members are
    extracted, so the 1.71 GB download expands to roughly 0.52 GB on disk. The
    archive is deleted afterwards unless keep_archive is set.
    """
    expected = {
        f"{relative[:-4] if relative.endswith('.dat') else relative}{suffix}"
        for relative in metadata["filename_lr"]
        for suffix in (".dat", ".hea")
    }
    missing = {name for name in expected if not (root / name).exists()}
    if not missing:
        print(f"All {len(expected):,} record files already present.")
        return

    print(f"{len(missing):,} of {len(expected):,} record files missing; fetching archive.")
    archive_path = root / "ptb-xl-1.0.3.zip"
    _download_archive(archive_path)

    with zipfile.ZipFile(archive_path) as archive:
        # Members are prefixed with the release directory name; map them onto the
        # paths the rest of this module expects.
        members = [m for m in archive.namelist() if "/records100/" in m and m.endswith((".dat", ".hea"))]
        if not members:
            raise RuntimeError("archive contained no records100 members; layout may have changed")

        for member in tqdm(members, desc="extracting", unit="file"):
            relative = member.split("/records100/", 1)[1]
            destination = root / "records100" / relative
            if destination.exists():
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, open(destination, "wb") as target:
                shutil.copyfileobj(source, target)

    if not keep_archive:
        archive_path.unlink(missing_ok=True)
        print("Removed the archive; extracted records retained.")

    still_missing = {name for name in expected if not (root / name).exists()}
    if still_missing:
        raise RuntimeError(f"{len(still_missing):,} record files still missing after extraction")


def load_ecg(root: Path, relative_path: str) -> np.ndarray:
    """Read one record and standardize each lead independently."""
    stem = relative_path[: -len(".dat")] if relative_path.endswith(".dat") else relative_path
    signal, _ = wfdb.rdsamp(str(root / stem))
    signal = signal.astype(np.float32).T  # [12 leads, 1000 samples]
    mean = signal.mean(axis=1, keepdims=True)
    std = signal.std(axis=1, keepdims=True) + 1e-6
    return (signal - mean) / std


def build_signal_cache(metadata: pd.DataFrame, root: Path = DATA_ROOT) -> np.ndarray:
    """Decode every record once into a single array, saved next to the raw data.

    Decoding a WFDB record costs far more than a forward pass for a model this
    small, so decoding lazily per access leaves the device waiting on I/O for every
    sample of every epoch. Decoding once removes that entirely.
    """
    cache_path = root / "signals_100hz.npy"
    index_path = root / "signals_index.npy"

    if cache_path.exists() and index_path.exists():
        cached_ids = np.load(index_path)
        if np.array_equal(cached_ids, metadata.index.to_numpy()):
            print(f"Reusing signal cache: {cache_path}")
            return np.load(cache_path, mmap_mode="r")
        print("Cache does not match the current record index; rebuilding.")

    signals = np.empty((len(metadata), NUM_LEADS, SIGNAL_LENGTH), dtype=np.float32)
    for i, relative in enumerate(tqdm(metadata["filename_lr"], desc="decoding", unit="rec")):
        signals[i] = load_ecg(root, relative)

    np.save(cache_path, signals)
    np.save(index_path, metadata.index.to_numpy())
    print(f"Wrote {cache_path} ({signals.nbytes / 1024**3:.2f} GB)")
    return signals


def split_indices(metadata: pd.DataFrame) -> dict[str, np.ndarray]:
    """Positions into the cache for the official patient-wise PTB-XL folds."""
    fold = metadata["strat_fold"].to_numpy()
    return {
        "train": np.flatnonzero((fold >= 1) & (fold <= 8)),
        "val": np.flatnonzero(fold == 9),
        "test": np.flatnonzero(fold == 10),
    }


def prepare(root: Path = DATA_ROOT) -> tuple[pd.DataFrame, np.ndarray, dict[str, np.ndarray]]:
    """Fetch, decode and split. Safe to call repeatedly; every stage resumes."""
    metadata = fetch_metadata(root)
    download_records(metadata, root)
    signals = build_signal_cache(metadata, root)
    return metadata, signals, split_indices(metadata)
