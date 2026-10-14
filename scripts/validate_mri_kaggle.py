"""External check of LV volumes and ejection fraction on the Kaggle cardiac cohort.

The Second Annual Data Science Bowl (Kaggle, 2016) released short-axis cine MRI
for about 1,140 patients (the mirror used here holds 1,139), each with
end-diastolic and end-systolic LV volumes measured by clinicians, but no
segmentation masks. That makes it unusable for training a segmenter and well
suited to testing one: none of these patients was seen in training, they come
from a different source than ACDC and M&Ms, and the reference is the quantity a
clinician reports rather than a mask.

For every patient the segmenter is run on every slice of every cine frame. LV
cavity volume per frame is the sum over slices; end-diastolic volume is the
largest and end-systolic the smallest, the usual approach when frame indices are
not given. Ejection fraction follows from the two.

Two cautions when reading the numbers. The reference volumes were measured by a
different protocol from ACDC's masks (for example, where the basal slice stops),
so some systematic bias is expected and is reported rather than corrected. And
a patient whose short-axis stack is incomplete in the mirror can produce a wrong
volume; those are reported, not silently dropped.

By default only the mirror's 439 test patients are scored, because every cine
frame is segmented and the full cohort takes several hours on a CPU. Each cine
file is about 8 MB; files are downloaded, scored and deleted one at a
time, and per-patient results are appended to a CSV so the run can resume.

Usage:
    python scripts/validate_mri_kaggle.py --checkpoint results/mri_segmentation_pooled/mri_segmenter.pt
    python scripts/validate_mri_kaggle.py --splits train val test   # all 1,139 patients
    python scripts/validate_mri_kaggle.py --limit 20     # quick check
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from cardiac_nexus.models import select_device
from cardiac_nexus.mri_data import _download, _fit
from cardiac_nexus.mri_dataset import normalize_volumes
from cardiac_nexus.mri_training import predict_volumes
from evaluate_mri import agreement, load_segmenter

BASE = "https://huggingface.co/datasets/viennh2012/cardiac_cine_kaggle/resolve/main"
DATA_ROOT = REPO_ROOT / "data" / "kaggle"
LV = 3


def patient_table(root: Path) -> pd.DataFrame:
    tables = []
    for split in ("train", "val", "test"):
        _download(f"{BASE}/{split}.csv", root / f"{split}.csv")
        table = pd.read_csv(root / f"{split}.csv")
        table["source_split"] = split
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


@torch.no_grad()
def lv_volume_curve(model, path: Path, device) -> tuple[np.ndarray, tuple[int, int]]:
    """LV cavity volume in ml for every cine frame of one patient."""
    image = nib.load(str(path))
    array = np.asarray(image.dataobj, dtype=np.float32)          # x, y, slices, frames
    zooms = image.header.get_zooms()
    voxel_ml = float(zooms[0] * zooms[1] * zooms[2]) / 1000.0
    slices, frames = array.shape[2], array.shape[3]
    stack = _fit(np.transpose(array, (3, 2, 0, 1)).reshape(frames * slices, *array.shape[:2]))
    frame = np.repeat(np.arange(frames, dtype=np.int16), slices)
    phase = np.zeros_like(frame, dtype=np.int8)
    images = normalize_volumes(stack, frame, phase)
    masks = np.zeros(stack.shape, dtype=np.uint8)
    spacing = np.tile(np.asarray(zooms[:3], dtype=np.float32), (len(frame), 1))
    volumes = predict_volumes(model, images, masks, frame, phase, spacing, np.arange(frames), device, batch_size=64)
    curve = np.array([(volumes[(f, 0)][0] == LV).sum() * voxel_ml for f in range(frames)])
    return curve, (slices, frames)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path,
                        default=REPO_ROOT / "results" / "mri_segmentation_pooled" / "mri_segmenter.pt")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--splits", nargs="+", default=["test"], choices=["train", "val", "test"],
                        help="which of the mirror's splits to score; none was used in training")
    parser.add_argument("--limit", type=int, default=None, help="score only the first N patients")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    output = args.output or args.checkpoint.parent
    output.mkdir(parents=True, exist_ok=True)
    rows_path = output / "kaggle_per_patient.csv"

    device = select_device(args.device)
    model, _ = load_segmenter(args.checkpoint, device)
    table = patient_table(DATA_ROOT)
    table = table[table["source_split"].isin(args.splits)]
    if args.limit:
        table = table.head(args.limit)

    done = set(pd.read_csv(rows_path)["pid"].astype(str)) if rows_path.exists() else set()
    for _, row in tqdm(table.iterrows(), total=len(table), desc="Kaggle patients"):
        if str(row["pid"]) in done:
            continue
        path = DATA_ROOT / row["sax_t"]
        record = {"pid": row["pid"], "source_split": row["source_split"],
                  "edv_true": row["diastole_volume"], "esv_true": row["systole_volume"], "ef_true": row["ef"]}
        try:
            _download(f"{BASE}/{row['sax_t']}", path)
            curve, (slices, frames) = lv_volume_curve(model, path, device)
            record.update({"edv_pred": float(curve.max()), "esv_pred": float(curve.min()),
                           "ef_pred": float(100 * (1 - curve.min() / curve.max())) if curve.max() > 0 else np.nan,
                           "slices": slices, "frames": frames, "error": ""})
        except Exception as error:  # a broken file is recorded, not fatal
            record.update({"edv_pred": np.nan, "esv_pred": np.nan, "ef_pred": np.nan, "error": repr(error)[:200]})
        finally:
            path.unlink(missing_ok=True)
        pd.DataFrame([record]).to_csv(rows_path, mode="a", header=not rows_path.exists(), index=False)

    results = pd.read_csv(rows_path)
    scored = results.dropna(subset=["edv_pred", "esv_pred", "ef_pred", "edv_true", "esv_true", "ef_true"])
    report = {"checkpoint": str(args.checkpoint), "patients": int(len(results)), "scored": int(len(scored)),
              "failed": results.loc[results["error"].fillna("") != "", ["pid", "error"]].to_dict("records"),
              "lv_end_diastolic_volume_ml": agreement(scored.edv_pred, scored.edv_true),
              "lv_end_systolic_volume_ml": agreement(scored.esv_pred, scored.esv_true),
              "lv_ejection_fraction_percent": agreement(scored.ef_pred, scored.ef_true)}
    (output / "kaggle_validation.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"\nKaggle cohort: {report['scored']} of {report['patients']} patients scored")
    for key in ("lv_end_diastolic_volume_ml", "lv_end_systolic_volume_ml", "lv_ejection_fraction_percent"):
        value = report[key]
        low, high = value["limits_of_agreement"]
        print(f"  {key:30} r={value['correlation']:.3f}  bias {value['bias']:+.2f}  "
              f"limits [{low:+.2f}, {high:+.2f}]  MAE {value['mean_absolute_error']:.2f}")


if __name__ == "__main__":
    main()
