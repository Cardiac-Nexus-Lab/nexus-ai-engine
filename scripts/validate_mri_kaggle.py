"""External check of LV volumes and ejection fraction on the Kaggle cardiac cohort.

The Second Annual Data Science Bowl (Kaggle, 2016) released short-axis cine MRI
for about 1,140 patients (the mirror used here holds 1,139), each with
end-diastolic and end-systolic LV volumes measured by clinicians, but no
segmentation masks. That makes it unusable for training a segmenter directly and
well suited to testing one: the reference is the quantity a clinician reports
rather than a mask.

For every patient each segmenter is run on every slice of every cine frame, the
predicted LV is reduced to its largest 3D connected component per frame, and the
LV cross-sectional area of every slice in every frame is stored. Volumes are then
computed two ways from the same areas:

* cylindrical: sum over slices of area x slice spacing;
* frustum: sum over adjacent slice pairs of dz/3 (A_i + A_i+1 + sqrt(A_i A_i+1)).
  As written this integrates only between the first and last slice centres, so
  it omits half a slice at each end and is expected to read lower.

End-diastole is the frame with the largest cylindrical LV volume and end-systole
the smallest, the usual choice when frame indices are not given.

Slice spacing is read from the NIfTI header and also recorded alongside the
mirror's stated original spacing, so a mismatch between the two is visible. EF is
a ratio of two volumes measured with the same spacing, so a wrong spacing biases
EDV and ESV but cancels in EF; that is why the volumes are reported separately.

The reference volumes were measured by a different protocol from ACDC's masks
(for example, where the basal slice stops), so some systematic bias is expected
and is reported rather than corrected.

By default only the mirror's 439 test patients are scored. With `--save-edes`,
the ED and ES frames of every scored patient are also saved with the first
checkpoint's predicted LV mask, for weak supervision on non-test patients; the
test patients' frames are never saved.

Each cine file is about 8 MB; files are downloaded, scored and deleted one at a
time, and per-patient results are appended to a CSV so the run can resume.

Usage:
    python scripts/validate_mri_kaggle.py --checkpoints results/mri_segmentation_pooled/mri_segmenter.pt \\
        results/mri_segmentation/mri_segmenter.pt
    python scripts/validate_mri_kaggle.py --splits train val --save-edes     # non-test patients
    python scripts/validate_mri_kaggle.py --limit 20                          # quick check
"""

from __future__ import annotations

import argparse
import contextlib
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
from scipy import ndimage
from tqdm.auto import tqdm

from cardiac_nexus.models import select_device
from cardiac_nexus.mri_data import _download, _fit
from cardiac_nexus.mri_dataset import neighbour_indices, normalize_volumes
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


def load_cine(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Cine stack as [frames, slices, H, W] and voxel spacing (x, y, z) in mm."""
    image = nib.load(str(path))
    array = np.asarray(image.dataobj, dtype=np.float32)          # x, y, slices, frames
    zooms = image.header.get_zooms()
    stack = _fit(np.transpose(array, (3, 2, 0, 1)))
    return stack, (float(zooms[0]), float(zooms[1]), float(zooms[2]))


@torch.no_grad()
def segment_cine(model, stack: np.ndarray, device, bf16: bool, batch_size: int = 64) -> np.ndarray:
    """Predicted labels for every slice of every frame, [frames, slices, H, W]."""
    frames, slices = stack.shape[:2]
    flat = stack.reshape(frames * slices, *stack.shape[2:])
    frame = np.repeat(np.arange(frames, dtype=np.int16), slices)
    phase = np.zeros_like(frame, dtype=np.int8)
    images = normalize_volumes(flat, frame, phase)
    previous, following = neighbour_indices(frame, phase)
    model.eval()
    autocast = (torch.autocast("cpu", dtype=torch.bfloat16) if bf16 and device.type == "cpu"
                else contextlib.nullcontext())
    out = []
    with autocast:
        for start in range(0, len(flat), batch_size):
            rows = np.arange(start, min(start + batch_size, len(flat)))
            batch = np.stack([images[previous[rows]], images[rows], images[following[rows]]], axis=1)
            out.append(model(torch.from_numpy(batch).to(device)).argmax(dim=1).cpu().numpy().astype(np.uint8))
    return np.concatenate(out).reshape(frames, slices, *stack.shape[2:])


def largest_lv(frame_labels: np.ndarray) -> np.ndarray:
    """Binary LV mask of one frame, largest 3D connected component only."""
    lv = frame_labels == LV
    components, count = ndimage.label(lv)
    if count > 1:
        lv = components == (np.argmax(np.bincount(components.ravel())[1:]) + 1)
    return lv


def volumes_ml(areas_mm2: np.ndarray, dz: float) -> dict[str, np.ndarray]:
    """Per-frame LV volume from per-slice areas [frames, slices], two ways."""
    cylinder = areas_mm2.sum(axis=1) * dz / 1000.0
    a, b = areas_mm2[:, :-1], areas_mm2[:, 1:]
    frustum = (dz / 3.0 * (a + b + np.sqrt(a * b))).sum(axis=1) / 1000.0
    return {"cylindrical": cylinder, "frustum": frustum}


def summarize(results: pd.DataFrame) -> dict:
    report = {"patients": int(len(results)),
              "failed": results.loc[results["error"].fillna("") != "", ["pid", "error"]].to_dict("records")}
    for method in ("cylindrical", "frustum"):
        columns = [f"edv_{method}", f"esv_{method}", f"ef_{method}", "edv_true", "esv_true", "ef_true"]
        scored = results.dropna(subset=columns)
        entry = {"scored": int(len(scored))}
        for quantity, unit in (("edv", "ml"), ("esv", "ml"), ("ef", "percent")):
            predicted, actual = scored[f"{quantity}_{method}"], scored[f"{quantity}_true"]
            entry[f"lv_{quantity}_{unit}"] = {
                **agreement(predicted, actual),
                "rmse": float(np.sqrt(np.mean((predicted - actual) ** 2)))}
        report[method] = entry
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoints", type=Path, nargs="+",
                        default=[REPO_ROOT / "results" / "mri_segmentation_pooled" / "mri_segmenter.pt"])
    parser.add_argument("--splits", nargs="+", default=["test"], choices=["train", "val", "test"])
    parser.add_argument("--save-edes", action="store_true",
                        help="save ED/ES frames and the first checkpoint's LV mask (never for test patients)")
    parser.add_argument("--bf16", action="store_true", help="bfloat16 autocast on CPU")
    parser.add_argument("--limit", type=int, default=None, help="score only the first N patients")
    parser.add_argument("--output-name", default="kaggle", help="folder created next to each checkpoint")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if args.save_edes and "test" in args.splits:
        sys.exit("--save-edes is for non-test patients only")
    device = select_device(args.device)
    models = [(path, load_segmenter(path, device)[0]) for path in args.checkpoints]
    table = patient_table(DATA_ROOT)
    table = table[table["source_split"].isin(args.splits)]
    if args.limit:
        table = table.head(args.limit)
    edes_dir = DATA_ROOT / "edes"
    edes_dir.mkdir(parents=True, exist_ok=True)

    tag = "_".join(args.splits)
    outputs, rows_paths = {}, {}
    for path in args.checkpoints:
        outputs[path] = path.parent / args.output_name
        (outputs[path] / "areas").mkdir(parents=True, exist_ok=True)
        rows_paths[path] = outputs[path] / f"kaggle_{tag}_per_patient.csv"
    done = None
    for path in args.checkpoints:
        finished = set(pd.read_csv(rows_paths[path])["pid"].astype(str)) if rows_paths[path].exists() else set()
        done = finished if done is None else done & finished

    for _, row in tqdm(table.iterrows(), total=len(table), desc="Kaggle patients"):
        if str(row["pid"]) in done:
            continue
        cine_path = DATA_ROOT / row["sax_t"]
        base = {"pid": row["pid"], "source_split": row["source_split"],
                "edv_true": row["diastole_volume"], "esv_true": row["systole_volume"], "ef_true": row["ef"],
                "dz_stated": row.get("original_sax_spacing_z", np.nan)}
        try:
            _download(f"{BASE}/{row['sax_t']}", cine_path)
            stack, (dx, dy, dz) = load_cine(cine_path)
            error = ""
        except Exception as caught:  # a broken file is recorded, not fatal
            stack, error = None, repr(caught)[:200]
        finally:
            cine_path.unlink(missing_ok=True)

        for index, (path, model) in enumerate(models):
            record = dict(base, error=error)
            if stack is not None:
                try:
                    labels = segment_cine(model, stack, device, args.bf16)
                    lv = np.stack([largest_lv(frame) for frame in labels])
                    areas = lv.sum(axis=(2, 3)) * dx * dy
                    np.save(outputs[path] / "areas" / f"{row['pid']}.npy", areas.astype(np.float32))
                    curves = volumes_ml(areas, dz)
                    ed, es = int(curves["cylindrical"].argmax()), int(curves["cylindrical"].argmin())
                    record.update({"dz_header": dz, "slices": stack.shape[1], "frames": stack.shape[0],
                                   "ed_frame": ed, "es_frame": es})
                    for method, curve in curves.items():
                        record[f"edv_{method}"] = float(curve[ed])
                        record[f"esv_{method}"] = float(curve[es])
                        record[f"ef_{method}"] = (float(100 * (1 - curve[es] / curve[ed])) if curve[ed] > 0
                                                  else np.nan)
                    if args.save_edes and index == 0:
                        np.savez_compressed(edes_dir / f"{row['pid']}.npz",
                                            ed_image=stack[ed].astype(np.uint8), es_image=stack[es].astype(np.uint8),
                                            ed_lv=lv[ed], es_lv=lv[es], spacing=np.array([dx, dy, dz], np.float32))
                except Exception as caught:
                    record["error"] = repr(caught)[:200]
            pd.DataFrame([record]).to_csv(rows_paths[path], mode="a", header=not rows_paths[path].exists(),
                                          index=False)

    for path in args.checkpoints:
        results = pd.read_csv(rows_paths[path])
        report = {"checkpoint": str(path), "splits": args.splits, "bf16": args.bf16, **summarize(results)}
        spacing = results.dropna(subset=["dz_header", "dz_stated"])
        report["slice_spacing_audit"] = {
            "patients": int(len(spacing)),
            "header_minus_stated_mm": {"max_abs": float((spacing.dz_header - spacing.dz_stated).abs().max()),
                                       "mean": float((spacing.dz_header - spacing.dz_stated).mean())},
            "header_dz_values_mm": {str(k): int(v) for k, v in
                                    spacing.dz_header.round(2).value_counts().head(10).items()}}
        (outputs[path] / f"kaggle_{tag}_validation.json").write_text(json.dumps(report, indent=2, default=str))
        print(f"\n{path}: Kaggle {'+'.join(args.splits)}, {report['patients']} patients")
        for method in ("cylindrical", "frustum"):
            for key in ("lv_edv_ml", "lv_esv_ml", "lv_ef_percent"):
                value = report[method][key]
                low, high = value["limits_of_agreement"]
                print(f"  {method:11} {key:14} r={value['correlation']:.3f}  bias {value['bias']:+.2f}  "
                      f"LoA [{low:+.2f}, {high:+.2f}]  MAE {value['mean_absolute_error']:.2f}  RMSE {value['rmse']:.2f}")


if __name__ == "__main__":
    main()
