"""The ECG image digitizer on real photographs and scans: PMcardio ECG Image Database (PM-ECG-ID).

PM-ECG-ID (Iring et al., Zenodo 13617673, GPL-3.0-or-later) prints 100 PTB-XL ECGs
in several layouts and photographs them with three phones, scans them, bends and
crumples them, and photographs them on screens. Only these real-world scenarios
("photos_*") are used here, not its synthetic augmentations, and only the images
needed are fetched from the 34 GB archive (src/cardiac_nexus/remote_zip.py).

Ground truth is the PTB-XL recording itself (records100, as the classifier sees
it). For each image the digitizer runs exactly as in the web portal, and the
script records:

* per-lead correlation of the digitized signal with the true one;
* whether the portal's plausibility checks would accept it;
* whether the classifier's leading finding matches its finding on the true signal.

The digitizer was trained on synthetic renders of PTB-XL folds 1-8, so results are
also reported for the ECGs from folds 9-10 alone, whose signals it never saw.

Usage:
    python scripts/validate_digitizer_real.py                 # 12x1 pages only (what the portal supports)
    python scripts/validate_digitizer_real.py --all-layouts   # every real-world image
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import cv2
import numpy as np
import pandas as pd
import torch

from cardiac_nexus import data
from cardiac_nexus.ecg_image.pipeline import digitize_page, load_localizer
from cardiac_nexus.remote_zip import open_remote_zip
from evaluate_ecg import load_checkpoint

URL = "https://zenodo.org/records/13617673/files/pmcardio_ecg_image_database.zip"
CACHE = REPO_ROOT / "data" / "pm_ecg_id"
OUTPUT = REPO_ROOT / "results" / "digitizer_real"
CHECKPOINT = REPO_ROOT / "results" / "local_xresnet18_full" / "ecg_multilabel.pt"
REPORT = REPO_ROOT / "results" / "local_xresnet18_full" / "evaluation_report.json"
LOCALIZER = REPO_ROOT / "results" / "digitizer" / "trace_localizer.pt"
PORTAL = REPO_ROOT.parent / "nexus-web-portal" / "backend"


def portal_checks():
    """The web portal's own plausibility check, so 'accepted' means what it means in the app."""
    if not (PORTAL / "cardiac_ai.py").exists():
        return None
    sys.path.insert(0, str(PORTAL))
    import cardiac_ai
    return cardiac_ai


def fetch(archive, relative: str) -> bytes:
    path = CACHE / relative
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(archive.read(f"final_data/visual_data/{relative}"))
    return path.read_bytes()


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() < 1e-6 or b.std() < 1e-6:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-layouts", action="store_true")
    args = parser.parse_args()

    archive = open_remote_zip(URL)
    metadata = pd.read_csv(io.BytesIO(archive.read("final_data/metadata.csv")))
    real = metadata[metadata["Image relative path"].str.startswith("photos_")].copy()
    real["scenario"] = real["Image relative path"].str.split("/").str[0]
    real["ecg_id"] = real["ECG ID"].str.extract(r"LPAE_(\d+)").astype(int)[0]
    if not args.all_layouts:
        real = real[real["ECG format"] == "12x1"]
    print(f"{len(real)} real-world images from {real['ecg_id'].nunique()} ECGs", flush=True)

    data.fetch_metadata()  # makes sure the PTB-XL tables are present
    # The full record table: fetch_metadata drops recordings without a diagnostic label,
    # which still have a waveform to compare a digitized image against.
    ptbxl = pd.read_csv(data.DATA_ROOT / "ptbxl_database.csv", index_col=0)
    device = torch.device("cpu")
    model, classes, _ = load_checkpoint(CHECKPOINT, device)
    calibration = json.loads(REPORT.read_text())["vector_scaling"]
    scale, bias = np.asarray(calibration["scales"]), np.asarray(calibration["biases"])
    localizer = load_localizer(LOCALIZER, device)
    portal = portal_checks()

    def classify(signal: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            logits = model(torch.from_numpy(signal[None].astype(np.float32))).numpy()[0]
        return 1.0 / (1.0 + np.exp(-(logits * scale + bias)))

    rows = []
    for _, item in real.iterrows():
        record = ptbxl.loc[item["ecg_id"]]
        truth = data.load_ecg(data.DATA_ROOT, record["filename_lr"])
        image = cv2.imdecode(np.frombuffer(fetch(archive, item["Image relative path"]), np.uint8), cv2.IMREAD_COLOR)
        digitized = digitize_page(image, localizer, device)
        signal = digitized.signal[:, :1000]
        low_confidence = float((digitized.confidence < 0.15).mean())
        accepted, reason = None, ""
        if portal is not None:
            try:
                portal.check_is_ecg(signal, from_image=True, low_confidence=low_confidence)
                accepted = True
            except portal.InputError as error:
                accepted, reason = False, str(error).split(":")[0]
        p_true, p_image = classify(truth), classify(signal)
        rows.append({
            "image": item["Image relative path"], "scenario": item["scenario"], "layout": item["ECG format"],
            "ecg_id": int(item["ecg_id"]), "fold": int(record["strat_fold"]),
            "median_lead_correlation": float(np.median([correlation(signal[i], truth[i]) for i in range(12)])),
            "low_confidence_share": low_confidence, "portal_accepts": accepted, "refusal": reason,
            "top_class_true": classes[int(np.argmax(p_true))], "top_class_image": classes[int(np.argmax(p_image))],
            "page_found": not any("No page border" in w for w in digitized.warnings),
        })
        if len(rows) % 25 == 0:
            print(f"  {len(rows)}/{len(real)}", flush=True)

    table = pd.DataFrame(rows)
    table["same_top_class"] = table["top_class_true"] == table["top_class_image"]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT / ("per_image_all_layouts.csv" if args.all_layouts else "per_image_12x1.csv"), index=False)

    def summary(part: pd.DataFrame) -> dict:
        return {"images": int(len(part)), "ecgs": int(part["ecg_id"].nunique()),
                "median_correlation": float(part["median_lead_correlation"].median()),
                "share_correlation_above_0.8": float((part["median_lead_correlation"] > 0.8).mean()),
                "portal_accepts": float(part["portal_accepts"].mean()) if part["portal_accepts"].notna().all() else None,
                "same_top_class": float(part["same_top_class"].mean())}

    report = {"dataset": "PM-ECG-ID real-world scenarios (photos_*), GPL-3.0-or-later",
              "all": summary(table),
              "unseen_folds_9_10": summary(table[table["fold"] >= 9]) if (table["fold"] >= 9).any() else None,
              "by_scenario": {s: summary(p) for s, p in table.groupby("scenario")},
              "by_layout": {s: summary(p) for s, p in table.groupby("layout")}}
    name = "report_all_layouts.json" if args.all_layouts else "report_12x1.json"
    (OUTPUT / name).write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("all", "unseen_folds_9_10")}, indent=1))
    for s, v in report["by_scenario"].items():
        print(f"  {s:16} n={v['images']:3}  corr {v['median_correlation']:.2f}  accepted {v['portal_accepts']}  same class {v['same_top_class']:.2f}")
    print(f"Wrote {OUTPUT / name}")


if __name__ == "__main__":
    main()
