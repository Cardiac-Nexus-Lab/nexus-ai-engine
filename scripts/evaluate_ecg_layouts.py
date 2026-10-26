"""What the ECG classifier loses when it only sees what a printed layout shows.

A 3x4 printout shows each lead for 2.5 of its 10 seconds (I, II, III for 0-2.5 s,
aVR, aVL, aVF for 2.5-5 s, and so on), plus any full-length rhythm strips. No
digitizer can recover the rest: it was never printed. The classifier, trained on
complete ten-second recordings, then sees a signal with most of every lead missing.

This measures that loss with no digitizer involved: PTB-XL test-fold signals
(fold 10) are reduced to what each layout prints, filled in one of two ways, and
scored with the deployed model and calibration. The drop is an upper bound on what
image input in that layout can reach, however good the digitizer is.

Fills compared:
* zeros: the unprinted time is 0, the per-lead mean after standardization;
* repeat: the printed window is repeated to fill ten seconds.
Each lead is standardized on its printed part only, as a digitized lead would be.

Usage:
    python scripts/evaluate_ecg_layouts.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np
import torch

from cardiac_nexus import data
from cardiac_nexus.ecg_layouts import LAYOUTS, reduce_to_layout
from cardiac_nexus.evaluate import evaluate_with_uncertainty
from evaluate_ecg import load_checkpoint

CHECKPOINT = REPO_ROOT / "results" / "local_xresnet18_full" / "ecg_multilabel.pt"
REPORT = REPO_ROOT / "results" / "local_xresnet18_full" / "evaluation_report.json"
OUTPUT = REPO_ROOT / "results" / "ecg_layouts"


def main() -> None:
    device = torch.device("cpu")
    model, classes, _ = load_checkpoint(CHECKPOINT, device)
    calibration = json.loads(REPORT.read_text())["vector_scaling"]
    scale, bias = np.asarray(calibration["scales"]), np.asarray(calibration["biases"])
    metadata, signals, splits = data.prepare()
    test = splits["test"]
    x = np.asarray(signals[test], dtype=np.float32)
    y = metadata[data.SUPERCLASSES].to_numpy(dtype=np.float32)[test]

    @torch.no_grad()
    def probabilities(batch: np.ndarray) -> np.ndarray:
        logits = np.concatenate([model(torch.from_numpy(batch[i:i + 256])).numpy() for i in range(0, len(batch), 256)])
        return 1.0 / (1.0 + np.exp(-(logits * scale + bias)))

    report = {"checkpoint": str(CHECKPOINT.relative_to(REPO_ROOT)), "test_records": int(len(test)), "layouts": {}}
    for layout in LAYOUTS:
        for fill in (("full",) if layout == "12x1" else ("zeros", "repeat")):
            batch = x if fill == "full" else reduce_to_layout(x, layout, fill)
            rows = evaluate_with_uncertainty(y, probabilities(batch), classes, resamples=1000)
            entry = {r["class"]: {"auroc": r["auroc"], "support": r["support"]} for r in rows}
            entry["macro_auroc"] = float(np.mean([r["auroc"]["estimate"] for r in rows]))
            report["layouts"][f"{layout} ({fill})"] = entry
            print(f"{layout:8} {fill:7} macro AUROC {entry['macro_auroc']:.3f}  " +
                  "  ".join(f"{r['class']} {r['auroc']['estimate']:.3f}" for r in rows), flush=True)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "layout_report.json").write_text(json.dumps(report, indent=2))
    print(f"Wrote {OUTPUT / 'layout_report.json'}")


if __name__ == "__main__":
    main()
