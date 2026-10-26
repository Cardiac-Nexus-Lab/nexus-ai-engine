"""External test of the PTB-XL ECG classifier on hospitals it never saw.

Data: PhysioNet/Computing in Cardiology Challenge 2021 (CC BY 4.0), the Georgia
(Emory, USA), Chapman-Shaoxing and Ningbo (China) sources only. Its PTB and PTB-XL
sources are not used: the classifier was trained on PTB-XL.

The model is frozen: no retraining and no recalibration. Each recording goes
through the web portal's own steps (first 10 s, anti-aliased resampling to 100 Hz,
per-lead standardisation, vector-scaling calibration from PTB-XL fold 9).

Labels are SNOMED-CT codes, which do not translate one to one into PTB-XL's five
superclasses, so the mapping below was fixed and committed before any external
result was seen. For each class a recording is positive if it has one of the
class's codes, left out of that class if it has only a related code that is not
the same thing (AMBIGUOUS), and negative otherwise. NORM is positive only when every
code is a plain sinus rhythm, and negative when the recording has any code of the
other four classes; recordings with other findings are left out of NORM.

Usage:
    python scripts/validate_ecg_external.py --download     # fetch the three sources once (resumable)
    python scripts/validate_ecg_external.py
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np
import pandas as pd
import scipy.io
import torch
from scipy.signal import resample_poly
from sklearn.metrics import average_precision_score

from cardiac_nexus.evaluate import bootstrap_metric, expected_calibration_error
from evaluate_ecg import load_checkpoint

BASE = "https://physionet.org/files/challenge-2021/1.0.3/training"
SOURCES = {"georgia": "Georgia (Emory, USA)", "chapman_shaoxing": "Chapman-Shaoxing (China)",
           "ningbo": "Ningbo (China)"}
DATA = REPO_ROOT / "data" / "challenge2021"
OUTPUT = REPO_ROOT / "results" / "ecg_external"
CHECKPOINT = REPO_ROOT / "results" / "local_xresnet18_full" / "ecg_multilabel.pt"
REPORT = REPO_ROOT / "results" / "local_xresnet18_full" / "evaluation_report.json"
SAMPLES, SECONDS = 1000, 10.0

# Fixed before testing. Codes and names from the Challenge 2021 dx_mapping tables.
SINUS = {"426783006", "427393009", "426177001", "427084000"}  # sinus rhythm, arrhythmia, bradycardia, tachycardia
CLASS_CODES = {
    "MI": {"164865005", "57054005", "54329005", "164867002"},  # MI, acute MI, anterior MI, old MI
    "STTC": {"164934002", "59931005", "429622005", "164930006", "428750005", "164931005", "55930002",
             "164861001", "426434006", "425419005", "425623009", "413444003", "413844008", "111975006",
             "704997005"},  # T abnormal/inversion, ST depression/abnormal/elevation/changes, ischaemia, long QT
    "CD": {"270492004", "195042002", "233917008", "27885002", "54016002", "426183003", "164909002",
           "733534002", "251120003", "59118001", "713427006", "713426002", "6374002", "445118002",
           "445211001", "698252002", "164947007", "74390002", "195060002", "82226007"},  # AV blocks, BBBs, fascicular, IVCD, WPW
    "HYP": {"164873001", "89792004", "67741000119109", "446813000", "446358003", "253352002", "253339007",
            "195126007", "266249003"},  # LVH, RVH, atrial enlargement/abnormality/hypertrophy
}
AMBIGUOUS = {
    "MI": {"164917005"},                                     # Q wave abnormal: often, not always, infarction
    "STTC": {"428417006", "164937009", "251259000", "370365005"},  # early repolarisation, U wave, high T, LV strain
    "CD": {"39732003", "47665007"},                          # axis deviation
    "HYP": {"55827005", "67751000119106", "251223006"},     # voltage criteria only, tall P waves
}


def records(source: str) -> list[str]:
    """Record paths like 'g1/E00001'. Only the project root has a folder list; each folder has its own."""
    root = BASE.rsplit("/training", 1)[0]
    folders = [line[len(f"training/{source}/"):] for line in
               urllib.request.urlopen(f"{root}/RECORDS", timeout=60).read().decode().split()
               if line.startswith(f"training/{source}/")]
    names = []
    for folder in folders:
        listing = urllib.request.urlopen(f"{BASE}/{source}/{folder}RECORDS", timeout=60).read().decode().split()
        names += [f"{folder}{name}" for name in listing]
    return names


def download(sources, workers: int = 16) -> None:
    """Fetch .hea and .mat for every record; files already present are skipped."""
    for source in sources:
        names = records(source)
        jobs = [(f"{BASE}/{source}/{name}{ext}", DATA / source / f"{name}{ext}")
                for name in names for ext in (".hea", ".mat")]
        jobs = [(url, path) for url, path in jobs if not path.exists()]
        print(f"{source}: {len(names)} records, {len(jobs)} files to fetch", flush=True)

        def fetch(job):
            url, path = job
            path.parent.mkdir(parents=True, exist_ok=True)
            for attempt in range(4):
                try:
                    partial = path.with_suffix(path.suffix + ".part")
                    urllib.request.urlretrieve(url, partial)
                    partial.replace(path)
                    return True
                except Exception:
                    time.sleep(2 ** attempt)
            return False

        done = failed = 0
        with concurrent.futures.ThreadPoolExecutor(workers) as pool:
            for ok in pool.map(fetch, jobs):
                done += 1
                failed += not ok
                if done % 2000 == 0:
                    print(f"  {source}: {done}/{len(jobs)} ({failed} failed)", flush=True)
        print(f"{source}: finished, {failed} files failed", flush=True)


def read_record(header: Path):
    """Signal in mV as [12, samples], sampling rate and SNOMED codes, from a Challenge record."""
    lines = header.read_text().splitlines()
    name, leads, fs, length = lines[0].split()[:4]
    gains, baselines = [], []
    for line in lines[1:1 + int(leads)]:
        fields = line.split()
        gains.append(float(fields[2].split("/")[0].split("(")[0]))
        baselines.append(float(fields[2].split("(")[1].split(")")[0]) if "(" in fields[2] else 0.0)
    codes = next((l.split(":", 1)[1].strip().split(",") for l in lines if l.startswith("# Dx")), [])
    values = scipy.io.loadmat(str(header.with_suffix(".mat")))["val"].astype(np.float64)
    signal = (values - np.asarray(baselines)[:, None]) / np.asarray(gains)[:, None]
    return signal, float(fs), {c.strip() for c in codes if c.strip()}


def prepare(signal: np.ndarray, fs: float) -> np.ndarray | None:
    """The portal's steps: first 10 s (refuse under 9.5 s), anti-aliased resampling, per-lead z-score."""
    if signal.shape[0] != 12 or signal.shape[1] / fs < 9.5 or not np.isfinite(signal).all():
        return None
    signal = signal[:, : int(round(SECONDS * fs))]
    from fractions import Fraction
    ratio = Fraction(SAMPLES, signal.shape[1]).limit_denominator(1000)
    out = resample_poly(signal, ratio.numerator, ratio.denominator, axis=1, padtype="line")
    out = out[:, :SAMPLES] if out.shape[1] >= SAMPLES else np.pad(out, ((0, 0), (0, SAMPLES - out.shape[1])), mode="edge")
    out = (out - out.mean(axis=1, keepdims=True)) / (out.std(axis=1, keepdims=True) + 1e-6)
    return out.astype(np.float32)


def labels_for(codes: set[str]) -> dict:
    """1 positive, 0 negative, None left out, for each class."""
    out = {}
    for name, positive in CLASS_CODES.items():
        out[name] = 1 if codes & positive else (None if codes & AMBIGUOUS[name] else 0)
    abnormal = any(codes & c for c in CLASS_CODES.values())
    out["NORM"] = 1 if codes and codes <= SINUS else (0 if abnormal else None)
    return out


def class_metrics(truth: np.ndarray, probability: np.ndarray, resamples: int) -> dict:
    keep = ~np.isnan(truth)
    y, p = truth[keep].astype(int), probability[keep]
    if y.sum() == 0 or y.sum() == len(y):
        return {"recordings": int(len(y)), "positives": int(y.sum()), "auroc": None}
    auroc = bootstrap_metric(y, p, resamples=resamples)
    return {"recordings": int(len(y)), "positives": int(y.sum()),
            "auroc": auroc["estimate"], "auroc_95ci": [auroc["low"], auroc["high"]],
            "average_precision": float(average_precision_score(y, p)),
            "prevalence": float(y.mean()),
            "ece": expected_calibration_error(y, p)["ece"]}


def evaluate(resamples: int) -> None:
    device = torch.device("cpu")
    model, classes, _ = load_checkpoint(CHECKPOINT, device)
    calibration = json.loads(REPORT.read_text())["vector_scaling"]
    scale, bias = np.asarray(calibration["scales"]), np.asarray(calibration["biases"])

    rows, skipped = [], {}
    for source in SOURCES:
        batch, meta = [], []
        headers = sorted((DATA / source).rglob("*.hea"))
        for header in headers:
            if not header.with_suffix(".mat").exists():
                skipped[source] = skipped.get(source, 0) + 1
                continue
            signal, fs, codes = read_record(header)
            prepared = prepare(signal, fs)
            if prepared is None:
                skipped[source] = skipped.get(source, 0) + 1
                continue
            batch.append(prepared)
            meta.append((header.stem, codes))
            if len(batch) == 256 or header == headers[-1]:
                with torch.no_grad():
                    logits = model(torch.from_numpy(np.stack(batch)).to(device)).numpy()
                probabilities = 1.0 / (1.0 + np.exp(-(logits * scale + bias)))
                for (record, codes), p in zip(meta, probabilities):
                    rows.append({"source": source, "record": record, "codes": " ".join(sorted(codes)),
                                 **labels_for(codes), **{f"p_{c}": float(v) for c, v in zip(classes, p)}})
                batch, meta = [], []
        print(f"{source}: {sum(r['source'] == source for r in rows)} recordings scored, "
              f"{skipped.get(source, 0)} skipped", flush=True)

    table = pd.DataFrame(rows)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT / "external_predictions.csv.gz", index=False)
    groups = {name: table[table["source"] == name] for name in SOURCES} | {"all three": table}
    report = {"sources": SOURCES, "checkpoint": str(CHECKPOINT.relative_to(REPO_ROOT)),
              "label_map": {"sinus_only_for_NORM": sorted(SINUS),
                            "positive": {k: sorted(v) for k, v in CLASS_CODES.items()},
                            "ambiguous_left_out": {k: sorted(v) for k, v in AMBIGUOUS.items()}},
              "skipped": skipped, "results": {}}
    for group, part in groups.items():
        report["results"][group] = {"recordings": int(len(part)),
                                    **{c: class_metrics(part[c].astype(float).to_numpy(), part[f"p_{c}"].to_numpy(),
                                                        resamples) for c in classes}}
        aurocs = [report["results"][group][c]["auroc"] for c in classes if report["results"][group][c]["auroc"]]
        print(f"{group}: " + ", ".join(f"{c} {report['results'][group][c]['auroc']:.3f}" if report["results"][group][c]["auroc"]
                                       else f"{c} n/a" for c in classes), flush=True)
        report["results"][group]["macro_auroc_over_available"] = float(np.mean(aurocs)) if aurocs else None
    (OUTPUT / "external_report.json").write_text(json.dumps(report, indent=2))
    print(f"Wrote {OUTPUT}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--resamples", type=int, default=1000)
    args = parser.parse_args()
    if args.download:
        download(SOURCES)
    else:
        evaluate(args.resamples)


if __name__ == "__main__":
    main()
