"""Train the multi-label ECG model locally.

Usage:
    python scripts/train_ecg.py                 # full run, best device available
    python scripts/train_ecg.py --epochs 2      # quick check
    python scripts/train_ecg.py --device cpu    # force CPU
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from cardiac_nexus import data
from cardiac_nexus.training import TrainConfig, train_ecg

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="cpu, mps, cuda; defaults to the fastest available")
    parser.add_argument("--limit", type=int, default=None, help="use only the first N records (smoke tests)")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "results" / "local")
    args = parser.parse_args()

    metadata, signals, splits = data.prepare()
    labels = metadata[data.SUPERCLASSES].to_numpy(dtype=np.float32)

    if args.limit:
        keep = np.arange(min(args.limit, len(metadata)))
        allowed = set(keep.tolist())
        splits = {name: np.array([i for i in idx if i in allowed]) for name, idx in splits.items()}
        print(f"Limited to the first {len(keep):,} records: " + str({k: len(v) for k, v in splits.items()}))

    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
        classes=data.SUPERCLASSES,
    )
    train_ecg(np.asarray(signals), labels, splits, config, args.output)


if __name__ == "__main__":
    main()
