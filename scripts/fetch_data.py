"""Fetch and cache PTB-XL. Resumable: re-run after an interruption."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cardiac_nexus import data

if __name__ == "__main__":
    metadata, signals, splits = data.prepare()
    print(f"\nReady: {len(metadata):,} records, cache {signals.shape}")
    print("Splits:", {name: len(idx) for name, idx in splits.items()})
