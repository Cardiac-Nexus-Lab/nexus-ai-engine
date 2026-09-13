"""Train the cardiac MRI segmentation model on ACDC.

Usage:
    python scripts/train_mri.py --epochs 120
    python scripts/train_mri.py --epochs 2 --limit-patients 10   # quick check
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cardiac_nexus import mri_data
from cardiac_nexus.mri_training import MRITrainConfig, train_segmenter

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--base-width", type=int, default=16,
                        help="channels at the first U-Net level; doubles per level")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit-patients", type=int, default=None,
                        help="use only this many training patients, for a quick pipeline check")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "results" / "mri_segmentation")
    args = parser.parse_args()

    metadata, cache, splits = mri_data.prepare()
    if args.limit_patients:
        splits = {"train": splits["train"][: args.limit_patients],
                  "val": splits["val"][: max(2, args.limit_patients // 4)],
                  "test": splits["test"]}

    config = MRITrainConfig(epochs=args.epochs, base_width=args.base_width, batch_size=args.batch_size,
                            learning_rate=args.learning_rate, weight_decay=args.weight_decay,
                            seed=args.seed, augment=not args.no_augment, device=args.device)
    result = train_segmenter(config, cache, splits, args.output)
    print(f"\nBest epoch {result['best_epoch']}: validation mean Dice {result['best_val_mean_dice']:.3f}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
