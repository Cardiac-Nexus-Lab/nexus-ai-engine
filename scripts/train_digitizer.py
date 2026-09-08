"""Train the trace localizer that reads a waveform off a photographed ECG.

Usage:
    python scripts/train_digitizer.py --epochs 20
    python scripts/train_digitizer.py --epochs 2 --records-per-epoch 200   # quick check
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from cardiac_nexus import data
from cardiac_nexus.ecg_image.dataset import StripDataset
from cardiac_nexus.ecg_image.digitize import (
    STRIP_HEIGHT,
    TraceLocalizer,
    TraceLoss,
    expected_rows,
)
from cardiac_nexus.models import select_device

REPO_ROOT = Path(__file__).resolve().parents[1]


def flatten(batch: torch.Tensor) -> torch.Tensor:
    """[records, strips, ...] -> [records*strips, ...]; each strip is independent."""
    return batch.reshape(-1, *batch.shape[2:])


def run_epoch(model, loader, criterion, optimizer, device, training: bool):
    model.train(training)
    losses, errors, counts = [], [], 0
    for images, rows, valid in tqdm(loader, leave=False, desc="train" if training else "eval"):
        images = flatten(images).to(device)
        rows = flatten(rows).to(device)
        valid = flatten(valid).to(device)

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, rows)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            predicted = expected_rows(logits)
            # Score only where the trace was actually inside the strip.
            absolute_error = ((predicted - rows).abs() * valid).sum() / valid.sum().clamp_min(1)

        losses.append(loss.item() * len(images))
        errors.append(absolute_error.item() * len(images))
        counts += len(images)

    return float(np.sum(losses) / counts), float(np.sum(errors) / counts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--records-per-epoch", type=int, default=500,
                        help="records sampled per epoch; each yields 48 strips (12 leads x 4 windows)")
    parser.add_argument("--val-records", type=int, default=120)
    parser.add_argument("--batch-records", type=int, default=1, help="records per batch (x48 strips)")
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "results" / "digitizer")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = select_device(args.device)

    metadata, signals, splits = data.prepare()
    signals = np.asarray(signals)

    train_pool = splits["train"]
    val_records = splits["val"][: args.val_records]
    print(f"Device: {device}")
    print(f"  train pool: {len(train_pool):,} records, sampling {args.records_per_epoch:,} per epoch")
    print(f"  validation: {len(val_records):,} records ({len(val_records) * 48:,} strips, fixed)")

    # Validation is deterministic so the metric moves only when the model does.
    val_loader = DataLoader(
        StripDataset(signals, val_records, seed=args.seed, deterministic=True),
        batch_size=args.batch_records, shuffle=False, num_workers=args.workers,
    )

    model = TraceLocalizer().to(device)
    print(f"  parameters: {sum(p.numel() for p in model.parameters()):,}")
    criterion = TraceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history, best_error, best_epoch, best_state = [], np.inf, None, None
    for epoch in range(1, args.epochs + 1):
        # Fresh records and fresh distortions each epoch: the training set is
        # generated rather than fixed, so there is no reason to reuse either.
        sample = rng.choice(train_pool, size=min(args.records_per_epoch, len(train_pool)), replace=False)
        train_loader = DataLoader(
            StripDataset(signals, sample, seed=args.seed + epoch, deterministic=False),
            batch_size=args.batch_records, shuffle=True, num_workers=args.workers,
        )

        started = time.perf_counter()
        train_loss, train_error = run_epoch(model, train_loader, criterion, optimizer, device, True)
        val_loss, val_error = run_epoch(model, val_loader, criterion, optimizer, device, False)
        scheduler.step()

        history.append({"epoch": epoch, "train_loss": train_loss, "train_px": train_error,
                        "val_loss": val_loss, "val_px": val_error})
        marker = ""
        if val_error < best_error:
            best_error, best_epoch = val_error, epoch
            best_state = copy.deepcopy(model.state_dict())
            marker = "  <- best"
        print(f"Epoch {epoch:02d}: train {train_loss:.4f} ({train_error:5.2f} px) | "
              f"val {val_loss:.4f} ({val_error:5.2f} px) | "
              f"{time.perf_counter() - started:5.0f}s{marker}")

    model.load_state_dict(best_state)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state_dict": model.state_dict(), "strip_height": STRIP_HEIGHT,
         "best_epoch": best_epoch, "val_pixel_error": best_error, "history": history},
        args.output / "trace_localizer.pt",
    )
    (args.output / "digitizer_metrics.json").write_text(
        json.dumps({"best_epoch": best_epoch, "val_pixel_error": best_error,
                    "history": history, "config": vars(args) | {"output": str(args.output)}},
                   indent=2, default=str)
    )
    print(f"\nBest epoch {best_epoch}: {best_error:.2f} px mean absolute error")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
