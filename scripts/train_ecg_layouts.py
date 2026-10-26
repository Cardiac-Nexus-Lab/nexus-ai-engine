"""Fine-tune the ECG classifier to read printed layouts, not only full recordings.

An image of a 3x4 printout holds 2.5 of each lead's 10 seconds, and the deployed
classifier, trained on complete recordings, loses macro AUROC 0.911 -> 0.83 on
exactly that input (results/ecg_layouts). This continues training the deployed
model with each training recording, at random, either left whole or reduced to
what one layout prints (cardiac_nexus.ecg_layouts, the same reduction
digitize_layout produces), on top of the original augmentation.

Model selection uses the mean validation macro AUROC (PTB-XL fold 9) over full
signals and all six layouts. Test results (fold 10) are reported per layout for
this model and for the deployed one on identical inputs, with one vector-scaling
calibration fitted on fold 9 across all inputs. The result replaces the deployed
model only by a separate, deliberate step, and should only if it is better on
every layout without a real loss on full recordings.

Usage:
    python scripts/train_ecg_layouts.py --epochs 12
    python scripts/train_ecg_layouts.py --epochs 1 --limit 600   # quick check
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from cardiac_nexus import data
from cardiac_nexus.augment import ECGAugment
from cardiac_nexus.ecg_layouts import LAYOUTS, reduce_to_layout
from cardiac_nexus.evaluate import VectorScaler, evaluate_with_uncertainty, expected_calibration_error
from cardiac_nexus.training import macro_auroc
from evaluate_ecg import load_checkpoint

DEPLOYED = REPO_ROOT / "results" / "local_xresnet18_full" / "ecg_multilabel.pt"
REDUCED = [layout for layout in LAYOUTS if layout != "12x1"]


class LayoutMaskedECG(Dataset):
    """Recordings, each either whole (probability `keep_whole`) or reduced to a random layout, then augmented."""

    def __init__(self, signals, labels, indices, keep_whole: float, seed: int):
        self.signals, self.labels, self.indices = signals, labels, np.asarray(indices)
        self.keep_whole = keep_whole
        self.rng = np.random.default_rng(seed)
        self.augment = ECGAugment(generator=np.random.default_rng(seed + 1))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        signal = np.asarray(self.signals[self.indices[item]], dtype=np.float32)
        if self.rng.random() >= self.keep_whole:
            signal = reduce_to_layout(signal, str(self.rng.choice(REDUCED)))
        return self.augment(torch.from_numpy(signal)), torch.from_numpy(self.labels[self.indices[item]])


@torch.no_grad()
def logits_for(model, x: np.ndarray, batch: int = 256) -> np.ndarray:
    model.eval()
    return np.concatenate([model(torch.from_numpy(x[i:i + batch])).numpy() for i in range(0, len(x), batch)])


def inputs_by_layout(x: np.ndarray) -> dict[str, np.ndarray]:
    return {"12x1": x} | {layout: reduce_to_layout(x, layout) for layout in REDUCED}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--keep-whole", type=float, default=0.3, help="share of training recordings left complete")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="use only the first N records of each split")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "results" / "ecg_layout_model")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cpu")

    model, classes, _ = load_checkpoint(DEPLOYED, device)
    deployed, _, _ = load_checkpoint(DEPLOYED, device)
    metadata, signals, splits = data.prepare()
    labels = metadata[data.SUPERCLASSES].to_numpy(dtype=np.float32)
    if args.limit:
        splits = {k: v[: args.limit] for k, v in splits.items()}
    val_inputs = inputs_by_layout(np.asarray(signals[splits["val"]], dtype=np.float32))
    val_y = labels[splits["val"]]

    train_labels = labels[splits["train"]]
    positives = train_labels.sum(axis=0)
    pos_weight = torch.tensor((len(train_labels) - positives) / np.maximum(positives, 1), dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    loader = DataLoader(LayoutMaskedECG(signals, labels, splits["train"], args.keep_whole, args.seed),
                        batch_size=args.batch_size, shuffle=True, num_workers=0)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.learning_rate, epochs=args.epochs,
                                                    steps_per_epoch=len(loader))

    def val_scores(m) -> dict[str, float]:
        return {layout: macro_auroc(val_y, 1 / (1 + np.exp(-logits_for(m, x)))) for layout, x in val_inputs.items()}

    start_scores = val_scores(model)
    history = [{"epoch": 0, "val": start_scores, "val_mean": float(np.mean(list(start_scores.values())))}]
    print(f"Deployed model, validation: mean {history[0]['val_mean']:.4f}  " +
          "  ".join(f"{k} {v:.3f}" for k, v in start_scores.items()), flush=True)
    best_mean, best_epoch, best_state = history[0]["val_mean"], 0, copy.deepcopy(model.state_dict())
    for epoch in range(1, args.epochs + 1):
        started, losses = time.perf_counter(), []
        model.train()
        for x, y in loader:
            loss = criterion(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())
        scores = val_scores(model)
        mean = float(np.mean(list(scores.values())))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val": scores, "val_mean": mean})
        marker = ""
        if mean > best_mean:
            best_mean, best_epoch, best_state = mean, epoch, copy.deepcopy(model.state_dict())
            marker = "  <- best"
        print(f"Epoch {epoch:02d}: loss {np.mean(losses):.4f} | val mean {mean:.4f}  " +
              "  ".join(f"{k} {v:.3f}" for k, v in scores.items()) + f" | {time.perf_counter() - started:.0f}s{marker}",
              flush=True)
    model.load_state_dict(best_state)

    # One calibration per model, fitted on validation logits across every input it will meet.
    scalers = {}
    for name, m in (("fine-tuned", model), ("deployed", deployed)):
        z = np.concatenate([logits_for(m, x) for x in val_inputs.values()])
        scaler = VectorScaler(len(classes))
        scaler.fit(z, np.concatenate([val_y] * len(val_inputs)))
        scalers[name] = scaler

    test_inputs = inputs_by_layout(np.asarray(signals[splits["test"]], dtype=np.float32))
    test_y = labels[splits["test"]]
    report = {"deployed_checkpoint": str(DEPLOYED.relative_to(REPO_ROOT)), "selected_epoch": best_epoch,
              "keep_whole": args.keep_whole, "epochs": args.epochs, "history": history, "test": {}}
    for layout, x in test_inputs.items():
        report["test"][layout] = {}
        for name, m in (("fine-tuned", model), ("deployed", deployed)):
            probabilities = scalers[name].apply(logits_for(m, x))
            rows = evaluate_with_uncertainty(test_y, probabilities, classes, resamples=1000)
            report["test"][layout][name] = {
                "macro_auroc": float(np.mean([r["auroc"]["estimate"] for r in rows])),
                "auroc": {r["class"]: r["auroc"] for r in rows},
                "mean_ece": float(np.mean([expected_calibration_error(test_y[:, i], probabilities[:, i])["ece"]
                                           for i in range(len(classes))])),
            }
        t = report["test"][layout]
        print(f"test {layout:8} fine-tuned {t['fine-tuned']['macro_auroc']:.3f} | deployed {t['deployed']['macro_auroc']:.3f}",
              flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "classes": classes, "architecture": "xresnet1d18",
                "selected_epoch": best_epoch, "val_mean_macro_auroc": best_mean,
                "vector_scaling": {"scales": scalers["fine-tuned"].scales.tolist(),
                                   "biases": scalers["fine-tuned"].biases.tolist()}},
               args.output / "ecg_multilabel.pt")
    (args.output / "layout_model_report.json").write_text(json.dumps(report, indent=2, default=float))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
