"""Training and evaluation for the single-modality ECG model."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from .augment import ECGAugment
from .models import ECGClassifier, select_device


@dataclass
class TrainConfig:
    epochs: int = 15
    batch_size: int = 64
    learning_rate: float = 1e-3
    seed: int = 42
    device: str | None = None
    architecture: str = "cnn"
    weight_decay: float = 0.0
    dropout: float = 0.0
    augment: bool = False
    one_cycle: bool = False
    classes: list[str] = field(default_factory=lambda: ["NORM", "MI", "STTC", "CD", "HYP"])


def build_model(architecture: str, num_classes: int):
    """Return the classifier for the named architecture.

    "cnn" is the original three-layer baseline; the xresnet1d variants follow
    Strodthoff et al., whose xresnet1d101 leads the PTB-XL superclass benchmark.
    The deeper variants are far slower to train, so the choice is a real tradeoff
    rather than a formality.
    """
    if architecture == "cnn":
        return ECGClassifier(num_classes=num_classes)
    from .xresnet1d import XResNet1dClassifier

    return XResNet1dClassifier(num_classes=num_classes, architecture=architecture)


class ECGDataset(Dataset):
    """Serves pre-decoded signals held in memory.

    Tensors are built once here rather than per access, so __getitem__ is a slice.
    """

    def __init__(self, signals: np.ndarray, labels: np.ndarray, indices: np.ndarray,
                 augment: ECGAugment | None = None):
        self.signals = torch.from_numpy(np.ascontiguousarray(signals[indices]))
        self.labels = torch.from_numpy(np.ascontiguousarray(labels[indices]))
        self.augment = augment
        assert len(self.signals) == len(self.labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        signal = self.signals[index]
        if self.augment is not None:
            signal = self.augment(signal)
        return signal, self.labels[index]


def macro_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    scores = [
        roc_auc_score(y_true[:, i], y_prob[:, i])
        for i in range(y_true.shape[1])
        if len(np.unique(y_true[:, i])) > 1
    ]
    return float(np.mean(scores)) if scores else float("nan")


def per_class_metrics(y_true: np.ndarray, y_prob: np.ndarray, classes: list[str], threshold: float = 0.5):
    y_pred = (y_prob >= threshold).astype(int)
    rows = []
    for i, name in enumerate(classes):
        true_i, prob_i, pred_i = y_true[:, i], y_prob[:, i], y_pred[:, i]
        tn, fp, fn, tp = confusion_matrix(true_i, pred_i, labels=[0, 1]).ravel()
        rows.append(
            {
                "class": name,
                "auroc": float(roc_auc_score(true_i, prob_i)) if len(np.unique(true_i)) > 1 else float("nan"),
                "average_precision": float(average_precision_score(true_i, prob_i)),
                "f1": float(f1_score(true_i, pred_i, zero_division=0)),
                "sensitivity": float(tp / max(tp + fn, 1)),
                "specificity": float(tn / max(tn + fp, 1)),
                "support": int(true_i.sum()),
                "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            }
        )
    return rows


def _run_epoch(model, loader, criterion, optimizer, device, training: bool, scheduler=None):
    model.train(training)
    losses, labels, probabilities = [], [], []
    for signals, targets in tqdm(loader, leave=False, desc="train" if training else "eval"):
        signals, targets = signals.to(device), targets.to(device)
        with torch.set_grad_enabled(training):
            logits = model(signals)
            loss = criterion(logits, targets)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
        losses.append(loss.item() * len(targets))
        labels.append(targets.detach().cpu().numpy())
        probabilities.append(torch.sigmoid(logits).detach().cpu().numpy())
    return (
        float(np.sum(losses) / len(loader.dataset)),
        np.concatenate(labels),
        np.concatenate(probabilities),
    )


def train_ecg(
    signals: np.ndarray,
    labels: np.ndarray,
    splits: dict[str, np.ndarray],
    config: TrainConfig,
    output_dir: Path,
):
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    device = select_device(config.device)
    print(f"Device: {device}")

    loaders = {
        name: DataLoader(
            ECGDataset(
                signals,
                labels,
                indices,
                # Augmentation is a training-time regulariser; validation and test
                # must stay fixed or their metrics stop being comparable.
                augment=ECGAugment(generator=np.random.default_rng(config.seed))
                if (config.augment and name == "train")
                else None,
            ),
            batch_size=config.batch_size,
            shuffle=(name == "train"),
            num_workers=0,  # data is already resident; workers would only add IPC cost
            pin_memory=(device.type == "cuda"),
        )
        for name, indices in splits.items()
    }
    for name, loader in loaders.items():
        print(f"  {name}: {len(loader.dataset):,} recordings")

    model = build_model(config.architecture, len(config.classes)).to(device)
    parameters = sum(p.numel() for p in model.parameters())
    print(f"  architecture: {config.architecture} ({parameters:,} parameters)")

    # Per-class positive weighting, since the superclasses are far from balanced.
    train_labels = labels[splits["train"]]
    positives = train_labels.sum(axis=0)
    negatives = len(train_labels) - positives
    pos_weight = torch.tensor(negatives / np.maximum(positives, 1), dtype=torch.float32, device=device)
    print("  positive weights:", dict(zip(config.classes, pos_weight.cpu().numpy().round(2))))

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)

    # The PTB-XL benchmark trained with fastai's one-cycle policy rather than a
    # flat rate. It warms up, then anneals towards zero, which lets the run use a
    # higher peak rate early and settle into a flatter minimum late.
    scheduler = None
    if config.one_cycle:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.learning_rate,
            epochs=config.epochs,
            steps_per_epoch=len(loaders["train"]),
        )

    # Validation AUROC fluctuates between epochs, so the last epoch is not reliably
    # the best model. Keep the best-validation weights and restore them before testing.
    history, best_auc, best_epoch, best_state = [], -np.inf, None, None
    for epoch in range(1, config.epochs + 1):
        train_loss, _, _ = _run_epoch(model, loaders["train"], criterion, optimizer, device, True, scheduler)
        val_loss, val_y, val_p = _run_epoch(model, loaders["val"], criterion, optimizer, device, False)
        val_auc = macro_auroc(val_y, val_p)
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_macro_auroc": val_auc}
        )
        marker = ""
        if val_auc > best_auc:
            best_auc, best_epoch = val_auc, epoch
            best_state = copy.deepcopy(model.state_dict())
            marker = "  <- best"
        print(
            f"Epoch {epoch:02d}: train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_macro_AUROC={val_auc:.4f}{marker}"
        )

    model.load_state_dict(best_state)
    print(f"\nRestored epoch {best_epoch} (val macro AUROC {best_auc:.4f})")

    test_loss, test_y, test_p = _run_epoch(model, loaders["test"], criterion, optimizer, device, False)
    rows = per_class_metrics(test_y, test_p, config.classes)
    test_macro = float(np.nanmean([r["auroc"] for r in rows]))

    print(f"\nTest loss {test_loss:.4f} | macro AUROC {test_macro:.4f}\n")
    header = f"{'class':>6} {'AUROC':>7} {'AP':>7} {'F1':>7} {'Sens':>7} {'Spec':>7} {'n':>6}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['class']:>6} {r['auroc']:7.4f} {r['average_precision']:7.4f} "
            f"{r['f1']:7.4f} {r['sensitivity']:7.4f} {r['specificity']:7.4f} {r['support']:6d}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "ecg_multilabel.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "classes": config.classes,
            "architecture": config.architecture,
            "augment": config.augment,
            "one_cycle": config.one_cycle,
            "weight_decay": config.weight_decay,
            "selected_epoch": best_epoch,
            "val_macro_auroc": best_auc,
            "test_macro_auroc": test_macro,
            "history": history,
        },
        checkpoint_path,
    )

    metrics = {
        "config": {**config.__dict__, "device": str(device)},
        "selected_epoch": best_epoch,
        "val_macro_auroc": best_auc,
        "test_loss": test_loss,
        "test_macro_auroc": test_macro,
        "per_class": rows,
        "history": history,
    }
    (output_dir / "ecg_multilabel_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved {checkpoint_path.name} and ecg_multilabel_metrics.json to {output_dir}")
    return metrics
