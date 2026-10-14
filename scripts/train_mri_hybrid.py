"""Hybrid fine-tuning: gold masks from three cohorts plus Kaggle volume supervision.

Starts from the pooled segmenter and trains on two kinds of patient at once:

* the gold-standard training patients of ACDC, M&Ms and M&Ms-2, with the usual
  Dice plus cross-entropy over RV, myocardium and LV, so those boundaries keep
  their pixel-level supervision;
* the Kaggle cohort's non-test patients (mirror train and val splits), which
  have clinician-measured LV volumes but no masks: every one contributes a
  differentiable volume loss (`mri_weak.volume_loss`), and those whose predicted
  volumes already match the clinician's closely also contribute their LV mask as
  a silver-standard target.

Each step takes one gold batch and one Kaggle patient (its ED and ES stacks).
The Kaggle test patients are never read. `scripts/validate_mri_kaggle.py
--splits train val --save-edes` must have been run first.

No epoch is selected on held-out data: the 94 gold validation patients are
scored after every epoch only to show whether the volume objective is eroding
segmentation, and the final epoch is the model reported.

Usage:
    python scripts/train_mri_hybrid.py --epochs 5
    python scripts/train_mri_hybrid.py --epochs 1 --limit-steps 20     # quick check
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from cardiac_nexus import mri_cohorts
from cardiac_nexus.models import select_device
from cardiac_nexus.mri_dataset import ACDCSlices, normalize_volumes
from cardiac_nexus.mri_models import DiceCELoss
from cardiac_nexus.mri_training import (MRITrainConfig, encoder_widths, predict_volumes, save_checkpoint,
                                        score_patients, summarize_dice)
from cardiac_nexus.mri_weak import KaggleVolumes, lv_mask_loss, silver_patients, volume_loss
from evaluate_mri import load_segmenter
from train_mri_cv import cleaned, predict_ensemble, print_comparison, summarize

POOLED = REPO_ROOT / "results" / "mri_segmentation_pooled" / "mri_segmenter.pt"
KAGGLE_RESULTS = REPO_ROOT / "results" / "mri_segmentation_pooled" / "kaggle" / "kaggle_train_val_per_patient.csv"
EDES = REPO_ROOT / "data" / "kaggle" / "edes"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weak-weight", type=float, default=0.5, help="weight of the Kaggle losses")
    parser.add_argument("--silver-method", default="frustum", choices=["frustum", "cylindrical"])
    parser.add_argument("--bf16", action="store_true", help="bfloat16 autocast for forward passes on CPU")
    parser.add_argument("--limit-steps", type=int, default=None, help="stop each epoch early, for a quick check")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "results" / "mri_segmentation_hybrid")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    metadata, cache, splits = mri_cohorts.prepare()
    kaggle_table = pd.read_csv(KAGGLE_RESULTS)
    assert not (kaggle_table["source_split"] == "test").any(), "Kaggle test patients must not be used"
    kaggle_table = kaggle_table[kaggle_table["error"].fillna("") == ""].dropna(subset=["edv_true", "esv_true"])
    # Every stored stack has a 10 mm header spacing, but some originals list impossible spacings (1 mm, or
    # 18-340 mm); their clinician volumes cannot be matched to the stored stack, so they are left out.
    plausible = kaggle_table["dz_stated"].between(5.0, 12.5)
    print(f"Leaving out {int((~plausible).sum())} Kaggle patients with implausible stated slice spacing")
    kaggle_table = kaggle_table[plausible]
    silver = silver_patients(kaggle_table, args.silver_method)
    kaggle = KaggleVolumes(EDES, kaggle_table, silver)
    print(f"Gold training patients {len(splits['train'])}; Kaggle weak patients {len(kaggle)} "
          f"({sum(i['silver'] for i in kaggle.items)} silver, method {args.silver_method})")

    device = select_device(args.device)
    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    images = normalize_volumes(cache["images"], cache["patient"], cache["phase"])
    gold = ACDCSlices(images, cache["masks"], cache["patient"], cache["phase"], splits["train"], augment=True, seed=42)
    loader = DataLoader(gold, batch_size=16, shuffle=True, num_workers=0, drop_last=True)
    model, start = load_segmenter(POOLED, device)
    model.train()
    criterion = DiceCELoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    steps = min(len(loader), args.limit_steps or len(loader))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.learning_rate, epochs=args.epochs,
                                                    steps_per_epoch=steps, pct_start=0.1)
    autocast = (lambda: torch.autocast("cpu", dtype=torch.bfloat16)) if args.bf16 and device.type == "cpu" \
        else contextlib.nullcontext
    config = MRITrainConfig(epochs=args.epochs, learning_rate=args.learning_rate, init_checkpoint=str(POOLED))
    assert tuple(start["widths"]) == encoder_widths(config.base_width)

    history = []
    order = rng.permutation(len(kaggle))
    cursor = 0
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        model.train()
        gold_losses, weak_losses, ef_errors = [], [], []
        for step, (inputs, targets) in enumerate(loader):
            if step >= steps:
                break
            item = kaggle[int(order[cursor % len(order)])]
            cursor += 1
            with autocast():
                gold_loss = criterion(model(inputs.to(device)).float(), targets.to(device))
                ed_logits, es_logits = model(item["ed"].to(device)), model(item["es"].to(device))
            weak, estimate = volume_loss(ed_logits, es_logits, item["voxel_ml"], item["edv"], item["esv"])
            if item["silver"]:
                weak = weak + lv_mask_loss(ed_logits, item["ed_lv"].to(device)) \
                    + lv_mask_loss(es_logits, item["es_lv"].to(device))
            loss = gold_loss + args.weak_weight * weak
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            gold_losses.append(float(gold_loss))
            weak_losses.append(float(weak))
            ef_errors.append(abs((1 - estimate["esv_hat"] / max(estimate["edv_hat"], 1.0))
                                 - (1 - item["esv"] / item["edv"])) * 100)

        scored = score_patients(predict_volumes(model, images, cache["masks"], cache["patient"], cache["phase"],
                                                cache["spacing"], splits["val"], device))
        val_dice = summarize_dice(scored)
        history.append({"epoch": epoch, "gold_loss": float(np.mean(gold_losses)),
                        "weak_loss": float(np.mean(weak_losses)),
                        "train_soft_ef_abs_error_points": float(np.mean(ef_errors)),
                        "val_dice": val_dice, "val_mean_dice": float(np.mean(list(val_dice.values())))})
        save_checkpoint(args.output, model, config, epoch, val_dice,
                        {k: v for k, v in splits.items() if k != "test_by_cohort"}, history)
        (args.output / "mri_training_metrics.json").write_text(json.dumps(
            {"config": {**asdict(config), "weak_weight": args.weak_weight, "bf16": args.bf16,
                        "silver_method": args.silver_method},
             "kaggle_patients": len(kaggle), "silver_patients": sum(i["silver"] for i in kaggle.items),
             "history": history}, indent=2))
        print(f"Epoch {epoch:02d}: gold {history[-1]['gold_loss']:.4f} weak {history[-1]['weak_loss']:.4f} "
              f"soft EF error {history[-1]['train_soft_ef_abs_error_points']:.2f} pts | val Dice "
              f"LV {val_dice['LV']:.3f} MYO {val_dice['MYO']:.3f} RV {val_dice['RV']:.3f} | "
              f"{time.perf_counter() - started:.0f}s", flush=True)

    models = {"pooled model": load_segmenter(POOLED, device)[0],
              "hybrid model": load_segmenter(args.output / "mri_segmenter.pt", device)[0]}
    report = {"test": {}}
    for cohort, patients in splits["test_by_cohort"].items():
        reports = {}
        for name, candidate in models.items():
            volumes = predict_ensemble([candidate], images, cache["masks"], cache["patient"], cache["phase"],
                                       cache["spacing"], patients, device)
            reports[name] = summarize(volumes)
            reports[name + " + LCC"] = summarize(cleaned(volumes))
        report["test"][cohort] = reports
        print(f"\n{cohort} test set, {len(patients)} patients:")
        print_comparison(reports)
    (args.output / "hybrid_test_comparison.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"\nWrote {args.output / 'hybrid_test_comparison.json'}")


if __name__ == "__main__":
    main()
