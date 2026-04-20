#!/usr/bin/env python3
"""
Unified trainer for the five manuscript configurations.

--method choices:
    tru            (Temporal Residual U-Net; direct regression, no noise)
    ia_nonlinear   (Inference-Alignment, nonlinear DDPM schedule, x̂₀-predict)
    ia_linear      (Inference-Alignment, linear schedule, x̂₀-predict)
    std_ddim       (Standard conditional DDIM; v-predict on noised target)

All four wrappers share the same MultiScaleTemporalUNet_v13 backbone.
History dropout defaults to 0.03 across all methods; LR schedule is
warmup + cosine to zero (no floor).
"""

import argparse
import csv
import math
import os
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src import (
    MultiScaleTemporalUNet_v13,
    MaskedLongitudinalDataset,
    METHOD_CHOICES,
    build_wrapper,
)
from src.metrics import compute_ssim


def get_lr_lambda(warmup_steps, total_steps):
    """Warmup + cosine-to-zero LR schedule (no floor)."""
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return lr_lambda


def patient_level_split(dataset, val_split, seed, exclude_patients_tsv=None):
    """Split eyes by patient (MRN) to prevent leakage, with optional exclusion TSV."""
    all_eye_ids = dataset.get_all_eye_ids()
    eye_to_mrn = dataset.get_eye_to_mrn_mapping()

    mrn_to_eyes = defaultdict(list)
    for eid in all_eye_ids:
        mrn_to_eyes[eye_to_mrn[eid]].append(eid)

    all_mrns = sorted(mrn_to_eyes.keys())

    if exclude_patients_tsv:
        exclude_ids = set()
        with open(exclude_patients_tsv) as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                exclude_ids.add(row.get("hashed_mrn", row.get("mrn", "")))
        before = len(all_mrns)
        all_mrns = [m for m in all_mrns if m not in exclude_ids]
        print(f"Excluded {before - len(all_mrns)} patients "
              f"({before} -> {len(all_mrns)}) via {exclude_patients_tsv}")

    rng = random.Random(seed)
    rng.shuffle(all_mrns)

    split_idx = int(len(all_mrns) * (1 - val_split))
    train_mrns = set(all_mrns[:split_idx])
    val_mrns = set(all_mrns[split_idx:])

    train_eye_ids = set()
    for mrn in train_mrns:
        train_eye_ids.update(mrn_to_eyes[mrn])
    val_eye_ids = set()
    for mrn in val_mrns:
        val_eye_ids.update(mrn_to_eyes[mrn])

    return train_mrns, val_mrns, train_eye_ids, val_eye_ids


def run_inference(method, wrapper, history, deltas, temporal_mask, args, val_seed):
    if method == "tru":
        return wrapper.predict(history, deltas, temporal_mask)
    if method == "ia_nonlinear":
        return wrapper.x0hat_predict(history, deltas, temporal_mask,
                                     t_infer=int(args.t_infer), seed=val_seed)
    if method == "ia_linear":
        return wrapper.x0hat_predict(history, deltas, temporal_mask,
                                     t_infer=args.ia_linear_t_infer, seed=val_seed)
    if method == "std_ddim":
        return wrapper.x0hat_predict(history, deltas, temporal_mask,
                                     t_infer=int(args.t_infer), seed=val_seed)
    raise ValueError(f"Unknown method: {method}")


def main():
    parser = argparse.ArgumentParser(
        description="Unified trainer for TRU / IA-Nonlinear / IA-Linear / Std-DDIM."
    )

    # Required
    parser.add_argument("--method", type=str, required=True, choices=METHOD_CHOICES)
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory of PNGs following the documented naming schema.")
    parser.add_argument("--output_dir", type=str, required=True)

    # Optional training
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--max_history", type=int, default=6)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--history_dropout", type=float, default=0.03,
                        help="Canonical value across all five manuscript configurations.")

    # Method-specific hyperparameters
    parser.add_argument("--t_min", type=int, default=0,
                        help="IA-Nonlinear: lower training-timestep bound.")
    parser.add_argument("--t_max", type=int, default=1000,
                        help="IA-Nonlinear: upper training-timestep bound (exclusive).")
    parser.add_argument("--t_infer", type=float, default=200,
                        help="IA-Nonlinear / Std-DDIM-1step: inference timestep (int index).")
    parser.add_argument("--t_fixed", type=int, default=200,
                        help="TRU: vestigial constant timestep (see TRUWrapper docstring).")
    parser.add_argument("--ia_linear_t_min", type=float, default=0.05)
    parser.add_argument("--ia_linear_t_max", type=float, default=0.95)
    parser.add_argument("--ia_linear_t_infer", type=float, default=0.20)

    # Split and validation
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_every", type=int, default=5)
    parser.add_argument("--exclude_patients_tsv", type=str, default=None,
                        help="Optional TSV with a 'hashed_mrn' column; "
                             "matching patients are removed before splitting.")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Method: {args.method}")
    os.makedirs(args.output_dir, exist_ok=True)
    args_dict = vars(args).copy()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # -------------------- Patient-level split --------------------
    index_dataset = MaskedLongitudinalDataset(
        args.data_dir, max_history=args.max_history,
        image_size=args.image_size, use_augmentation=False,
    )
    train_mrns, val_mrns, train_eye_ids, val_eye_ids = patient_level_split(
        index_dataset, args.val_split, args.seed, args.exclude_patients_tsv,
    )
    print(f"Train: {len(train_mrns)} patients, {len(train_eye_ids)} eyes")
    print(f"Val:   {len(val_mrns)} patients, {len(val_eye_ids)} eyes")
    del index_dataset

    train_dataset = MaskedLongitudinalDataset(
        args.data_dir, max_history=args.max_history,
        image_size=args.image_size, use_augmentation=True,
        eye_ids=train_eye_ids,
    )
    val_dataset = MaskedLongitudinalDataset(
        args.data_dir, max_history=args.max_history,
        image_size=args.image_size, use_augmentation=False,
        eye_ids=val_eye_ids,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=4)
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # -------------------- Model + wrapper + EMA --------------------
    model = MultiScaleTemporalUNet_v13(
        max_history=args.max_history, base_channels=64,
        delta_emb_dim=256, history_dropout=args.history_dropout,
    ).to(device)
    wrapper = build_wrapper(args.method, model, args, device)

    ema_model = MultiScaleTemporalUNet_v13(
        max_history=args.max_history, base_channels=64,
        delta_emb_dim=256, history_dropout=0.0,
    ).to(device)
    ema_model.load_state_dict(model.state_dict())
    for p in ema_model.parameters():
        p.requires_grad = False

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # -------------------- Optimizer + schedule --------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total_steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, get_lr_lambda(args.warmup_steps, total_steps)
    )

    start_epoch = 0
    best_ssim = -1.0

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        ema_model.load_state_dict(ckpt["ema_model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_ssim = ckpt.get("best_ssim", -1.0)
        print(f"Resumed from epoch {start_epoch}, best_ssim={best_ssim:.4f}")

    # -------------------- Training loop --------------------
    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_losses = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            target = batch["target"].to(device)
            target_mask = batch["target_pixel_mask"].to(device)
            history = batch["history"].to(device)
            deltas = batch["deltas"].to(device)
            temporal_mask = batch["temporal_mask"].to(device)

            optimizer.zero_grad()
            loss = wrapper(target, target_mask, history, deltas, temporal_mask)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                for ema_p, p in zip(ema_model.parameters(), model.parameters()):
                    ema_p.data.mul_(0.999).add_(p.data, alpha=0.001)

            train_losses.append(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        epoch_loss = float(np.mean(train_losses))
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch+1}: train_loss={epoch_loss:.4f}, "
              f"lr={current_lr:.2e}", flush=True)

        # -------------------- Validation --------------------
        if (epoch + 1) % args.val_every == 0:
            ema_model.eval()
            ema_wrapper = build_wrapper(args.method, ema_model, args, device)
            ema_wrapper.eval()

            val_ssims = []
            val_seed = 12345
            with torch.no_grad():
                for vi, batch in enumerate(val_loader):
                    history = batch["history"].to(device)
                    deltas = batch["deltas"].to(device)
                    temporal_mask = batch["temporal_mask"].to(device)
                    target = batch["target"].to(device)

                    pred = run_inference(
                        args.method, ema_wrapper, history, deltas, temporal_mask,
                        args, val_seed + vi,
                    )

                    pred_np = pred.squeeze(1).cpu().numpy()
                    tgt_np = target.squeeze(1).cpu().numpy()
                    for bi in range(pred_np.shape[0]):
                        s = compute_ssim(tgt_np[bi], pred_np[bi])
                        if not np.isnan(s):
                            val_ssims.append(s)

            mean_ssim = float(np.mean(val_ssims)) if val_ssims else 0.0
            print(f"  Val SSIM: {mean_ssim:.4f} (n={len(val_ssims)})", flush=True)

            ckpt = {
                "epoch": epoch,
                "model": model.state_dict(),
                "ema_model": ema_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "args": args_dict,
                "best_ssim": max(best_ssim, mean_ssim),
            }
            if mean_ssim > best_ssim:
                best_ssim = mean_ssim
                torch.save(ckpt, os.path.join(args.output_dir, "checkpoint_best.pt"))
                print(f"  New best SSIM: {best_ssim:.4f} -> checkpoint_best.pt")
            torch.save(ckpt, os.path.join(args.output_dir, "checkpoint_latest.pt"))

    print(f"\nTraining complete. Best SSIM: {best_ssim:.4f}")
    print(f"Outputs in: {args.output_dir}")


if __name__ == "__main__":
    main()
