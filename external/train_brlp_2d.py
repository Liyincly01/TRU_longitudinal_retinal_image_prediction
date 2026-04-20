#!/usr/bin/env python3
"""
2D-BrLP benchmark training script for longitudinal FAF prediction.

Adapts BrLP (Puglisi et al., MIA 2025) from 3D Brain MRI to 2D Retinal FAF.
Three-phase pipeline:
  Phase 1: Train 2D VAE on individual FAF frames (256×256 → 32×32×4 latent).
  Phase 2: Train latent diffusion (U-Net + ControlNet) with DDIM sampling.
  Inference: DDIM + Latent Average Stabilization (LAS) to reduce per-sample noise.

The dataset is reused from `src.backbone.MaskedLongitudinalDataset` so that
patient-level splitting, augmentation, and the filename schema are identical
to the TRU pipeline. A lightweight `SingleFrameDataset` wraps individual
frames for VAE pretraining.
"""

import argparse
import csv
import glob
import math
import os
import random
import re
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from skimage.metrics import structural_similarity as ssim_func
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode

# Allow `from src import ...` when invoking as `python external/train_brlp_2d.py`.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.backbone import MaskedLongitudinalDataset, extract_last_frame
from src.metrics import (
    calculate_atrophy_dice,
    calculate_copy_metrics,
    calculate_masked_metrics,
)

from external.models_brlp import (
    ContinuousTimeEmbedding,
    ControlNet2D,
    DDIMScheduler,
    DiffusionUNet2D,
    PatchDiscriminator,
    VAE2D,
)


# =============================================================================
# SINGLE-FRAME DATASET (for VAE pretraining)
# =============================================================================

class SingleFrameDataset(Dataset):
    """All individual frames, filtered by eye ID for patient-level splits."""

    def __init__(self, data_dir, image_size=256, use_augmentation=True, eye_ids=None):
        super().__init__()
        self.image_size = image_size
        self.use_augmentation = use_augmentation

        all_files = glob.glob(os.path.join(data_dir, "*.png"))
        image_files = [f for f in all_files if not f.endswith("_mask.png")]

        if eye_ids is not None:
            filtered = []
            for f in image_files:
                fn = os.path.basename(f)
                base = fn.replace("_reg.png", "").replace("_anchor.png", "")
                m = re.match(r"^(\d+)_(\d+)_([\d.]+)", base)
                if m:
                    eid = f"{m.group(1)}_{m.group(2)}"
                    if eid in eye_ids:
                        filtered.append(f)
                else:
                    parts = base.split("_")
                    if len(parts) >= 3:
                        eid = f"{parts[0]}_{parts[1]}"
                        if eid in eye_ids:
                            filtered.append(f)
            image_files = filtered

        self.image_files = image_files
        print(f"SingleFrameDataset: {len(self.image_files)} frames")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img = Image.open(self.image_files[idx]).convert("L")
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)

        if self.use_augmentation:
            if random.random() > 0.5:
                img = TF.hflip(img)
            angle = random.uniform(-10, 10)
            img = TF.affine(img, angle=angle, translate=(0, 0), scale=1.0, shear=0,
                            interpolation=InterpolationMode.BILINEAR, fill=0)

        img = TF.to_tensor(img)
        img = img * 2 - 1
        return img


# =============================================================================
# VISUALIZATION
# =============================================================================

def save_comparison_grid(history, target, generated, target_mask, path, epoch):
    history = (history + 1) / 2
    target = (target + 1) / 2
    generated = (generated + 1) / 2
    _, N, _, _, _ = history.shape
    B = history.shape[0]
    rows = min(B, 4)
    cols = N + 3
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    if rows == 1:
        axes = axes[None, :]
    for b in range(rows):
        for n in range(N):
            axes[b, n].imshow(history[b, n, 0].cpu(), cmap="gray", vmin=0, vmax=1)
            axes[b, n].axis("off")
            if b == 0:
                axes[b, n].set_title(f"t-{N-n}")
        axes[b, N].imshow(generated[b, 0].cpu(), cmap="gray", vmin=0, vmax=1)
        axes[b, N].axis("off")
        if b == 0:
            axes[b, N].set_title("BrLP")
        axes[b, N+1].imshow(target[b, 0].cpu(), cmap="gray", vmin=0, vmax=1)
        axes[b, N+1].axis("off")
        if b == 0:
            axes[b, N+1].set_title("Target")
        diff = torch.abs(generated[b, 0] - target[b, 0]).cpu()
        axes[b, N+2].imshow(diff, cmap="hot", vmin=0, vmax=0.5)
        axes[b, N+2].axis("off")
        if b == 0:
            axes[b, N+2].set_title("|Diff|")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_vae_samples(originals, reconstructions, path, epoch):
    originals = (originals + 1) / 2
    reconstructions = (reconstructions + 1) / 2
    rows = min(originals.shape[0], 4)
    fig, axes = plt.subplots(rows, 3, figsize=(9, rows * 3))
    if rows == 1:
        axes = axes[None, :]
    for i in range(rows):
        axes[i, 0].imshow(originals[i, 0].cpu(), cmap="gray", vmin=0, vmax=1)
        axes[i, 0].axis("off")
        if i == 0:
            axes[i, 0].set_title("Original")
        axes[i, 1].imshow(reconstructions[i, 0].cpu(), cmap="gray", vmin=0, vmax=1)
        axes[i, 1].axis("off")
        if i == 0:
            axes[i, 1].set_title("Recon")
        diff = torch.abs(originals[i, 0] - reconstructions[i, 0]).cpu()
        axes[i, 2].imshow(diff, cmap="hot", vmin=0, vmax=0.3)
        axes[i, 2].axis("off")
        if i == 0:
            axes[i, 2].set_title("|Diff|")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# =============================================================================
# PHASE 1: VAE TRAINING
# =============================================================================

def _warmup_cosine_floor_1e2(warmup_steps, total_steps):
    """Warmup + cosine schedule with floor at 1e-2 of peak (VAE default)."""
    def lr_lambda(step):
        if step < warmup_steps:
            return max(float(step) / float(max(1, warmup_steps)), 1e-2)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.5 * (1.0 + math.cos(math.pi * progress)), 1e-2)
    return lr_lambda


def train_vae(args, train_eye_ids, val_eye_ids, device):
    print("\n" + "=" * 70)
    print("PHASE 1: Training 2D VAE")
    print("=" * 70 + "\n", flush=True)

    vae_dir = os.path.join(args.output_dir, "vae")
    os.makedirs(vae_dir, exist_ok=True)

    train_ds = SingleFrameDataset(args.data_dir, image_size=args.image_size,
                                  use_augmentation=True, eye_ids=train_eye_ids)
    val_ds = SingleFrameDataset(args.data_dir, image_size=args.image_size,
                                use_augmentation=False, eye_ids=val_eye_ids)
    train_loader = DataLoader(train_ds, batch_size=args.vae_batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.vae_batch_size, shuffle=False,
                            num_workers=4)

    vae = VAE2D(in_channels=1, latent_channels=args.latent_channels,
                num_channels=(64, 128, 256, 256), num_res_blocks=2).to(device)
    discriminator = PatchDiscriminator(in_channels=1, num_channels=32, num_layers=3).to(device)

    print(f"VAE: {sum(p.numel() for p in vae.parameters())/1e6:.2f}M params, "
          f"Discriminator: {sum(p.numel() for p in discriminator.parameters())/1e6:.2f}M params")

    opt_vae = torch.optim.AdamW(vae.parameters(), lr=args.vae_lr, weight_decay=1e-4)
    opt_disc = torch.optim.AdamW(discriminator.parameters(), lr=args.vae_lr, weight_decay=1e-4)

    total_steps = args.vae_epochs * len(train_loader)
    warmup_steps = min(1000, total_steps // 10)
    sched_vae = torch.optim.lr_scheduler.LambdaLR(
        opt_vae, _warmup_cosine_floor_1e2(warmup_steps, total_steps))
    sched_disc = torch.optim.lr_scheduler.LambdaLR(
        opt_disc, _warmup_cosine_floor_1e2(warmup_steps, total_steps))

    kl_weight = 1e-7
    adv_weight = 0.025
    adv_start_epoch = max(1, args.vae_epochs // 10)

    best_val_loss = float("inf")
    start_epoch = 0
    ckpt_path = os.path.join(vae_dir, "checkpoint_latest.pt")
    if os.path.exists(ckpt_path) and not args.force_retrain_vae:
        print(f"Resuming VAE from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        vae.load_state_dict(ckpt["vae"])
        discriminator.load_state_dict(ckpt["discriminator"])
        opt_vae.load_state_dict(ckpt["opt_vae"])
        opt_disc.load_state_dict(ckpt["opt_disc"])
        sched_vae.load_state_dict(ckpt["sched_vae"])
        sched_disc.load_state_dict(ckpt["sched_disc"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"  Resumed at epoch {start_epoch}, best_val_loss={best_val_loss:.4f}", flush=True)

    for epoch in range(start_epoch, args.vae_epochs):
        vae.train()
        discriminator.train()
        train_losses = []

        for batch in train_loader:
            x = batch.to(device)

            recon, mean, logvar = vae(x)
            recon_loss = F.l1_loss(recon, x)
            kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
            vae_loss = recon_loss + kl_weight * kl_loss

            if epoch >= adv_start_epoch:
                disc_fake = discriminator(recon)
                adv_loss = F.mse_loss(disc_fake, torch.ones_like(disc_fake))
                vae_loss = vae_loss + adv_weight * adv_loss

            opt_vae.zero_grad()
            vae_loss.backward()
            nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
            opt_vae.step()
            sched_vae.step()

            if epoch >= adv_start_epoch:
                opt_disc.zero_grad()
                disc_real = discriminator(x)
                disc_fake = discriminator(recon.detach())
                d_loss = 0.5 * (
                    F.mse_loss(disc_real, torch.ones_like(disc_real))
                    + F.mse_loss(disc_fake, torch.zeros_like(disc_fake))
                )
                d_loss.backward()
                opt_disc.step()
                sched_disc.step()

            train_losses.append(vae_loss.item())

        epoch_loss = float(np.mean(train_losses))

        if (epoch + 1) % 5 == 0:
            vae.eval()
            val_losses, val_ssims = [], []
            with torch.no_grad():
                for batch in val_loader:
                    x = batch.to(device)
                    recon, _, _ = vae(x)
                    val_losses.append(F.l1_loss(recon, x).item())
                    x_01 = (x.cpu() + 1) / 2
                    r_01 = (recon.cpu().clamp(-1, 1) + 1) / 2
                    for i in range(x.shape[0]):
                        try:
                            val_ssims.append(
                                ssim_func(x_01[i, 0].numpy(), r_01[i, 0].numpy(),
                                          data_range=1.0))
                        except Exception:
                            pass
            val_loss = float(np.mean(val_losses))
            val_ssim = float(np.mean(val_ssims)) if val_ssims else 0.0
            print(f"VAE Epoch {epoch+1}/{args.vae_epochs}: "
                  f"train_loss={epoch_loss:.4f}, val_recon={val_loss:.4f}, "
                  f"val_ssim={val_ssim:.4f}", flush=True)

            if (epoch + 1) % 20 == 0:
                with torch.no_grad():
                    sample = next(iter(val_loader))[:8].to(device)
                    recon, _, _ = vae(sample)
                    save_vae_samples(sample.cpu(), recon.cpu(),
                                     os.path.join(vae_dir, f"vae_epoch{epoch+1:03d}.png"),
                                     epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({"vae": vae.state_dict(), "epoch": epoch,
                            "best_val_loss": best_val_loss},
                           os.path.join(vae_dir, "checkpoint_best.pt"))

            torch.save({
                "vae": vae.state_dict(),
                "discriminator": discriminator.state_dict(),
                "opt_vae": opt_vae.state_dict(),
                "opt_disc": opt_disc.state_dict(),
                "sched_vae": sched_vae.state_dict(),
                "sched_disc": sched_disc.state_dict(),
                "epoch": epoch, "best_val_loss": best_val_loss,
            }, os.path.join(vae_dir, "checkpoint_latest.pt"))

    # Load best
    best_ckpt = torch.load(os.path.join(vae_dir, "checkpoint_best.pt"),
                           map_location=device, weights_only=False)
    vae.load_state_dict(best_ckpt["vae"])
    print(f"\nVAE training complete. Best val_recon={best_val_loss:.4f}")
    return vae


# =============================================================================
# PHASE 2: LATENT DIFFUSION
# =============================================================================

def train_diffusion(args, vae, train_eye_ids, val_eye_ids, device):
    print("\n" + "=" * 70)
    print("PHASE 2: Training Latent Diffusion (U-Net + ControlNet)")
    print("=" * 70 + "\n", flush=True)

    diff_dir = os.path.join(args.output_dir, "diffusion")
    os.makedirs(diff_dir, exist_ok=True)

    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    with torch.no_grad():
        dummy = torch.randn(1, 1, args.image_size, args.image_size, device=device)
        dummy_latent = vae.encode_to_latent(dummy)
        latent_shape = dummy_latent.shape[1:]
        print(f"Latent shape: {tuple(latent_shape)}")

    train_ds = MaskedLongitudinalDataset(
        args.data_dir, max_history=args.max_history,
        image_size=args.image_size, use_augmentation=True,
        eye_ids=train_eye_ids,
    )
    val_ds = MaskedLongitudinalDataset(
        args.data_dir, max_history=args.max_history,
        image_size=args.image_size, use_augmentation=False,
        eye_ids=val_eye_ids,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=4)

    delta_embed = ContinuousTimeEmbedding(dim=args.context_dim).to(device)

    unet = DiffusionUNet2D(
        in_channels=args.latent_channels, out_channels=args.latent_channels,
        num_channels=(256, 512, 768), num_res_blocks=2,
        attention_levels=(False, True, True), context_dim=args.context_dim,
    ).to(device)
    controlnet = ControlNet2D(
        in_channels=args.latent_channels, cond_channels=args.latent_channels + 1,
        num_channels=(256, 512, 768), num_res_blocks=2,
        attention_levels=(False, True, True), context_dim=args.context_dim,
    ).to(device)
    controlnet.load_state_dict(unet.state_dict(), strict=False)

    scheduler = DDIMScheduler(num_train_timesteps=1000,
                              beta_start=0.0015, beta_end=0.0205,
                              schedule="scaled_linear")

    print(f"U-Net: {sum(p.numel() for p in unet.parameters())/1e6:.2f}M, "
          f"ControlNet: {sum(p.numel() for p in controlnet.parameters())/1e6:.2f}M, "
          f"DeltaEmbed: {sum(p.numel() for p in delta_embed.parameters())/1e6:.2f}M",
          flush=True)

    all_params = (list(unet.parameters())
                  + list(controlnet.parameters())
                  + list(delta_embed.parameters()))
    optimizer = torch.optim.AdamW(all_params, lr=args.lr, weight_decay=1e-4)
    total_steps = args.epochs * len(train_loader)
    warmup_steps = args.warmup_steps

    sched = torch.optim.lr_scheduler.LambdaLR(
        optimizer, _warmup_cosine_floor_1e2(warmup_steps, total_steps))

    ema_unet = DiffusionUNet2D(
        in_channels=args.latent_channels, out_channels=args.latent_channels,
        num_channels=(256, 512, 768), num_res_blocks=2,
        attention_levels=(False, True, True), context_dim=args.context_dim,
    ).to(device)
    ema_controlnet = ControlNet2D(
        in_channels=args.latent_channels, cond_channels=args.latent_channels + 1,
        num_channels=(256, 512, 768), num_res_blocks=2,
        attention_levels=(False, True, True), context_dim=args.context_dim,
    ).to(device)
    ema_delta_embed = ContinuousTimeEmbedding(dim=args.context_dim).to(device)
    ema_unet.load_state_dict(unet.state_dict())
    ema_controlnet.load_state_dict(controlnet.state_dict())
    ema_delta_embed.load_state_dict(delta_embed.state_dict())
    for p in ema_unet.parameters():
        p.requires_grad = False
    for p in ema_controlnet.parameters():
        p.requires_grad = False
    for p in ema_delta_embed.parameters():
        p.requires_grad = False

    best_ssim = -1.0
    best_latent_mse = float("inf")
    training_history = {
        "train_loss": [], "val_latent_mse": [], "ssim": [], "psnr": [],
        "dice": [], "copy_sim": [], "lr": [],
    }

    start_epoch = 0
    ckpt_path = os.path.join(diff_dir, "checkpoint_latest.pt")
    if os.path.exists(ckpt_path) and not args.force_retrain_diff:
        print(f"Resuming diffusion from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        unet.load_state_dict(ckpt["unet"])
        controlnet.load_state_dict(ckpt["controlnet"])
        delta_embed.load_state_dict(ckpt["delta_embed"])
        ema_unet.load_state_dict(ckpt["ema_unet"])
        ema_controlnet.load_state_dict(ckpt["ema_controlnet"])
        ema_delta_embed.load_state_dict(ckpt["ema_delta_embed"])
        optimizer.load_state_dict(ckpt["optimizer"])
        sched.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_ssim = ckpt.get("best_ssim", -1.0)
        best_latent_mse = ckpt.get("best_latent_mse", float("inf"))
        training_history = ckpt.get("training_history", training_history)
        print(f"  Resumed at epoch {start_epoch}, best_ssim={best_ssim:.4f}", flush=True)

    ema_decay = 0.999

    for epoch in range(start_epoch, args.epochs):
        unet.train()
        controlnet.train()
        delta_embed.train()
        train_losses = []

        for batch in train_loader:
            target = batch["target"].to(device)
            history = batch["history"].to(device)
            deltas = batch["deltas"].to(device)
            temporal_mask = batch["temporal_mask"].to(device)
            B = target.shape[0]

            with torch.no_grad():
                z_target = vae.encode_to_latent(target)
                last_frame = extract_last_frame(history, temporal_mask)
                z_baseline = vae.encode_to_latent(last_frame)

            recent_deltas = []
            for b in range(B):
                valid = (~temporal_mask[b]).nonzero(as_tuple=True)[0]
                recent_deltas.append(deltas[b, valid[-1]] if len(valid) > 0
                                      else deltas[b, -1])
            recent_deltas = torch.stack(recent_deltas)

            dt_emb = delta_embed(recent_deltas).unsqueeze(1)
            dt_spatial = recent_deltas.view(B, 1, 1, 1).expand(
                B, 1, z_baseline.shape[2], z_baseline.shape[3]
            )
            controlnet_cond = torch.cat([z_baseline, dt_spatial], dim=1)

            z_residual = z_target - z_baseline
            noise = torch.randn_like(z_residual)
            t = torch.randint(0, 1000, (B,), device=device)
            z_noisy = scheduler.add_noise(z_residual, noise, t)

            cn_residuals = controlnet(z_noisy, t, dt_emb, controlnet_cond)
            noise_pred = unet(z_noisy, t, dt_emb, cn_residuals)

            loss = F.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(all_params, 1.0)
            optimizer.step()
            sched.step()

            with torch.no_grad():
                for ema_p, p in zip(ema_unet.parameters(), unet.parameters()):
                    ema_p.data.mul_(ema_decay).add_(p.data, alpha=1 - ema_decay)
                for ema_p, p in zip(ema_controlnet.parameters(), controlnet.parameters()):
                    ema_p.data.mul_(ema_decay).add_(p.data, alpha=1 - ema_decay)
                for ema_p, p in zip(ema_delta_embed.parameters(), delta_embed.parameters()):
                    ema_p.data.mul_(ema_decay).add_(p.data, alpha=1 - ema_decay)

            train_losses.append(loss.item())

        epoch_loss = float(np.mean(train_losses))
        current_lr = optimizer.param_groups[0]["lr"]
        training_history["train_loss"].append(epoch_loss)
        training_history["lr"].append(current_lr)
        print(f"\nEpoch {epoch+1}/{args.epochs}: train_loss={epoch_loss:.6f}, "
              f"lr={current_lr:.2e}", flush=True)

        if (epoch + 1) % 5 == 0:
            ema_unet.eval()
            ema_controlnet.eval()
            ema_delta_embed.eval()

            val_latent_mses = []
            all_tgt, all_pred, all_mask, all_hist, all_tmask = [], [], [], [], []
            max_val_batches = 10

            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    if batch_idx >= max_val_batches:
                        break
                    target = batch["target"].to(device)
                    history = batch["history"].to(device)
                    delta_vals = batch["deltas"].to(device)
                    temporal_mask_b = batch["temporal_mask"].to(device)
                    target_mask_b = batch["target_pixel_mask"].to(device)
                    B = target.shape[0]

                    z_target = vae.encode_to_latent(target)
                    last_frame = extract_last_frame(history, temporal_mask_b)
                    z_baseline = vae.encode_to_latent(last_frame)

                    recent_deltas = []
                    for b in range(B):
                        valid = (~temporal_mask_b[b]).nonzero(as_tuple=True)[0]
                        recent_deltas.append(delta_vals[b, valid[-1]] if len(valid) > 0
                                              else delta_vals[b, -1])
                    recent_deltas = torch.stack(recent_deltas)

                    dt_emb = ema_delta_embed(recent_deltas).unsqueeze(1)
                    dt_spatial = recent_deltas.view(B, 1, 1, 1).expand(
                        B, 1, z_baseline.shape[2], z_baseline.shape[3]
                    )
                    controlnet_cond = torch.cat([z_baseline, dt_spatial], dim=1)

                    z_pred_residual = torch.randn_like(z_baseline)
                    timesteps = scheduler.get_timesteps(args.num_inference_steps).to(device)
                    for i, t_val in enumerate(timesteps):
                        t_batch = torch.full((B,), t_val.item(), device=device, dtype=torch.long)
                        t_prev = timesteps[i + 1].item() if i + 1 < len(timesteps) else 0
                        cn_res = ema_controlnet(z_pred_residual, t_batch, dt_emb, controlnet_cond)
                        noise_pred = ema_unet(z_pred_residual, t_batch, dt_emb, cn_res)
                        z_pred_residual, _ = scheduler.step(
                            noise_pred, t_val.item(), z_pred_residual, t_prev)

                    z_pred = z_baseline + z_pred_residual
                    val_latent_mses.append(F.mse_loss(z_pred, z_target).item())
                    pred_pixel = vae.decode(z_pred).clamp(-1, 1)

                    all_tgt.append(target.cpu())
                    all_pred.append(pred_pixel.cpu())
                    all_mask.append(target_mask_b.cpu())
                    all_hist.append(history.cpu())
                    all_tmask.append(temporal_mask_b.cpu())

            val_latent_mse = float(np.mean(val_latent_mses))
            all_tgt = torch.cat(all_tgt, dim=0)
            all_pred = torch.cat(all_pred, dim=0)
            all_mask = torch.cat(all_mask, dim=0)
            all_hist = torch.cat(all_hist, dim=0)
            all_tmask = torch.cat(all_tmask, dim=0)

            psnr_val, ssim_val = calculate_masked_metrics(all_tgt, all_pred, all_mask)
            dice_val = calculate_atrophy_dice(all_tgt, all_pred, all_mask)
            copy_sim = calculate_copy_metrics(all_pred, all_hist, all_tmask)

            training_history["val_latent_mse"].append(val_latent_mse)
            training_history["ssim"].append(ssim_val)
            training_history["psnr"].append(psnr_val)
            training_history["dice"].append(dice_val)
            training_history["copy_sim"].append(copy_sim)

            print(f"  Val: latent_mse={val_latent_mse:.6f}, "
                  f"SSIM={ssim_val:.4f}, PSNR={psnr_val:.2f}, "
                  f"Dice={dice_val:.4f}, CopySim={copy_sim:.4f}", flush=True)

            if (epoch + 1) % 20 == 0:
                n_vis = min(4, all_tgt.shape[0])
                save_comparison_grid(
                    all_hist[:n_vis], all_tgt[:n_vis],
                    all_pred[:n_vis], all_mask[:n_vis],
                    os.path.join(diff_dir, f"diff_epoch{epoch+1:03d}.png"), epoch,
                )

            if ssim_val > best_ssim:
                best_ssim = ssim_val
                torch.save({
                    "unet": unet.state_dict(),
                    "controlnet": controlnet.state_dict(),
                    "delta_embed": delta_embed.state_dict(),
                    "ema_unet": ema_unet.state_dict(),
                    "ema_controlnet": ema_controlnet.state_dict(),
                    "ema_delta_embed": ema_delta_embed.state_dict(),
                    "vae": vae.state_dict(), "epoch": epoch,
                    "best_ssim": best_ssim, "best_latent_mse": best_latent_mse,
                }, os.path.join(diff_dir, "checkpoint_best_ssim.pt"))
                print(f"  -> New best SSIM: {best_ssim:.4f}")

            if val_latent_mse < best_latent_mse:
                best_latent_mse = val_latent_mse
                torch.save({
                    "unet": unet.state_dict(),
                    "controlnet": controlnet.state_dict(),
                    "delta_embed": delta_embed.state_dict(),
                    "ema_unet": ema_unet.state_dict(),
                    "ema_controlnet": ema_controlnet.state_dict(),
                    "ema_delta_embed": ema_delta_embed.state_dict(),
                    "vae": vae.state_dict(), "epoch": epoch,
                    "best_ssim": best_ssim, "best_latent_mse": best_latent_mse,
                }, os.path.join(diff_dir, "checkpoint_best_latent_mse.pt"))
                print(f"  -> New best Latent MSE: {best_latent_mse:.6f}")

            torch.save({
                "unet": unet.state_dict(),
                "controlnet": controlnet.state_dict(),
                "delta_embed": delta_embed.state_dict(),
                "ema_unet": ema_unet.state_dict(),
                "ema_controlnet": ema_controlnet.state_dict(),
                "ema_delta_embed": ema_delta_embed.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": sched.state_dict(),
                "epoch": epoch, "best_ssim": best_ssim,
                "best_latent_mse": best_latent_mse,
                "training_history": training_history,
            }, os.path.join(diff_dir, "checkpoint_latest.pt"))

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes[0, 0].plot(training_history["train_loss"]);  axes[0, 0].set_title("Train Loss (Noise MSE)")
    axes[0, 1].plot(training_history["val_latent_mse"]); axes[0, 1].set_title("Val Latent MSE")
    axes[0, 2].plot(training_history["ssim"]);        axes[0, 2].set_title("Val SSIM")
    axes[1, 0].plot(training_history["psnr"]);        axes[1, 0].set_title("Val PSNR")
    axes[1, 1].plot(training_history["dice"]);        axes[1, 1].set_title("Val Dice")
    axes[1, 2].plot(training_history["lr"]);          axes[1, 2].set_title("Learning Rate")
    plt.tight_layout()
    plt.savefig(os.path.join(diff_dir, "training_curves.png"), dpi=150)
    plt.close()

    print(f"\nDiffusion training complete. Best SSIM={best_ssim:.4f}, "
          f"Best Latent MSE={best_latent_mse:.6f}")


# =============================================================================
# PATIENT-LEVEL SPLIT (mirrors train.py)
# =============================================================================

def patient_level_split(data_dir, val_split, seed, max_history, image_size,
                        exclude_patients_tsv=None):
    idx_ds = MaskedLongitudinalDataset(
        data_dir, max_history=max_history, image_size=image_size,
        use_augmentation=False,
    )
    all_eye_ids = idx_ds.get_all_eye_ids()
    eye_to_mrn = idx_ds.get_eye_to_mrn_mapping()

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
        print(f"Excluded {before - len(all_mrns)} patients")

    rng = random.Random(seed)
    rng.shuffle(all_mrns)
    split_idx = int(len(all_mrns) * (1 - val_split))
    train_mrns = set(all_mrns[:split_idx])
    val_mrns = set(all_mrns[split_idx:])
    train_eye_ids = {e for m in train_mrns for e in mrn_to_eyes[m]}
    val_eye_ids = {e for m in val_mrns for e in mrn_to_eyes[m]}
    print(f"Train: {len(train_mrns)} patients, {len(train_eye_ids)} eyes")
    print(f"Val:   {len(val_mrns)} patients, {len(val_eye_ids)} eyes", flush=True)
    return train_eye_ids, val_eye_ids


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="2D-BrLP: Latent Diffusion for Longitudinal FAF Prediction."
    )

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_history", type=int, default=6)
    parser.add_argument("--image_size", type=int, default=256)

    parser.add_argument("--vae_epochs", type=int, default=100)
    parser.add_argument("--vae_batch_size", type=int, default=16)
    parser.add_argument("--vae_lr", type=float, default=1e-4)
    parser.add_argument("--latent_channels", type=int, default=4)
    parser.add_argument("--force_retrain_vae", action="store_true")

    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--context_dim", type=int, default=256)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--las_m", type=int, default=5,
                        help="Number of LAS samples for inference averaging.")
    parser.add_argument("--force_retrain_diff", action="store_true")

    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude_patients_tsv", type=str, default=None)
    parser.add_argument("--skip_vae", action="store_true",
                        help="Skip VAE training (load from checkpoint_best.pt).")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("2D-BrLP: Latent Diffusion for Longitudinal FAF Prediction")
    print("  Adapted from BrLP (Puglisi et al., MIA 2025)")
    print(f"  Image: {args.image_size}x{args.image_size} -> latent (8x downsample)")
    print(f"  VAE: {args.vae_epochs} epochs, Diffusion: {args.epochs} epochs")
    print(f"  DDIM: {args.num_inference_steps} steps, LAS m={args.las_m}")
    print("=" * 70, flush=True)

    train_eye_ids, val_eye_ids = patient_level_split(
        args.data_dir, args.val_split, args.seed,
        args.max_history, args.image_size,
        exclude_patients_tsv=args.exclude_patients_tsv,
    )

    vae_best_path = os.path.join(args.output_dir, "vae", "checkpoint_best.pt")
    if args.skip_vae and os.path.exists(vae_best_path):
        print(f"\nLoading pre-trained VAE from {vae_best_path}")
        vae = VAE2D(in_channels=1, latent_channels=args.latent_channels,
                    num_channels=(64, 128, 256, 256), num_res_blocks=2).to(device)
        ckpt = torch.load(vae_best_path, map_location=device, weights_only=False)
        vae.load_state_dict(ckpt["vae"])
    else:
        vae = train_vae(args, train_eye_ids, val_eye_ids, device)

    train_diffusion(args, vae, train_eye_ids, val_eye_ids, device)

    print("\n" + "=" * 70)
    print(f"2D-BrLP training complete. Results in: {args.output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
