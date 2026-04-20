"""
Shared U-Net backbone and dataset for all five configurations.

The backbone `MultiScaleTemporalUNet_v13` is used by every training wrapper
(TRU, IA-Nonlinear, IA-Linear, Std-DDIM). This keeps the parameter count
and compute identical across configurations so that Table 3 comparisons
isolate the effect of the noise/alignment strategy rather than capacity.
"""

import glob
import math
import os
import random
import re
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


# =============================================================================
# BUILDING BLOCKS
# =============================================================================

class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ContinuousTimeEmbedding(nn.Module):
    """Embed continuous inter-visit intervals (years) via log1p + sinusoidal + MLP."""
    def __init__(self, dim=256):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, delta):
        device = delta.device
        delta_log = torch.log1p(delta)
        half_dim = self.dim // 2
        freqs = torch.exp(torch.linspace(0, -math.log(100), half_dim, device=device))
        args = delta_log.unsqueeze(-1) * freqs
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        return self.mlp(emb)


class ResBlock(nn.Module):
    """ResBlock with dual FiLM conditioning: timestep + inter-visit delta.

    Both scale/shift pairs are composed multiplicatively so that a constant
    timestep (TRU case) collapses the t-branch to a learned bias, leaving the
    delta-branch as the only effective FiLM pathway.
    """
    def __init__(self, in_ch, out_ch, time_emb_dim, delta_emb_dim=None, groups=8):
        super().__init__()

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch * 2),
        )

        self.use_delta = delta_emb_dim is not None
        if self.use_delta:
            self.delta_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(delta_emb_dim, out_ch * 2),
            )

        self.block1 = nn.Sequential(
            nn.GroupNorm(groups, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(groups, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        )
        self.res_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb, delta_emb=None):
        t_scale, t_shift = self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1).chunk(2, dim=1)

        if self.use_delta and delta_emb is not None:
            d_scale, d_shift = self.delta_mlp(delta_emb).unsqueeze(-1).unsqueeze(-1).chunk(2, dim=1)
        else:
            d_scale = torch.zeros_like(t_scale)
            d_shift = torch.zeros_like(t_shift)

        h = self.block1(x)
        h = h * (1 + t_scale) * (1 + d_scale) + t_shift + d_shift
        h = self.block2(h)
        return h + self.res_conv(x)


class HistoryFeatureExtractor(nn.Module):
    """Extracts features from history frames at three spatial scales (1, 1/2, 1/4)."""
    def __init__(self, in_channels=1, base_dim=64):
        super().__init__()

        self.scale1_encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_dim, 3, padding=1),
            nn.GroupNorm(8, base_dim),
            nn.SiLU(),
        )
        self.scale2_encoder = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(base_dim, base_dim * 2, 3, padding=1),
            nn.GroupNorm(8, base_dim * 2),
            nn.SiLU(),
        )
        self.scale3_encoder = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(base_dim * 2, base_dim * 4, 3, padding=1),
            nn.GroupNorm(8, base_dim * 4),
            nn.SiLU(),
        )

    def forward(self, history):
        B, N, C, H, W = history.shape
        history_flat = history.view(B * N, C, H, W)

        feat1 = self.scale1_encoder(history_flat)
        feat1 = feat1.view(B, N, -1, H, W)

        feat2 = self.scale2_encoder(feat1.view(B * N, -1, H, W))
        feat2 = feat2.view(B, N, -1, H // 2, W // 2)

        feat3 = self.scale3_encoder(feat2.view(B * N, -1, H // 2, W // 2))
        feat3 = feat3.view(B, N, -1, H // 4, W // 4)

        return {"scale1": feat1, "scale2": feat2, "scale3": feat3}


class DeltaWeightedAttn(nn.Module):
    """Delta-weighted aggregation of history frames (scores depend only on dt)."""
    def __init__(self, dim, delta_dim=256):
        super().__init__()
        self.delta_bias = nn.Linear(delta_dim, 1)

    def forward(self, x, h_frames, delta_emb, temporal_mask):
        B, N, C, H, W = h_frames.shape
        scores = self.delta_bias(delta_emb).view(B, N, 1, 1).expand(B, N, H, W)
        if temporal_mask is not None:
            scores = scores.masked_fill(temporal_mask.view(B, N, 1, 1), float("-inf"))
        weights = F.softmax(scores, dim=1)
        return (h_frames * weights.unsqueeze(2)).sum(dim=1)


# =============================================================================
# MULTI-SCALE TEMPORAL U-NET v13 (~22.83M params)
# =============================================================================

class MultiScaleTemporalUNet_v13(nn.Module):
    """Four-level U-Net (64/128/256/512) with delta-weighted history fusion
    at scales 1/2/3, dual-FiLM ResBlocks, near-zero output init, and a
    residual-to-last-frame output head.

    Encoder ResBlock counts: 2-2-1-1.  Decoder ResBlock counts: 1-1-2-2.
    Bottleneck: 2 ResBlocks at 512 channels.

    The `t` argument is a diffusion timestep when the wrapper is a
    diffusion variant, and a constant dummy (e.g. 200) when the wrapper is
    TRU — in which case the t-branch of each ResBlock collapses to a
    learned constant bias.
    """

    def __init__(self, max_history=6, base_channels=64, delta_emb_dim=256,
                 history_dropout=0.03):
        super().__init__()
        self.max_history = max_history
        self.base_channels = base_channels
        self.time_dim = base_channels * 4
        self.delta_emb_dim = delta_emb_dim
        self.history_dropout = history_dropout

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(base_channels),
            nn.Linear(base_channels, self.time_dim),
            nn.GELU(),
            nn.Linear(self.time_dim, self.time_dim),
        )
        self.delta_embed = ContinuousTimeEmbedding(dim=delta_emb_dim)
        self.feature_extractor = HistoryFeatureExtractor(
            in_channels=1, base_dim=base_channels
        )

        self.history_attn1 = DeltaWeightedAttn(dim=base_channels, delta_dim=delta_emb_dim)
        self.history_attn2 = DeltaWeightedAttn(dim=base_channels * 2, delta_dim=delta_emb_dim)
        self.history_attn3 = DeltaWeightedAttn(dim=base_channels * 4, delta_dim=delta_emb_dim)

        self.fuse_scale1 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, kernel_size=1),
            nn.GroupNorm(8, base_channels), nn.SiLU(),
        )
        self.fuse_scale2 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 2, kernel_size=1),
            nn.GroupNorm(16, base_channels * 2), nn.SiLU(),
        )
        self.fuse_scale3 = nn.Sequential(
            nn.Conv2d(base_channels * 8, base_channels * 4, kernel_size=1),
            nn.GroupNorm(32, base_channels * 4), nn.SiLU(),
        )

        self.inc = nn.Conv2d(1, base_channels, 3, padding=1)

        self.down1 = nn.ModuleList([
            ResBlock(base_channels, base_channels, self.time_dim, delta_emb_dim),
            nn.MaxPool2d(2),
        ])
        self.down1b = ResBlock(base_channels, base_channels, self.time_dim, delta_emb_dim)

        self.down2 = nn.ModuleList([
            ResBlock(base_channels, base_channels * 2, self.time_dim, delta_emb_dim),
            nn.MaxPool2d(2),
        ])
        self.down2b = ResBlock(base_channels * 2, base_channels * 2, self.time_dim, delta_emb_dim)

        self.down3 = nn.ModuleList([
            ResBlock(base_channels * 2, base_channels * 4, self.time_dim, delta_emb_dim),
            nn.MaxPool2d(2),
        ])
        self.down4 = nn.ModuleList([
            ResBlock(base_channels * 4, base_channels * 8, self.time_dim, delta_emb_dim),
            nn.MaxPool2d(2),
        ])

        self.bot1 = ResBlock(base_channels * 8, base_channels * 8, self.time_dim, delta_emb_dim)
        self.bot2 = ResBlock(base_channels * 8, base_channels * 8, self.time_dim, delta_emb_dim)

        self.up1 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, 2)
        self.up_res1 = ResBlock(base_channels * 8 + base_channels * 4,
                                base_channels * 4, self.time_dim, delta_emb_dim)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, 2)
        self.up_res2 = ResBlock(base_channels * 4 + base_channels * 2,
                                base_channels * 2, self.time_dim, delta_emb_dim)

        self.up3 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, 2)
        self.up_res3 = ResBlock(base_channels * 2 + base_channels,
                                base_channels, self.time_dim, delta_emb_dim)
        self.up_res3b = ResBlock(base_channels, base_channels, self.time_dim, delta_emb_dim)

        self.up4 = nn.ConvTranspose2d(base_channels, base_channels, 2, 2)
        self.up_res4 = ResBlock(base_channels * 2, base_channels, self.time_dim, delta_emb_dim)
        self.up_res4b = ResBlock(base_channels, base_channels, self.time_dim, delta_emb_dim)

        self.outc = nn.Conv2d(base_channels, 1, 1)
        nn.init.normal_(self.outc.weight, std=1e-3)
        nn.init.zeros_(self.outc.bias)

    @staticmethod
    def _extract_last_frame(history, temporal_mask):
        """Most-recent non-padded frame per batch element: (B, 1, H, W)."""
        B, N, C, H, W = history.shape
        last_frames = []
        for b in range(B):
            valid = (~temporal_mask[b]).nonzero(as_tuple=True)[0]
            if len(valid) > 0:
                last_frames.append(history[b, valid[-1]])
            else:
                last_frames.append(torch.zeros(C, H, W, device=history.device))
        return torch.stack(last_frames)

    def forward(self, x_noisy, history, deltas, temporal_mask, t):
        t_emb = self.time_mlp(t)
        delta_emb = self.delta_embed(deltas)
        delta_recent_emb = delta_emb[:, -1]

        raw_features = self.feature_extractor(history)

        if self.training and random.random() < self.history_dropout:
            raw_features = {k: torch.zeros_like(v) for k, v in raw_features.items()}

        x = self.inc(x_noisy)

        h1 = self.history_attn1(x, raw_features["scale1"], delta_emb, temporal_mask)
        x = self.fuse_scale1(torch.cat([x, h1], dim=1))

        x = self.down1[0](x, t_emb, delta_recent_emb)
        x = self.down1b(x, t_emb, delta_recent_emb)
        x1 = x
        x = self.down1[1](x)

        x = self.down2[0](x, t_emb, delta_recent_emb)
        h2 = self.history_attn2(x, raw_features["scale2"], delta_emb, temporal_mask)
        x = self.fuse_scale2(torch.cat([x, h2], dim=1))
        x = self.down2b(x, t_emb, delta_recent_emb)
        d1 = x
        x = self.down2[1](x)

        x = self.down3[0](x, t_emb, delta_recent_emb)
        h3 = self.history_attn3(x, raw_features["scale3"], delta_emb, temporal_mask)
        x = self.fuse_scale3(torch.cat([x, h3], dim=1))
        d2 = x
        x = self.down3[1](x)

        x5 = self.down4[0](x, t_emb, delta_recent_emb)
        d4 = self.down4[1](x5)

        b = self.bot1(d4, t_emb, delta_recent_emb)
        b = self.bot2(b, t_emb, delta_recent_emb)

        u1 = self.up1(b)
        u1 = torch.cat([u1, x5], dim=1)
        u1 = self.up_res1(u1, t_emb, delta_recent_emb)

        u2 = self.up2(u1)
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.up_res2(u2, t_emb, delta_recent_emb)

        u3 = self.up3(u2)
        u3 = torch.cat([u3, d1], dim=1)
        u3 = self.up_res3(u3, t_emb, delta_recent_emb)
        u3 = self.up_res3b(u3, t_emb, delta_recent_emb)

        u4 = self.up4(u3)
        u4 = torch.cat([u4, x1], dim=1)
        u4 = self.up_res4(u4, t_emb, delta_recent_emb)
        u4 = self.up_res4b(u4, t_emb, delta_recent_emb)

        residual = self.outc(u4)
        last_frame = self._extract_last_frame(history, temporal_mask)
        return last_frame + residual


# =============================================================================
# DATASET
# =============================================================================

class MaskedLongitudinalDataset(Dataset):
    """Longitudinal dataset with eye-level grouping and per-sample history.

    Expected filename pattern: `{mrn}_{eye_laterality}_{time}[_reg|_anchor][_mask].png`
    where `mrn` and `eye_laterality` are integers, `time` is a float in years,
    and optional `_mask.png` companions mark valid pixels. See docs/data_format.md.
    """

    PADDING_DELTA = 100.0

    def __init__(self, data_dir, max_history=6, image_size=256,
                 use_augmentation=True, eye_ids=None):
        super().__init__()
        self.data_dir = data_dir
        self.max_history = max_history
        self.image_size = image_size
        self.use_augmentation = use_augmentation

        all_files = glob.glob(os.path.join(data_dir, "*.png"))
        self.image_files = [f for f in all_files if not f.endswith("_mask.png")]
        print(f"Found {len(self.image_files)} images")

        self.eye_sequences, self.eye_to_mrn = self._parse_and_group_files()

        if eye_ids is not None:
            self.eye_sequences = {
                eid: visits for eid, visits in self.eye_sequences.items()
                if eid in eye_ids
            }
            self.eye_to_mrn = {
                eid: mrn for eid, mrn in self.eye_to_mrn.items()
                if eid in eye_ids
            }
            print(f"Filtered to {len(self.eye_sequences)} eyes")

        self.samples = self._build_samples()
        unique_mrns = len(set(self.eye_to_mrn.values()))
        print(f"Built {len(self.samples)} samples from "
              f"{len(self.eye_sequences)} eyes ({unique_mrns} patients)")

    def _parse_filename(self, filepath):
        filename = os.path.basename(filepath)
        base = filename.replace("_reg.png", "").replace("_anchor.png", "")
        m = re.match(r"^(\d+)_(\d+)_([\d.]+)", base)
        if m:
            return m.group(1), f"{m.group(1)}_{m.group(2)}", float(m.group(3))
        parts = base.split("_")
        if len(parts) >= 3:
            try:
                return parts[0], f"{parts[0]}_{parts[1]}", float(parts[2])
            except ValueError:
                pass
        return None, None, None

    def _parse_and_group_files(self):
        eye_data = defaultdict(list)
        eye_to_mrn = {}
        for filepath in self.image_files:
            mrn, eye_id, t_val = self._parse_filename(filepath)
            if eye_id is not None:
                eye_data[eye_id].append({"path": filepath, "time": t_val})
                eye_to_mrn[eye_id] = mrn
        for eye_id in eye_data:
            eye_data[eye_id].sort(key=lambda x: x["time"])
        eye_sequences = {eid: v for eid, v in eye_data.items() if len(v) >= 2}
        eye_to_mrn = {eid: mrn for eid, mrn in eye_to_mrn.items() if eid in eye_sequences}
        return eye_sequences, eye_to_mrn

    def _build_samples(self):
        samples = []
        for eye_id, visits in self.eye_sequences.items():
            for target_idx in range(1, len(visits)):
                samples.append({
                    "eye_id": eye_id,
                    "target_idx": target_idx,
                    "history_indices": list(range(target_idx)),
                })
        return samples

    def _get_mask_path(self, image_path):
        fn = os.path.basename(image_path)
        d = os.path.dirname(image_path)
        for old, new in [("_reg.png", "_reg_mask.png"), ("_anchor.png", "_anchor_mask.png")]:
            if fn.endswith(old):
                mp = os.path.join(d, fn.replace(old, new))
                if os.path.exists(mp):
                    return mp
        if fn.endswith(".png"):
            mp = os.path.join(d, fn[:-4] + "_mask.png")
            if os.path.exists(mp):
                return mp
        return None

    def _load_image_and_mask(self, image_path):
        img = Image.open(image_path).convert("L")
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)

        mask_path = self._get_mask_path(image_path)
        if mask_path is not None and os.path.exists(mask_path):
            mask = Image.open(mask_path).convert("L")
            mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)
        else:
            mask = Image.new("L", (self.image_size, self.image_size), 255)
        return img, mask

    def _apply_augmentation(self, images, masks):
        do_hflip = random.random() > 0.5
        angle = random.uniform(-15, 15)
        aug_images, aug_masks = [], []
        for img, mask in zip(images, masks):
            if do_hflip:
                img = TF.hflip(img)
                mask = TF.hflip(mask)
            img = TF.affine(img, angle=angle, translate=(0, 0), scale=1.05, shear=0,
                            interpolation=InterpolationMode.BILINEAR, fill=0)
            mask = TF.affine(mask, angle=angle, translate=(0, 0), scale=1.05, shear=0,
                             interpolation=InterpolationMode.NEAREST, fill=0)
            aug_images.append(TF.to_tensor(img))
            aug_masks.append((TF.to_tensor(mask) > 0.5).float())
        return aug_images, aug_masks

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        eye_id = sample["eye_id"]
        visits = self.eye_sequences[eye_id]

        target_visit = visits[sample["target_idx"]]
        history_visits = [visits[i] for i in sample["history_indices"]]

        target_img, target_mask = self._load_image_and_mask(target_visit["path"])
        target_time = target_visit["time"]

        history_images, history_masks, deltas = [], [], []
        for visit in history_visits:
            img, mask = self._load_image_and_mask(visit["path"])
            history_images.append(img)
            history_masks.append(mask)
            deltas.append(target_time - visit["time"])

        actual_history_length = len(history_images)

        while len(history_images) < self.max_history:
            history_images.insert(0, Image.new("L", (self.image_size, self.image_size), 0))
            history_masks.insert(0, Image.new("L", (self.image_size, self.image_size), 0))
            deltas.insert(0, self.PADDING_DELTA)

        history_images = history_images[-self.max_history:]
        history_masks = history_masks[-self.max_history:]
        deltas = deltas[-self.max_history:]
        actual_history_length = min(actual_history_length, self.max_history)

        all_images = history_images + [target_img]
        all_masks = history_masks + [target_mask]

        if self.use_augmentation:
            all_images, all_masks = self._apply_augmentation(all_images, all_masks)
        else:
            all_images = [TF.to_tensor(img) for img in all_images]
            all_masks = [(TF.to_tensor(m) > 0.5).float() for m in all_masks]

        history_tensors = all_images[:-1]
        history_mask_tensors = all_masks[:-1]
        target_tensor = all_images[-1]
        target_pixel_mask = all_masks[-1]

        history = torch.stack(history_tensors, dim=0)
        history_pixel_masks = torch.stack(history_mask_tensors, dim=0)
        deltas_tensor = torch.tensor(deltas, dtype=torch.float32)

        temporal_mask = torch.zeros(self.max_history, dtype=torch.bool)
        num_padded = self.max_history - actual_history_length
        if num_padded > 0:
            temporal_mask[:num_padded] = True

        target_tensor = target_tensor * 2 - 1
        history = history * 2 - 1

        return {
            "target": target_tensor,
            "target_pixel_mask": target_pixel_mask,
            "history": history,
            "history_pixel_masks": history_pixel_masks,
            "deltas": deltas_tensor,
            "temporal_mask": temporal_mask,
            "eye_id": eye_id,
        }

    def get_all_eye_ids(self):
        return list(self.eye_sequences.keys())

    def get_eye_to_mrn_mapping(self):
        return self.eye_to_mrn.copy()


def extract_last_frame(history, temporal_mask):
    """Standalone helper mirroring `MultiScaleTemporalUNet_v13._extract_last_frame`."""
    return MultiScaleTemporalUNet_v13._extract_last_frame(history, temporal_mask)
