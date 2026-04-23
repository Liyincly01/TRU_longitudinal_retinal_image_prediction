#!/usr/bin/env python3
"""
Benchmark the five paper configurations against the copy-last baseline
on a test set.

Reports MAE / NMSE / PSNR / SSIM / Delta-SSIM per-eye; additionally
reports hypo-AF atrophy Dice and HD95 for images with macular atrophy. Pairwise Wilcoxon p-values are appended.

This script does NOT ship weights or data. Every checkpoint path and
data directory must be supplied via CLI flags. Checkpoints trained
with `train.py` can be loaded directly — the 'args' dict in the
checkpoint provides the max_history the model was trained with.
"""

import argparse
import glob
import os
import re
import sys
import warnings
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import ndimage
from scipy.stats import wilcoxon
from skimage.metrics import structural_similarity as ssim_func

# Allow `from src import ...` when invoking as `python evaluate.py`.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import (
    MultiScaleTemporalUNet_v13,
    IALinearDiffusion,
    IANonlinearDiffusion,
    StdVPredDiffusion,
    extract_last_frame,
)

warnings.filterwarnings("ignore", category=FutureWarning)


IMAGE_SIZE = 256
MAX_HISTORY = 6
DDIM_SEED = 42
BATCH_SIZE = 8

METHOD_ORDER = [
    "copy_last",
    "Std_DDIM_50step",
    "Std_DDIM_1step",
    "IA_Nonlinear",
    "IA_Linear",
    "TRU",
]

FAF_METRICS = ["mae", "nmse", "psnr", "ssim", "delta_ssim",
               "atrophy_dice", "atrophy_hd95"]
SLO_METRICS = ["mae", "nmse", "psnr", "ssim", "delta_ssim"]

FMT = {
    "mae": ".4f", "nmse": ".4f", "psnr": ".2f",
    "ssim": ".4f", "delta_ssim": ".4f",
    "atrophy_dice": ".4f", "atrophy_hd95": ".2f",
}

PVALUE_COMPARISONS = [
    ("Std_DDIM_50step", "Std_DDIM_1step", "Std_50step vs Std_1step"),
    ("IA_Nonlinear",    "Std_DDIM_1step", "IA_Nonlinear vs Std_DDIM_1step"),
    ("TRU",             "IA_Nonlinear",   "TRU vs IA_Nonlinear"),
    ("TRU",             "IA_Linear",      "TRU vs IA_Linear"),
]

DELTA_SCALE = 1.0
PAD_DELTA = 100.0


# =============================================================================
# Data scanning
# =============================================================================

def parse_faf_filename(filepath):
    fn = os.path.basename(filepath)
    base = fn.replace("_reg.png", "").replace("_anchor.png", "")
    m = re.match(r"^(\d+)_(\d+)_([\d.]+)", base)
    if not m:
        return None, None, None
    return m.group(1), f"{m.group(1)}_{m.group(2)}", float(m.group(3))


def scan_faf_sequences(root_dir):
    files = [p for p in glob.glob(os.path.join(root_dir, "*.png"))
             if not p.endswith("_mask.png")]
    eye_data = defaultdict(list)
    for p in files:
        _, eye_id, t = parse_faf_filename(p)
        if eye_id is None:
            continue
        eye_data[eye_id].append({"path": p, "time": t})
    for eid in eye_data:
        eye_data[eid].sort(key=lambda x: x["time"])
    return {k: v for k, v in eye_data.items() if len(v) >= 2}


def scan_faf_sequences_multi(root_dirs):
    merged = {}
    for d in root_dirs:
        seqs = scan_faf_sequences(d)
        for eid, visits in seqs.items():
            if eid not in merged:
                merged[eid] = list(visits)
            else:
                existing_times = {v["time"] for v in merged[eid]}
                for v in visits:
                    if v["time"] not in existing_times:
                        merged[eid].append(v)
                        existing_times.add(v["time"])
                merged[eid].sort(key=lambda x: x["time"])
    return {k: v for k, v in merged.items() if len(v) >= 2}


def parse_slo_filename(filepath):
    fn = os.path.basename(filepath)
    if not fn.endswith(".png"):
        return None, None, None
    base = fn[:-4]
    parts = base.split("_", 4)
    if len(parts) < 4:
        return None, None, None
    try:
        return parts[0], f"{parts[0]}_{parts[1]}", float(parts[2])
    except (ValueError, IndexError):
        return None, None, None


def scan_slo_sequences(data_dir, patient_list_path):
    with open(patient_list_path) as f:
        patients = set(line.strip() for line in f if line.strip())
    all_files = glob.glob(os.path.join(data_dir, "*.png"))
    eye_data = defaultdict(list)
    for fp in all_files:
        pid, eid, t = parse_slo_filename(fp)
        if eid is None or pid not in patients:
            continue
        eye_data[eid].append({"path": fp, "time": t})
    for eid in eye_data:
        eye_data[eid].sort(key=lambda x: x["time"])
    return {k: v for k, v in eye_data.items() if len(v) >= 2}


def build_anchor_tasks(eye_seqs):
    """For each eye, use all-but-last visits as history; last visit is target."""
    tasks = []
    for eid in sorted(eye_seqs.keys()):
        visits = eye_seqs[eid]
        if len(visits) < 2:
            continue
        tasks.append({"eye_id": eid, "history": visits[:-1], "target": visits[-1]})
    return tasks


# =============================================================================
# Image / mask loading
# =============================================================================

def load_image(path, size=IMAGE_SIZE):
    img = Image.open(path).convert("L").resize((size, size), Image.BILINEAR)
    return np.array(img, dtype=np.float32) / 255.0 * 2.0 - 1.0


def load_faf_mask(path, size=IMAGE_SIZE):
    fn = os.path.basename(path)
    d = os.path.dirname(path)
    for old, new in [("_reg.png", "_reg_mask.png"), ("_anchor.png", "_anchor_mask.png")]:
        if fn.endswith(old):
            mp = os.path.join(d, fn.replace(old, new))
            if os.path.exists(mp):
                m = Image.open(mp).convert("L").resize((size, size), Image.NEAREST)
                return np.array(m, dtype=np.float32) / 255.0 > 0.5
    mp = os.path.join(d, fn[:-4] + "_mask.png")
    if os.path.exists(mp):
        m = Image.open(mp).convert("L").resize((size, size), Image.NEAREST)
        return np.array(m, dtype=np.float32) / 255.0 > 0.5
    return np.ones((size, size), dtype=bool)


def generate_slo_registration_mask(img_np):
    """Flood-fill the border-zero region to derive a valid-pixel mask for SLO."""
    zero_mask = (img_np == 0)
    if not zero_mask.any():
        return np.ones(img_np.shape, dtype=np.uint8)
    dilated = ndimage.binary_dilation(zero_mask, iterations=1)
    border = np.zeros_like(dilated)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    seed = dilated & border
    struct = ndimage.generate_binary_structure(2, 2)
    prev = seed.copy()
    while True:
        grown = ndimage.binary_dilation(prev, structure=struct) & dilated
        if np.array_equal(grown, prev):
            break
        prev = grown
    return (~prev).astype(np.uint8)


def load_slo_mask(path, size=IMAGE_SIZE):
    img = Image.open(path).convert("L")
    mask_np = generate_slo_registration_mask(np.array(img))
    mask = Image.fromarray(mask_np * 255, mode="L").resize((size, size), Image.NEAREST)
    return np.array(mask, dtype=np.float32) / 255.0 > 0.5


def make_diffusion_inputs(history_visits, target_time, max_h=MAX_HISTORY):
    n = len(history_visits)
    imgs = [load_image(v["path"]) for v in history_visits]
    deltas = [(target_time - v["time"]) / DELTA_SCALE for v in history_visits]
    while len(imgs) < max_h:
        imgs.insert(0, np.full((IMAGE_SIZE, IMAGE_SIZE), -1.0, dtype=np.float32))
        deltas.insert(0, PAD_DELTA)
    imgs = imgs[-max_h:]
    deltas = deltas[-max_h:]
    n_actual = min(n, max_h)
    n_pad = max_h - n_actual
    tmask = np.zeros(max_h, dtype=bool)
    if n_pad > 0:
        tmask[:n_pad] = True
    return (np.stack(imgs, axis=0)[:, None, :, :],
            np.array(deltas, dtype=np.float32),
            tmask)


# =============================================================================
# Atrophy segmentation + metrics
# =============================================================================

def segment_atrophy(img_uint8, roi_radius_frac=0.40, seed_radius_frac=0.15,
                    threshold_sigma=1.5, cap_frac=0.70, min_area_px=20,
                    morph_kernel=5):
    """Adaptive dark-region segmentation for hypo-AF atrophy."""
    H, W = img_uint8.shape
    cx, cy = W // 2, H // 2
    Y, X = np.ogrid[:H, :W]
    valid = img_uint8 > 10
    roi_r = int(min(H, W) * roi_radius_frac)
    roi_circle = ((X - cx) ** 2 + (Y - cy) ** 2) <= roi_r ** 2
    central = img_uint8[valid & roi_circle]
    if central.size < 100:
        return np.zeros((H, W), dtype=np.uint8)
    mean_i, std_i = float(central.mean()), float(central.std())
    threshold = min(mean_i - threshold_sigma * std_i, mean_i * cap_frac)
    threshold = max(threshold, 1.0)
    dark = (img_uint8.astype(np.float32) < threshold) & valid & roi_circle
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
    dark_u8 = dark.astype(np.uint8) * 255
    dark_u8 = cv2.morphologyEx(dark_u8, cv2.MORPH_CLOSE, kernel)
    dark_u8 = cv2.morphologyEx(dark_u8, cv2.MORPH_OPEN, kernel)
    seed_r = int(min(H, W) * seed_radius_frac)
    seed = ((X - cx) ** 2 + (Y - cy) ** 2) <= seed_r ** 2
    num_labels, labels = cv2.connectedComponents(dark_u8)
    final = np.zeros((H, W), dtype=np.uint8)
    for lbl in range(1, num_labels):
        comp = labels == lbl
        if comp.sum() >= min_area_px and (comp & seed).any():
            final[comp] = 255
    return final


def compute_atrophy_metrics(pred_mask, gt_mask):
    nan_result = dict(atrophy_dice=np.nan, atrophy_hd95=np.nan)
    pred_b, gt_b = pred_mask.astype(bool), gt_mask.astype(bool)
    if pred_b.sum() == 0 or gt_b.sum() == 0:
        return nan_result
    inter = np.logical_and(pred_b, gt_b).sum()
    dice = float(2.0 * inter / (pred_b.sum() + gt_b.sum()))
    hd95 = np.nan
    try:
        from medpy.metric.binary import hd95 as _hd95
        hd95 = float(_hd95(pred_b, gt_b))
    except Exception:
        pass
    return dict(atrophy_dice=dice, atrophy_hd95=hd95)


def to01(arr):
    return np.clip((arr + 1.0) / 2.0, 0.0, 1.0)


def compute_metrics(target, pred, mask, last_hist=None):
    t = np.clip(to01(target), 0, 1)
    p = np.clip(to01(pred), 0, 1)
    m = mask.astype(bool)
    nan_dict = dict(mae=np.nan, nmse=np.nan, psnr=np.nan,
                    ssim=np.nan, delta_ssim=np.nan)
    if m.sum() == 0:
        return nan_dict
    tv, pv = t[m], p[m]
    mae = float(np.mean(np.abs(tv - pv)))
    mse = float(np.mean((tv - pv) ** 2))
    psnr = float(10 * np.log10(1.0 / max(mse, 1e-10)))
    nmse = float(np.sum((tv - pv) ** 2) / max(np.sum(tv ** 2), 1e-10))
    ys, xs = np.where(m)
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
    box_ok = (y1 - y0) > 16 and (x1 - x0) > 16
    ssim = np.nan
    if box_ok:
        try:
            ssim = float(ssim_func(t[y0:y1+1, x0:x1+1],
                                   p[y0:y1+1, x0:x1+1], data_range=1.0))
        except Exception:
            pass
    delta_ssim = np.nan
    if last_hist is not None and box_ok:
        lh = np.clip(to01(last_hist), 0, 1)
        try:
            delta_ssim = float(ssim_func(
                (t - lh)[y0:y1+1, x0:x1+1],
                (p - lh)[y0:y1+1, x0:x1+1], data_range=2.0))
        except Exception:
            pass
    return dict(mae=mae, nmse=nmse, psnr=psnr, ssim=ssim, delta_ssim=delta_ssim)


# =============================================================================
# Model loaders
# =============================================================================

def _load_unet(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = ckpt.get("args", {})
    def get(k, d=None):
        return args.get(k, d) if isinstance(args, dict) else getattr(args, k, d)
    max_history = get("max_history", MAX_HISTORY)

    model = MultiScaleTemporalUNet_v13(
        max_history=max_history, base_channels=64,
        delta_emb_dim=256, history_dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["ema_model"])
    model.eval()
    return model, args, max_history, ckpt.get("epoch", "?")


def load_std_ddim_50step(ckpt_path, device, n_steps=50):
    print(f"  Loading Std_DDIM_50step: {os.path.basename(ckpt_path)} ...", end=" ", flush=True)
    model, _, max_h, epoch = _load_unet(ckpt_path, device)
    wrap = StdVPredDiffusion(model, device=str(device)).to(device).eval()

    def predict(history, deltas, temporal_mask, seed=None):
        return wrap.ddim_sample(history, deltas, temporal_mask,
                                n_steps=n_steps, seed=seed)

    print(f"epoch={epoch}, max_h={max_h}, steps={n_steps}  OK")
    return predict, max_h


def load_std_ddim_1step(ckpt_path, device, t_infer=200):
    print(f"  Loading Std_DDIM_1step: {os.path.basename(ckpt_path)} ...", end=" ", flush=True)
    model, _, max_h, epoch = _load_unet(ckpt_path, device)
    wrap = StdVPredDiffusion(model, device=str(device)).to(device).eval()

    def predict(history, deltas, temporal_mask, seed=None):
        return wrap.x0hat_predict(history, deltas, temporal_mask,
                                  t_infer=t_infer, seed=seed)

    print(f"epoch={epoch}, max_h={max_h}, t_infer={t_infer}  OK")
    return predict, max_h


def load_ia_nonlinear(ckpt_path, device):
    print(f"  Loading IA_Nonlinear: {os.path.basename(ckpt_path)} ...", end=" ", flush=True)
    model, args, max_h, epoch = _load_unet(ckpt_path, device)
    def get(k, d=None):
        return args.get(k, d) if isinstance(args, dict) else getattr(args, k, d)
    t_infer = int(get("t_infer", 200))
    wrap = IANonlinearDiffusion(
        model, t_min=get("t_min", 0), t_max=get("t_max", 1000),
        device=str(device),
    ).to(device).eval()

    def predict(history, deltas, temporal_mask, seed=None):
        out = wrap.x0hat_predict(history, deltas, temporal_mask,
                                 t_infer=t_infer, seed=seed)
        wrap.model.eval()
        return out

    print(f"epoch={epoch}, max_h={max_h}, t_infer={t_infer}  OK")
    return predict, max_h


def load_ia_linear(ckpt_path, device):
    print(f"  Loading IA_Linear: {os.path.basename(ckpt_path)} ...", end=" ", flush=True)
    model, args, max_h, epoch = _load_unet(ckpt_path, device)
    def get(k, d=None):
        return args.get(k, d) if isinstance(args, dict) else getattr(args, k, d)
    t_infer = float(get("ia_linear_t_infer", get("t_infer", 0.20)))
    wrap = IALinearDiffusion(
        model,
        t_min=get("ia_linear_t_min", 0.05),
        t_max=get("ia_linear_t_max", 0.95),
        t_infer=t_infer, device=str(device),
    ).to(device).eval()

    def predict(history, deltas, temporal_mask, seed=None):
        out = wrap.x0hat_predict(history, deltas, temporal_mask,
                                 t_infer=t_infer, seed=seed)
        wrap.model.eval()
        return out

    print(f"epoch={epoch}, max_h={max_h}, t_infer={t_infer}  OK")
    return predict, max_h


def load_tru(ckpt_path, device):
    print(f"  Loading TRU: {os.path.basename(ckpt_path)} ...", end=" ", flush=True)
    model, args, max_h, epoch = _load_unet(ckpt_path, device)
    def get(k, d=None):
        return args.get(k, d) if isinstance(args, dict) else getattr(args, k, d)
    t_fixed = get("t_fixed", 200)

    def predict(history, deltas, temporal_mask, seed=None):
        model.eval()
        B = history.shape[0]
        last_frame = extract_last_frame(history, temporal_mask)
        t_batch = torch.full((B,), t_fixed, device=history.device, dtype=torch.float)
        return torch.clamp(model(last_frame, history, deltas, temporal_mask, t_batch),
                           -1, 1)

    print(f"epoch={epoch}, max_h={max_h}, t_fixed={t_fixed}  OK")
    return predict, max_h


# =============================================================================
# Evaluation loop
# =============================================================================

@torch.no_grad()
def run_model_batch(predict_fn, hist_batch, delta_batch, tmask_batch, device, seed=42):
    h = torch.from_numpy(hist_batch).float().to(device)
    d = torch.from_numpy(delta_batch).float().to(device)
    t = torch.from_numpy(tmask_batch.astype(np.float32)).bool().to(device)
    preds = predict_fn(h, d, t, seed=seed)
    return preds.squeeze(1).cpu().numpy()


def evaluate_dataset(dataset_name, tasks, models, device, load_mask_fn,
                     batch_size=BATCH_SIZE, use_atrophy=True):
    results = []
    n = len(tasks)

    print(f"  Pre-computing GT data for {n} eyes ...")
    gt_data = []
    for task in tasks:
        tgt_img = load_image(task["target"]["path"])
        tgt_mask = load_mask_fn(task["target"]["path"])
        last_hist_img = load_image(task["history"][-1]["path"])
        dt = task["target"]["time"] - task["history"][-1]["time"]

        gt_atrophy = None
        last_atrophy_area = 0
        if use_atrophy:
            tgt_u8 = np.clip(to01(tgt_img) * 255, 0, 255).astype(np.uint8)
            gt_atrophy = segment_atrophy(tgt_u8)
            last_u8 = np.clip(to01(last_hist_img) * 255, 0, 255).astype(np.uint8)
            last_atrophy_area = int(segment_atrophy(last_u8).astype(bool).sum())

        gt_data.append(dict(
            tgt_img=tgt_img, tgt_mask=tgt_mask, last_hist_img=last_hist_img,
            dt=dt, gt_atrophy=gt_atrophy, last_atrophy_area=last_atrophy_area))

    print(f"  [copy_last] ...")
    for ti, task in enumerate(tasks):
        g = gt_data[ti]
        pred = load_image(task["history"][-1]["path"])
        m = compute_metrics(g["tgt_img"], pred, g["tgt_mask"], last_hist=g["last_hist_img"])
        row = {"eye_id": task["eye_id"], "n_history": len(task["history"]),
               "target_time": task["target"]["time"], "delta_t_years": g["dt"]}
        for k, v in m.items():
            row[f"copy_last_{k}"] = v

        if use_atrophy and g["gt_atrophy"] is not None:
            pred_u8 = np.clip(to01(pred) * 255, 0, 255).astype(np.uint8)
            pred_at = segment_atrophy(pred_u8)
            for k, v in compute_atrophy_metrics(pred_at, g["gt_atrophy"]).items():
                row[f"copy_last_{k}"] = v
            row["gt_atrophy_area"] = int(g["gt_atrophy"].astype(bool).sum())
            row["last_hist_atrophy_area"] = g["last_atrophy_area"]

        results.append(row)

    for model_name, (predict_fn, max_h) in models.items():
        print(f"\n  [{model_name}] batched inference (max_h={max_h}) ...")
        for i in range(0, n, batch_size):
            batch_tasks = tasks[i:i + batch_size]
            B = len(batch_tasks)
            hist_batch = np.zeros((B, max_h, 1, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
            delta_batch = np.full((B, max_h), PAD_DELTA, dtype=np.float32)
            tmask_batch = np.ones((B, max_h), dtype=bool)

            for bi, task in enumerate(batch_tasks):
                h_arr, d_arr, tm_arr = make_diffusion_inputs(
                    task["history"], task["target"]["time"], max_h=max_h)
                hist_batch[bi] = h_arr
                delta_batch[bi] = d_arr
                tmask_batch[bi] = tm_arr

            preds = run_model_batch(predict_fn, hist_batch, delta_batch, tmask_batch,
                                    device, seed=DDIM_SEED)

            for bi, task in enumerate(batch_tasks):
                idx = i + bi
                g = gt_data[idx]
                m = compute_metrics(g["tgt_img"], preds[bi], g["tgt_mask"],
                                    last_hist=g["last_hist_img"])
                for k, v in m.items():
                    results[idx][f"{model_name}_{k}"] = v
                if use_atrophy and g["gt_atrophy"] is not None:
                    pred_u8 = np.clip(to01(preds[bi]) * 255, 0, 255).astype(np.uint8)
                    pred_at = segment_atrophy(pred_u8)
                    for k, v in compute_atrophy_metrics(pred_at, g["gt_atrophy"]).items():
                        results[idx][f"{model_name}_{k}"] = v

            if (i // batch_size) % 10 == 0:
                print(f"    {min(i+B, n)}/{n}", end="\r", flush=True)
        print(f"    Done ({n} tasks).")

    return results


def build_summary(df, method_names, metrics):
    rows = []
    for method in method_names:
        test_col = f"{method}_psnr"
        if test_col not in df.columns or df[test_col].dropna().empty:
            continue
        row = {"method": method}
        for metric in metrics:
            col = f"{method}_{metric}"
            fmt = FMT[metric]
            if col in df.columns:
                vals = df[col].dropna()
                if len(vals) > 0:
                    row[metric] = f"{float(vals.mean()):{fmt}}\u00b1{float(vals.std()):{fmt}}"
                    row[f"{metric}_n"] = int(len(vals))
                else:
                    row[metric] = "nan"
            else:
                row[metric] = "nan"
        rows.append(row)
    return pd.DataFrame(rows)


def compute_pairwise_significance(df, comparisons, metrics):
    rows = []
    for method_a, method_b, label in comparisons:
        row = {"comparison": label}
        for metric in metrics:
            col_a, col_b = f"{method_a}_{metric}", f"{method_b}_{metric}"
            if col_a not in df.columns or col_b not in df.columns:
                row[metric] = "n/a"
                continue
            paired = df[[col_a, col_b]].dropna()
            if len(paired) < 10:
                row[metric] = "n/a"
                continue
            diffs = paired[col_a].values - paired[col_b].values
            if np.allclose(diffs, 0):
                row[metric] = "p=1.000"
                continue
            try:
                _, p = wilcoxon(paired[col_a].values, paired[col_b].values,
                                alternative="two-sided", zero_method="wilcox")
                row[metric] = f"p={p:.2e}" if p < 0.001 else f"p={p:.4f}"
            except Exception:
                row[metric] = "n/a"
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Five-configuration benchmark on held-out FAF (+ optional SLO)."
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)

    # FAF datasets (at least one is required)
    parser.add_argument("--faf_holdout_dir", type=str, action="append", default=None,
                        help="FAF holdout directory (may be repeated to merge multiple dirs).")
    parser.add_argument("--faf_holdout_name", type=str, default="holdout",
                        help="Dataset label used in output CSV filenames.")

    # Checkpoints (at least one; missing ones are skipped gracefully)
    parser.add_argument("--ckpt_std_ddim", type=str, default=None,
                        help="Checkpoint for Std-DDIM (powers both 50-step and 1-step).")
    parser.add_argument("--ckpt_ia_nonlinear", type=str, default=None)
    parser.add_argument("--ckpt_ia_linear", type=str, default=None)
    parser.add_argument("--ckpt_tru", type=str, default=None)

    # SLO external set (optional)
    parser.add_argument("--slo_data_dir", type=str, default=None,
                        help="Optional SLO data directory for external evaluation.")
    parser.add_argument("--slo_patient_list", type=str, default=None,
                        help="Optional line-separated patient-ID file; only listed patients used.")

    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # ---------------- Load models ----------------
    print("\n=== Loading models ===")
    models = {}

    if args.ckpt_std_ddim and os.path.isfile(args.ckpt_std_ddim):
        for name, loader in [
            ("Std_DDIM_50step", lambda c, d: load_std_ddim_50step(c, d, n_steps=50)),
            ("Std_DDIM_1step",  lambda c, d: load_std_ddim_1step(c, d, t_infer=200)),
        ]:
            try:
                fn, mh = loader(args.ckpt_std_ddim, device)
                models[name] = (fn, mh)
            except Exception as e:
                print(f"  [WARN] Could not load {name}: {e}")

    if args.ckpt_ia_nonlinear and os.path.isfile(args.ckpt_ia_nonlinear):
        try:
            fn, mh = load_ia_nonlinear(args.ckpt_ia_nonlinear, device)
            models["IA_Nonlinear"] = (fn, mh)
        except Exception as e:
            print(f"  [WARN] Could not load IA_Nonlinear: {e}")

    if args.ckpt_ia_linear and os.path.isfile(args.ckpt_ia_linear):
        try:
            fn, mh = load_ia_linear(args.ckpt_ia_linear, device)
            models["IA_Linear"] = (fn, mh)
        except Exception as e:
            print(f"  [WARN] Could not load IA_Linear: {e}")

    if args.ckpt_tru and os.path.isfile(args.ckpt_tru):
        try:
            fn, mh = load_tru(args.ckpt_tru, device)
            models["TRU"] = (fn, mh)
        except Exception as e:
            print(f"  [WARN] Could not load TRU: {e}")

    active_methods = [m for m in METHOD_ORDER if m == "copy_last" or m in models]
    print(f"\nActive methods: {active_methods}")

    # ---------------- FAF evaluation ----------------
    if args.faf_holdout_dir:
        print(f"\n{'='*60}")
        print(f"Dataset: {args.faf_holdout_name}")
        print(f"{'='*60}")

        eye_seqs = scan_faf_sequences_multi(args.faf_holdout_dir)
        tasks = build_anchor_tasks(eye_seqs)
        print(f"  Eyes: {len(eye_seqs)}, Tasks: {len(tasks)}")

        rows = evaluate_dataset(
            args.faf_holdout_name, tasks, models, device, load_faf_mask,
            batch_size=args.batch_size, use_atrophy=True,
        )
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(args.output_dir,
                               f"results_per_eye_{args.faf_holdout_name}.csv"), index=False)

        summary_df = build_summary(df, active_methods, FAF_METRICS)
        sig_df = compute_pairwise_significance(df, PVALUE_COMPARISONS, FAF_METRICS)
        summary_df = pd.concat([summary_df, sig_df], ignore_index=True)
        summary_df.to_csv(os.path.join(args.output_dir,
                                       f"summary_{args.faf_holdout_name}.csv"), index=False)

        print(f"\n  -- {args.faf_holdout_name} Summary --")
        avail = [c for c in (["method"] + FAF_METRICS) if c in summary_df.columns]
        print(summary_df[avail].to_string(index=False))

    # ---------------- SLO evaluation (optional) ----------------
    if args.slo_data_dir and args.slo_patient_list:
        ds_name = "SLO_external"
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*60}")

        eye_seqs = scan_slo_sequences(args.slo_data_dir, args.slo_patient_list)
        tasks = build_anchor_tasks(eye_seqs)
        print(f"  Eyes: {len(eye_seqs)}, Tasks: {len(tasks)}")

        rows = evaluate_dataset(
            ds_name, tasks, models, device, load_slo_mask,
            batch_size=args.batch_size, use_atrophy=False,
        )
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(args.output_dir,
                               f"results_per_eye_{ds_name}.csv"), index=False)

        summary_df = build_summary(df, active_methods, SLO_METRICS)
        sig_df = compute_pairwise_significance(df, PVALUE_COMPARISONS, SLO_METRICS)
        summary_df = pd.concat([summary_df, sig_df], ignore_index=True)
        summary_df.to_csv(os.path.join(args.output_dir,
                                       f"summary_{ds_name}.csv"), index=False)

        print(f"\n  -- {ds_name} Summary --")
        avail = [c for c in (["method"] + SLO_METRICS) if c in summary_df.columns]
        print(summary_df[avail].to_string(index=False))

    print(f"\n\nAll done. Outputs in: {args.output_dir}")


if __name__ == "__main__":
    main()
