#!/usr/bin/env python3
"""
Posterior collapse (multi-sample diversity) analysis.

Generates K independent samples per test eye for IA-Nonlinear and IA-Linear,
then measures inter-sample SSIM, per-pixel variance, and bias-variance
decomposition to show that the learned stochastic posteriors have
collapsed to near-point-masses.

This is a code-only release: both model checkpoints and a data directory
must be supplied via CLI flags. No weights or images are shipped.
"""

import argparse
import glob
import os
import re
import sys
import warnings
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
import torch
from PIL import Image
from skimage.metrics import structural_similarity as ssim_func

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import (
    MultiScaleTemporalUNet_v13,
    IALinearDiffusion,
    IANonlinearDiffusion,
)


plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

IMAGE_SIZE = 256
MAX_HISTORY = 6
DELTA_SCALE = 1.0
PAD_DELTA = 100.0


# -----------------------------------------------------------------------------
# Data utilities
# -----------------------------------------------------------------------------

def parse_filename(filepath):
    fn = os.path.basename(filepath)
    base = fn.replace("_reg.png", "").replace("_anchor.png", "")
    m = re.match(r"^(\d+)_(\d+)_([\d.]+)", base)
    if not m:
        return None, None, None
    return m.group(1), f"{m.group(1)}_{m.group(2)}", float(m.group(3))


def get_mask_path(image_path):
    fn = os.path.basename(image_path)
    d = os.path.dirname(image_path)
    for old, new in [("_reg.png", "_reg_mask.png"), ("_anchor.png", "_anchor_mask.png")]:
        if fn.endswith(old):
            mp = os.path.join(d, fn.replace(old, new))
            if os.path.exists(mp):
                return mp
    mp = os.path.join(d, fn[:-4] + "_mask.png")
    return mp if os.path.exists(mp) else None


def load_image(path, size=IMAGE_SIZE):
    img = Image.open(path).convert("L").resize((size, size), Image.BILINEAR)
    return np.array(img, dtype=np.float32) / 255.0 * 2.0 - 1.0


def load_image_01(path, size=IMAGE_SIZE):
    img = Image.open(path).convert("L").resize((size, size), Image.BILINEAR)
    return np.array(img, dtype=np.float32) / 255.0


def load_mask(path, size=IMAGE_SIZE):
    mp = get_mask_path(path)
    if mp:
        m = Image.open(mp).convert("L").resize((size, size), Image.NEAREST)
        return np.array(m, dtype=np.float32) / 255.0 > 0.5
    return np.ones((size, size), dtype=bool)


def to01(arr):
    return np.clip((arr + 1.0) / 2.0, 0.0, 1.0)


def scan_sequences(root_dir):
    files = [p for p in glob.glob(os.path.join(root_dir, "*.png"))
             if not p.endswith("_mask.png")]
    eye_data = defaultdict(list)
    for p in files:
        _, eye_id, t = parse_filename(p)
        if eye_id is None:
            continue
        eye_data[eye_id].append({"path": p, "time": t})
    for eid in eye_data:
        eye_data[eid].sort(key=lambda x: x["time"])
    return {k: v for k, v in eye_data.items() if len(v) >= 2}


def build_tasks(eye_seqs):
    tasks = []
    for eid in sorted(eye_seqs.keys()):
        visits = eye_seqs[eid]
        tasks.append({"eye_id": eid, "history": visits[:-1], "target": visits[-1]})
    return tasks


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


# -----------------------------------------------------------------------------
# Model loaders
# -----------------------------------------------------------------------------

def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = ckpt.get("args", {})
    def get(k, d=None):
        return args.get(k, d) if isinstance(args, dict) else getattr(args, k, d)
    max_h = get("max_history", MAX_HISTORY)

    model = MultiScaleTemporalUNet_v13(
        max_history=max_h, base_channels=64,
        delta_emb_dim=256, history_dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["ema_model"])
    model.eval()
    return model, args, max_h


# -----------------------------------------------------------------------------
# Multi-sample inference
# -----------------------------------------------------------------------------

@torch.no_grad()
def generate_k_samples_ia_nonlinear(model, history, deltas, tmask, device, k,
                                    t_infer=200):
    ia = IANonlinearDiffusion(
        model, t_min=100, t_max=300, device=str(device)
    ).to(device).eval()

    H = torch.from_numpy(history[None]).float().to(device)
    D = torch.from_numpy(deltas[None]).float().to(device)
    T = torch.from_numpy(tmask[None]).bool().to(device)

    samples = []
    for seed in range(k):
        pred = ia.x0hat_predict(H, D, T, t_infer=t_infer, seed=seed * 1000 + 42)
        samples.append(to01(pred.squeeze().cpu().numpy()))
    return np.stack(samples, axis=0)


@torch.no_grad()
def generate_k_samples_ia_linear(model, history, deltas, tmask, device, k):
    fm = IALinearDiffusion(
        model, t_min=0.05, t_max=0.95, t_infer=0.20, device=str(device)
    ).to(device).eval()

    H = torch.from_numpy(history[None]).float().to(device)
    D = torch.from_numpy(deltas[None]).float().to(device)
    T = torch.from_numpy(tmask[None]).bool().to(device)

    samples = []
    for seed in range(k):
        pred = fm.x0hat_predict(H, D, T, t_infer=0.20, seed=seed * 1000 + 42)
        samples.append(to01(pred.squeeze().cpu().numpy()))
    return np.stack(samples, axis=0)


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def compute_pairwise_ssim(samples, mask):
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return np.nan
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
    if (y1 - y0) < 16 or (x1 - x0) < 16:
        return np.nan

    K = samples.shape[0]
    ssims = []
    for i, j in combinations(range(K), 2):
        try:
            s = ssim_func(samples[i, y0:y1+1, x0:x1+1],
                          samples[j, y0:y1+1, x0:x1+1], data_range=1.0)
            ssims.append(float(s))
        except Exception:
            pass
    return float(np.mean(ssims)) if ssims else np.nan


def save_variance_map(output_dir, eye_id, model_name, last_frame, target,
                      mean_pred, pixel_sd, mask):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    titles = ["Last Frame (I_N)", "Target (I*)",
              f"Mean Prediction\n({model_name})",
              f"Pixel SD Across K\n({model_name})"]
    imgs = [last_frame, target, mean_pred, pixel_sd]
    cmaps = ["gray", "gray", "gray", "inferno"]

    for ax, img, title, cmap in zip(axes, imgs, titles, cmaps):
        if cmap == "inferno":
            im = ax.imshow(img, cmap=cmap, vmin=0, vmax=max(0.01, img[mask].max()))
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="SD")
        else:
            ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    safe_eid = eye_id.replace("/", "_")
    safe_model = model_name.replace(" ", "_")
    fig.suptitle(f"Eye: {eye_id}", fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{safe_eid}_{safe_model}_variance_map.png"))
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Posterior-collapse (K-sample) analysis.")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory of held-out FAF images (same schema as training).")
    parser.add_argument("--ckpt_ia_nonlinear", type=str, required=True)
    parser.add_argument("--ckpt_ia_linear", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--k_samples", type=int, default=10)
    parser.add_argument("--max_eyes", type=int, default=None,
                        help="Limit eyes for faster testing.")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    K = args.k_samples
    print(f"Device: {device}, K={K} samples per eye")

    fig_dir = os.path.join(args.output_dir, "figures")
    varmap_dir = os.path.join(fig_dir, "spatial_variance_maps")
    stat_dir = os.path.join(args.output_dir, "statistics")
    for d in (fig_dir, varmap_dir, stat_dir):
        os.makedirs(d, exist_ok=True)

    print("\nLoading models ...")
    model_ia, _, max_h_ia = load_model(args.ckpt_ia_nonlinear, device)
    print(f"  IA-Nonlinear: loaded (max_h={max_h_ia})")
    model_fm, _, max_h_fm = load_model(args.ckpt_ia_linear, device)
    print(f"  IA-Linear: loaded (max_h={max_h_fm})")

    print("\nScanning holdout set ...")
    eye_seqs = scan_sequences(args.data_dir)
    tasks = build_tasks(eye_seqs)
    if args.max_eyes:
        tasks = tasks[:args.max_eyes]
    print(f"  Eyes: {len(tasks)}")

    models_config = [
        ("IA_Nonlinear", model_ia, max_h_ia, generate_k_samples_ia_nonlinear),
        ("IA_Linear",    model_fm, max_h_fm, generate_k_samples_ia_linear),
    ]

    all_per_eye = []
    vis_indices = []

    for task_idx, task in enumerate(tasks):
        hist_visits = task["history"]
        tgt_visit = task["target"]

        tgt_01 = load_image_01(tgt_visit["path"])
        mask = load_mask(tgt_visit["path"])
        if int(mask.sum()) < 100:
            continue

        row = {"eye_id": task["eye_id"], "n_history": len(hist_visits),
               "delta_t": tgt_visit["time"] - hist_visits[-1]["time"]}

        for model_name, model, max_h, gen_fn in models_config:
            h_arr, d_arr, tm_arr = make_diffusion_inputs(
                hist_visits, tgt_visit["time"], max_h=max_h)
            samples = gen_fn(model, h_arr, d_arr, tm_arr, device, k=K)

            mean_pred = samples.mean(axis=0)
            pixel_var = samples.var(axis=0)
            pixel_sd = np.sqrt(pixel_var)

            mean_pixel_var = float(pixel_var[mask].mean())
            mean_pixel_sd = float(pixel_sd[mask].mean())
            pw_ssim = compute_pairwise_ssim(samples, mask)

            sq_error = (mean_pred - tgt_01) ** 2
            mean_sq_error = float(sq_error[mask].mean())
            mean_abs_error = float(np.abs(mean_pred - tgt_01)[mask].mean())
            var_mse_ratio = mean_pixel_var / max(mean_sq_error, 1e-10)

            row[f"{model_name}_pw_ssim"] = pw_ssim
            row[f"{model_name}_mean_pixel_var"] = mean_pixel_var
            row[f"{model_name}_mean_pixel_sd"] = mean_pixel_sd
            row[f"{model_name}_pred_mse"] = mean_sq_error
            row[f"{model_name}_pred_mae"] = mean_abs_error
            row[f"{model_name}_var_mse_ratio"] = var_mse_ratio

            if len(vis_indices) < 4 and model_name == "IA_Nonlinear":
                vis_indices.append(task_idx)

            if task_idx in vis_indices:
                last_frame_01 = load_image_01(hist_visits[-1]["path"])
                save_variance_map(
                    varmap_dir, task["eye_id"], model_name,
                    last_frame_01, tgt_01, mean_pred, pixel_sd, mask)

        all_per_eye.append(row)

        if (task_idx + 1) % 100 == 0:
            print(f"  {task_idx + 1}/{len(tasks)}", flush=True)

    print(f"  Processed {len(all_per_eye)} eyes")

    df = pd.DataFrame(all_per_eye)
    df.to_csv(os.path.join(stat_dir, "per_eye_statistics.csv"), index=False)

    # ---------------- Summary ----------------
    print("\n=== Summary ===")
    summary_rows = []
    for model_name in ["IA_Nonlinear", "IA_Linear"]:
        pw = df[f"{model_name}_pw_ssim"].dropna()
        var = df[f"{model_name}_mean_pixel_var"].dropna()
        sd = df[f"{model_name}_mean_pixel_sd"].dropna()
        mse = df[f"{model_name}_pred_mse"].dropna()
        ratio = df[f"{model_name}_var_mse_ratio"].dropna()

        total_var = var.mean()
        total_mse = mse.mean()
        total_bias = max(0, total_mse - total_var)
        bias_pct = total_bias / max(total_mse, 1e-10) * 100
        var_pct = total_var / max(total_mse, 1e-10) * 100

        summary_rows.append({
            "Model": model_name,
            "Inter-sample SSIM": f"{pw.mean():.6f}+/-{pw.std():.6f}",
            "Mean pixel variance": f"{var.mean():.2e}+/-{var.std():.2e}",
            "Mean pixel SD": f"{sd.mean():.6f}+/-{sd.std():.6f}",
            "Prediction MSE": f"{mse.mean():.6f}+/-{mse.std():.6f}",
            "Var/MSE ratio": f"{ratio.mean():.6f}+/-{ratio.std():.6f}",
            "Variance % of error": f"{var_pct:.2f}%",
            "Bias % of error": f"{bias_pct:.2f}%",
            "n_eyes": len(pw),
        })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(os.path.join(stat_dir, "summary_table.csv"), index=False)
    print(df_summary.to_string(index=False))

    # ---------------- Figures ----------------
    print("\nGenerating figures ...")

    # 1) Inter-sample SSIM histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    for model_name, color, label in [("IA_Nonlinear", "#2196F3", "IA-Nonlinear"),
                                     ("IA_Linear",    "#FF9800", "IA-Linear")]:
        vals = df[f"{model_name}_pw_ssim"].dropna()
        ax.hist(vals, bins=80, alpha=0.6, color=color, label=label, edgecolor="none")
        ax.axvline(vals.mean(), color=color, linestyle="--", linewidth=1.5)
    ax.set_xlabel("Pairwise SSIM Between K Independent Samples")
    ax.set_ylabel("Number of Eyes")
    ax.set_title("Inter-Sample Agreement (Higher = Less Diversity)")
    ax.legend()
    ax.set_xlim(left=max(0.95, ax.get_xlim()[0]))
    for fmt in ("png", "pdf"):
        fig.savefig(os.path.join(fig_dir, f"intersample_ssim_histogram.{fmt}"))
    plt.close(fig)

    # 2) Variance/MSE ratio histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    for model_name, color, label in [("IA_Nonlinear", "#2196F3", "IA-Nonlinear"),
                                     ("IA_Linear",    "#FF9800", "IA-Linear")]:
        vals = df[f"{model_name}_var_mse_ratio"].dropna()
        ax.hist(vals, bins=80, alpha=0.6, color=color, label=label, edgecolor="none")
        ax.axvline(vals.mean(), color=color, linestyle="--", linewidth=1.5)
    ax.set_xlabel("Variance / MSE Ratio")
    ax.set_ylabel("Number of Eyes")
    ax.set_title("Stochastic Variance as Fraction of Total Error")
    ax.legend()
    for fmt in ("png", "pdf"):
        fig.savefig(os.path.join(fig_dir, f"variance_mse_ratio_histogram.{fmt}"))
    plt.close(fig)

    # 3) MAE vs inter-sample SD scatter
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (model_name, color, label) in zip(axes, [
            ("IA_Nonlinear", "#2196F3", "IA-Nonlinear"),
            ("IA_Linear",    "#FF9800", "IA-Linear")]):
        mae_vals = df[f"{model_name}_pred_mae"].dropna()
        sd_vals = df[f"{model_name}_mean_pixel_sd"].dropna()
        idx = mae_vals.index.intersection(sd_vals.index)
        ax.scatter(mae_vals[idx], sd_vals[idx], alpha=0.3, s=10, color=color,
                   rasterized=True)
        lim = max(mae_vals[idx].max(), sd_vals[idx].max()) * 1.05
        ax.plot([0, lim], [0, lim], "k--", linewidth=1, alpha=0.5, label="Identity")
        mean_ratio = float(sd_vals[idx].mean() / mae_vals[idx].mean())
        ax.set_xlabel("Prediction MAE")
        ax.set_ylabel("Inter-Sample SD")
        ax.set_title(f"{label}\nMean SD/MAE ratio: {mean_ratio:.4f}")
        ax.legend()
        ax.set_aspect("equal")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
    fig.suptitle("Prediction Error vs. Stochastic Spread", fontsize=14, y=1.02)
    plt.tight_layout()
    for fmt in ("png", "pdf"):
        fig.savefig(os.path.join(fig_dir, f"mae_vs_sd_scatter.{fmt}"))
    plt.close(fig)

    # 4) Bias-variance bar chart
    fig, ax = plt.subplots(figsize=(6, 5))
    labels = ["IA-Nonlinear", "IA-Linear"]
    x = np.arange(len(labels))
    width = 0.35
    bias_vals, var_vals = [], []
    for model_name in ["IA_Nonlinear", "IA_Linear"]:
        v = df[f"{model_name}_mean_pixel_var"].mean()
        m = df[f"{model_name}_pred_mse"].mean()
        bias_vals.append(max(0, m - v))
        var_vals.append(v)
    ax.bar(x, bias_vals, width, label="Bias^2", color="#2196F3")
    ax.bar(x, var_vals, width, bottom=bias_vals, label="Variance", color="#FF9800")
    ax.set_ylabel("MSE Decomposition")
    ax.set_title("Bias-Variance Decomposition")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    for i, (b, v) in enumerate(zip(bias_vals, var_vals)):
        total = b + v
        ax.text(i, total * 1.02, f"Var: {v/total*100:.2f}%", ha="center", fontsize=10)
    for fmt in ("png", "pdf"):
        fig.savefig(os.path.join(fig_dir, f"bias_variance_decomposition.{fmt}"))
    plt.close(fig)

    print(f"\nAll outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
