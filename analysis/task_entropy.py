#!/usr/bin/env python3
"""
Model-free task-entropy characterization.

Measures the intrinsic entropy of the (I_N, I*) prediction task at the
pixel and image level from the training data alone — no model is
invoked. Outputs per-pixel change distributions, image-level summaries,
and stratified statistics by inter-visit interval, plus figures.

Code-only release: a data directory (PNGs following the documented
naming schema) must be supplied via `--data_dir`.
"""

import argparse
import glob
import os
import re
import warnings
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib.gridspec import GridSpec
from scipy import stats
from skimage.metrics import structural_similarity as ssim_func

warnings.filterwarnings("ignore", category=FutureWarning)


plt.rcParams.update({
    "font.size": 12, "axes.labelsize": 13, "axes.titlesize": 14,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
})

IMAGE_SIZE = 256


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


def load_image_01(path, size=IMAGE_SIZE):
    img = Image.open(path).convert("L").resize((size, size), Image.BILINEAR)
    return np.array(img, dtype=np.float32) / 255.0


def load_mask(path, size=IMAGE_SIZE):
    mp = get_mask_path(path)
    if mp:
        m = Image.open(mp).convert("L").resize((size, size), Image.NEAREST)
        return np.array(m, dtype=np.float32) / 255.0 > 0.5
    return np.ones((size, size), dtype=bool)


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


def build_pairs(eye_seqs):
    """Per eye, pair the second-to-last and last visit as (I_N, I*)."""
    pairs = []
    for eid, visits in sorted(eye_seqs.items()):
        i_n = visits[-2]
        i_star = visits[-1]
        pairs.append({
            "eye_id": eid,
            "path_in": i_n["path"],
            "path_istar": i_star["path"],
            "delta_t": i_star["time"] - i_n["time"],
        })
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Task-entropy characterization.")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory of PNGs following the documented naming schema.")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_pairs", type=int, default=None,
                        help="Limit number of pairs for testing.")
    args = parser.parse_args()

    fig_dir = os.path.join(args.output_dir, "figures")
    stat_dir = os.path.join(args.output_dir, "statistics")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(stat_dir, exist_ok=True)

    print("Scanning dataset ...")
    eye_seqs = scan_sequences(args.data_dir)
    pairs = build_pairs(eye_seqs)
    if args.max_pairs:
        pairs = pairs[:args.max_pairs]
    print(f"  Eyes with >=2 visits: {len(eye_seqs)}")
    print(f"  (I_N, I*) pairs: {len(pairs)}")

    # ---------------- Step 1: per-pixel change distribution ----------------
    print("\nStep 1: Per-pixel change distribution ...")
    all_abs_deltas = []
    image_stats = []

    for pi, pair in enumerate(pairs):
        i_n = load_image_01(pair["path_in"])
        i_star = load_image_01(pair["path_istar"])
        mask_n = load_mask(pair["path_in"])
        mask_star = load_mask(pair["path_istar"])
        mask = mask_n & mask_star

        n_valid = int(mask.sum())
        if n_valid < 100:
            continue

        abs_delta = np.abs(i_star.astype(np.float64) - i_n.astype(np.float64))
        valid_deltas = abs_delta[mask]

        if len(valid_deltas) > 10000:
            idx = np.random.RandomState(pi).choice(len(valid_deltas), 10000, replace=False)
            all_abs_deltas.append(valid_deltas[idx].astype(np.float32))
        else:
            all_abs_deltas.append(valid_deltas.astype(np.float32))

        changed_frac = float((valid_deltas > 0.05).mean())
        mean_delta = float(valid_deltas.mean())

        ys, xs = np.where(mask)
        y0, y1, x0, x1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
        try:
            ssim_val = float(ssim_func(
                i_star[y0:y1+1, x0:x1+1], i_n[y0:y1+1, x0:x1+1], data_range=1.0))
        except Exception:
            ssim_val = np.nan

        image_stats.append({
            "eye_id": pair["eye_id"],
            "delta_t": pair["delta_t"],
            "n_valid_pixels": n_valid,
            "changed_fraction": changed_frac,
            "mean_abs_delta": mean_delta,
            "copy_last_ssim": ssim_val,
            "frac_lt_001": float((valid_deltas < 0.01).mean()),
            "frac_lt_002": float((valid_deltas < 0.02).mean()),
            "frac_lt_005": float((valid_deltas < 0.05).mean()),
            "frac_lt_010": float((valid_deltas < 0.10).mean()),
            "median_delta": float(np.median(valid_deltas)),
            "p95_delta": float(np.percentile(valid_deltas, 95)),
            "p99_delta": float(np.percentile(valid_deltas, 99)),
        })

        if (pi + 1) % 1000 == 0:
            print(f"    {pi+1}/{len(pairs)}", flush=True)

    print(f"  Processed {len(image_stats)} valid pairs")
    all_abs_deltas = np.concatenate(all_abs_deltas)
    print(f"  Pooled pixel samples: {len(all_abs_deltas):,}")

    df_img = pd.DataFrame(image_stats)
    df_img.to_csv(os.path.join(stat_dir, "image_level_stats.csv"), index=False)

    total_pixels = sum(s["n_valid_pixels"] for s in image_stats)
    pixel_summary = {
        "total_pixels_analyzed": total_pixels,
        "total_pairs": len(image_stats),
        "frac_lt_001": float((all_abs_deltas < 0.01).mean()),
        "frac_lt_002": float((all_abs_deltas < 0.02).mean()),
        "frac_lt_005": float((all_abs_deltas < 0.05).mean()),
        "frac_lt_010": float((all_abs_deltas < 0.10).mean()),
        "median_abs_delta": float(np.median(all_abs_deltas)),
        "p95_abs_delta": float(np.percentile(all_abs_deltas, 95)),
        "p99_abs_delta": float(np.percentile(all_abs_deltas, 99)),
    }
    pd.DataFrame([pixel_summary]).to_csv(
        os.path.join(stat_dir, "pixel_change_summary.csv"), index=False)

    print("\n  === Pixel-Level Change Summary ===")
    for k, v in pixel_summary.items():
        print(f"    {k}: {v:.6f}" if isinstance(v, float) else f"    {k}: {v:,}")

    # ---------------- Figures 1 & 2: pixel histograms / CDF ----------------
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(all_abs_deltas, bins=400, range=(0, 1), color="#2196F3", alpha=0.8,
            edgecolor="none")
    ax.set_yscale("log")
    ax.set_xlabel("Per-pixel absolute change |I* - I_N|")
    ax.set_ylabel("Count (log scale)")
    ax.set_title("Distribution of Per-Pixel Intensity Change")
    for thresh, color, label in [(0.01, "#4CAF50", "1%"), (0.05, "#FF9800", "5%"),
                                  (0.10, "#F44336", "10%")]:
        frac = float((all_abs_deltas < thresh).mean())
        ax.axvline(thresh, color=color, linestyle="--", linewidth=1.5)
        ax.text(thresh + 0.005, ax.get_ylim()[1] * 0.5,
                f"{frac:.1%} < {label}", fontsize=9, color=color, rotation=90, va="center")
    ax.set_xlim(0, 0.5)
    for fmt in ("png", "pdf"):
        fig.savefig(os.path.join(fig_dir, f"pixel_change_histogram.{fmt}"))
    plt.close(fig)

    sorted_d = np.sort(all_abs_deltas)
    cdf = np.arange(1, len(sorted_d) + 1) / len(sorted_d)
    fig, ax = plt.subplots(figsize=(8, 5))
    step = max(1, len(sorted_d) // 5000)
    ax.plot(sorted_d[::step], cdf[::step], color="#2196F3", linewidth=2)
    ax.set_xlabel("Per-pixel absolute change |I* - I_N|")
    ax.set_ylabel("Cumulative fraction of pixels")
    ax.set_title("Empirical CDF of Per-Pixel Change")
    ax.set_xlim(0, 0.3)
    ax.set_ylim(0.5, 1.005)
    ax.grid(True, alpha=0.3)
    for thresh, color in [(0.01, "#4CAF50"), (0.02, "#8BC34A"),
                           (0.05, "#FF9800"), (0.10, "#F44336")]:
        frac = float((all_abs_deltas < thresh).mean())
        ax.axvline(thresh, color=color, linestyle="--", linewidth=1, alpha=0.7)
        ax.axhline(frac, color=color, linestyle=":", linewidth=0.8, alpha=0.5)
        ax.plot(thresh, frac, "o", color=color, markersize=6)
        ax.annotate(f"{frac:.1%}", (thresh, frac), textcoords="offset points",
                    xytext=(8, -5), fontsize=9, color=color)
    for fmt in ("png", "pdf"):
        fig.savefig(os.path.join(fig_dir, f"pixel_change_cdf.{fmt}"))
    plt.close(fig)

    # ---------------- Image-level stats + delta_t scatter ----------------
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df_img["changed_fraction"], bins=80, color="#9C27B0", alpha=0.8,
            edgecolor="none")
    ax.set_xlabel("Fraction of pixels with |I* - I_N| > 0.05")
    ax.set_ylabel("Number of eyes")
    ax.set_title("Distribution of Changed Pixel Fraction Across Eyes")
    med = df_img["changed_fraction"].median()
    ax.axvline(med, color="black", linestyle="--", linewidth=1.5)
    ax.text(med + 0.01, ax.get_ylim()[1] * 0.9, f"Median: {med:.3f}", fontsize=11)
    for fmt in ("png", "pdf"):
        fig.savefig(os.path.join(fig_dir, f"changed_fraction_histogram.{fmt}"))
    plt.close(fig)

    valid = df_img[["delta_t", "changed_fraction", "mean_abs_delta"]].dropna()
    slope, intercept, r, p, _ = stats.linregress(valid["delta_t"], valid["changed_fraction"])
    x_line = np.linspace(0, min(10, valid["delta_t"].max()), 100)

    fig = plt.figure(figsize=(8, 6))
    gs = GridSpec(2, 1, height_ratios=[1, 5], hspace=0.05)
    ax_hist = fig.add_subplot(gs[0])
    ax_main = fig.add_subplot(gs[1], sharex=ax_hist)
    ax_hist.hist(valid["delta_t"], bins=100, color="#90CAF9", edgecolor="none",
                 range=(0, min(10, valid["delta_t"].max())))
    ax_hist.set_ylabel("Count", fontsize=9)
    ax_hist.tick_params(labelbottom=False, labelsize=9)
    ax_hist.set_title("Temporal Change vs. Inter-Visit Interval", fontsize=14)
    hb = ax_main.hexbin(valid["delta_t"], valid["changed_fraction"],
                         gridsize=60, cmap="Blues", mincnt=1,
                         extent=[0, min(10, valid["delta_t"].max()),
                                 0, valid["changed_fraction"].quantile(0.995)])
    plt.colorbar(hb, ax=ax_main, pad=0.02, fraction=0.04, label="Count")
    ax_main.plot(x_line, slope * x_line + intercept, color="#F44336", linewidth=2,
                 label=f"r={r:.3f}, p={p:.1e}")
    ax_main.set_xlabel("Inter-visit interval (years)")
    ax_main.set_ylabel("Fraction of pixels with |I* - I_N| > 0.05")
    ax_main.legend(fontsize=11, loc="upper left")
    ax_main.set_xlim(0, min(10, valid["delta_t"].max() * 1.05))
    for fmt in ("png", "pdf"):
        fig.savefig(os.path.join(fig_dir, f"delta_t_vs_changed_fraction.{fmt}"))
    plt.close(fig)

    slope2, intercept2, r2, p2, _ = stats.linregress(valid["delta_t"], valid["mean_abs_delta"])
    fig = plt.figure(figsize=(8, 6))
    gs = GridSpec(2, 1, height_ratios=[1, 5], hspace=0.05)
    ax_hist = fig.add_subplot(gs[0])
    ax_main = fig.add_subplot(gs[1], sharex=ax_hist)
    ax_hist.hist(valid["delta_t"], bins=100, color="#FFE0B2", edgecolor="none",
                 range=(0, min(10, valid["delta_t"].max())))
    ax_hist.set_ylabel("Count", fontsize=9)
    ax_hist.tick_params(labelbottom=False, labelsize=9)
    ax_hist.set_title("Mean Pixel Change vs. Inter-Visit Interval", fontsize=14)
    hb = ax_main.hexbin(valid["delta_t"], valid["mean_abs_delta"],
                         gridsize=60, cmap="Oranges", mincnt=1,
                         extent=[0, min(10, valid["delta_t"].max()),
                                 0, valid["mean_abs_delta"].quantile(0.995)])
    plt.colorbar(hb, ax=ax_main, pad=0.02, fraction=0.04, label="Count")
    ax_main.plot(x_line, slope2 * x_line + intercept2, color="#F44336", linewidth=2,
                 label=f"r={r2:.3f}, p={p2:.1e}")
    ax_main.set_xlabel("Inter-visit interval (years)")
    ax_main.set_ylabel("Mean |I* - I_N| per eye")
    ax_main.legend(fontsize=11, loc="upper left")
    ax_main.set_xlim(0, min(10, valid["delta_t"].max() * 1.05))
    for fmt in ("png", "pdf"):
        fig.savefig(os.path.join(fig_dir, f"delta_t_vs_mean_change.{fmt}"))
    plt.close(fig)

    # ---------------- Stratified statistics ----------------
    strata_defs = [
        ("Short (dt < 0.5y)", df_img["delta_t"] < 0.5),
        ("Medium (0.5 <= dt < 1.5y)", (df_img["delta_t"] >= 0.5) & (df_img["delta_t"] < 1.5)),
        ("Long (dt >= 1.5y)", df_img["delta_t"] >= 1.5),
        ("All", pd.Series(True, index=df_img.index)),
    ]
    strat_rows = []
    for name, mask_s in strata_defs:
        subset = df_img[mask_s]
        if len(subset) == 0:
            continue
        strat_rows.append({
            "stratum": name,
            "n_pairs": len(subset),
            "frac_pixels_lt_005": f"{subset['frac_lt_005'].mean():.4f}",
            "median_changed_frac": f"{subset['changed_fraction'].median():.4f}",
            "mean_abs_delta": f"{subset['mean_abs_delta'].mean():.4f}+/-{subset['mean_abs_delta'].std():.4f}",
            "copy_last_ssim": f"{subset['copy_last_ssim'].mean():.4f}+/-{subset['copy_last_ssim'].std():.4f}",
        })
    df_strat = pd.DataFrame(strat_rows)
    df_strat.to_csv(os.path.join(stat_dir, "stratified_by_delta_t.csv"), index=False)

    print("\n  === Stratified Statistics ===")
    print(df_strat.to_string(index=False))

    print("\n" + "=" * 70)
    print("Key Numbers:")
    print("=" * 70)
    print(f"  Dataset: {args.data_dir}")
    print(f"  Pairs: {len(image_stats):,}, total pixels: {total_pixels:,}")
    print(f"  {pixel_summary['frac_lt_001']:.1%} change by < 1%")
    print(f"  {pixel_summary['frac_lt_005']:.1%} change by < 5%")
    print(f"  {pixel_summary['frac_lt_010']:.1%} change by < 10%")
    print(f"  Median per-pixel |delta|: {pixel_summary['median_abs_delta']:.4f}")
    print(f"  Correlation (delta_t vs changed_fraction): r={r:.3f}")
    print(f"\nAll outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
