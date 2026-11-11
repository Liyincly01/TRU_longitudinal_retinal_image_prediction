#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_strata(path, output):
    frame = pd.read_csv(path)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    groups = [(["visits_ge_2", "visits_ge_3", "visits_ge_4"], ["≥2 visits", "≥3 visits", "≥4 visits"]),
              (["severity_T1", "severity_T2", "severity_T3"], ["T1", "T2", "T3"])]
    for ax, (strata, labels) in zip(axes, groups):
        for method, rows in frame.groupby("method", sort=False):
            if method == "copy_last":
                continue
            rows = rows.set_index("stratum").reindex(strata)
            x = np.arange(3)
            ax.plot(x, rows.estimate, marker="o", label=method)
            ax.vlines(x, rows.ci_low, rows.ci_high, alpha=.65)
        ax.set_xticks(range(3), labels)
        ax.axhline(0, color="gray", linewidth=.7)
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Spearman correlation for distance change")
    axes[0].set_xlabel("Minimum available visits")
    axes[1].set_xlabel("First-visit atrophy-area tertile")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "stratified_robustness.pdf")
    plt.close(fig)


def plot_images(path, output):
    with np.load(path, allow_pickle=False) as cache:
        history, target = cache["history"], cache["target"]
        methods = cache["methods"].astype(str)
        images = [history[-1], target] + [cache[f"method_{i}"][0] for i in range(len(methods))]
        labels = ["Last history", "Target"] + list(methods)
        fig, axes = plt.subplots(2, len(images), figsize=(2 * len(images), 4), squeeze=False)
        for i, (image, label) in enumerate(zip(images, labels)):
            axes[0, i].imshow(image, cmap="gray", vmin=0, vmax=1)
            axes[0, i].set_title(label, fontsize=8)
            axes[1, i].imshow(image - history[-1], cmap="RdBu_r", vmin=-.3, vmax=.3)
        for ax in axes.flat:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(output / "change_maps.pdf")
        plt.close(fig)
        fig, axes = plt.subplots(1, len(images), figsize=(2 * len(images), 2), squeeze=False)
        for i, (image, label) in enumerate(zip(images, labels)):
            ax = axes[0, i]
            if i < 2:
                ax.imshow(image, cmap="gray", vmin=0, vmax=1)
            else:
                ax.imshow(image - target, cmap="RdBu_r", vmin=-.3, vmax=.3)
            ax.set_title(label, fontsize=8)
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(output / "prediction_errors.pdf")
        plt.close(fig)


def plot_screen(path, output):
    frame = pd.read_csv(path)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].scatter(frame.delta_t, frame.changed_fraction, s=5, alpha=.25)
    axes[0].set_xlabel("Elapsed interval (years)")
    axes[0].set_ylabel("Changed-pixel fraction")
    if frame.delta_t.nunique() > 1:
        slope, intercept = np.polyfit(frame.delta_t, frame.changed_fraction, 1)
        x = np.array([frame.delta_t.min(), frame.delta_t.max()])
        axes[0].plot(x, intercept + slope*x, color="black")
    strata = [frame.loc[frame.delta_t < .5, "changed_fraction"],
              frame.loc[(frame.delta_t >= .5) & (frame.delta_t < 1.5), "changed_fraction"],
              frame.loc[frame.delta_t >= 1.5, "changed_fraction"]]
    axes[1].boxplot(strata)
    axes[1].set_xticks([1, 2, 3], ["<0.5 y", "0.5–1.5 y", "≥1.5 y"])
    axes[1].set_ylabel("Changed-pixel fraction")
    fig.tight_layout()
    fig.savefig(output / "elapsed_time_screen.pdf")
    plt.close(fig)


def plot_alignment(path, output):
    frame = pd.read_csv(path)
    order = ["Std_DDIM_50step", "Std_DDIM_1step", "IA_Nonlinear", "IA_Linear", "TRU"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for ax, metric in zip(axes, ["delta_ssim", "psnr"]):
        rows = frame[frame.metric == metric].set_index("method").reindex(order)
        ax.plot(range(len(order)), rows.estimate, marker="o")
        if {"ci_low", "ci_high"} <= set(rows):
            ax.vlines(range(len(order)), rows.ci_low, rows.ci_high)
        ax.set_xticks(range(len(order)), order, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("Change-map SSIM" if metric == "delta_ssim" else "PSNR (dB)")
    fig.tight_layout()
    fig.savefig(output / "alignment.pdf")
    plt.close(fig)


def plot_posterior(noise_path, audit_path, output):
    frame = pd.read_csv(noise_path)
    audit = json.loads(Path(audit_path).read_text())
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    axes[0].scatter(frame.sigma_meas, frame.seed_sd_distance, s=9, alpha=.5)
    limit = max(frame.sigma_meas.max(), frame.seed_sd_distance.max()) * 1.05
    axes[0].plot([0, limit], [0, limit], color="gray", linewidth=1)
    axes[0].set(xlabel="Per-eye measurement SD (pixels)", ylabel="Seed SD (pixels)",
                xlim=(0, limit), ylim=(0, limit))
    gain = audit["gain"]
    axes[1].plot(gain["estimate"], 0, "o")
    axes[1].hlines(0, gain["ci_low"], gain["ci_high"])
    axes[1].axvline(0, color="gray", linewidth=1)
    axes[1].axvline(audit["delta"], linestyle="--", color="black", label="Required gain")
    axes[1].set(xlabel="Aligned stochastic − TRU endpoint correlation", yticks=[])
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "stochastic_audit.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Render longitudinal FAF analysis outputs.")
    parser.add_argument("--strata")
    parser.add_argument("--prediction")
    parser.add_argument("--screen")
    parser.add_argument("--summary")
    parser.add_argument("--noise")
    parser.add_argument("--audit")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if args.strata:
        plot_strata(args.strata, output)
    if args.prediction:
        plot_images(args.prediction, output)
    if args.screen:
        plot_screen(args.screen, output)
    if args.summary:
        plot_alignment(args.summary, output)
    if bool(args.noise) != bool(args.audit):
        parser.error("Use --noise and --audit together.")
    if args.noise:
        plot_posterior(args.noise, args.audit, output)


if __name__ == "__main__":
    main()
