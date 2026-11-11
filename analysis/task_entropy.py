#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import sequences, load_image, load_mask, write_json
from src.statistics import correlation


def build_pairs(seqs):
    return [(eye, a, b) for eye, visits in seqs.items() for a, b in zip(visits, visits[1:])]


def histogram_quantile(counts, q):
    n = int(counts.sum())
    position = (n - 1) * q
    cumulative = np.cumsum(counts)
    low = np.searchsorted(cumulative, int(np.floor(position)) + 1)
    high = np.searchsorted(cumulative, int(np.ceil(position)) + 1)
    return (low + (high - low) * (position % 1)) / 255


def analyze(seqs, output_dir, draws=2000, seed=42):
    rows = []
    counts = np.zeros(256, np.int64)
    for eye, a, b in build_pairs(seqs):
        mask = load_mask(a["mask"]) & load_mask(b["mask"])
        if not mask.any():
            raise ValueError(f"No valid overlap for eye {eye}.")
        difference = np.abs(load_image(b["image"]) - load_image(a["image"]))[mask]
        counts += np.bincount(np.rint(difference * 255).astype(int), minlength=256)
        rows.append(dict(eye_id=eye, patient_id=a["patient_id"], start_time=a["time"],
                         end_time=b["time"], delta_t=b["time"] - a["time"],
                         changed_fraction=float(np.mean(difference > .05)),
                         mean_absolute_change=float(np.mean(difference)), n_valid_pixels=int(mask.sum())))
    frame = pd.DataFrame(rows)
    if len(frame) < 2 or frame.delta_t.nunique() < 2:
        raise ValueError("Regression requires multiple pairs with varying intervals.")
    fit = linregress(frame.delta_t, frame.changed_fraction)
    horizon = float(frame.delta_t.median())
    denominator = fit.intercept + fit.slope * horizon
    inference = correlation(frame.delta_t, frame.changed_fraction, frame.patient_id, draws, seed)
    inference["n_pairs"] = inference.pop("n_eyes")
    summary = dict(n_pairs=len(frame), n_eyes=frame.eye_id.nunique(),
                   n_patients=frame.patient_id.nunique(), threshold=.05,
                   slope=fit.slope, intercept=fit.intercept, median_interval=horizon,
                   frac_time=fit.slope * horizon / denominator if denominator > 0 else np.nan,
                   spearman=inference, median_absolute_pixel_change=histogram_quantile(counts, .5))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "pair_statistics.csv", index=False)
    pd.DataFrame({"absolute_change": np.arange(256) / 255, "pixels": counts}).to_csv(
        output / "pixel_distribution.csv", index=False)
    write_json(output / "screen.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Elapsed-time screen on consecutive FAF visits.")
    data = parser.add_mutually_exclusive_group(required=True)
    data.add_argument("--data_dir", action="append")
    data.add_argument("--manifest")
    parser.add_argument("--cohort")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    seqs = sequences(args.data_dir, args.manifest, args.split, args.cohort)
    result = analyze(seqs, args.output_dir, args.bootstrap, args.seed)
    result.update(split=args.split, cohort=args.cohort)
    write_json(Path(args.output_dir) / "screen.json", result)


if __name__ == "__main__":
    main()
