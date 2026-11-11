#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.clinical_metrics import measurements
from src.data import checksum, write_json
from src.statistics import cluster_indices, spearman


def paired_reliability(observed_draws):
    n = len(observed_draws) // 2
    a = rankdata(observed_draws[:2*n:2], axis=1)
    b = rankdata(observed_draws[1:2*n:2], axis=1)
    a -= a.mean(axis=1, keepdims=True)
    b -= b.mean(axis=1, keepdims=True)
    denominator = np.sqrt(np.sum(a*a, axis=1) * np.sum(b*b, axis=1))
    return np.divide(np.sum(a*b, axis=1), denominator,
                     out=np.full(n, np.nan), where=denominator > 0)


def margin(observed_draws):
    rhos = paired_reliability(observed_draws)
    if not np.isfinite(rhos).all():
        return np.nan, np.nan
    rho = float(np.median(rhos))
    return rho, float(1 / np.sqrt(min(rho, 1.)) - 1) if rho > 0 else np.nan


def perturb_eye(cache, residuals, draws, rng, threshold_sd, include_predictions=True):
    methods = cache["methods"].astype(str).tolist() if include_predictions else []
    base = cache["history"][-1]
    target = cache["target"]
    mask, fovea = cache["valid_mask"], cache["fovea"]
    border = int(cache["border_px"])
    original = np.concatenate([base[None], target[None]] +
                              [cache[f"method_{i}"] for i in range(len(methods))])
    _, distances = measurements(original, mask, fovea, border_px=border)
    if not np.isfinite(distances).all():
        return None
    if np.isin(original, [0., 1.]).all():
        raise ValueError("Threshold perturbation requires probability maps, not binary masks.")
    observed = np.full(draws, np.nan)
    predicted = np.full((draws, len(methods)), np.nan)
    areas = np.full(draws, np.nan)
    for b in range(draws):
        delta = residuals[rng.integers(len(residuals))]
        threshold = rng.normal(.5, threshold_sd)
        _, d = measurements(np.stack([base, target]), mask, fovea + delta,
                            threshold, border)
        observed[b] = d[1] - d[0]
        area, _ = measurements(target[None], mask, fovea + delta, threshold, border)
        areas[b] = area[0]
        for j in range(len(methods)):
            _, dist = measurements(cache[f"method_{j}"], mask, fovea + delta,
                                   threshold, border)
            predicted[b, j] = np.mean(dist) - d[0]
    return observed, predicted, areas


def analyze(index_path, residual_path, output_dir, mode, draws=1000, bootstrap=2000,
            seed=42, threshold_sd=.02):
    if draws < 4 or draws % 2:
        raise ValueError("Use an even number of at least four perturbation draws.")
    if threshold_sd < 0 or bootstrap < 1:
        raise ValueError("Threshold SD must be nonnegative and bootstrap count positive.")
    index_path = Path(index_path)
    index = pd.read_csv(index_path, dtype={"patient_id": str, "eye_id": str})
    if not len(index) or index.duplicated(["cohort", "eye_id"]).any():
        raise ValueError("Probability index must contain unique eyes.")
    if index.cohort.nunique() != 1:
        raise ValueError("Analyze one cohort at a time.")
    if mode == "calibrate" and not set(index.split) <= {"train", "val"}:
        raise ValueError("Calibration requires development train/val data.")
    if mode == "sensitivity" and set(index.split) & {"train", "val"}:
        raise ValueError("Sensitivity mode requires held-out or transfer data.")
    errors = pd.read_csv(residual_path)
    residuals = errors[["dx", "dy"]].to_numpy(float)
    if not len(residuals) or not np.isfinite(residuals).all():
        raise ValueError("Localization residuals must be finite paired dx/dy vectors.")
    rng = np.random.default_rng(seed)
    observations, predictions, area_draws, included = [], [], [], []
    methods = None
    for row in index.to_dict("records"):
        with np.load(index_path.parent / row["path"], allow_pickle=False) as cache:
            names = cache["methods"].astype(str).tolist() if mode == "sensitivity" else []
            if methods is None:
                methods = names
            if methods != names:
                raise ValueError("Cached method order differs between eyes.")
            result = perturb_eye(cache, residuals, draws, rng, threshold_sd,
                                 include_predictions=mode == "sensitivity")
        if result is not None:
            observed, predicted, areas = result
            observations.append(observed)
            predictions.append(predicted)
            area_draws.append(areas)
            included.append(row)
    if len(included) < 2:
        raise ValueError("Too few eyes have well-defined endpoints.")
    observed = np.stack(observations, axis=1)
    predicted = np.stack(predictions, axis=1)
    areas = np.stack(area_draws, axis=1)
    ids = np.array([r["patient_id"] for r in included])
    eye_ids = np.array([r["eye_id"] for r in included])
    noise = np.std(observed, axis=0, ddof=1)
    complete_draws = np.isfinite(observed).all(axis=1) & np.isfinite(predicted).all(axis=(1, 2))
    complete = bool(complete_draws.all() and np.isfinite(noise).all())
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(dict(cohort=str(index.cohort.iloc[0]), eye_id=eye_ids, patient_id=ids, sigma_meas=noise,
                      target_area_sd=np.std(areas, axis=0, ddof=1))).to_csv(
                          output / "measurement_noise_per_eye.csv", index=False)
    np.savez_compressed(output / "perturbations.npz", observed=observed,
                        predicted=predicted, methods=np.array(methods),
                        patient_ids=ids, eye_ids=eye_ids)
    summary = dict(mode=mode, cohort=str(index.cohort.iloc[0]), n_input_eyes=len(index),
                   n_eligible_eyes=len(included), n_patients=len(set(ids)), draws=draws,
                   complete_draws=int(complete_draws.sum()), complete=complete,
                   threshold_mean=.5, threshold_sd=threshold_sd, seed=seed,
                   sigma_meas=float(np.median(noise)) if np.isfinite(noise).all() else np.nan,
                   median_target_area_sd=float(np.median(np.std(areas, axis=0, ddof=1))),
                   residuals_sha256=checksum(residual_path), index_sha256=checksum(index_path),
                   patient_ids=sorted(set(ids)), endpoint="observed_distance_change")
    if mode == "calibrate":
        rho, delta = margin(observed) if complete else (np.nan, np.nan)
        boot = []
        if complete and len(set(ids)) >= 2:
            for rows in cluster_indices(ids, np.random.default_rng(seed + 1), bootstrap):
                _, value = margin(observed[:, rows])
                if np.isfinite(value):
                    boot.append(value)
        summary.update(rho_test_retest=rho, delta=delta, reliability_pairs=draws // 2,
                       delta_ci_low=float(np.percentile(boot, 2.5)) if boot else np.nan,
                       delta_ci_high=float(np.percentile(boot, 97.5)) if boot else np.nan,
                       delta_valid_bootstraps=len(boot))
        summary["complete"] = complete and np.isfinite(delta) and np.isfinite(summary["sigma_meas"])
        write_json(output / "calibration.json", summary)
    else:
        correlations = np.full((draws, len(methods)), np.nan)
        for b in np.flatnonzero(complete_draws):
            for j in range(len(methods)):
                correlations[b, j] = spearman(observed[b], predicted[b, :, j])
        records = []
        nonconstant = [j for j, method in enumerate(methods) if method != "copy_last"]
        valid_rank = complete_draws & np.isfinite(correlations[:, nonconstant]).all(axis=1)
        for j, method in enumerate(methods):
            values = correlations[:, j]
            finite = values[np.isfinite(values)]
            top = (np.isclose(values[valid_rank],
                   correlations[valid_rank][:, nonconstant].max(axis=1)).mean()
                   if valid_rank.any() and j in nonconstant else np.nan)
            records.append(dict(method=method, median=float(np.median(finite)) if len(finite) else np.nan,
                                band_low=float(np.percentile(finite, 2.5)) if len(finite) else np.nan,
                                band_high=float(np.percentile(finite, 97.5)) if len(finite) else np.nan,
                                valid_draws=len(finite), top_rank_fraction=top,
                                ranking_draws=int(valid_rank.sum())))
        pd.DataFrame(records).to_csv(output / "sensitivity.csv", index=False)
        write_json(output / "run.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Propagate measured foveal and segmentation error.")
    parser.add_argument("--index", required=True)
    parser.add_argument("--residuals", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--mode", required=True, choices=["calibrate", "sensitivity"])
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    analyze(args.index, args.residuals, args.output_dir, args.mode,
            args.draws, args.bootstrap, args.seed)


if __name__ == "__main__":
    main()
