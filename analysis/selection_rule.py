#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import write_json
from src.statistics import correlation_difference


def audit(frame, calibration, reference="IA_Nonlinear", draws=2000, seed=42):
    if not calibration.get("complete"):
        raise ValueError("Calibration is incomplete.")
    if set(frame.patient_id.astype(str)) & set(calibration["patient_ids"]):
        raise ValueError("Calibration and audit patients overlap.")
    if frame.cohort.nunique() != 1:
        raise ValueError("Audit one cohort at a time.")
    stochastic = frame[frame.method == reference]
    deterministic = frame[frame.method == "TRU"]
    joined = stochastic.merge(deterministic, on=["cohort", "eye_id", "patient_id"],
                              suffixes=("_r", "_d"), validate="one_to_one")
    if not len(joined):
        raise ValueError("No paired aligned-reference/TRU predictions.")
    if not (joined.n_samples_r == 10).all():
        raise ValueError("The manuscript audit requires K=10 samples per eye.")
    if not np.allclose(joined.observed_distance_change_r, joined.observed_distance_change_d, equal_nan=True):
        raise ValueError("Observed endpoints differ between methods.")
    fields = ["observed_distance_change_r", "predicted_distance_change_r",
              "predicted_distance_change_d", "seed_sd_distance_r"]
    paired = joined[np.isfinite(joined[fields].to_numpy(float)).all(axis=1)]
    if len(paired) < 2:
        raise ValueError("Too few endpoint-eligible paired eyes.")
    gain = correlation_difference(paired.observed_distance_change_r,
                                  paired.predicted_distance_change_r,
                                  paired.predicted_distance_change_d, paired.patient_id, draws, seed)
    sd = float(np.median(paired.seed_sd_distance_r))
    sigma, delta = calibration["sigma_meas"], calibration["delta"]
    if not np.isfinite([sigma, delta, gain["estimate"], gain["ci_low"]]).all():
        raise ValueError("Selection criteria are not estimable.")
    resolvable = sd >= sigma
    useful_gain = gain["estimate"] >= delta and gain["ci_low"] > 0
    return dict(reference=reference, n_eyes=len(paired), n_patients=paired.patient_id.nunique(),
                median_seed_sd=sd, sigma_meas=sigma, delta=delta,
                fraction_below_development_floor=float(np.mean(paired.seed_sd_distance_r < sigma)),
                median_ratio_to_development_floor=float(np.median(paired.seed_sd_distance_r / sigma))
                if sigma > 0 else None,
                gain=gain, resolvable=bool(resolvable), useful_gain=bool(useful_gain),
                selected_class="stochastic" if resolvable and useful_gain else "deterministic")


def compare_noise(frame, noise, reference):
    reference_rows = frame[frame.method == reference]
    columns = ["cohort", "eye_id", "patient_id"]
    joined = reference_rows.merge(noise, on=columns, validate="one_to_one")
    required = ["observed_distance_change", "predicted_distance_change", "seed_sd_distance", "sigma_meas"]
    joined = joined[np.isfinite(joined[required].to_numpy(float)).all(axis=1)].copy()
    if not len(joined):
        raise ValueError("No audit eyes match per-eye measurement noise.")
    joined["below_noise_floor"] = joined.seed_sd_distance < joined.sigma_meas
    joined["seed_to_noise_ratio"] = np.divide(
        joined.seed_sd_distance, joined.sigma_meas,
        out=np.full(len(joined), np.nan), where=joined.sigma_meas > 0)
    return joined[columns + ["seed_sd_distance", "sigma_meas", "below_noise_floor", "seed_to_noise_ratio"]]


def criterion_table(result, screen=None):
    rows = []
    if screen:
        rows.append(dict(criterion="elapsed_time_screen", estimate=screen["frac_time"],
                         threshold="descriptive", passed=None))
    rows.extend([
        dict(criterion="resolvable_diversity", estimate=result["median_seed_sd"],
             threshold=result["sigma_meas"], passed=result["resolvable"]),
        dict(criterion="endpoint_gain", estimate=result["gain"]["estimate"],
             ci_low=result["gain"]["ci_low"], ci_high=result["gain"]["ci_high"],
             p=result["gain"]["p"], threshold=result["delta"], passed=result["useful_gain"]),
        dict(criterion="selected_class", estimate=result["selected_class"],
             threshold="both Stage-2 criteria")])
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Apply the endpoint-based model-class selection rule.")
    parser.add_argument("--per_eye", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--reference", default="IA_Nonlinear", choices=["IA_Nonlinear", "IA_Linear"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--measurement_noise", help="Per-eye noise from the audit cohort perturbation run.")
    parser.add_argument("--screen", help="Elapsed-time screen JSON.")
    args = parser.parse_args()
    frame = pd.read_csv(args.per_eye, dtype={"patient_id": str, "eye_id": str})
    calibration = json.loads(Path(args.calibration).read_text())
    result = audit(frame, calibration, args.reference, args.bootstrap, args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.measurement_noise:
        noise = pd.read_csv(args.measurement_noise, dtype={"patient_id": str, "eye_id": str})
        paired_noise = compare_noise(frame, noise, args.reference)
        paired_noise.to_csv(output.with_name(output.stem + "_noise.csv"), index=False)
        ratios = paired_noise.seed_to_noise_ratio.dropna()
        result["per_eye_noise"] = dict(
            n_eyes=len(paired_noise), fraction_below_floor=float(paired_noise.below_noise_floor.mean()),
            median_ratio=float(ratios.median()) if len(ratios) else None,
            n_positive_noise_eyes=len(ratios))
    screen = json.loads(Path(args.screen).read_text()) if args.screen else None
    criterion_table(result, screen).to_csv(output.with_suffix(".csv"), index=False)
    write_json(args.output, result)


if __name__ == "__main__":
    main()
