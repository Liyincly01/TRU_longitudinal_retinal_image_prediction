#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.reporting import write_results


def extrapolate(times, distances, target_time):
    times, distances = np.asarray(times, float), np.asarray(distances, float)
    if len(times) < 2 or not np.isfinite(distances).all() or np.any(np.diff(times) <= 0):
        return None
    horizon = target_time - times[-1]
    last_two = distances[-1] + horizon * (distances[-1] - distances[-2]) / (times[-1] - times[-2])
    centered = times - times[-1]
    slope, intercept = np.linalg.lstsq(np.column_stack([centered, np.ones(len(times))]),
                                      distances, rcond=None)[0]
    return {"trend_last_two": float(last_two - distances[-1]),
            "trend_ols": float(slope * horizon + intercept - distances[-1])}


def analyze(history, per_eye, max_history=None):
    if max_history is not None and max_history < 1:
        raise ValueError("max_history must be positive.")
    rows, eligible = [], set()
    tru = per_eye[per_eye.method == "TRU"]
    for record in tru.to_dict("records"):
        recorded_limit = record.get("max_history", np.nan)
        if np.isfinite(recorded_limit) and max_history is not None and recorded_limit != max_history:
            raise ValueError("Trend history limit must match the evaluated TRU history limit.")
        limit = max_history if max_history is not None else (
            int(recorded_limit) if np.isfinite(recorded_limit) else 6)
        group = history[(history.cohort == record["cohort"]) &
                        (history.eye_id == record["eye_id"])].sort_values("time").tail(limit)
        if not np.isfinite(record["observed_distance_change"]) or len(group) < 2:
            continue
        if not np.isclose(group.distance.iloc[-1], record["last_distance"], equal_nan=True):
            raise ValueError("History and endpoint baseline distances differ.")
        predicted = extrapolate(group.time, group.distance, record["target_time"])
        if predicted is None:
            continue
        key = (record["cohort"], record["eye_id"])
        eligible.add(key)
        for method, value in predicted.items():
            row = {k: record[k] for k in ["cohort", "eye_id", "patient_id", "observed_distance_change",
                                          "last_distance", "n_visits", "target_time"]}
            row.update(method=method, predicted_distance_change=value, max_history=limit,
                       n_history_used=len(group))
            rows.append(row)
    keep = tru[[ (r.cohort, r.eye_id) in eligible for r in tru.itertuples() ]]
    return pd.concat([keep, pd.DataFrame(rows)], ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description="Extrapolate per-eye fovea-distance histories.")
    parser.add_argument("--history", required=True)
    parser.add_argument("--per_eye", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_history", type=int,
                        help="History limit; defaults to the recorded TRU evaluation setting.")
    args = parser.parse_args()
    types = {"eye_id": str, "patient_id": str}
    result = analyze(pd.read_csv(args.history, dtype=types), pd.read_csv(args.per_eye, dtype=types),
                     args.max_history)
    if result.empty:
        parser.error("No eyes have two usable prior visits and a TRU endpoint.")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_results(result, output, args.bootstrap, args.seed)


if __name__ == "__main__":
    main()
