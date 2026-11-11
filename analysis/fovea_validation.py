#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.clinical_metrics import measurements
from src.data import write_json
from src.reporting import summarize


def localization_errors(automatic, manual):
    if automatic.faf_file.duplicated().any() or manual.faf_file.duplicated().any():
        raise ValueError("Fovea tables must contain one row per image.")
    joined = automatic.merge(manual, on="faf_file", validate="one_to_one")
    columns = ["fovea_x_256", "fovea_y_256", "manual_x_256", "manual_y_256"]
    joined = joined[np.isfinite(joined[columns].to_numpy(float)).all(axis=1)].copy()
    if "in_faf" in joined:
        joined = joined[joined.in_faf.astype(str).str.lower().isin(["1", "1.0", "true"])]
    if not len(joined):
        raise ValueError("No paired valid foveal coordinates.")
    joined["dx"] = joined.manual_x_256 - joined.fovea_x_256
    joined["dy"] = joined.manual_y_256 - joined.fovea_y_256
    joined["error_px"] = np.hypot(joined.dx, joined.dy)
    return joined


def endpoint_remeasurement(index_path, manual):
    index_path = Path(index_path)
    index = pd.read_csv(index_path, dtype={"eye_id": str, "patient_id": str})
    lookup = manual.set_index("faf_file")[["manual_x_256", "manual_y_256"]].to_dict("index")
    rows = []
    for row in index.to_dict("records"):
        point = lookup.get(row["fovea_image"])
        if point is None:
            continue
        xy = np.array([point["manual_x_256"], point["manual_y_256"]], float)
        with np.load(index_path.parent / row["path"], allow_pickle=False) as cache:
            for label, fovea in [("automatic", cache["fovea"]), ("manual", xy)]:
                _, distances = measurements(np.stack([cache["history"][-1], cache["target"]]),
                                             cache["valid_mask"], fovea,
                                             border_px=int(cache["border_px"]))
                for i, method in enumerate(cache["methods"].astype(str)):
                    _, predicted = measurements(cache[f"method_{i}"], cache["valid_mask"],
                                                fovea, border_px=int(cache["border_px"]))
                    rows.append(dict(cohort=row["cohort"], eye_id=row["eye_id"],
                                     patient_id=row["patient_id"], method=method, coordinate=label,
                                     observed_distance_change=distances[1] - distances[0],
                                     predicted_distance_change=float(np.mean(predicted) - distances[0])))
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("No manual coordinates match the cached last-history frame.")
    valid = frame.groupby(["cohort", "eye_id"])[["observed_distance_change", "predicted_distance_change"]].agg(
        lambda s: np.isfinite(s).all()).all(axis=1)
    keys = set(valid[valid].index)
    return frame[[(r.cohort, r.eye_id) in keys for r in frame.itertuples()]]


def main():
    parser = argparse.ArgumentParser(description="Validate foveal localization against manual coordinates.")
    parser.add_argument("--automatic", required=True)
    parser.add_argument("--manual", required=True)
    parser.add_argument("--index")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    automatic = pd.read_csv(args.automatic)
    manual = pd.read_csv(args.manual)
    paired = localization_errors(automatic, manual)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paired.to_csv(output / "localization_residuals.csv", index=False)
    q = np.percentile(paired.error_px, [25, 50, 75])
    write_json(output / "localization.json", dict(n_pairs=len(paired),
               median_px=q[1], q25_px=q[0], q75_px=q[2],
               n_automatic=len(automatic), n_manual=len(manual)))
    if args.index:
        frame = endpoint_remeasurement(args.index, manual)
        frame.to_csv(output / "remeasured_per_eye.csv", index=False)
        tables = []
        for coordinate, subset in frame.groupby("coordinate"):
            table = summarize(subset, args.bootstrap, args.seed)
            table["coordinate"] = coordinate
            table["rank"] = table.groupby(["cohort", "metric"]).estimate.rank(
                ascending=False, method="min")
            tables.append(table)
        pd.concat(tables, ignore_index=True).to_csv(output / "manual_endpoint_comparison.csv", index=False)


if __name__ == "__main__":
    main()
