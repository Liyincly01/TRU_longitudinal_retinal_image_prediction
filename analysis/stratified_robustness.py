#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import write_json
from src.statistics import correlation


def analyze(frame, draws=2000, seed=42):
    if frame.cohort.nunique() != 1:
        raise ValueError("Stratify one hold-out cohort at a time.")
    tru = frame[frame.method == "TRU"]
    finite = np.isfinite(tru[["observed_distance_change", "first_visit_area"]].to_numpy(float)).all(axis=1)
    eligible = tru.loc[finite]
    if eligible.empty:
        raise ValueError("No TRU endpoint-eligible eyes.")
    cutpoints = np.quantile(eligible.first_visit_area, [1/3, 2/3])
    strata = [(f"visits_ge_{n}", eligible[eligible.n_visits >= n].eye_id) for n in (2, 3, 4)]
    tertile = np.searchsorted(cutpoints, eligible.first_visit_area, side="left") + 1
    strata += [(f"severity_T{i}", eligible.loc[tertile == i, "eye_id"]) for i in (1, 2, 3)]
    rows = []
    for stratum, eyes in strata:
        subset = frame[frame.eye_id.isin(eyes)]
        for method, group in subset.groupby("method", sort=False):
            stats = correlation(group.observed_distance_change, group.predicted_distance_change,
                                group.patient_id, draws, seed)
            rows.append(dict(stratum=stratum, method=method, **stats))
    return pd.DataFrame(rows), dict(cutpoints=cutpoints, n_eligible_eyes=len(eligible),
                                    baseline="first_recorded_visit", tie_rule="lower_tertile")


def main():
    parser = argparse.ArgumentParser(description="History-length and baseline-severity robustness.")
    parser.add_argument("--per_eye", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    frame = pd.read_csv(args.per_eye, dtype={"eye_id": str, "patient_id": str})
    result, metadata = analyze(frame, args.bootstrap, args.seed)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result.to_csv(output / "strata.csv", index=False)
    write_json(output / "strata.json", metadata)


if __name__ == "__main__":
    main()
