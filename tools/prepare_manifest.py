#!/usr/bin/env python3
import argparse
import csv
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import sequences, read_manifest


def split_patients(records, cohort, holdout_seed=1337, validation_seed=42):
    selected = [r for r in records if r["cohort"] == cohort]
    if not selected or {r["split"] for r in selected} != {"unsplit"}:
        raise ValueError("Development splitting requires an unsplit cohort.")
    patients = sorted({r["patient_id"] for r in selected})
    if len(patients) < 10:
        raise ValueError("Too few patients for a 90/10 development split.")
    n_test = max(1, round(.10 * len(patients)))
    indices = np.random.default_rng(holdout_seed).choice(len(patients), n_test, replace=False)
    test = {patients[i] for i in indices}
    remaining = sorted(set(patients) - test)
    random.Random(validation_seed).shuffle(remaining)
    position = int(len(remaining) * .85)
    train = set(remaining[:position])
    for row in selected:
        row["split"] = "test" if row["patient_id"] in test else (
            "train" if row["patient_id"] in train else "val")
    return records


def main():
    parser = argparse.ArgumentParser(description="Index registered FAF images and patient-level splits.")
    parser.add_argument("--source", action="append", required=True, metavar="COHORT:SPLIT=DIRECTORY")
    parser.add_argument("--exclude_patients_tsv")
    parser.add_argument("--split_development", metavar="COHORT")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    records = []
    for source in args.source:
        label, directory = source.split("=", 1)
        cohort, split = label.split(":", 1)
        seqs = sequences([directory], cohort=cohort, split=split)
        records.extend(v for visits in seqs.values() for v in visits)
    if args.exclude_patients_tsv:
        with open(args.exclude_patients_tsv, newline="") as handle:
            excluded = {r.get("patient_id", r.get("hashed_mrn", r.get("mrn", "")))
                        for r in csv.DictReader(handle, delimiter="\t")}
        records = [r for r in records if r["patient_id"] not in excluded]
    if args.split_development:
        records = split_patients(records, args.split_development)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    frame = pd.DataFrame(records)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", dir=output.parent) as handle:
        frame.to_csv(handle, index=False)
        handle.flush()
        read_manifest(handle.name)
    frame.to_csv(output, index=False)
    summary = frame.groupby(["cohort", "split"]).agg(
        patients=("patient_id", "nunique"), eyes=("eye_id", "nunique"), images=("image", "size"))
    summary.to_csv(output.with_name(output.stem + "_counts.csv"))
    print(summary.to_string())


if __name__ == "__main__":
    main()
