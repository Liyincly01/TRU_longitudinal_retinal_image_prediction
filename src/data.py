import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


def parse_filename(path):
    stem = Path(path).stem
    match = re.fullmatch(r"(\d+)_(\d+)_([0-9]+(?:\.[0-9]+)?)(?:_.+)?", stem)
    if not match or stem.endswith("_mask"):
        raise ValueError(f"Unsupported image filename: {Path(path).name}")
    patient, side, time = match.groups()
    return patient, f"{patient}_{side}", float(time)


def mask_path(image):
    candidate = Path(image).with_name(Path(image).stem + "_mask.png")
    return str(candidate) if candidate.is_file() else None


def load_image(path, size=256):
    with Image.open(path) as image:
        return np.asarray(image.convert("L").resize((size, size), Image.Resampling.BILINEAR),
                          dtype=np.float32) / 255


def load_mask(path, size=256):
    if not path:
        raise ValueError("A registered-image validity mask is required.")
    with Image.open(path) as image:
        return np.asarray(image.convert("L").resize((size, size), Image.Resampling.NEAREST)) > 127


def read_manifest(path):
    base = Path(path).resolve().parent
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"patient_id", "eye_id", "time", "image", "mask", "split", "cohort"}
        if not required <= set(reader.fieldnames or []):
            raise ValueError("Manifest requires " + ", ".join(sorted(required)))
        records = list(reader)
    if not records:
        raise ValueError("Empty manifest.")
    for row in records:
        if any(not str(row[k]).strip() for k in required):
            raise ValueError("Manifest contains empty required fields.")
        row["time"] = float(row["time"])
        if not np.isfinite(row["time"]):
            raise ValueError("Visit time must be finite.")
        for key in ("image", "mask"):
            row[key] = str((base / row[key]).resolve())
            if not Path(row[key]).is_file():
                raise FileNotFoundError(row[key])
    patient_splits = defaultdict(set)
    for row in records:
        patient_splits[row["patient_id"]].add(row["split"])
    overlap = [p for p, splits in patient_splits.items() if len(splits) > 1]
    if overlap:
        raise ValueError(f"{len(overlap)} patients occur in multiple splits.")
    seen = set()
    eye_patients = {}
    for row in records:
        key = (row["cohort"], row["eye_id"], row["time"])
        if key in seen:
            raise ValueError("Duplicate eye/visit entry.")
        seen.add(key)
        eye = (row["cohort"], row["eye_id"])
        previous = eye_patients.setdefault(eye, row["patient_id"])
        if previous != row["patient_id"]:
            raise ValueError("An eye is assigned to multiple patients.")
    return records


def sequences(data_dirs=None, manifest=None, split=None, cohort=None):
    if bool(data_dirs) == bool(manifest):
        raise ValueError("Supply either image directories or a manifest.")
    if manifest:
        records = read_manifest(manifest)
        records = [r for r in records if (split is None or r["split"] == split)
                   and (cohort is None or r["cohort"] == cohort)]
        cohorts = {r["cohort"] for r in records}
        if len(cohorts) > 1:
            raise ValueError("Select a single cohort.")
    else:
        records = []
        for folder in data_dirs:
            paths = sorted(Path(folder).glob("*.png"))
            if not paths:
                raise ValueError(f"No PNG images in {folder}.")
            for path in paths:
                if path.stem.endswith("_mask"):
                    continue
                patient, eye, time = parse_filename(path)
                records.append(dict(patient_id=patient, eye_id=eye, time=time,
                                    image=str(path.resolve()), mask=mask_path(path),
                                    cohort=cohort or "cohort", split=split or "unspecified"))
    result = defaultdict(list)
    for row in records:
        result[row["eye_id"]].append(row)
    for eye, visits in result.items():
        visits.sort(key=lambda v: v["time"])
        if len({v["patient_id"] for v in visits}) != 1:
            raise ValueError(f"Inconsistent patient for eye {eye}.")
        if len(visits) > 1 and np.any(np.diff([v["time"] for v in visits]) <= 0):
            raise ValueError(f"Visit times must be strictly increasing for eye {eye}.")
    result = {k: v for k, v in sorted(result.items()) if len(v) >= 2}
    if not result:
        raise ValueError("No longitudinal eyes found.")
    return result


def checksum(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def training_eyes(manifest, cohort=None, exclude_patients_tsv=None):
    records = read_manifest(manifest)
    records = [r for r in records if cohort is None or r["cohort"] == cohort]
    if len({r["cohort"] for r in records}) != 1:
        raise ValueError("Select one development cohort for training.")
    excluded = set()
    if exclude_patients_tsv:
        with open(exclude_patients_tsv, newline="") as handle:
            excluded = {r.get("patient_id", r.get("hashed_mrn", r.get("mrn", "")))
                        for r in csv.DictReader(handle, delimiter="\t")}
    sets = []
    for split in ("train", "val"):
        selected = [r for r in records if r["split"] == split and r["patient_id"] not in excluded]
        sets.append(({r["patient_id"] for r in selected}, {r["eye_id"] for r in selected}))
    if not sets[0][1] or not sets[1][1]:
        raise ValueError("Manifest must contain nonempty train and val splits.")
    return sets[0][0], sets[1][0], sets[0][1], sets[1][1]


def write_json(path, payload):
    def clean(value):
        if isinstance(value, dict):
            return {str(k): clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, np.ndarray)):
            return [clean(v) for v in value]
        if isinstance(value, (float, np.floating)):
            return float(value) if np.isfinite(value) else None
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, Path):
            return str(value)
        return value
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(clean(payload), indent=2, allow_nan=False) + "\n")
