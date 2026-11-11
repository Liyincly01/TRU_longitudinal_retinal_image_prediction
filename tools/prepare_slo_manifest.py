#!/usr/bin/env python3
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage


def registration_mask(image):
    dilated = ndimage.binary_dilation(np.asarray(image) == 0, iterations=1)
    border = np.zeros_like(dilated)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    invalid = ndimage.binary_propagation(dilated & border, mask=dilated,
                                         structure=np.ones((3, 3)))
    return ~invalid


def main():
    parser = argparse.ArgumentParser(description="Index an optional SLO image-evaluation cohort.")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--patient_list", required=True)
    parser.add_argument("--cohort", default="SLO_external")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    folder = output.with_name(output.stem + "_masks")
    if output.exists() or folder.exists():
        raise FileExistsError("Choose a new manifest and mask directory.")
    patients = {line.strip() for line in Path(args.patient_list).read_text().splitlines() if line.strip()}
    rows = []
    for image in sorted(Path(args.data_dir).glob("*.png")):
        parts = image.stem.split("_")
        if image.stem.endswith("_mask") or len(parts) < 4 or parts[0] not in patients:
            continue
        rows.append(dict(patient_id=parts[0], eye_id="_".join(parts[:2]), time=float(parts[2]),
                         image=str(image.resolve()), mask=str(folder / image.name),
                         split="transfer", cohort=args.cohort))
    counts = Counter(row["eye_id"] for row in rows)
    rows = [row for row in rows if counts[row["eye_id"]] >= 2]
    if not rows:
        raise ValueError("No longitudinal SLO eyes match the patient list.")
    folder.mkdir(parents=True)
    for row in rows:
        with Image.open(row["image"]) as image:
            mask = registration_mask(np.asarray(image.convert("L")))
        Image.fromarray(mask.astype(np.uint8) * 255).save(row["mask"])
    pd.DataFrame(rows).to_csv(output, index=False)


if __name__ == "__main__":
    main()
