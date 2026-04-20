# Data format

No data is shipped with this repository. All scripts consume a directory
of PNG images that follow a fixed naming convention so that eyes and
time-ordered visits can be recovered without a separate metadata file.

## Filename schema

Every image filename matches one of two shapes:

```
{mrn}_{eye_laterality}_{time}[_reg|_anchor].png
{mrn}_{eye_laterality}_{time}.png
```

Rules:

- `mrn` — integer subject/patient identifier (may be a hash).
- `eye_laterality` — integer eye indicator (e.g. `0` for left, `1` for right, or any project-specific scheme).
- `time` — float, in **years**, measured relative to a project-specific anchor (typically the subject's first visit or an earlier fixed date). Only the relative differences between visits of the same eye are used by the models, so the anchor choice does not affect training.
- Optional suffix `_reg` or `_anchor` is allowed and stripped during parsing.
- `eye_id` is derived as `f"{mrn}_{eye_laterality}"`.
- Eyes with fewer than **2** visits are dropped from training and evaluation.

Example:

```
30482_0_0.0_anchor.png      # eye 30482-left, baseline
30482_0_1.47_reg.png        # eye 30482-left, +1.47 years, registered
30482_1_0.0_anchor.png      # eye 30482-right, baseline
30482_1_2.11_reg.png        # eye 30482-right, +2.11 years, registered
```

## Images

- Grayscale PNG, any resolution (loaders resize to `--image_size`, default 256).
- Values in the native file range (uint8 `[0, 255]`); pipeline rescales to `[-1, 1]` internally.

## Optional per-image masks

A sibling mask file can be provided to restrict loss and metrics to valid pixels:

```
{base}_reg.png        → {base}_reg_mask.png
{base}_anchor.png     → {base}_anchor_mask.png
{base}.png            → {base}_mask.png
```

- Grayscale PNG; the loader thresholds the resized mask at `> 0.5` to produce a boolean mask.
- Any filename ending in `_mask.png` is excluded from the image list.
- If no mask is found, the loader falls back to an all-ones mask.

## Patient-level splitting and exclusion

The trainer splits eyes by **patient (MRN)** — all eyes from a given patient go
entirely to either the train or validation set. This prevents leakage
between a patient's two eyes.

An optional `--exclude_patients_tsv` flag removes entire patients before
splitting. The expected format is a tab-separated file with a header row
and a column named either `hashed_mrn` or `mrn`:

```
hashed_mrn
30482
31115
32206
...
```

Any patient whose ID matches a value in this file is dropped from both
train and validation. This is intended for e.g. removing Stargardt
patients from a mixed-etiology training set, or removing consented-out
patients after-the-fact.

## History construction (training)

For each eye with N sorted visits, the dataset generates N − 1 training
samples: target index `i ∈ {1, ..., N-1}` with history indices
`{0, 1, ..., i-1}`. History is left-padded to `--max_history` with
zero images and a padding-delta of 100.0; the corresponding positions
are flagged in a boolean `temporal_mask` so that `DeltaWeightedAttn`
can mask them out of the softmax.

## Evaluation tasks (benchmark)

`evaluate.py` builds **anchor tasks**: per eye, the last visit is the
target `I*` and all preceding visits form the history.
