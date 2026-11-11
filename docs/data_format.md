# Data format

## Visits

A CSV manifest contains one row per visit:

```csv
patient_id,eye_id,time,image,mask,split,cohort
example01,example01_left,0.0,images/baseline.png,masks/baseline.png,train,optos
example01,example01_left,1.2,images/followup.png,masks/followup.png,train,optos
```

Identifiers are strings. Times are in years and must increase strictly within an eye. Paths are absolute or relative to the manifest. Both eyes of each patient must share a split. Use `train`, `val`, `test` or `transfer`; training requires nonempty `train` and `val` sets. Select one cohort per run. Eyes require at least two visits.

Images must already share a registered analysis frame and consistent intensity harmonization. Supply grayscale uint8 PNG images and binary validity masks (0/255); loaders resize images bilinearly and masks by nearest neighbor to 256×256. Inference outputs use [0,1]; model tensors use [-1,1]. Evaluation does not register images or histogram-match predictions to their target.

For directory input, filenames follow `patient_laterality_time[_suffix].png`, with numeric patient and laterality identifiers. For example, `1_0_1.25_reg.png` requires `1_0_1.25_reg_mask.png`. A manifest supports arbitrary image names.

To index already assigned directories:

```bash
python tools/prepare_manifest.py \
  --source optos:train=PATH_TRAIN --source optos:val=PATH_VAL \
  --source optos:test=PATH_TEST --source optos_stargardt:transfer=PATH_TRANSFER \
  --output private/visits.csv
```

`--exclude_patients_tsv` accepts a `patient_id`, `hashed_mrn` or `mrn` column. Apply the manuscript's diagnosis exclusions before development splitting. For a new cohort, `--source optos:unsplit=PATH --split_development optos` makes a 90/10 patient split (seed 1337), then an 85/15 development split (seed 42). This creates new assignments; it does not identify the original study split. A companion counts CSV reports patients, longitudinal eyes and images.

Training creates one task for each visit after the first. Evaluation predicts the final visit. Both retain the most recent prior visits up to `--max_history`: training defaults to six, and TRU-family inference defaults to the checkpoint's saved limit unless overridden. Earlier visits remain available for first-visit severity and the consecutive-pair screen. Directory-mode training assumes the supplied directory contains development patients only and creates its own validation split.

## Foveal coordinates

```csv
faf_file,fovea_x_256,fovea_y_256,in_faf
baseline.png,126.2,129.7,1
```

Coordinates are `(x, y)` pixels in the registered 256×256 frame, after any orientation changes. `faf_file` must uniquely identify an image basename. `in_faf=0`, nonfinite coordinates, out-of-frame coordinates and coordinates outside valid overlap are ineligible for distance analysis. The last-history coordinate is shared by baseline, target and all predictions. Manual tables use `faf_file,manual_x_256,manual_y_256` in the same frame.

## Atrophy segmentation

`--segmentor_weights` loads the FAF DynUNet followed by sigmoid: four levels with filters 16/32/64/128, 5×5 kernels and unit strides. Per-image normalization clips at the 0.5/99.5 percentiles, centers, divides by twice the standard deviation and clips to [-1,1]. Weights must match this architecture.

Alternatively, `--seg_module package:function` supplies a callable accepting an `N×256×256` uint8 array and returning finite probability maps of the same shape in [0,1]. Clinical endpoints use probabilities greater than 0.5 within valid overlap. No minimum lesion size or prediction-horizon filter is applied. `--seg_border_px` is an explicit, optional top/bottom-border component exclusion; its default is zero.

## External predictions

```csv
eye_id,method,path
example01_left,MyModel,predictions/example01_left.npy
```

Provide exactly one prediction for every evaluated eye and method. NPY arrays must be 256×256 floats in [0,1]; PNG predictions are uint8. Paths are relative to this CSV or absolute. Multi-sample endpoint audits use the model sampling interface.

Probability caches contain raw segmentation probabilities, the common validity mask, foveal coordinates and per-sample predictions. Their index records cohort, split, patient and eye IDs. These caches and all per-eye outputs are private analysis artifacts.

## Optional SLO evaluation

`tools/prepare_slo_manifest.py --data_dir PATH --patient_list private/patients.txt --output private/slo.csv` preserves image-only SLO evaluation. It accepts `patient_eye_time_suffix.png` filenames and derives validity masks from border-connected zero padding. Evaluate this manifest with `--split transfer`, without a FAF segmentor. SLO results are separate from the manuscript's FAF cohorts.
