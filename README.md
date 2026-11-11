# Longitudinal retinal image prediction

Code for *Diagnostic-Guided Longitudinal Modeling for Forecasting Retinal Atrophy Progression*.

Temporal Retinal U-Net (TRU) predicts a future FAF image from recent registered visits and their time intervals. The history limit is configurable. The repository includes training, image and clinical endpoint evaluation, and diagnostic model-selection analyses.

## Installation

Install PyTorch and torchvision for your hardware, then:

```bash
pip install -r requirements.txt
```

Clinical evaluation with the DynUNet segmentor additionally requires `requirements-clinical.txt`. ImageFlowNet requires its upstream dependencies. CPU checks use Python 3.13, PyTorch 2.6.0, torchvision 0.21.0, NumPy 2.4.4, SciPy 1.16.3, pandas 2.2.3 and scikit-image 0.25.0. These are validation versions, not a record of the original training environment.

## Data

Supply registered, intensity-harmonized grayscale FAF images, validity masks and a patient-level visit manifest. See [data format](docs/data_format.md). Patient images, clinical records and cohort manifests are not distributed here. See [TRU weights](docs/weights.md) for the inference checkpoint.

The development cohort excludes diagnosed Stargardt disease. Keep the 10% patient hold-out separate; split the remaining patients 85/15 into training and validation. Transfer cohorts are evaluated without retraining. Supply existing study split assignments directly.

## Training

```bash
python train.py --method tru --manifest private/visits.csv --cohort optos \
  --max_history 10 --output_dir runs/tru
```

Defaults: 256×256 images, six history visits, 150 epochs, batch size 12, AdamW at 1e-4 with weight decay 1e-4, 1,000 warmup steps, cosine decay, gradient clipping at 1, EMA 0.999 and history dropout 0.03. Validation SSIM selects the checkpoint. Augmentations are shared across visits.

Set `--max_history N` to retain the N most recent visits. The command above uses ten, matching the shared checkpoint's training configuration.

| Training method | Input | Evaluation configurations |
| --- | --- | --- |
| `tru` | Clean latest history image | TRU, one deterministic pass |
| `std_ddim` | Noised target, v-prediction | 50-step Gaussian initialization; one-step history initialization |
| `ia_nonlinear` | Noised latest history image, nonlinear schedule | One-step IA-Nonlinear |
| `ia_linear` | `(1-t) × latest + t × noise`, `t ∈ [0.05, 0.95]` | One-step IA-Linear |

Train the other configurations by changing `--method`. Both Std-DDIM evaluations use the same checkpoint. The one-step nonlinear timestep defaults to 200; linear inference uses 0.20. Inference reads checkpoint configuration.

```bash
python external/train_brlp_2d.py --manifest private/visits.csv --cohort optos \
  --output_dir runs/brlp
```

The 2D-BrLP diffusion stage uses 150 epochs, batch size 12, learning rate 2e-4, 50 DDIM steps and five latent samples for validation and inference. Its VAE stage has separate settings exposed by `--help`.

## Evaluation

```bash
python evaluate.py --manifest private/visits.csv --cohort optos --split test \
  --ckpt_tru weights/tru_v1.0.pt --segmentor_weights weights/atrophy.pt \
  --fovea_lookup private/fovea.csv --save_probabilities --save_predictions \
  --output_dir outputs/optos
```

Each eye's final visit is the target. Copy-last and pixelwise linear extrapolation are included automatically. Outputs include `per_eye.csv`, `summary.csv`, `comparisons.csv` and input/checkpoint metadata. Without a segmentor, evaluation reports image metrics only. Missing endpoint measurements remain undefined.

Inference uses each TRU-family checkpoint's history limit by default (ten for `tru_v1.0.pt`). Supply `--max_history 6`, for example, to override it. Both the training and inference limits are recorded in `run.json`; per-eye outputs report the number used. The model API also accepts `MultiScaleTemporalUNet_v13(max_history=N)`.

Optional checkpoint flags are `--ckpt_std_ddim`, `--ckpt_ia_nonlinear`, `--ckpt_ia_linear`, `--ckpt_tadm`, `--ckpt_brlp` and `--ckpt_imageflownet`. TADM uses five DDIM steps. ImageFlowNet also requires `--ifn_repo` and `--ifn_time_multiplier`, fixed from training as `5 / maximum_training_interval_in_years`. Checkpoints must match the provided architectures. Predictions from other implementations can be supplied with `--predictions_csv`.

Manuscript comparator training uses 150 epochs and effective batch size 12, AdamW, weight decay 1e-4, 1,000-step warmup and cosine decay. ImageFlowNet uses learning rate 1e-4 and validation PSNR selection; TADM uses 2e-4 and validation SSIM selection. This repository supplies inference adapters for these two models; their FAF training pipelines are external.

See [analysis commands and definitions](docs/analysis.md) for the stochastic audit, measurement-error propagation, endpoint trends, manual-coordinate validation and figures. Reported paper values are not embedded in the analysis scripts.

## Validation and weights

```bash
python -m unittest discover -s tests -v
python tools/checkpoint.py --checkpoint weights/training_checkpoint.pt --method tru \
  --output outputs/tru_checkpoint.json --export_weights weights/tru.pt
```

The export retains EMA weights and inference configuration, omitting optimizer state and machine paths. Tests exercise synthetic data and statistical behavior; reproducing study results requires the exact study manifests, preprocessing, checkpoints and annotation assets.

## Citation

[Manuscript preprint](https://doi.org/10.48550/arXiv.2604.16955)

## License and comparators

Project code is covered by [MIT](LICENSE), except the TADM adaptation in `external/models_tadm.py`, which is covered by [GPLv3](external/LICENSE-TADM). Upstream components retain their respective licenses. Please cite the methods you use:

- [TADM](https://github.com/MattiaLitrico/TADM-Temporally-Aware-Diffusion-Model-for-Neurodegenerative-Progression-on-Brain-MRI), Litrico et al., MICCAI 2024.
- [ImageFlowNet](https://github.com/KrishnaswamyLab/ImageFlowNet), Liu et al., ICASSP 2025.
- BrLP, Puglisi et al., *Medical Image Analysis*, 2025; adapted here to 2D FAF.
