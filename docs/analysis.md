# Analyses

Run each cohort separately. All coordinates and distances use the registered 256×256 frame. Use a new output directory for each evaluation.

## Endpoints and inference

Area change is `area(target) − area(last history)`, in pixels squared. Distance change is `distance(target) − distance(last history)`, in pixels. Predictions use the same baseline and the last-history foveal coordinate. Distance is zero inside atrophy and undefined when no atrophy is detected or the coordinate is invalid. Areas and distances use the intersection of target and last-history validity masks.

Image PSNR and MAE use valid pixels. SSIM uses 7×7 windows wholly contained in the valid region. Change-map SSIM compares signed observed and predicted changes with data range 2, after scaling images to [0,1].

Endpoint agreement uses Spearman correlation. Copy-last has zero change and undefined correlation. Confidence intervals and paired differences use 2,000 patient-cluster bootstrap resamples, retaining both eyes and repeated sampled clusters. Comparisons report paired eye counts and valid resample counts. Two-sided p-values have a finite-resampling floor of `1/(B_valid+1)`; they cannot establish `p < 1e-4` with 2,000 resamples. Holm adjustment is applied within cohort, metric and comparison family over estimable contrasts. Supply all planned baselines for a complete benchmark family.

## Stage 1: elapsed-time screen

```bash
python analysis/task_entropy.py --manifest private/visits.csv --cohort optos \
  --split train --output_dir outputs/screen
```

All consecutive training-visit pairs contribute. Changed pixels have absolute intensity difference greater than 0.05. The script reports the linear slope and intercept, Spearman correlation and clustered interval, and `frac_time = slope × median_interval / (intercept + slope × median_interval)`. This screen is descriptive; no numerical cutoff is imposed for “much less.”

## Coordinate validation

```bash
python analysis/fovea_validation.py --automatic private/fovea.csv \
  --manual private/manual_fovea.csv --index outputs/transfer/probability_index.csv \
  --output_dir outputs/manual
```

Outputs include paired localization residuals (`dx = manual_x − automatic_x`, likewise for y), median/IQR localization error, and automatic-versus-manual endpoint rankings on a common eligible set. Omit `--index` to compute localization errors only. Supply one vendor/cohort at a time.

## Measurement error

First cache development probabilities using `evaluate.py --split val --save_probabilities`, with the same segmentor and coordinate settings as the audit. Calibration does not require model predictions. Use empirical paired residuals applicable to the cohort; the choice of residual pool must be specified before analysis.

```bash
python analysis/measurement_error.py --mode calibrate \
  --index outputs/development/probability_index.csv \
  --residuals private/development_residuals.csv --output_dir outputs/calibration
```

Residual CSVs require finite `dx,dy` columns. Each of 1,000 draws samples one paired displacement and a threshold from `N(0.5, 0.02²)` for each eye. The same displacement and threshold are applied to its baseline, target and predictions. Probability maps are required; binary masks cannot support threshold jitter.

The observed distance-change SD gives each eye's measurement noise. Its development median defines the representative floor. The median Spearman correlation across 500 independent pairs of endpoint draws gives test–retest reliability; the gain margin is `delta = 1/sqrt(reliability) − 1`, with a patient-clustered bootstrap interval. Neither parameter is fitted to reported manuscript values.

Calibration accepts only development `train`/`val` eyes. The selection audit rejects overlapping calibration patients. Sensitivity analysis uses a fixed, unperturbed eligible eye set and reports correlation bands and top-rank frequencies across draws:

```bash
python analysis/measurement_error.py --mode sensitivity \
  --index outputs/audit/probability_index.csv \
  --residuals private/audit_residuals.csv --output_dir outputs/sensitivity
```

Eligibility requires finite baseline, target and all compared prediction distances. Invalid perturbations are reported, without redefining the eye set on each draw. Incomplete calibration cannot drive model selection. Sensitivity bands describe measurement-error propagation, not patient-bootstrap confidence intervals.

## Stage 2: stochastic audit

```bash
python analysis/posterior_collapse.py --manifest private/visits.csv --cohort optos \
  --split test --ckpt_tru weights/tru.pt --ckpt_ia_nonlinear weights/ia_nonlinear.pt \
  --segmentor_weights weights/atrophy.pt --fovea_lookup private/fovea.csv \
  --save_probabilities --k_samples 10 --output_dir outputs/audit

python analysis/selection_rule.py --per_eye outputs/audit/per_eye.csv \
  --calibration outputs/calibration/calibration.json \
  --measurement_noise outputs/sensitivity/measurement_noise_per_eye.csv \
  --screen outputs/screen/screen.json --output outputs/selection.json
```

For each eye, the prediction endpoint is the mean of ten sample endpoints. Diversity is their sample SD. Selection requires both median seed SD at least the development noise floor and an endpoint-correlation gain over TRU at least `delta`, with a positive lower 95% confidence bound. Missing or unestimable criteria stop the audit. Per-eye noise ratios are reported separately from the development-floor decision.

Outputs also include inter-sample SSIM, target-versus-last-history SSIM and pixel-variance diagnostics. Add `--ckpt_ia_linear` for the second aligned reference; run the selection script with `--reference IA_Linear`. The selection JSON and companion criterion CSV provide the Stage-2 decision and its supporting statistics.

## Endpoint trends and robustness

```bash
python analysis/endpoint_trends.py --history outputs/optos/history_measurements.csv \
  --per_eye outputs/optos/per_eye.csv --output_dir outputs/trends
python analysis/stratified_robustness.py --per_eye outputs/optos/per_eye.csv \
  --output_dir outputs/strata
```

Scalar trends use actual visit times: a last-two-visit slope and an OLS fit over the same recent history supplied to TRU. The history limit is read from the TRU evaluation rows. For older tables without this field, `--max_history N` sets the limit (default six). At least two valid history distances are required. Extrapolated values are not clipped. TRU and the trends are compared on the same eye set.

Robustness analyses report distance-change agreement for minimum total visit counts of 2, 3 and 4, and severity tertiles based on the first recorded visit's atrophy area. Cutpoints are computed once within the cohort's observed-endpoint-eligible set; ties enter the lower tertile.

## Figures and tables

```bash
python analysis/render_results.py --screen outputs/screen/pair_statistics.csv \
  --summary outputs/alignment/summary.csv --noise outputs/selection_noise.csv \
  --audit outputs/selection.json --strata outputs/strata/strata.csv \
  --prediction outputs/optos/predictions/000000.npz --output_dir outputs/figures
```

Each input is optional; `--noise` and `--audit` are supplied together. Figures are standalone PDFs. The summary input should contain the five alignment configurations from one cohort. Qualitative panels show the first seeded prediction, signed change maps and prediction-minus-target maps (±0.3). The elapsed-time plot groups intervals at 0.5 and 1.5 years for display.

Dataset counts support Table I; evaluation summaries and paired contrasts support Table II; the screen and selection outputs support Table III. Figure inputs cover elapsed-time sensitivity and alignment (Fig. 2A/C), endpoint diversity and gain (Fig. 2D), qualitative changes/errors (Figs. 3–4), and robustness strata (Fig. 5). Conceptual diagrams in Figs. 1 and 2B are not analysis outputs.
