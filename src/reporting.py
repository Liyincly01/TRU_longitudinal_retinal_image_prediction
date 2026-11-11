import numpy as np
import pandas as pd

from .statistics import bootstrap, correlation, correlation_difference, mean_difference, holm


IMAGE_METRICS = ("psnr", "delta_ssim", "ssim", "mae")
ENDPOINTS = {"area_change": ("observed_area_change", "predicted_area_change"),
             "distance_change": ("observed_distance_change", "predicted_distance_change")}
BASELINES = ("copy_last", "linear_spline", "TADM", "ImageFlowNet", "2D_BrLP",
             "trend_last_two", "trend_ols")


def summarize(frame, draws=2000, seed=42):
    rows = []
    for (cohort, method), group in frame.groupby(["cohort", "method"], sort=False):
        for metric in IMAGE_METRICS:
            if metric not in group:
                continue
            values = group[metric].to_numpy(float)
            valid = np.isfinite(values)
            stats = bootstrap(values, group.patient_id, np.mean, draws, seed)
            rows.append(dict(cohort=cohort, method=method, metric=metric, **stats,
                             sd=float(np.std(values[valid], ddof=1)) if valid.sum() > 1 else np.nan))
        for metric, (gt, pred) in ENDPOINTS.items():
            if gt in group and pred in group:
                stats = correlation(group[gt], group[pred], group.patient_id, draws, seed)
                rows.append(dict(cohort=cohort, method=method, metric=metric, **stats))
    return pd.DataFrame(rows)


def compare(frame, draws=2000, seed=42):
    rows = []
    for cohort, group in frame.groupby("cohort", sort=False):
        methods = set(group.method)
        contrasts = [("TRU", b, "benchmark") for b in BASELINES if b in methods]
        contrasts += [("IA_Nonlinear", "Std_DDIM_1step", "alignment"),
                      ("Std_DDIM_1step", "Std_DDIM_50step", "alignment"),
                      ("IA_Nonlinear", "TRU", "alignment"), ("IA_Linear", "TRU", "alignment")]
        for a, b, family in contrasts:
            if a not in methods or b not in methods:
                continue
            joined = group[group.method == a].merge(group[group.method == b],
                on=["cohort", "eye_id", "patient_id"], suffixes=("_a", "_b"), validate="one_to_one")
            for metric in IMAGE_METRICS:
                if metric + "_a" in joined:
                    result = mean_difference(joined[metric + "_a"], joined[metric + "_b"],
                                             joined.patient_id, draws, seed)
                    rows.append(dict(cohort=cohort, a=a, b=b, metric=metric, family=family, **result))
            for metric, (gt, pred) in ENDPOINTS.items():
                if gt + "_a" in joined:
                    if not np.allclose(joined[gt + "_a"], joined[gt + "_b"], equal_nan=True):
                        raise ValueError("Paired methods have different observed endpoints.")
                    result = correlation_difference(joined[gt + "_a"], joined[pred + "_a"],
                                                     joined[pred + "_b"], joined.patient_id, draws, seed)
                    rows.append(dict(cohort=cohort, a=a, b=b, metric=metric, family=family, **result))
    result = pd.DataFrame(rows)
    if len(result):
        result["p_holm"] = result.groupby(["cohort", "metric", "family"], sort=False)["p"].transform(holm)
    return result


def write_results(frame, output_dir, draws=2000, seed=42):
    frame.to_csv(output_dir / "per_eye.csv", index=False)
    summarize(frame, draws, seed).to_csv(output_dir / "summary.csv", index=False)
    compare(frame, draws, seed).to_csv(output_dir / "comparisons.csv", index=False)
