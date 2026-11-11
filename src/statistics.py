import warnings

import numpy as np
from scipy.stats import ConstantInputWarning, spearmanr


def spearman(observed, predicted):
    x, y = np.asarray(observed, float), np.asarray(predicted, float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        return float(spearmanr(x, y).statistic)


def cluster_indices(patient_ids, rng, draws):
    ids = np.asarray(patient_ids).astype(str)
    patients = np.unique(ids)
    groups = [np.flatnonzero(ids == p) for p in patients]
    for _ in range(draws):
        yield np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])


def bootstrap(values, patient_ids, statistic, draws=2000, seed=42):
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    ids = np.asarray(patient_ids).astype(str)
    if len(values) != len(ids) or draws < 1:
        raise ValueError("Invalid bootstrap inputs.")
    valid = np.isfinite(values).all(axis=1)
    values, ids = values[valid], ids[valid]
    n_patients = len(np.unique(ids))
    result = dict(estimate=np.nan, ci_low=np.nan, ci_high=np.nan, p=np.nan,
                  n_eyes=len(values), n_patients=n_patients, valid_resamples=0,
                  requested_resamples=draws)
    if not len(values):
        return result
    result["estimate"] = float(statistic(values))
    if not np.isfinite(result["estimate"]) or n_patients < 2:
        return result
    samples = np.array([statistic(values[i]) for i in
                        cluster_indices(ids, np.random.default_rng(seed), draws)])
    samples = samples[np.isfinite(samples)]
    result["valid_resamples"] = len(samples)
    if len(samples):
        result["ci_low"], result["ci_high"] = np.percentile(samples, [2.5, 97.5])
        p = 2 * min(np.mean(samples <= 0), np.mean(samples >= 0))
        result["p"] = min(1.0, max(float(p), 1 / (len(samples) + 1)))
    return result


def correlation(observed, predicted, patient_ids, draws=2000, seed=42):
    return bootstrap(np.column_stack([observed, predicted]), patient_ids,
                     lambda v: spearman(v[:, 0], v[:, 1]), draws, seed)


def correlation_difference(observed, a, b, patient_ids, draws=2000, seed=42):
    return bootstrap(np.column_stack([observed, a, b]), patient_ids,
                     lambda v: spearman(v[:, 0], v[:, 1]) -
                     spearman(v[:, 0], v[:, 2]), draws, seed)


def mean_difference(a, b, patient_ids, draws=2000, seed=42):
    return bootstrap(np.column_stack([a, b]), patient_ids,
                     lambda v: np.mean(v[:, 0] - v[:, 1]), draws, seed)


def holm(pvalues):
    p = np.asarray(pvalues, dtype=float)
    result = np.full_like(p, np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    order = valid[np.argsort(p[valid], kind="stable")]
    if len(order):
        result[order] = np.minimum(1, np.maximum.accumulate(
            p[order] * np.arange(len(order), 0, -1)))
    return result
