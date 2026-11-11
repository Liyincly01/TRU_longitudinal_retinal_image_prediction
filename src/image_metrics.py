import numpy as np
from scipy.ndimage import binary_erosion
from skimage.metrics import structural_similarity


def masked_ssim(a, b, mask, data_range):
    mask = np.asarray(mask, bool)
    if min(mask.shape) < 7:
        return np.nan
    centers = binary_erosion(mask, structure=np.ones((7, 7)), border_value=0)
    if not centers.any():
        return np.nan
    a, b = np.asarray(a, float), np.asarray(b, float)
    if not np.isfinite(a[mask]).all() or not np.isfinite(b[mask]).all():
        raise ValueError("Nonfinite valid image pixels.")
    _, similarity = structural_similarity(np.where(mask, a, 0), np.where(mask, b, 0),
                                          data_range=data_range, win_size=7, full=True)
    return float(similarity[centers].mean())


def image_metrics(target, prediction, last, mask):
    mask = np.asarray(mask, bool)
    if not mask.any():
        return dict(mae=np.nan, psnr=np.nan, ssim=np.nan, delta_ssim=np.nan)
    error = np.asarray(prediction)[mask] - np.asarray(target)[mask]
    if not np.isfinite(error).all():
        raise ValueError("Nonfinite image prediction.")
    mse = np.mean(error ** 2)
    return dict(mae=float(np.mean(np.abs(error))),
                psnr=float(-10 * np.log10(max(mse, 1e-10))),
                ssim=masked_ssim(target, prediction, mask, 1.0),
                delta_ssim=masked_ssim(target - last, prediction - last, mask, 2.0))
