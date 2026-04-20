"""
Metrics for longitudinal FAF prediction.

Image-level: masked PSNR / SSIM, hypo-AF Dice, cosine similarity to last
history frame, temporal consistency (MAE of change maps, copy ratio).
All metrics operate in the `[-1, 1]` → `[0, 1]` convention used by the
dataset: inputs are model outputs in `[-1, 1]`, masks are `{0, 1}` float.
"""

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim_func


def to01(arr):
    return np.clip((arr + 1.0) / 2.0, 0.0, 1.0)


def compute_ssim(target_np, pred_np):
    """Scalar SSIM between two [-1,1] numpy arrays (or [0,1] after to01)."""
    t = np.clip(to01(target_np), 0, 1)
    p = np.clip(to01(pred_np), 0, 1)
    try:
        return float(ssim_func(t, p, data_range=1.0))
    except Exception:
        return float("nan")


def calculate_masked_metrics(target, prediction, mask):
    """Masked PSNR and SSIM over a batch of (B, 1, H, W) tensors."""
    target = (target.detach().cpu().clamp(-1, 1) + 1) / 2
    prediction = (prediction.detach().cpu().clamp(-1, 1) + 1) / 2
    mask = mask.detach().cpu()

    psnr_vals, ssim_vals = [], []
    for i in range(target.shape[0]):
        t = target[i, 0].numpy()
        p = prediction[i, 0].numpy()
        m = mask[i, 0].numpy()

        valid_t = t[m > 0.5]
        valid_p = p[m > 0.5]
        if len(valid_t) > 100:
            mse = np.mean((valid_t - valid_p) ** 2)
            psnr = 10 * np.log10(1.0 / max(mse, 1e-10))
            psnr_vals.append(psnr)

        ys, xs = np.where(m > 0.5)
        if len(ys) > 100:
            y0, y1 = ys.min(), ys.max()
            x0, x1 = xs.min(), xs.max()
            if (y1 - y0) > 16 and (x1 - x0) > 16:
                try:
                    ssim_vals.append(
                        ssim_func(t[y0:y1+1, x0:x1+1],
                                  p[y0:y1+1, x0:x1+1], data_range=1.0)
                    )
                except Exception:
                    pass

    return (float(np.mean(psnr_vals)) if psnr_vals else 0.0,
            float(np.mean(ssim_vals)) if ssim_vals else 0.0)


def calculate_atrophy_dice(target, prediction, mask, threshold=0.25):
    """Dice overlap of hypoautofluorescent (dark) regions inside the mask."""
    target = (target.detach().cpu().clamp(-1, 1) + 1) / 2
    prediction = (prediction.detach().cpu().clamp(-1, 1) + 1) / 2
    mask = mask.detach().cpu()

    dice_scores = []
    for i in range(target.shape[0]):
        t = target[i, 0].numpy()
        p = prediction[i, 0].numpy()
        m = mask[i, 0].numpy() > 0.5

        t_dark = (t < threshold) & m
        p_dark = (p < threshold) & m
        intersection = np.sum(t_dark & p_dark)
        denom = np.sum(t_dark) + np.sum(p_dark)
        if denom > 0:
            dice_scores.append(2.0 * intersection / denom)

    return float(np.mean(dice_scores)) if dice_scores else 0.0


def calculate_copy_metrics(generated, history, temporal_mask):
    """Cosine similarity of generated frame to most-recent history frame."""
    B = generated.shape[0]
    similarities = []
    for b in range(B):
        valid = (~temporal_mask[b]).nonzero(as_tuple=True)[0]
        if len(valid) > 0:
            hist_frame = history[b, valid[-1]].flatten()
            gen_frame = generated[b, 0].flatten()
            cos = F.cosine_similarity(
                hist_frame.unsqueeze(0), gen_frame.unsqueeze(0)
            ).item()
            similarities.append(cos)
    return float(np.mean(similarities)) if similarities else 0.0


def calculate_temporal_consistency(prediction, target, history, temporal_mask):
    """MAE between predicted and true per-pixel change, plus near-zero-change ratio."""
    B = prediction.shape[0]
    pred_np = (prediction.detach().cpu().clamp(-1, 1) + 1) / 2
    target_np = (target.detach().cpu().clamp(-1, 1) + 1) / 2
    history_np = (history.detach().cpu().clamp(-1, 1) + 1) / 2

    maes, copy_ratios = [], []
    for b in range(B):
        valid = (~temporal_mask[b]).nonzero(as_tuple=True)[0]
        if len(valid) > 0:
            last_hist = history_np[b, valid[-1], 0].numpy()
            pred = pred_np[b, 0].numpy()
            tgt = target_np[b, 0].numpy()

            pred_change = np.abs(pred - last_hist)
            true_change = np.abs(tgt - last_hist)

            maes.append(np.mean(np.abs(pred_change - true_change)))
            copy_ratios.append(np.mean(pred_change < 0.01))

    return (float(np.mean(maes)) if maes else 0.0,
            float(np.mean(copy_ratios)) if copy_ratios else 0.0)
