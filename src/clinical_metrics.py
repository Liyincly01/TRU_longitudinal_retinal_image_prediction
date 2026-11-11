import importlib

import numpy as np
import pandas as pd
from scipy import ndimage


def load_fovea_lookup(path):
    if path is None:
        return {}
    frame = pd.read_csv(path, dtype={"faf_file": str})
    required = {"faf_file", "fovea_x_256", "fovea_y_256"}
    if not required <= set(frame):
        raise ValueError("Fovea CSV requires " + ", ".join(sorted(required)))
    if frame.faf_file.duplicated().any():
        raise ValueError("Duplicate image keys in fovea CSV.")
    lookup = {}
    for row in frame.to_dict("records"):
        flag = str(row.get("in_faf", "1")).lower()
        if flag not in {"1", "1.0", "true"}:
            continue
        xy = np.array([row["fovea_x_256"], row["fovea_y_256"]], float)
        if np.isfinite(xy).all() and ((xy >= 0) & (xy < 256)).all():
            lookup[str(row["faf_file"])] = xy
    return lookup


def binary_mask(probability, valid_mask, threshold=0.5, border_px=0):
    probability = np.asarray(probability, float)
    valid = np.asarray(valid_mask, bool)
    if probability.shape != valid.shape:
        raise ValueError("Probability and validity masks have different shapes.")
    mask = (probability > threshold) & valid
    if border_px:
        labels, _ = ndimage.label(mask, structure=np.ones((3, 3)))
        border_labels = np.unique(np.concatenate([
            labels[:border_px].ravel(), labels[-border_px:].ravel()]))
        mask &= ~np.isin(labels, border_labels[border_labels != 0])
    return mask


def distance_to_atrophy(mask, fovea):
    mask = np.asarray(mask, bool)
    xy = np.asarray(fovea, float)
    if xy.shape != (2,) or not np.isfinite(xy).all() or not mask.any():
        return np.nan
    x, y = xy
    if not (0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]):
        return np.nan
    ix, iy = min(int(x + 0.5), mask.shape[1] - 1), min(int(y + 0.5), mask.shape[0] - 1)
    if mask[iy, ix]:
        return 0.0
    ys, xs = np.where(mask)
    return float(np.hypot(xs - x, ys - y).min())


def measurements(probabilities, valid_mask, fovea, threshold=0.5, border_px=0):
    masks = [binary_mask(p, valid_mask, threshold, border_px) for p in probabilities]
    areas = np.array([m.sum() for m in masks], float)
    distances = np.array([distance_to_atrophy(m, fovea) for m in masks])
    if not np.any(valid_mask):
        areas[:] = np.nan
    xy = np.asarray(fovea, float)
    if xy.shape == (2,) and np.isfinite(xy).all():
        height, width = np.shape(valid_mask)
        x, y = np.clip(np.floor(xy + .5).astype(int), [0, 0], [width - 1, height - 1])
        if not (0 <= xy[0] < width and 0 <= xy[1] < height and valid_mask[y, x]):
            distances[:] = np.nan
    return areas, distances


def validate_probabilities(result, shape):
    result = np.asarray(result, np.float32)
    if result.shape != shape:
        raise ValueError(f"Segmentor returned {result.shape}; expected {shape}.")
    if not np.isfinite(result).all() or result.min() < 0 or result.max() > 1:
        raise ValueError("Segmentor must return finite probabilities in [0, 1].")
    return result


def load_segmentor(weights=None, module=None, device="cpu"):
    if weights and module:
        raise ValueError("Choose segmentor weights or a segmentor callable.")
    if module:
        name, function = module.split(":", 1)
        fn = getattr(importlib.import_module(name), function)
        return lambda images: validate_probabilities(fn(images), images.shape)
    if weights is None:
        return None
    import torch
    from monai.networks.nets import DynUNet

    model = torch.nn.Sequential(DynUNet(
        spatial_dims=2, in_channels=1, out_channels=1,
        kernel_size=[5] * 4, filters=[16, 32, 64, 128],
        strides=[1] * 4, upsample_kernel_size=[1] * 3),
        torch.nn.Sigmoid()).to(device)
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    model.eval()

    def segment(images):
        normalized = []
        for image in images:
            image = image.astype(np.float32)
            lo, hi = np.percentile(image, [0.5, 99.5])
            image = np.clip(image, lo, hi)
            image = (image - image.mean()) / max(2 * image.std(), 1e-8)
            normalized.append(np.clip(image, -1, 1))
        with torch.inference_mode():
            tensor = torch.from_numpy(np.stack(normalized)[:, None]).to(device=device, dtype=torch.float32)
            result = model(tensor).squeeze(1).cpu().numpy()
        return validate_probabilities(result, images.shape)
    return segment
