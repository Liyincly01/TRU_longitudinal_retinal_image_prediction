from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd

from .clinical_metrics import measurements, load_fovea_lookup
from .data import load_image, load_mask, checksum, write_json
from .image_metrics import image_metrics, masked_ssim
from .predictors import eye_seed


def prediction_sources(path):
    if not path:
        return {}
    frame = pd.read_csv(path, dtype={"eye_id": str, "method": str})
    if not {"eye_id", "method", "path"} <= set(frame):
        raise ValueError("Prediction CSV requires eye_id, method, path.")
    if frame.duplicated(["eye_id", "method"]).any():
        raise ValueError("Duplicate external predictions.")
    result = {}
    for row in frame.to_dict("records"):
        result.setdefault(row["method"], {})[row["eye_id"]] = str(
            (Path(path).resolve().parent / row["path"]).resolve())
    return result


def load_prediction(path):
    if str(path).endswith(".npy"):
        image = np.load(path, allow_pickle=False)
    else:
        image = load_image(path)
    image = np.asarray(image, np.float32)
    if image.shape != (256, 256) or not np.isfinite(image).all():
        raise ValueError("Predictions must be finite 256x256 images.")
    if image.min() < 0 or image.max() > 1:
        raise ValueError("Predictions must be in [0, 1].")
    return image


def evaluate(seqs, models, output_dir, segmentor=None, fovea_csv=None,
             external_predictions=None, cohort="cohort", seed=42,
             save_probabilities=False, save_predictions=False,
             border_px=0, k_samples=1):
    if k_samples < 1:
        raise ValueError("k_samples must be positive.")
    output_dir = Path(output_dir)
    if (output_dir / "per_eye.csv").exists():
        raise FileExistsError("Output already contains per_eye.csv; choose a new directory.")
    output_dir.mkdir(parents=True, exist_ok=True)
    lookup = load_fovea_lookup(fovea_csv)
    external_predictions = external_predictions or {}
    overlap = set(models) & set(external_predictions)
    if overlap:
        raise ValueError(f"Duplicate prediction methods: {sorted(overlap)}")
    for method, entries in external_predictions.items():
        if set(entries) != set(seqs):
            raise ValueError(f"{method} prediction eye set does not match evaluation.")
    if save_probabilities and segmentor is None:
        raise ValueError("Saving probabilities requires a segmentor.")
    rows, visits_rows, index = [], [], []
    method_names = list(models) + list(external_predictions)
    for number, (eye, visits) in enumerate(seqs.items()):
        history = np.stack([load_image(v["image"]) for v in visits[:-1]])
        target = load_image(visits[-1]["image"])
        times = np.array([v["time"] for v in visits[:-1]], float)
        target_time = visits[-1]["time"]
        mask = load_mask(visits[-1]["mask"]) & load_mask(visits[-2]["mask"])
        fovea = lookup.get(Path(visits[-2]["image"]).name, np.array([np.nan, np.nan]))
        if np.isfinite(fovea).all():
            x, y = np.clip(np.floor(fovea + .5).astype(int), 0, 255)
            if not mask[y, x]:
                fovea = np.array([np.nan, np.nan])
        patient = visits[0]["patient_id"]
        prefix = f"{number:06d}"
        predictions = {}
        for method, predictor in models.items():
            count = k_samples if method in ("IA_Nonlinear", "IA_Linear") else 1
            images = [predictor(history, times, target_time, eye_seed(seed + k, eye))
                      for k in range(count)]
            image = np.asarray(images, np.float32)
            if image.shape != (count, 256, 256) or not np.isfinite(image).all():
                raise ValueError(f"Invalid {method} predictions.")
            if image.min() < 0 or image.max() > 1:
                raise ValueError(f"{method} predictions are outside [0, 1].")
            predictions[method] = image
        for method, entries in external_predictions.items():
            predictions[method] = load_prediction(entries[eye])[None]
        history_probs = target_probs = None
        prediction_probs = {}
        if segmentor:
            history_probs = segmentor(np.rint(history * 255).astype(np.uint8))
            target_probs = segmentor(np.rint(target[None] * 255).astype(np.uint8))[0]
            gt_areas, gt_distances = measurements(
                np.stack([history_probs[-1], target_probs]), mask, fovea, border_px=border_px)
            for i, visit in enumerate(visits[:-1]):
                visit_mask = load_mask(visit["mask"])
                area, _ = measurements(history_probs[i:i+1], visit_mask, fovea, border_px=border_px)
                _, distance = measurements(history_probs[i:i+1], visit_mask & mask, fovea,
                                            border_px=border_px)
                visits_rows.append(dict(cohort=cohort, eye_id=eye, patient_id=patient,
                                        visit_index=i, time=visit["time"], area=area[0],
                                        distance=distance[0], n_history=len(history)))
            first_area = visits_rows[-len(history)]["area"]
            for method, images in predictions.items():
                prediction_probs[method] = segmentor(np.rint(images * 255).astype(np.uint8))
        for method, images in predictions.items():
            metrics = [image_metrics(target, image, history[-1], mask) for image in images]
            row = dict(cohort=cohort, eye_id=eye, patient_id=patient, method=method,
                       n_visits=len(visits), n_history=len(history),
                       target_time=target_time, last_time=times[-1],
                       delta_t_years=target_time - times[-1], n_samples=len(images))
            history_limit = getattr(models.get(method), "max_history", None)
            if history_limit is not None:
                row.update(max_history=history_limit, n_history_used=min(len(history), history_limit))
            for metric in metrics[0]:
                row[metric] = float(np.mean([m[metric] for m in metrics]))
            if len(images) > 1 and mask.any():
                variance = float(images.var(axis=0)[mask].mean())
                mean_error = float(((images.mean(axis=0) - target)[mask] ** 2).mean())
                row.update(mean_pixel_variance=variance, mean_prediction_mse=mean_error,
                           variance_fraction=variance / max(variance + mean_error, 1e-12),
                           intersample_ssim=float(np.mean([masked_ssim(images[i], images[j], mask, 1.)
                               for i, j in combinations(range(len(images)), 2)])),
                           intervisit_ssim=masked_ssim(target, history[-1], mask, 1.))
            if segmentor:
                areas, distances = measurements(prediction_probs[method], mask, fovea,
                                                border_px=border_px)
                row.update(first_visit_area=first_area, last_area=gt_areas[0],
                           target_area=gt_areas[1], last_distance=gt_distances[0],
                           target_distance=gt_distances[1],
                           observed_area_change=gt_areas[1] - gt_areas[0],
                           predicted_area_change=float(areas.mean() - gt_areas[0]),
                           observed_distance_change=gt_distances[1] - gt_distances[0],
                           predicted_distance_change=float(distances.mean() - gt_distances[0]),
                           seed_sd_distance=float(np.std(distances, ddof=1)) if len(distances) > 1 else np.nan,
                           fovea_x=fovea[0], fovea_y=fovea[1])
            rows.append(row)
        if save_probabilities:
            folder = output_dir / "probabilities"
            folder.mkdir(exist_ok=True)
            payload = dict(history=history_probs, target=target_probs,
                           valid_mask=mask, fovea=fovea, history_times=times,
                           target_time=target_time, methods=np.array(method_names),
                           border_px=border_px, schema_version=1)
            for i, method in enumerate(method_names):
                payload[f"method_{i}"] = prediction_probs[method]
            filename = folder / (prefix + ".npz")
            np.savez_compressed(filename, **payload)
            index.append(dict(cohort=cohort, eye_id=eye, patient_id=patient,
                              split=visits[0]["split"], fovea_image=Path(visits[-2]["image"]).name,
                              path=str(filename.relative_to(output_dir))))
        if save_predictions:
            folder = output_dir / "predictions"
            folder.mkdir(exist_ok=True)
            payload = dict(history=history, history_times=times, target=target,
                           target_time=target_time, valid_mask=mask, methods=np.array(method_names))
            for i, method in enumerate(method_names):
                payload[f"method_{i}"] = predictions[method]
            np.savez_compressed(folder / (prefix + ".npz"), **payload)
        print(f"{number + 1}/{len(seqs)} eyes", flush=True)
    if visits_rows:
        pd.DataFrame(visits_rows).to_csv(output_dir / "history_measurements.csv", index=False)
    if index:
        pd.DataFrame(index).to_csv(output_dir / "probability_index.csv", index=False)
    return pd.DataFrame(rows)


def run_metadata(args, seqs):
    assets = {}
    for key, value in vars(args).items():
        if value and (key.startswith("ckpt_") or key in ("manifest", "fovea_lookup", "segmentor_weights")):
            assets[key] = {"sha256": checksum(value), "filename": Path(value).name}
    return dict(arguments=vars(args), assets=assets, n_eyes=len(seqs),
                n_patients=len({v[0]["patient_id"] for v in seqs.values()}),
                n_images=sum(len(v) for v in seqs.values()),
                endpoint_fovea="last_history", severity_baseline="first_visit",
                stochastic_endpoint_aggregation="mean_of_sample_endpoints")
