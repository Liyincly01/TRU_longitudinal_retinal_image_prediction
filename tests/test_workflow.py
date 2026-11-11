import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from analysis.endpoint_trends import analyze as trends
from analysis.fovea_validation import endpoint_remeasurement, localization_errors
from analysis.measurement_error import analyze as measurement_error
from analysis.render_results import plot_alignment, plot_images, plot_posterior, plot_screen, plot_strata
from analysis.selection_rule import audit, compare_noise, criterion_table
from analysis.stratified_robustness import analyze as stratify
from analysis.task_entropy import analyze as screen
from src.backbone import MaskedLongitudinalDataset
from src.data import read_manifest, sequences, training_eyes, write_json
from src.evaluation import evaluate
from src.predictors import linear_spline, tensors
from src.reporting import write_results
from tools.prepare_slo_manifest import registration_mask


class WorkflowTests(unittest.TestCase):
    def test_slo_mask_excludes_border_padding_only(self):
        image = np.ones((32, 32), np.uint8)
        image[:3] = 0
        image[16, 16] = 0
        mask = registration_mask(image)
        self.assertFalse(mask[:4].any())
        self.assertTrue(mask[16, 16])
        self.assertTrue(mask[4:].all())

    def test_manifest_prevents_patient_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "image.png"
            Image.fromarray(np.zeros((8, 8), np.uint8)).save(image)
            rows = [dict(patient_id="same", eye_id=str(i), time=0, image=image, mask=image,
                         split=split, cohort="cohort") for i, split in enumerate(["train", "test"])]
            manifest = root / "visits.csv"
            pd.DataFrame(rows).to_csv(manifest, index=False)
            with self.assertRaisesRegex(ValueError, "multiple splits"):
                read_manifest(manifest)

    def test_configurable_recent_visits(self):
        history = np.stack([np.full((8, 8), i / 10, np.float32) for i in range(8)])
        for limit in (1, 3, 6, 10):
            h, delta, padded = tensors(history, np.arange(8), 10, limit, "cpu")
            valid = ~padded[0]
            start = max(0, 8-limit)
            np.testing.assert_allclose(h[0, valid, 0, 0, 0], np.arange(start, 8) / 5 - 1, atol=1e-7)
            np.testing.assert_allclose(delta[0, valid], 10-np.arange(start, 8))
            self.assertEqual(int(padded.sum()), max(0, limit-8))
        with self.assertRaises(ValueError):
            tensors(history, np.arange(8), 10, 0, "cpu")

    def test_endpoint_trends_follow_evaluation_history_limit(self):
        history = pd.DataFrame(dict(cohort="cohort", eye_id="eye", time=[0., 1., 2., 3.],
                                    distance=[30., 15., 9., 8.]))
        frame = pd.DataFrame([dict(cohort="cohort", eye_id="eye", patient_id="patient", method="TRU",
                                   observed_distance_change=-2., predicted_distance_change=-1.,
                                   last_distance=8., target_time=4., n_visits=5, max_history=2)])
        result = trends(history, frame)
        np.testing.assert_allclose(result.loc[result.method != "TRU", "predicted_distance_change"], [-1., -1.])
        with self.assertRaisesRegex(ValueError, "must match"):
            trends(history, frame, max_history=4)

    def test_evaluation_to_analysis_outputs(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            y, x = np.ogrid[:256, :256]
            valid = ((x - 128)**2 + (y - 128)**2 < 120**2).astype(np.uint8) * 255
            mask = root / "valid.png"
            Image.fromarray(valid).save(mask)
            rows, coordinates = [], []
            for split in ["train", "val", "test"]:
                for i in range(6):
                    eye = f"{split}_{i}"
                    times = np.array([0., .8 + .1 * i, 2. + .2 * i, 3.5 + .3 * i])
                    for j, time in enumerate(times):
                        radius = 14 + 2 * i + time * (.8 + .2 * i)
                        probability = 1 / (1 + np.exp(np.clip(np.hypot(x-120, y-128) - radius, -80, 80)))
                        image = root / f"{eye}_visit{j}.png"
                        Image.fromarray(np.rint((1-probability)*255).astype(np.uint8)).save(image)
                        rows.append(dict(patient_id=eye, eye_id=eye, time=time, image=image,
                                         mask=mask, split=split, cohort="synthetic"))
                        coordinates.append(dict(faf_file=image.name, fovea_x_256=30.,
                                                fovea_y_256=128., in_faf=1,
                                                manual_x_256=30.2, manual_y_256=128.1))
            manifest = root / "visits.csv"
            pd.DataFrame(rows).to_csv(manifest, index=False)
            fovea = pd.DataFrame(coordinates)
            fovea_csv = root / "fovea.csv"
            fovea.to_csv(fovea_csv, index=False)
            train_patients, val_patients, train_eyes, val_eyes = training_eyes(manifest)
            self.assertFalse(train_patients & val_patients)
            dataset = MaskedLongitudinalDataset(manifest=manifest, eye_ids=train_eyes,
                                                use_augmentation=False)
            self.assertEqual(len(dataset), 18)
            self.assertEqual(tuple(dataset[0]["history"].shape), (6, 1, 256, 256))

            def circular_forecast(history, times, target, seed):
                radii = np.sqrt(((1-history[-2:]) > .5).sum(axis=(1, 2)) / np.pi)
                radius = radii[-1] + (target-times[-1]) * np.diff(radii)[0] / np.diff(times[-2:])[0]
                return (1 - 1 / (1 + np.exp(np.clip(np.hypot(x-120, y-128) - radius, -80, 80)))).astype(np.float32)

            def stochastic(history, times, target, seed):
                prediction = circular_forecast(history, times, target, seed)
                return np.clip(prediction + np.random.default_rng(seed).normal(0, .01), 0, 1)

            models = {"copy_last": lambda h, t, target, seed: h[-1], "TRU": circular_forecast,
                      "IA_Nonlinear": stochastic}
            segmentor = lambda images: 1 - images.astype(np.float32) / 255
            outputs = {}
            for split in ["val", "test"]:
                output = root / split
                frame = evaluate(sequences(manifest=manifest, split=split), models, output,
                                 segmentor, fovea_csv, cohort="synthetic",
                                 save_probabilities=True, save_predictions=True, k_samples=10)
                write_results(frame, output, draws=30)
                outputs[split] = frame
                copy = frame[frame.method == "copy_last"]
                np.testing.assert_array_equal(copy.predicted_distance_change, np.zeros(6))
                np.testing.assert_array_equal(copy.predicted_area_change, np.zeros(6))
                with np.load(output / "probabilities/000000.npz") as cache:
                    from src.clinical_metrics import measurements
                    _, d = measurements(cache["method_2"], cache["valid_mask"], cache["fovea"])
                    row = frame[(frame.eye_id == split + "_0") & (frame.method == "IA_Nonlinear")].iloc[0]
                    self.assertAlmostEqual(row.predicted_distance_change, d.mean() - row.last_distance)

            residuals = root / "residuals.csv"
            pd.DataFrame(dict(dx=[-.4, .2, .6], dy=[.3, -.2, .1])).to_csv(residuals, index=False)
            calibration = measurement_error(root / "val/probability_index.csv", residuals,
                                            root / "calibration", "calibrate", draws=10, bootstrap=20)
            self.assertTrue(calibration["complete"])
            sensitivity = measurement_error(root / "test/probability_index.csv", residuals,
                                            root / "sensitivity", "sensitivity", draws=10, bootstrap=20)
            self.assertEqual(sensitivity["n_eligible_eyes"], 6)
            result = audit(outputs["test"], calibration, draws=30)
            write_json(root / "audit.json", result)
            self.assertEqual(len(criterion_table(result)), 3)
            noise = pd.read_csv(root / "sensitivity/measurement_noise_per_eye.csv")
            paired_noise = compare_noise(outputs["test"], noise, "IA_Nonlinear")
            paired_noise.to_csv(root / "noise.csv", index=False)
            history = pd.read_csv(root / "test/history_measurements.csv")
            trend = trends(history, outputs["test"])
            self.assertEqual(set(trend.method), {"TRU", "trend_last_two", "trend_ols"})
            self.assertTrue((trend.groupby("method").size() == 6).all())
            strata, metadata = stratify(outputs["test"], draws=30)
            self.assertEqual(metadata["baseline"], "first_recorded_visit")
            strata.to_csv(root / "strata.csv", index=False)
            remeasured = endpoint_remeasurement(root / "test/probability_index.csv", fovea)
            self.assertEqual(set(remeasured.coordinate), {"automatic", "manual"})
            auto = fovea.drop(columns=["manual_x_256", "manual_y_256"])
            manual = fovea[["faf_file", "manual_x_256", "manual_y_256"]]
            self.assertEqual(len(localization_errors(auto, manual)), len(fovea))
            screening = screen(sequences(manifest=manifest, split="train"), root / "screen", draws=30)
            self.assertEqual(screening["n_pairs"], 18)
            plot_strata(root / "strata.csv", root)
            plot_images(root / "test/predictions/000000.npz", root)
            plot_screen(root / "screen/pair_statistics.csv", root)
            plot_alignment(root / "test/summary.csv", root)
            plot_posterior(root / "noise.csv", root / "audit.json", root)
            self.assertEqual(len(list(root.glob("*.pdf"))), 6)


if __name__ == "__main__":
    unittest.main()
