import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.clinical_metrics import distance_to_atrophy, measurements
from src.data import parse_filename, read_manifest
from src.image_metrics import image_metrics
from src.statistics import cluster_indices, correlation_difference, holm
from analysis.task_entropy import build_pairs
from analysis.endpoint_trends import extrapolate
from analysis.measurement_error import paired_reliability, perturb_eye
from analysis.selection_rule import audit


class ScientificMethodsTests(unittest.TestCase):
    def test_absolute_changes_and_subpixel_inside(self):
        masks = np.zeros((3, 32, 32))
        masks[0, 10:20, 10:20] = 1
        masks[1, 8:20, 8:20] = 1
        masks[2] = masks[0]
        area, distance = measurements(masks, np.ones((32, 32), bool), [4, 14])
        np.testing.assert_equal(area - area[0], [0, 44, 0])
        np.testing.assert_allclose(distance - distance[0], [0, -2, 0])
        self.assertEqual(distance_to_atrophy(masks[0], [15.2, 15.3]), 0)
        self.assertTrue(np.isnan(distance_to_atrophy(np.zeros((32, 32)), [4, 14])))

    def test_ssim_ignores_invalid_pixels(self):
        y, x = np.ogrid[:64, :64]
        mask = (x-32)**2 + (y-32)**2 < 25**2
        target = np.zeros((64, 64))
        predicted = target.copy()
        before = image_metrics(target, predicted, target, mask)
        predicted[~mask] = 1
        after = image_metrics(target, predicted, target, mask)
        self.assertEqual(before, after)
        self.assertEqual(after["delta_ssim"], 1)

    def test_patient_clusters_and_holm(self):
        ids = np.array(["a", "a", "b", "b", "c"])
        for rows in cluster_indices(ids, np.random.default_rng(42), 100):
            self.assertEqual(np.sum(rows == 0), np.sum(rows == 1))
            self.assertEqual(np.sum(rows == 2), np.sum(rows == 3))
        np.testing.assert_allclose(holm([.01, .04, .03, np.nan]),
                                   [.03, .06, .06, np.nan], equal_nan=True)
        x = np.arange(10)
        result = correlation_difference(x, x, -x, np.repeat(np.arange(5), 2), draws=100)
        self.assertAlmostEqual(result["estimate"], 2)
        self.assertAlmostEqual(result["ci_low"], 2)
        self.assertAlmostEqual(result["p"], 1/101)

    def test_filename_and_consecutive_pairs(self):
        self.assertEqual(parse_filename("1_0_1.25.png"), ("1", "1_0", 1.25))
        self.assertEqual(parse_filename("1_0_1.25_AF_1024_reg.png"), ("1", "1_0", 1.25))
        seq = {"eye": [{"time": 0}, {"time": 1}, {"time": 4}]}
        self.assertEqual(len(build_pairs(seq)), 2)

    def test_irregular_scalar_trends(self):
        result = extrapolate([0, 1, 4], [10, 8, 2], 5)
        self.assertAlmostEqual(result["trend_last_two"], -2)
        self.assertAlmostEqual(result["trend_ols"], -2)
        self.assertIsNone(extrapolate([0], [5], 1))

    def test_reliability_margin_inputs(self):
        draw = np.array([[1, 3, 2, 4], [1, 3, 2, 4], [4, 3, 2, 1], [4, 3, 2, 1]])
        np.testing.assert_allclose(paired_reliability(draw), [1, 1])

    def test_shared_perturbations_keep_copy_last_zero(self):
        y, x = np.ogrid[:32, :32]
        base = 1 / (1 + np.exp(np.hypot(x-18, y-16) - 4))
        target = 1 / (1 + np.exp(np.hypot(x-19, y-16) - 6))
        cache = dict(methods=np.array(["copy_last"]), history=base[None], target=target,
                     valid_mask=np.ones((32, 32), bool), fovea=np.array([4., 16.]),
                     border_px=0, method_0=base[None])
        observed, predicted, area = perturb_eye(cache, np.array([[0., 0.], [1., .3]]),
                                                20, np.random.default_rng(1), .02)
        np.testing.assert_array_equal(predicted, np.zeros((20, 1)))
        self.assertEqual(len(observed), 20)

    def test_calibration_does_not_require_valid_model_predictions(self):
        y, x = np.ogrid[:32, :32]
        p = 1 / (1 + np.exp(np.hypot(x-18, y-16) - 4))
        cache = dict(methods=np.array(["bad"]), history=p[None], target=p,
                     valid_mask=np.ones((32, 32), bool), fovea=np.array([4., 16.]),
                     border_px=0, method_0=np.zeros((1, 32, 32)))
        result = perturb_eye(cache, np.array([[0., 0.]]), 10,
                             np.random.default_rng(2), .02, include_predictions=False)
        self.assertIsNotNone(result)

    def test_selection_requires_both_criteria_and_disjoint_patients(self):
        rows = []
        for i in range(10):
            for method in ["TRU", "IA_Nonlinear"]:
                rows.append(dict(cohort="holdout", eye_id=str(i), patient_id=str(i//2),
                                 method=method, observed_distance_change=i,
                                 predicted_distance_change=-i if method == "TRU" else i,
                                 seed_sd_distance=6., n_samples=10 if method != "TRU" else 1))
        frame = pd.DataFrame(rows)
        calibration = dict(complete=True, patient_ids=["calibration"], sigma_meas=5., delta=.064)
        self.assertEqual(audit(frame, calibration, draws=100)["selected_class"], "stochastic")
        frame.loc[frame.method != "TRU", "seed_sd_distance"] = .1
        self.assertEqual(audit(frame, calibration, draws=100)["selected_class"], "deterministic")
        calibration["patient_ids"] = ["0"]
        with self.assertRaises(ValueError):
            audit(frame, calibration, draws=100)


if __name__ == "__main__":
    unittest.main()
