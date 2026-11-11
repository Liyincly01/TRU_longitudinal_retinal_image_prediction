import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from external.models_tadm import TADMUnet, ConditionEncoder
from src.clinical_metrics import load_segmentor
from src.predictors import load_tadm
from src.backbone import MultiScaleTemporalUNet_v13


class AdapterTests(unittest.TestCase):
    def test_model_history_limit_ignores_older_frames(self):
        torch.manual_seed(42)
        model = MultiScaleTemporalUNet_v13(max_history=2, base_channels=8,
                                           delta_emb_dim=32, history_dropout=0).eval()
        history = torch.randn(1, 4, 1, 32, 32)
        times = torch.tensor([[4., 3., 2., 1.]])
        padding = torch.zeros((1, 4), dtype=torch.bool)
        with torch.inference_mode():
            expected = model(history[:, -1], history[:, -2:], times[:, -2:], padding[:, -2:], torch.tensor([200.]))
            actual = model(history[:, -1], history, times, padding, torch.tensor([200.]))
            history[:, :2] += 100
            altered = model(history[:, -1], history, times, padding, torch.tensor([200.]))
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        torch.testing.assert_close(expected, altered, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            MultiScaleTemporalUNet_v13(max_history=0)

    def test_tadm_checkpoint_and_sampling(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tadm.pt"
            unet = TADMUnet(dim=8, out_dim=1, dim_mults=(1, 2, 4, 8), cond_dim=8)
            condition = ConditionEncoder(in_channels=1, base_dim=8, num_blocks=4)
            torch.save(dict(ema_unet=unet.state_dict(), ema_cond_net=condition.state_dict(),
                            args=dict(dim=8)), path)
            predict = load_tadm(path, torch.device("cpu"), steps=5)
            result = predict(np.full((2, 32, 32), .5, np.float32), np.array([0., 1.]), 2., 42)
            self.assertEqual(result.shape, (32, 32))
            self.assertTrue(np.isfinite(result).all())
            self.assertTrue(((result >= 0) & (result <= 1)).all())

    @unittest.skipUnless(importlib.util.find_spec("monai"), "MONAI is optional")
    def test_segmentor_normalization_and_probabilities(self):
        from monai.networks.nets import DynUNet
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "segmentor.pt"
            model = torch.nn.Sequential(DynUNet(
                spatial_dims=2, in_channels=1, out_channels=1, kernel_size=[5] * 4,
                filters=[16, 32, 64, 128], strides=[1] * 4, upsample_kernel_size=[1] * 3),
                torch.nn.Sigmoid())
            torch.save(model.state_dict(), path)
            segment = load_segmentor(path, device="cpu")
            images = np.random.default_rng(1).integers(0, 256, (1, 32, 32), dtype=np.uint8)
            result = segment(images)
            self.assertEqual(result.shape, images.shape)
            self.assertTrue(np.isfinite(result).all())
            self.assertTrue(((result >= 0) & (result <= 1)).all())


if __name__ == "__main__":
    unittest.main()
