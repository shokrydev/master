import json
import importlib.util
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import save_file

from src.models.bigearthnet_s1s2_encoder import (
    BIGEARTHNET_S1S2_10M20M_BANDS,
    BigEarthNetS1S2Encoder,
)


class _FakeVisionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Module()
        self.head.fc = nn.Linear(640, 19)

    def forward_features(self, imagery):
        b = imagery.shape[0]
        values = torch.arange(b * 640 * 5 * 7, dtype=imagery.dtype, device=imagery.device)
        return values.view(b, 640, 5, 7)

    def forward_head(self, features, pre_logits=False):
        if not pre_logits:
            raise AssertionError("wrapper should request pre_logits=True")
        return features.mean(dim=(-2, -1))


class _FakeBigEarthNetModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.vision_encoder = _FakeVisionEncoder()


class BigEarthNetS1S2EncoderTest(unittest.TestCase):
    def test_spatial_mode_returns_fixed_grid_tokens_and_freezes_model(self):
        model = _FakeBigEarthNetModel()
        encoder = BigEarthNetS1S2Encoder(model=model, feature_mode="spatial_4x4")
        imagery = torch.randn(2, 12, 120, 120)

        tokens = encoder(imagery, BIGEARTHNET_S1S2_10M20M_BANDS)

        self.assertEqual(tokens.shape, (2, 16, 640))
        self.assertEqual(encoder.feature_dim, 640)
        self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))

    def test_pooled_prelogit_mode_returns_pre_classifier_embedding(self):
        encoder = BigEarthNetS1S2Encoder(
            model=_FakeBigEarthNetModel(),
            feature_mode="pooled_prelogit",
        )
        imagery = torch.randn(2, 12, 120, 120)

        features = encoder(imagery, BIGEARTHNET_S1S2_10M20M_BANDS)

        self.assertEqual(features.shape, (2, 640))
        self.assertEqual(encoder.feature_dim, 640)

    def test_rejects_wrong_band_order(self):
        encoder = BigEarthNetS1S2Encoder(model=_FakeBigEarthNetModel())
        imagery = torch.randn(1, 12, 120, 120)
        wrong_bands = list(reversed(BIGEARTHNET_S1S2_10M20M_BANDS))

        with self.assertRaisesRegex(ValueError, "expected non_rgb_bands"):
            encoder(imagery, wrong_bands)

    def test_rejects_empty_band_order(self):
        encoder = BigEarthNetS1S2Encoder(model=_FakeBigEarthNetModel())
        imagery = torch.randn(1, 12, 120, 120)

        with self.assertRaisesRegex(ValueError, "expected non_rgb_bands"):
            encoder(imagery, [])

    def test_rejects_wrong_channel_count(self):
        encoder = BigEarthNetS1S2Encoder(model=_FakeBigEarthNetModel())
        imagery = torch.randn(1, 10, 120, 120)

        with self.assertRaisesRegex(ValueError, "Expected 12 non-RGB channels"):
            encoder(imagery, BIGEARTHNET_S1S2_10M20M_BANDS)

    def test_loads_local_timm_mobilevit_checkpoint(self):
        if importlib.util.find_spec("timm") is None:
            self.skipTest("timm is not installed")
        import timm

        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            config = {
                "timm_model_name": "mobilevit_s",
                "channels": 12,
                "classes": 19,
                "drop_rate": 0.0,
                "drop_path_rate": 0.0,
            }
            (model_dir / "config.json").write_text(json.dumps(config))
            model = timm.create_model("mobilevit_s", in_chans=12, num_classes=19)
            state_dict = {
                f"model.vision_encoder.{key}": value
                for key, value in model.state_dict().items()
            }
            save_file(state_dict, model_dir / "model.safetensors")

            encoder = BigEarthNetS1S2Encoder(model_dir=model_dir, feature_mode="spatial_4x4")
            imagery = torch.randn(1, 12, 120, 120)

            tokens = encoder(imagery, BIGEARTHNET_S1S2_10M20M_BANDS)

            self.assertEqual(tokens.shape, (1, 16, 640))
            self.assertEqual(encoder.feature_dim, 640)

    def test_local_checkpoint_loader_warns_for_ignored_non_vision_weights(self):
        if importlib.util.find_spec("timm") is None:
            self.skipTest("timm is not installed")
        import timm

        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            config = {
                "timm_model_name": "mobilevit_s",
                "channels": 12,
                "classes": 19,
            }
            (model_dir / "config.json").write_text(json.dumps(config))
            model = timm.create_model("mobilevit_s", in_chans=12, num_classes=19)
            state_dict = {
                f"model.vision_encoder.{key}": value
                for key, value in model.state_dict().items()
            }
            state_dict["model.fusion_layer.weight"] = torch.randn(1)
            save_file(state_dict, model_dir / "model.safetensors")

            with self.assertWarnsRegex(UserWarning, "ignores non-vision checkpoint weights"):
                BigEarthNetS1S2Encoder(model_dir=model_dir)


if __name__ == "__main__":
    unittest.main()
