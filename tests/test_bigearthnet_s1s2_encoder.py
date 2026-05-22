import unittest

import torch
import torch.nn as nn

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


if __name__ == "__main__":
    unittest.main()
