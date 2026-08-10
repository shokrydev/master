import unittest

import torch

from src.models.additive_location_projection import AdditiveLocationProjection


class AdditiveLocationProjectionTest(unittest.TestCase):
    def test_output_shape_parameter_count_and_bias(self):
        bridge = AdditiveLocationProjection(
            feature_dim=4,
            hidden_size=8,
            scale=0.1,
        )
        output = bridge(torch.tensor([[1.0, 2.0, 3.0, 4.0]]))

        self.assertEqual(output.shape, (1, 8))
        self.assertEqual(output.dtype, torch.float32)
        self.assertIsNone(bridge.projection.bias)
        self.assertEqual(sum(p.numel() for p in bridge.parameters()), 32)

    def test_positive_feature_rescaling_does_not_change_output(self):
        bridge = AdditiveLocationProjection(feature_dim=4, hidden_size=8)
        features = torch.tensor([[1.0, -2.0, 3.0, -4.0]])

        torch.testing.assert_close(
            bridge(features),
            bridge(7.0 * features),
            rtol=1e-5,
            atol=1e-6,
        )

    def test_invalid_features_fail_clearly(self):
        bridge = AdditiveLocationProjection(feature_dim=4, hidden_size=8)

        with self.assertRaisesRegex(ValueError, "shape"):
            bridge(torch.ones(1, 3))
        with self.assertRaisesRegex(ValueError, "finite"):
            bridge(torch.tensor([[1.0, 2.0, float("nan"), 4.0]]))
        with self.assertRaisesRegex(ValueError, "non-zero RMS"):
            bridge(torch.zeros(1, 4))

    def test_same_seed_produces_identical_bridge(self):
        torch.manual_seed(42)
        first = AdditiveLocationProjection(feature_dim=4, hidden_size=8)
        torch.manual_seed(42)
        second = AdditiveLocationProjection(feature_dim=4, hidden_size=8)

        self.assertTrue(
            torch.equal(first.projection.weight, second.projection.weight)
        )

    def test_manifest_records_architecture_and_source(self):
        bridge = AdditiveLocationProjection(feature_dim=4, hidden_size=8)

        manifest = bridge.manifest(
            feature_source="direct",
            scope="s1s2",
            source_config={"encoding_type": "test"},
        )

        self.assertEqual(manifest["feature_source"], "direct")
        self.assertEqual(manifest["scope"], "s1s2")
        self.assertEqual(manifest["feature_dim"], 4)
        self.assertEqual(manifest["hidden_size"], 8)
        self.assertEqual(manifest["feature_normalization"], "rms")
        self.assertFalse(manifest["projection_bias"])


if __name__ == "__main__":
    unittest.main()
