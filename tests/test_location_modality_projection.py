import unittest

import torch
from torch import nn

from src.models.location_modality_projection import LocationModalityProjection


class LocationModalityProjectionTest(unittest.TestCase):
    def test_default_mlp_preserves_original_architecture_and_initialization(self):
        torch.manual_seed(42)
        projection = LocationModalityProjection(
            satclip_dim=4,
            hidden_size=8,
            num_tokens=2,
        )
        torch.manual_seed(42)
        original = nn.Sequential(
            nn.Linear(4, 8),
            nn.GELU(),
            nn.Linear(8, 16),
        )

        self.assertEqual(projection.architecture, "mlp")
        self.assertEqual(
            set(projection.state_dict()),
            {"proj.0.weight", "proj.0.bias", "proj.2.weight", "proj.2.bias"},
        )
        self.assertEqual(
            set(projection.proj.state_dict()),
            set(original.state_dict()),
        )
        for key, expected in original.state_dict().items():
            torch.testing.assert_close(projection.proj.state_dict()[key], expected)

        features = torch.randn(3, 4)
        torch.testing.assert_close(
            projection(features),
            original(features).view(3, 2, 8),
        )

    def test_compact_linear_shape_and_parameter_count(self):
        projection = LocationModalityProjection(
            satclip_dim=256,
            hidden_size=2048,
            num_tokens=8,
            architecture="linear",
        )

        output = projection(torch.randn(2, 256))

        self.assertEqual(output.shape, (2, 8, 2048))
        self.assertEqual(
            sum(parameter.numel() for parameter in projection.parameters()),
            4_210_688,
        )
        self.assertIsInstance(projection.proj, nn.Linear)
        self.assertIsNotNone(projection.proj.bias)

    def test_manifest_distinguishes_projection_architectures(self):
        mlp = LocationModalityProjection(4, 8, 2, architecture="mlp")
        linear = LocationModalityProjection(4, 8, 2, architecture="linear")

        self.assertEqual(mlp.manifest()["architecture"], "mlp")
        self.assertEqual(linear.manifest()["architecture"], "linear")
        self.assertEqual(linear.manifest()["satclip_dim"], 4)
        self.assertEqual(linear.manifest()["hidden_size"], 8)
        self.assertEqual(linear.manifest()["num_tokens"], 2)

    def test_invalid_configuration_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "satclip_dim"):
            LocationModalityProjection(satclip_dim=0)
        with self.assertRaisesRegex(ValueError, "hidden_size"):
            LocationModalityProjection(hidden_size=0)
        with self.assertRaisesRegex(ValueError, "num_tokens"):
            LocationModalityProjection(num_tokens=0)
        with self.assertRaisesRegex(ValueError, "architecture"):
            LocationModalityProjection(architecture="deep")


if __name__ == "__main__":
    unittest.main()
