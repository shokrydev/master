import unittest

import torch

from src.models.non_rgb_modality_projection import NonRGBModalityProjection


class NonRGBModalityProjectionTest(unittest.TestCase):
    def test_projects_pooled_encoder_features_to_token_sequence(self):
        projection = NonRGBModalityProjection(encoder_dim=5, hidden_size=7, num_tokens=3)
        features = torch.randn(2, 5)

        tokens = projection(features)

        self.assertEqual(tokens.shape, (2, 3, 7))

    def test_projects_spatial_encoder_tokens_individually(self):
        projection = NonRGBModalityProjection(encoder_dim=5, hidden_size=7, num_tokens=3)
        features = torch.randn(2, 3, 5)

        tokens = projection(features)

        self.assertEqual(tokens.shape, (2, 3, 7))

    def test_rejects_wrong_spatial_token_count(self):
        projection = NonRGBModalityProjection(encoder_dim=5, hidden_size=7, num_tokens=3)
        features = torch.randn(2, 4, 5)

        with self.assertRaisesRegex(ValueError, "Spatial non-RGB features"):
            projection(features)


if __name__ == "__main__":
    unittest.main()
