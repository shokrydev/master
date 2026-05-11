import unittest

import torch

from src.models.non_rgb_modality_projection import NonRGBModalityProjection


class NonRGBModalityProjectionTest(unittest.TestCase):
    def test_projects_encoder_features_to_token_sequence(self):
        projection = NonRGBModalityProjection(encoder_dim=5, hidden_size=7, num_tokens=3)
        features = torch.randn(2, 5)

        tokens = projection(features)

        self.assertEqual(tokens.shape, (2, 3, 7))


if __name__ == "__main__":
    unittest.main()
