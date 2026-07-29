import unittest

import torch

from src.models.scene_location_encoding import SceneLocationEncoding


class SceneLocationEncodingTest(unittest.TestCase):
    def test_output_shape_and_coordinate_order_are_deterministic(self):
        encoder = SceneLocationEncoding(
            hidden_size=8,
            scale_init=0.1,
            learned_scale=True,
        )
        lat = torch.tensor([48.0, 49.0], dtype=torch.float64)
        lon = torch.tensor([12.0, 13.0], dtype=torch.float64)

        first = encoder(lat, lon)
        second = encoder(lat, lon)

        self.assertEqual(first.shape, (2, 8))
        self.assertEqual(first.dtype, torch.float32)
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(first[0], first[1]))
        self.assertFalse(torch.equal(first, encoder(lon, lat)))

    def test_scale_is_the_only_trainable_parameter(self):
        encoder = SceneLocationEncoding(
            hidden_size=8,
            scale_init=0.1,
            learned_scale=True,
        )

        named_parameters = dict(encoder.named_parameters())

        self.assertEqual(set(named_parameters), {"scale"})
        self.assertAlmostEqual(float(named_parameters["scale"].detach()), 0.1)

    def test_fixed_scale_has_no_trainable_parameters(self):
        encoder = SceneLocationEncoding(
            hidden_size=8,
            scale_init=0.2,
            learned_scale=False,
        )

        self.assertEqual(list(encoder.parameters()), [])
        self.assertAlmostEqual(float(encoder.scale), 0.2)

    def test_invalid_shapes_and_values_fail(self):
        encoder = SceneLocationEncoding(hidden_size=8)

        with self.assertRaisesRegex(ValueError, "shape"):
            encoder(torch.zeros(1, 1), torch.zeros(1))
        with self.assertRaisesRegex(ValueError, "matching"):
            encoder(torch.zeros(2), torch.zeros(1))
        with self.assertRaisesRegex(ValueError, "finite"):
            encoder(torch.tensor([float("nan")]), torch.zeros(1))
        with self.assertRaisesRegex(ValueError, "latitude"):
            encoder(torch.tensor([91.0]), torch.zeros(1))
        with self.assertRaisesRegex(ValueError, "longitude"):
            encoder(torch.zeros(1), torch.tensor([181.0]))

    def test_hidden_size_must_be_divisible_by_four(self):
        with self.assertRaisesRegex(ValueError, "divisible by four"):
            SceneLocationEncoding(hidden_size=6)


if __name__ == "__main__":
    unittest.main()
