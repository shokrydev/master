import unittest

import torch
from PIL import Image

from src.data_modules.ben_txt_datamodule import _sentinel2_rgb_tensor_to_pil, collate_normalized


class TestBENTxTDataBoundary(unittest.TestCase):
    def test_sentinel2_rgb_tensor_is_rendered_to_pil_with_copernicus_scale(self) -> None:
        tensor = torch.tensor(
            [
                [[0.0, 1500.0], [3000.0, 4500.0]],
                [[3000.0, 1500.0], [0.0, 1500.0]],
                [[1500.0, 0.0], [1500.0, 3000.0]],
            ],
            dtype=torch.float32,
        )

        image = _sentinel2_rgb_tensor_to_pil(tensor)

        self.assertIsInstance(image, Image.Image)
        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.size, (2, 2))
        self.assertEqual(image.getpixel((0, 0)), (0, 215, 107))
        self.assertEqual(image.getpixel((1, 1)), (255, 107, 215))

    def test_normalized_collate_preserves_shared_sample_fields(self) -> None:
        image = Image.new("RGB", (4, 4), color=(12, 34, 56))
        batch = collate_normalized(
            [
                {
                    "image": image,
                    "input_text": "Question 1",
                    "target_texts": ["Answer 1"],
                    "lat": 10.5,
                    "lon": 20.5,
                },
                {
                    "image": image,
                    "input_text": "Question 2",
                    "target_texts": ["Answer 2a", "Answer 2b"],
                    "lat": -1.0,
                    "lon": 42.0,
                },
            ]
        )

        self.assertEqual(len(batch["image"]), 2)
        self.assertEqual(batch["input_text"], ["Question 1", "Question 2"])
        self.assertEqual(batch["target_texts"], [["Answer 1"], ["Answer 2a", "Answer 2b"]])
        self.assertTrue(
            torch.equal(batch["lat"], torch.tensor([10.5, -1.0], dtype=torch.float64))
        )
        self.assertTrue(
            torch.equal(batch["lon"], torch.tensor([20.5, 42.0], dtype=torch.float64))
        )


if __name__ == "__main__":
    unittest.main()
