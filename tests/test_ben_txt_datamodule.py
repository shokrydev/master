import unittest
from tempfile import TemporaryDirectory

import pandas as pd
import torch
from PIL import Image

from src.data_modules.ben_txt_datamodule import (
    _load_location_redacted_captions,
    _sentinel2_rgb_tensor_to_pil,
    collate_normalized,
)


class TestBENTxTDataBoundary(unittest.TestCase):
    def test_location_redacted_caption_file_loads_patch_mapping(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/captions.parquet"
            pd.DataFrame(
                {
                    "patch_id": ["patch-a", "patch-b"],
                    "refined_caption": ["caption a", "caption b"],
                }
            ).to_parquet(path)

            captions = _load_location_redacted_captions(path)

        self.assertEqual(captions, {"patch-a": "caption a", "patch-b": "caption b"})

    def test_location_redacted_caption_file_rejects_missing_columns(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/captions.parquet"
            pd.DataFrame({"patch_id": ["patch-a"], "caption": ["caption a"]}).to_parquet(path)

            with self.assertRaisesRegex(ValueError, "missing columns"):
                _load_location_redacted_captions(path)

    def test_location_redacted_caption_file_rejects_duplicate_patch_ids(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/captions.parquet"
            pd.DataFrame(
                {
                    "patch_id": ["patch-a", "patch-a"],
                    "refined_caption": ["caption a", "caption b"],
                }
            ).to_parquet(path)

            with self.assertRaisesRegex(ValueError, "duplicate patch_id"):
                _load_location_redacted_captions(path)

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
                    "sample_id": "row-1",
                    "patch_id": "patch-1",
                    "task_type": "captioning",
                    "task_category": "caption",
                    "lat": 10.5,
                    "lon": 20.5,
                },
                {
                    "image": image,
                    "input_text": "Question 2",
                    "target_texts": ["Answer 2a", "Answer 2b"],
                    "sample_id": "row-2",
                    "patch_id": "patch-2",
                    "task_type": "binary",
                    "task_category": "presence",
                    "lat": -1.0,
                    "lon": 42.0,
                },
            ]
        )

        self.assertEqual(len(batch["image"]), 2)
        self.assertEqual(batch["input_text"], ["Question 1", "Question 2"])
        self.assertEqual(batch["target_texts"], [["Answer 1"], ["Answer 2a", "Answer 2b"]])
        self.assertEqual(batch["sample_id"], ["row-1", "row-2"])
        self.assertEqual(batch["patch_id"], ["patch-1", "patch-2"])
        self.assertEqual(batch["task_type"], ["captioning", "binary"])
        self.assertEqual(batch["task_category"], ["caption", "presence"])
        self.assertTrue(
            torch.equal(batch["lat"], torch.tensor([10.5, -1.0], dtype=torch.float64))
        )
        self.assertTrue(
            torch.equal(batch["lon"], torch.tensor([20.5, 42.0], dtype=torch.float64))
        )


if __name__ == "__main__":
    unittest.main()
