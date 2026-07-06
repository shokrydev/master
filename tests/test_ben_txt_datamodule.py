import unittest
from tempfile import TemporaryDirectory

import pandas as pd
import torch
from PIL import Image

from src.data_modules.ben_txt_datamodule import (
    BENTxTDataModule,
    _apply_coordinate_perturbation,
    _fixed_random_subset,
    _load_location_redacted_captions,
    _sentinel2_rgb_tensor_to_pil,
    collate_normalized,
)


class TestBENTxTDataBoundary(unittest.TestCase):
    def test_fixed_validation_subset_is_seeded_and_not_a_leading_slice(self) -> None:
        dataset = list(range(100))

        first = _fixed_random_subset(dataset, size=12, seed=42)
        second = _fixed_random_subset(dataset, size=12, seed=42)

        self.assertEqual(first.indices, second.indices)
        self.assertEqual(len(first), 12)
        self.assertNotEqual(first.indices, list(range(12)))

    def test_datamodule_uses_explicit_test_splits(self) -> None:
        datamodule = BENTxTDataModule(
            image_lmdb_file="images.lmdb",
            metadata_file="metadata.parquet",
            test_splits=("bench",),
        )

        self.assertEqual(datamodule.test_splits, ("bench",))

    def test_datamodule_rejects_unknown_test_split(self) -> None:
        with self.assertRaisesRegex(ValueError, "test_splits"):
            BENTxTDataModule(
                image_lmdb_file="images.lmdb",
                metadata_file="metadata.parquet",
                test_splits=("unknown",),
            )

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
                    "split": "validation",
                    "country": "Austria",
                    "season": "Spring",
                    "climate_zone": "Cfb",
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
                    "split": "validation",
                    "country": "Portugal",
                    "season": "Summer",
                    "climate_zone": "Csa",
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
        self.assertEqual(batch["split"], ["validation", "validation"])
        self.assertEqual(batch["country"], ["Austria", "Portugal"])
        self.assertEqual(batch["season"], ["Spring", "Summer"])
        self.assertEqual(batch["climate_zone"], ["Cfb", "Csa"])
        self.assertTrue(
            torch.equal(batch["lat"], torch.tensor([10.5, -1.0], dtype=torch.float64))
        )
        self.assertTrue(
            torch.equal(batch["lon"], torch.tensor([20.5, 42.0], dtype=torch.float64))
        )

    def test_coordinate_perturbation_is_deterministic_and_preserves_rows(self) -> None:
        metadata = pd.DataFrame(
            {
                "ID": ["row-a", "row-b", "row-c"],
                "latitude": [10.0, 20.0, 30.0],
                "longitude": [1.0, 2.0, 3.0],
            }
        )

        shuffled = _apply_coordinate_perturbation(metadata, "shuffled")
        shuffled_again = _apply_coordinate_perturbation(metadata, "shuffled")

        self.assertEqual(shuffled["ID"].tolist(), ["row-a", "row-b", "row-c"])
        self.assertEqual(shuffled["latitude"].tolist(), shuffled_again["latitude"].tolist())
        self.assertCountEqual(shuffled["latitude"].tolist(), [10.0, 20.0, 30.0])
        self.assertCountEqual(shuffled["longitude"].tolist(), [1.0, 2.0, 3.0])

    def test_antipodal_coordinate_perturbation_changes_only_coordinates(self) -> None:
        metadata = pd.DataFrame(
            {
                "ID": ["row-a", "row-b"],
                "latitude": [10.0, -20.0],
                "longitude": [30.0, -40.0],
            }
        )

        perturbed = _apply_coordinate_perturbation(metadata, "antipodal")

        self.assertEqual(perturbed["ID"].tolist(), ["row-a", "row-b"])
        self.assertEqual(perturbed["latitude"].tolist(), [-10.0, 20.0])
        self.assertEqual(perturbed["longitude"].tolist(), [-150.0, 140.0])


if __name__ == "__main__":
    unittest.main()
