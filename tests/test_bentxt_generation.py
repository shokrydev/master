import unittest
from pathlib import Path

import yaml

from src.bentxt_generation import (
    BOUNDING_BOX_BUCKET,
    CAPTION_BUCKET,
    DEFAULT_MAX_NEW_TOKENS_BY_BUCKET,
    SHORT_ANSWER_BUCKET,
    bucket_indices,
    generation_bucket_for_task,
    validate_bucket_values,
)


class BENTxTGenerationTest(unittest.TestCase):
    def test_task_mapping_uses_three_generation_buckets(self):
        self.assertEqual(generation_bucket_for_task("binary"), SHORT_ANSWER_BUCKET)
        self.assertEqual(generation_bucket_for_task("mcq"), SHORT_ANSWER_BUCKET)
        self.assertEqual(generation_bucket_for_task("bounding box"), BOUNDING_BOX_BUCKET)
        self.assertEqual(generation_bucket_for_task("captioning"), CAPTION_BUCKET)

    def test_bucket_indices_are_complete_and_duplicate_free(self):
        task_types = [
            "binary",
            "bounding box",
            "captioning",
            "mcq",
            "binary",
            "bounding box",
            "captioning",
            "mcq",
            "binary",
        ]
        indices = bucket_indices(task_types)
        flattened = [index for bucket in indices.values() for index in bucket]
        self.assertCountEqual(flattened, range(len(task_types)))
        self.assertEqual(len(flattened), len(set(flattened)))
        for bucket, bucket_rows in indices.items():
            self.assertTrue(
                all(
                    generation_bucket_for_task(task_types[index]) == bucket for index in bucket_rows
                )
            )

    def test_bucket_indices_preserve_dataset_order(self):
        indices = bucket_indices(["captioning", "binary", "mcq", "bounding box"])
        self.assertEqual(indices[SHORT_ANSWER_BUCKET], [1, 2])
        self.assertEqual(indices[BOUNDING_BOX_BUCKET], [3])
        self.assertEqual(indices[CAPTION_BUCKET], [0])

    def test_bucket_values_require_the_complete_contract(self):
        with self.assertRaisesRegex(ValueError, "missing"):
            validate_bucket_values(
                {SHORT_ANSWER_BUCKET: 32},
                label="test_values",
            )

    def test_evaluation_config_uses_the_shared_generation_caps(self):
        config_path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "evaluation"
            / "bigearthnet_txt.yaml"
        )
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

        self.assertEqual(
            config["model"]["init_args"]["generation_max_new_tokens_by_bucket"],
            DEFAULT_MAX_NEW_TOKENS_BY_BUCKET,
        )


if __name__ == "__main__":
    unittest.main()
