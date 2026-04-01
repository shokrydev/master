import importlib
import sys
import types
import unittest

import torch


def _install_qwen3_test_stubs():
    lightning = types.ModuleType("lightning")

    class LightningModule:
        def __init__(self, *args, **kwargs):
            self.device = torch.device("cpu")

        def save_hyperparameters(self, *args, **kwargs):
            return None

        def print(self, *args, **kwargs):
            return None

        def log(self, *args, **kwargs):
            return None

    class LightningDataModule:
        pass

    lightning.LightningModule = LightningModule
    lightning.LightningDataModule = LightningDataModule
    sys.modules["lightning"] = lightning

    bitsandbytes = types.ModuleType("bitsandbytes")

    class AdamW8bit:
        def __init__(self, *args, **kwargs):
            pass

    bitsandbytes.optim = types.SimpleNamespace(AdamW8bit=AdamW8bit)
    sys.modules["bitsandbytes"] = bitsandbytes

    unsloth = types.ModuleType("unsloth")

    class FastVisionModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise NotImplementedError

        @staticmethod
        def get_peft_model(*args, **kwargs):
            raise NotImplementedError

        @staticmethod
        def for_training(*args, **kwargs):
            return None

        @staticmethod
        def for_inference(*args, **kwargs):
            return None

    unsloth.FastVisionModel = FastVisionModel
    sys.modules["unsloth"] = unsloth

    trainer = types.ModuleType("unsloth.trainer")

    class UnslothVisionDataCollator:
        def __init__(self, *args, **kwargs):
            pass

    trainer.UnslothVisionDataCollator = UnslothVisionDataCollator
    sys.modules["unsloth.trainer"] = trainer


_install_qwen3_test_stubs()
qwen3_module = importlib.import_module("src.lightning_modules.qwen3_vl_module")
Qwen3VLModule = qwen3_module.Qwen3VLModule


def _build_encoder_test_module(num_location_tokens: int = 2):
    module = object.__new__(Qwen3VLModule)
    module.loc_mode = "encoder"
    module.num_location_tokens = num_location_tokens
    module.device = torch.device("cpu")
    module._current_batch_geo = None

    config = types.SimpleNamespace(image_token_id=999, video_token_id=998)
    language_model = types.SimpleNamespace()
    qwen_model = types.SimpleNamespace(config=config, language_model=language_model)
    wrapper = types.SimpleNamespace(model=qwen_model)
    module.model = types.SimpleNamespace(base_model=types.SimpleNamespace(model=wrapper))
    return module


class InsertTokenHelpersTest(unittest.TestCase):
    def test_insert_tokens_2d_preserves_order_and_positions(self):
        tensor = torch.tensor([[1, 2, 3, 4], [10, 11, 12, 13]])
        insert = torch.tensor([[90, 91], [80, 81]])
        positions = torch.tensor([2, 0])

        out = Qwen3VLModule._insert_tokens_2d(tensor, insert, positions)

        expected = torch.tensor([[1, 2, 90, 91, 3, 4], [80, 81, 10, 11, 12, 13]])
        self.assertTrue(torch.equal(out, expected))

    def test_insert_tokens_3d_preserves_order_and_shapes(self):
        tensor = torch.tensor(
            [
                [[1.0, 1.1], [2.0, 2.1], [3.0, 3.1]],
                [[10.0, 10.1], [11.0, 11.1], [12.0, 12.1]],
            ]
        )
        insert = torch.tensor(
            [
                [[90.0, 90.1], [91.0, 91.1]],
                [[80.0, 80.1], [81.0, 81.1]],
            ]
        )
        positions = torch.tensor([1, 3])

        out = Qwen3VLModule._insert_tokens_3d(tensor, insert, positions)

        self.assertEqual(out.shape, (2, 5, 2))
        expected_first = torch.tensor(
            [[1.0, 1.1], [90.0, 90.1], [91.0, 91.1], [2.0, 2.1], [3.0, 3.1]]
        )
        expected_second = torch.tensor(
            [[10.0, 10.1], [11.0, 11.1], [12.0, 12.1], [80.0, 80.1], [81.0, 81.1]]
        )
        self.assertTrue(torch.equal(out[0], expected_first))
        self.assertTrue(torch.equal(out[1], expected_second))

    def test_insert_position_ids_inserts_contiguous_positions_and_shifts_suffix(self):
        position_ids = torch.tensor(
            [
                [[0, 1, 2, 3], [5, 6, 7, 8]],
                [[10, 11, 12, 13], [15, 16, 17, 18]],
                [[20, 21, 22, 23], [25, 26, 27, 28]],
            ]
        )
        positions = torch.tensor([2, 0])

        out = Qwen3VLModule._insert_position_ids(position_ids, positions, n=2)

        expected = torch.tensor(
            [
                [[0, 1, 2, 3, 4, 5], [0, 1, 7, 8, 9, 10]],
                [[10, 11, 12, 13, 14, 15], [0, 1, 17, 18, 19, 20]],
                [[20, 21, 22, 23, 24, 25], [0, 1, 27, 28, 29, 30]],
            ]
        )
        self.assertTrue(torch.equal(out, expected))


class PrepareModelInputsTest(unittest.TestCase):
    def test_prepare_model_inputs_inserts_ignore_labels_at_visual_boundary(self):
        module = _build_encoder_test_module(num_location_tokens=2)
        batch = {
            "input_ids": torch.tensor([[101, 102, 999, 999, 201], [301, 999, 302, 303, 304]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]),
            "labels": torch.tensor([[11, 12, 13, 14, 15], [21, 22, 23, 24, 25]]),
            "lat": torch.tensor([52.5, -33.9], dtype=torch.float64),
            "lon": torch.tensor([13.4, 151.2], dtype=torch.float64),
            "references": [["a"], ["b"]],
            "image_ids": ["img1", "img2"],
        }

        model_batch, references, image_ids, lat, lon = module._prepare_model_inputs(batch)

        expected_labels = torch.tensor(
            [[11, 12, -100, -100, 13, 14, 15], [21, -100, -100, 22, 23, 24, 25]]
        )
        self.assertTrue(torch.equal(model_batch["labels"], expected_labels))
        self.assertEqual(module._current_batch_geo["insert_positions"].tolist(), [2, 1])
        self.assertEqual(references, [["a"], ["b"]])
        self.assertEqual(image_ids, ["img1", "img2"])
        self.assertTrue(torch.equal(lat, torch.tensor([52.5, -33.9], dtype=torch.float64)))
        self.assertTrue(torch.equal(lon, torch.tensor([13.4, 151.2], dtype=torch.float64)))

    def test_prepare_model_inputs_falls_back_to_sequence_end_without_visual_tokens(self):
        module = _build_encoder_test_module(num_location_tokens=1)
        batch = {
            "input_ids": torch.tensor([[101, 102, 103, 0, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 0, 0]]),
            "labels": torch.tensor([[11, 12, 13, -100, -100]]),
            "lat": torch.tensor([1.0], dtype=torch.float64),
            "lon": torch.tensor([2.0], dtype=torch.float64),
        }

        model_batch, _, _, _, _ = module._prepare_model_inputs(batch)

        expected_labels = torch.tensor([[11, 12, 13, -100, -100, -100]])
        self.assertTrue(torch.equal(model_batch["labels"], expected_labels))
        self.assertEqual(module._current_batch_geo["insert_positions"].tolist(), [3])

    def test_prepare_model_inputs_is_invariant_for_non_encoder_modes(self):
        module = object.__new__(Qwen3VLModule)
        module.loc_mode = "text"
        module.num_location_tokens = 2
        module.device = torch.device("cpu")
        module._current_batch_geo = None

        batch = {
            "input_ids": torch.tensor([[101, 102, 103]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
            "labels": torch.tensor([[11, 12, 13]]),
            "lat": torch.tensor([1.0], dtype=torch.float64),
            "lon": torch.tensor([2.0], dtype=torch.float64),
            "references": [["ref"]],
            "image_ids": ["img"],
        }

        model_batch, references, image_ids, lat, lon = module._prepare_model_inputs(batch)

        self.assertTrue(torch.equal(model_batch["labels"], torch.tensor([[11, 12, 13]])))
        self.assertIsNone(module._current_batch_geo)
        self.assertEqual(references, [["ref"]])
        self.assertEqual(image_ids, ["img"])
        self.assertTrue(torch.equal(lat, torch.tensor([1.0], dtype=torch.float64)))
        self.assertTrue(torch.equal(lon, torch.tensor([2.0], dtype=torch.float64)))


if __name__ == "__main__":
    unittest.main()
