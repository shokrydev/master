import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file


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


def _install_captioning_stub():
    captioning = types.ModuleType("src.metrics.captioning")

    class CaptioningMetrics:
        def __init__(self):
            self.predictions = []

        def update(self, *args, **kwargs):
            return None

        def compute(self):
            return {}

        def reset(self):
            self.predictions = []

    captioning.CaptioningMetrics = CaptioningMetrics
    sys.modules["src.metrics.captioning"] = captioning


def _build_encoder_test_module(num_location_tokens: int = 2):
    module = object.__new__(Qwen3VLModule)
    module.loc_mode = "loc_embed"
    module.num_location_tokens = num_location_tokens
    module.device = torch.device("cpu")
    module._location_insertion_state = None

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
            "target_texts": [["a"], ["b"]],
        }

        model_batch, target_texts, lat, lon = module._prepare_model_inputs(batch)

        expected_labels = torch.tensor(
            [[11, 12, -100, -100, 13, 14, 15], [21, -100, -100, 22, 23, 24, 25]]
        )
        self.assertTrue(torch.equal(model_batch["labels"], expected_labels))
        self.assertEqual(module._location_insertion_state["insert_positions"].tolist(), [2, 1])
        self.assertEqual(target_texts, [["a"], ["b"]])
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

        model_batch, _, _, _ = module._prepare_model_inputs(batch)

        expected_labels = torch.tensor([[11, 12, 13, -100, -100, -100]])
        self.assertTrue(torch.equal(model_batch["labels"], expected_labels))
        self.assertEqual(module._location_insertion_state["insert_positions"].tolist(), [3])

    def test_prepare_model_inputs_is_invariant_for_non_encoder_modes(self):
        module = object.__new__(Qwen3VLModule)
        module.loc_mode = "loc_text"
        module.num_location_tokens = 2
        module.device = torch.device("cpu")
        module._location_insertion_state = None

        batch = {
            "input_ids": torch.tensor([[101, 102, 103]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
            "labels": torch.tensor([[11, 12, 13]]),
            "lat": torch.tensor([1.0], dtype=torch.float64),
            "lon": torch.tensor([2.0], dtype=torch.float64),
            "target_texts": [["ref"]],
            "multispectral": torch.ones(1, 3, 2, 2),
            "multispectral_bands": ["B04", "B03", "B02"],
        }

        model_batch, target_texts, lat, lon = module._prepare_model_inputs(batch)

        self.assertTrue(torch.equal(model_batch["labels"], torch.tensor([[11, 12, 13]])))
        self.assertNotIn("multispectral", model_batch)
        self.assertNotIn("multispectral_bands", model_batch)
        self.assertIsNone(module._location_insertion_state)
        self.assertEqual(target_texts, [["ref"]])
        self.assertTrue(torch.equal(lat, torch.tensor([1.0], dtype=torch.float64)))
        self.assertTrue(torch.equal(lon, torch.tensor([2.0], dtype=torch.float64)))


class AdapterArtifactSetupTest(unittest.TestCase):
    def test_fit_rejects_adapter_dir(self):
        module = Qwen3VLModule(adapter_dir="/tmp/adapter")
        module.trainer = types.SimpleNamespace(datamodule=None)

        with self.assertRaisesRegex(ValueError, "cannot be set for fit"):
            module.setup("fit")

    def test_validate_requires_adapter_dir(self):
        module = Qwen3VLModule()
        module.trainer = types.SimpleNamespace(datamodule=None)

        with self.assertRaisesRegex(ValueError, "adapter_dir"):
            module.setup("validate")

    def test_validate_loads_saved_adapters_without_wrapping_peft_again(self):
        _install_captioning_stub()
        calls = {}

        class FakeModel:
            def __init__(self):
                self._param = torch.nn.Parameter(torch.ones(1))
                config = types.SimpleNamespace(
                    image_token_id=999,
                    video_token_id=998,
                    text_config=types.SimpleNamespace(hidden_size=16),
                )
                language_model = types.SimpleNamespace(
                    register_forward_pre_hook=lambda *args, **kwargs: types.SimpleNamespace(remove=lambda: None)
                )
                inner = types.SimpleNamespace(config=config, language_model=language_model)
                self.base_model = types.SimpleNamespace(model=types.SimpleNamespace(model=inner))

            def parameters(self):
                return [self._param]

            def named_parameters(self):
                return [("dummy_weight", self._param)]

        original_from_pretrained = qwen3_module.FastVisionModel.from_pretrained
        original_get_peft_model = qwen3_module.FastVisionModel.get_peft_model
        original_for_training = qwen3_module.FastVisionModel.for_training

        try:
            def fake_from_pretrained(*args, **kwargs):
                calls["model_name"] = kwargs["model_name"]
                return FakeModel(), types.SimpleNamespace()

            def fake_get_peft_model(*args, **kwargs):
                calls["wrapped_with_peft"] = True
                return args[0]

            qwen3_module.FastVisionModel.from_pretrained = staticmethod(fake_from_pretrained)
            qwen3_module.FastVisionModel.get_peft_model = staticmethod(fake_get_peft_model)
            qwen3_module.FastVisionModel.for_training = staticmethod(lambda *args, **kwargs: None)

            module = Qwen3VLModule(adapter_dir="/tmp/adapter", loc_mode="no_loc")
            module.trainer = types.SimpleNamespace(datamodule=None)
            module.setup("validate")

            self.assertEqual(calls["model_name"], "/tmp/adapter")
            self.assertNotIn("wrapped_with_peft", calls)
        finally:
            qwen3_module.FastVisionModel.from_pretrained = original_from_pretrained
            qwen3_module.FastVisionModel.get_peft_model = original_get_peft_model
            qwen3_module.FastVisionModel.for_training = original_for_training


class LocationProjectionArtifactLoadTest(unittest.TestCase):
    def test_load_location_projection_artifacts_restores_saved_weights(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            saved_projection = torch.nn.Linear(3, 4)
            projection_path = Path(tmpdir) / "location_modality_projection.safetensors"
            save_file(saved_projection.state_dict(), projection_path)

            module = object.__new__(Qwen3VLModule)
            module.adapter_dir = tmpdir
            module.device = torch.device("cpu")
            module.location_modality_projection = torch.nn.Linear(3, 4)

            for parameter in module.location_modality_projection.parameters():
                torch.nn.init.zeros_(parameter)

            module._load_location_projection_artifacts()

            expected = saved_projection.state_dict()
            actual = module.location_modality_projection.state_dict()
            for key in expected:
                self.assertTrue(torch.equal(actual[key], expected[key]))

    def test_load_location_projection_artifacts_requires_saved_file(self):
        module = object.__new__(Qwen3VLModule)
        module.adapter_dir = "/tmp/missing-adapter-dir"
        module.device = torch.device("cpu")
        module.location_modality_projection = torch.nn.Linear(3, 4)

        with self.assertRaises(FileNotFoundError):
            module._load_location_projection_artifacts()


if __name__ == "__main__":
    unittest.main()
