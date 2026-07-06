import importlib
import importlib.machinery
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file


def _install_qwen3_test_stubs():
    lightning = types.ModuleType("lightning")
    lightning.__path__ = []
    lightning.__spec__ = importlib.machinery.ModuleSpec("lightning", loader=None, is_package=True)

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
    sys.modules["lightning.pytorch"] = lightning

    bitsandbytes = types.ModuleType("bitsandbytes")
    bitsandbytes.__spec__ = importlib.machinery.ModuleSpec("bitsandbytes", loader=None)

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
_slice_batch_indices = qwen3_module._slice_batch_indices

def _build_encoder_test_module(num_location_tokens: int = 2):
    module = object.__new__(Qwen3VLModule)
    module.loc_mode = "loc_embed"
    module.non_rgb_conditioning = "disabled"
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
    def test_slice_batch_indices_preserves_shared_band_order(self):
        batch = {
            "input_ids": torch.arange(12).reshape(4, 3),
            "lat": torch.tensor([1.0, 2.0, 3.0, 4.0]),
            "input_text": ["a", "b", "c", "d"],
            "target_texts": [["ta"], ["tb"], ["tc"], ["td"]],
            "non_rgb_bands": ["VV", "VH", "B04", "B03"],
        }

        sliced = _slice_batch_indices(batch, [2, 0])

        self.assertTrue(torch.equal(sliced["input_ids"], torch.tensor([[6, 7, 8], [0, 1, 2]])))
        self.assertTrue(torch.equal(sliced["lat"], torch.tensor([3.0, 1.0])))
        self.assertEqual(sliced["input_text"], ["c", "a"])
        self.assertEqual(sliced["target_texts"], [["tc"], ["ta"]])
        self.assertEqual(sliced["non_rgb_bands"], ["VV", "VH", "B04", "B03"])

    def test_validation_trace_indices_use_explicit_sample_ids(self):
        module = object.__new__(Qwen3VLModule)
        module.validation_trace_path = "trace.jsonl"
        module.validation_trace_sample_ids = ("row-c", "row-a")
        module._validation_trace_sample_id_set = set(module.validation_trace_sample_ids)
        module._trainer_or_none = lambda: types.SimpleNamespace(is_global_zero=True)
        batch = {
            "input_ids": torch.ones(4, 3),
            "sample_id": ["row-a", "row-b", "row-c", "row-d"],
        }

        self.assertEqual(module._validation_trace_indices(batch, batch_idx=5), [0, 2])

    def test_write_validation_trace_appends_jsonl_records(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            module = object.__new__(Qwen3VLModule)
            module.validation_trace_path = str(Path(tmpdir) / "trace.jsonl")
            module.loc_mode = "loc_text"
            module.model_name_or_path = "qwen-test"
            module.max_new_tokens = 32
            module._trainer_or_none = lambda: types.SimpleNamespace(global_step=12)

            module._write_validation_trace(
                predictions=["yes", "no"],
                target_texts=[["yes"], ["no"]],
                lat=torch.tensor([48.1, 49.2]),
                lon=torch.tensor([12.3, 13.4]),
                sample_metadata={
                    "input_text": ["Question A", "Question B"],
                    "sample_id": ["row-a", "row-b"],
                    "patch_id": ["patch-a", "patch-b"],
                    "task_type": ["binary", "binary"],
                    "task_category": ["country", "season"],
                    "split": ["validation", "validation"],
                },
                batch_idx=0,
            )

            lines = (Path(tmpdir) / "trace.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            self.assertIn('"global_step": 12', lines[0])
            self.assertIn('"prediction": "yes"', lines[0])
            self.assertIn('"sample_id": "row-a"', lines[0])

    def test_write_prediction_export_appends_jsonl_records(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            module = object.__new__(Qwen3VLModule)
            module.prediction_export_path = str(Path(tmpdir) / "predictions.jsonl")
            module.loc_mode = "loc_embed"
            module.model_name_or_path = "qwen-test"
            module.adapter_dir = "/tmp/adapter"
            module._prediction_export_count = 0

            module._write_prediction_export(
                predictions=["yes", "no"],
                target_texts=[["yes"], ["no"]],
                lat=torch.tensor([48.1, 49.2]),
                lon=torch.tensor([12.3, 13.4]),
                sample_metadata={
                    "input_text": ["Question A", "Question B"],
                    "sample_id": ["row-a", "row-b"],
                    "patch_id": ["patch-a", "patch-b"],
                    "task_type": ["binary", "binary"],
                    "task_category": ["country", "season"],
                    "split": ["bench", "bench"],
                },
            )

            lines = (Path(tmpdir) / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
            records = [json.loads(line) for line in lines]
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0]["prediction"], "yes")
            self.assertEqual(records[0]["sample_id"], "row-a")
            self.assertEqual(records[0]["location_condition"], "loc_embed")
            self.assertEqual(module._prediction_export_count, 2)

    def test_prediction_export_rejects_distributed_test(self):
        module = object.__new__(Qwen3VLModule)
        module.prediction_export_path = "predictions.jsonl"
        module._trainer_or_none = lambda: types.SimpleNamespace(world_size=2, is_global_zero=True)

        with self.assertRaisesRegex(ValueError, "single-process"):
            module.on_test_start()

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

    def test_insert_projected_tokens_updates_decoder_kwargs(self):
        module = object.__new__(Qwen3VLModule)
        kwargs = {
            "inputs_embeds": torch.tensor(
                [
                    [[1.0], [2.0], [3.0]],
                    [[10.0], [11.0], [12.0]],
                ]
            ),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]]),
            "position_ids": torch.tensor(
                [
                    [[0, 1, 2], [5, 6, 7]],
                    [[10, 11, 12], [15, 16, 17]],
                    [[20, 21, 22], [25, 26, 27]],
                ]
            ),
            "visual_pos_masks": torch.tensor([[False, True, True], [False, True, True]]),
        }
        tokens = torch.tensor([[[90.0], [91.0]], [[80.0], [81.0]]])
        positions = torch.tensor([1, 3])

        module._insert_projected_tokens_in_kwargs(kwargs, tokens, positions)

        expected_embeds = torch.tensor(
            [
                [[1.0], [90.0], [91.0], [2.0], [3.0]],
                [[10.0], [11.0], [12.0], [80.0], [81.0]],
            ]
        )
        expected_attention = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])
        expected_visual_mask = torch.tensor(
            [[False, False, False, True, True], [False, True, True, False, False]]
        )

        self.assertTrue(torch.equal(kwargs["inputs_embeds"], expected_embeds))
        self.assertTrue(torch.equal(kwargs["attention_mask"], expected_attention))
        self.assertTrue(torch.equal(kwargs["visual_pos_masks"], expected_visual_mask))
        self.assertEqual(kwargs["position_ids"].shape, (3, 2, 5))

    def test_projected_token_hook_inserts_non_rgb_tokens(self):
        module = object.__new__(Qwen3VLModule)
        module.device = torch.device("cpu")
        module._location_insertion_state = None
        module._non_rgb_insertion_state = {
            "tensor": torch.ones(2, 12, 2, 2),
            "bands": ["VV", "VH"],
            "insert_positions": torch.tensor([1, 3]),
        }

        class FakeEncoder:
            def __call__(self, imagery, bands):
                self.imagery = imagery
                self.bands = bands
                return torch.zeros(imagery.shape[0], 2, 5)

        class FakeProjection:
            def __call__(self, features):
                self.features = features
                return torch.tensor([[[90.0], [91.0]], [[80.0], [81.0]]])

        encoder = FakeEncoder()
        projection = FakeProjection()
        module.non_rgb_encoder = encoder
        module.non_rgb_modality_projection = projection

        kwargs = {
            "inputs_embeds": torch.tensor(
                [
                    [[1.0], [2.0], [3.0]],
                    [[10.0], [11.0], [12.0]],
                ]
            ),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]]),
        }

        module._projected_token_insertion_hook(None, (), kwargs)

        expected_embeds = torch.tensor(
            [
                [[1.0], [90.0], [91.0], [2.0], [3.0]],
                [[10.0], [11.0], [12.0], [80.0], [81.0]],
            ]
        )
        self.assertTrue(torch.equal(kwargs["inputs_embeds"], expected_embeds))
        self.assertTrue(torch.equal(kwargs["attention_mask"], torch.ones(2, 5, dtype=torch.long)))
        self.assertTrue(torch.equal(encoder.imagery, torch.ones(2, 12, 2, 2)))
        self.assertEqual(encoder.bands, ["VV", "VH"])
        self.assertEqual(projection.features.shape, (2, 2, 5))

    def test_projected_token_hook_orders_location_before_non_rgb(self):
        module = object.__new__(Qwen3VLModule)
        module.device = torch.device("cpu")
        module._non_rgb_insertion_state = {
            "tensor": torch.ones(1, 12, 2, 2),
            "bands": None,
            "insert_positions": torch.tensor([1]),
        }
        module._location_insertion_state = {
            "lat": torch.tensor([1.0], dtype=torch.float64),
            "lon": torch.tensor([2.0], dtype=torch.float64),
            "insert_positions": torch.tensor([1]),
        }
        module.non_rgb_encoder = lambda imagery, bands: torch.zeros(1, 1, 5)
        module.non_rgb_modality_projection = lambda features: torch.tensor([[[80.0]]])
        module.satclip = lambda coords: torch.zeros(1, 3)
        module.location_modality_projection = lambda features: torch.tensor([[[90.0]]])

        kwargs = {"inputs_embeds": torch.tensor([[[1.0], [2.0]]])}

        module._projected_token_insertion_hook(None, (), kwargs)

        expected = torch.tensor([[[1.0], [90.0], [80.0], [2.0]]])
        self.assertTrue(torch.equal(kwargs["inputs_embeds"], expected))


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
            "sample_id": ["row-1", "row-2"],
            "patch_id": ["patch-1", "patch-2"],
            "task_type": ["captioning", "binary"],
            "task_category": ["caption", "presence"],
        }

        model_batch, target_texts, lat, lon, non_rgb_imagery, metadata = (
            module._prepare_model_inputs(batch)
        )

        expected_labels = torch.tensor(
            [[11, 12, -100, -100, 13, 14, 15], [21, -100, -100, 22, 23, 24, 25]]
        )
        self.assertTrue(torch.equal(model_batch["labels"], expected_labels))
        self.assertEqual(module._location_insertion_state["insert_positions"].tolist(), [2, 1])
        self.assertEqual(target_texts, [["a"], ["b"]])
        self.assertTrue(torch.equal(lat, torch.tensor([52.5, -33.9], dtype=torch.float64)))
        self.assertTrue(torch.equal(lon, torch.tensor([13.4, 151.2], dtype=torch.float64)))
        self.assertIsNone(non_rgb_imagery["tensor"])
        self.assertIsNone(non_rgb_imagery["bands"])
        self.assertEqual(metadata["sample_id"], ["row-1", "row-2"])
        self.assertEqual(metadata["patch_id"], ["patch-1", "patch-2"])
        self.assertEqual(metadata["task_type"], ["captioning", "binary"])
        self.assertEqual(metadata["task_category"], ["caption", "presence"])
        self.assertNotIn("task_type", model_batch)

    def test_prepare_model_inputs_falls_back_to_sequence_end_without_visual_tokens(self):
        module = _build_encoder_test_module(num_location_tokens=1)
        batch = {
            "input_ids": torch.tensor([[101, 102, 103, 0, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 0, 0]]),
            "labels": torch.tensor([[11, 12, 13, -100, -100]]),
            "lat": torch.tensor([1.0], dtype=torch.float64),
            "lon": torch.tensor([2.0], dtype=torch.float64),
        }

        model_batch, _, _, _, _, _ = module._prepare_model_inputs(batch)

        expected_labels = torch.tensor([[11, 12, 13, -100, -100, -100]])
        self.assertTrue(torch.equal(model_batch["labels"], expected_labels))
        self.assertEqual(module._location_insertion_state["insert_positions"].tolist(), [3])

    def test_prepare_model_inputs_is_invariant_for_non_encoder_modes(self):
        module = object.__new__(Qwen3VLModule)
        module.loc_mode = "loc_text"
        module.non_rgb_conditioning = "disabled"
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
            "non_rgb_imagery": torch.ones(1, 3, 2, 2),
            "non_rgb_bands": ["VV", "VH", "B04", "B03", "B02"],
        }

        model_batch, target_texts, lat, lon, non_rgb_imagery, metadata = (
            module._prepare_model_inputs(batch)
        )

        self.assertTrue(torch.equal(model_batch["labels"], torch.tensor([[11, 12, 13]])))
        self.assertNotIn("non_rgb_imagery", model_batch)
        self.assertNotIn("non_rgb_bands", model_batch)
        self.assertTrue(torch.equal(non_rgb_imagery["tensor"], torch.ones(1, 3, 2, 2)))
        self.assertEqual(non_rgb_imagery["bands"], ["VV", "VH", "B04", "B03", "B02"])
        self.assertIsNone(module._location_insertion_state)
        self.assertEqual(target_texts, [["ref"]])
        self.assertTrue(torch.equal(lat, torch.tensor([1.0], dtype=torch.float64)))
        self.assertTrue(torch.equal(lon, torch.tensor([2.0], dtype=torch.float64)))
        self.assertIsNone(metadata["task_type"])

    def test_prepare_model_inputs_sets_non_rgb_state_and_masks_labels(self):
        module = object.__new__(Qwen3VLModule)
        module.loc_mode = "no_loc"
        module.non_rgb_conditioning = "enabled"
        module.num_non_rgb_tokens = 2
        module.device = torch.device("cpu")
        module._location_insertion_state = None
        module._non_rgb_insertion_state = None

        config = types.SimpleNamespace(image_token_id=999, video_token_id=998)
        inner = types.SimpleNamespace(config=config)
        module.model = types.SimpleNamespace(base_model=types.SimpleNamespace(model=types.SimpleNamespace(model=inner)))

        imagery = torch.ones(1, 12, 2, 2)
        batch = {
            "input_ids": torch.tensor([[101, 999, 201]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
            "labels": torch.tensor([[11, 12, 13]]),
            "non_rgb_imagery": imagery,
            "non_rgb_bands": ["VV", "VH"],
        }

        model_batch, _, _, _, non_rgb_imagery, _ = module._prepare_model_inputs(batch)

        expected_labels = torch.tensor([[11, -100, -100, 12, 13]])
        self.assertTrue(torch.equal(model_batch["labels"], expected_labels))
        self.assertNotIn("non_rgb_imagery", model_batch)
        self.assertNotIn("non_rgb_bands", model_batch)
        self.assertTrue(torch.equal(non_rgb_imagery["tensor"], imagery))
        self.assertEqual(non_rgb_imagery["bands"], ["VV", "VH"])
        self.assertTrue(torch.equal(module._non_rgb_insertion_state["tensor"], imagery))
        self.assertEqual(module._non_rgb_insertion_state["bands"], ["VV", "VH"])
        self.assertEqual(module._non_rgb_insertion_state["insert_positions"].tolist(), [1])
        self.assertIsNone(module._location_insertion_state)

    def test_prepare_model_inputs_rejects_missing_enabled_non_rgb_imagery(self):
        module = object.__new__(Qwen3VLModule)
        module.loc_mode = "no_loc"
        module.non_rgb_conditioning = "enabled"
        module.num_non_rgb_tokens = 2
        module.device = torch.device("cpu")
        module._location_insertion_state = None
        module._non_rgb_insertion_state = None

        batch = {
            "input_ids": torch.tensor([[101, 102, 103]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
            "labels": torch.tensor([[11, 12, 13]]),
        }

        with self.assertRaisesRegex(ValueError, "requires non_rgb_imagery"):
            module._prepare_model_inputs(batch)


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

    def test_invalid_non_rgb_conditioning_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported non_rgb_conditioning"):
            Qwen3VLModule(non_rgb_conditioning="spectral")

    def test_invalid_loc_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported loc_mode"):
            Qwen3VLModule(loc_mode="coordinates")

    def test_loc_text_requires_location_text_template(self):
        with self.assertRaisesRegex(ValueError, "location_text_template"):
            Qwen3VLModule(loc_mode="loc_text")

    def test_location_text_template_requires_loc_text_mode(self):
        with self.assertRaisesRegex(ValueError, "loc_mode='loc_text'"):
            Qwen3VLModule(location_text_template="lat {lat}")

    def test_coordinates_decimal_places_requires_loc_text_mode(self):
        with self.assertRaisesRegex(ValueError, "loc_mode='loc_text'"):
            Qwen3VLModule(coordinates_decimal_places=2)

    def test_coordinates_decimal_places_must_be_non_negative(self):
        with self.assertRaisesRegex(ValueError, "coordinates_decimal_places"):
            Qwen3VLModule(
                loc_mode="loc_text",
                location_text_template="Scene coordinates: {location}.",
                coordinates_decimal_places=-1,
            )

    def test_loc_embed_requires_location_embed_marker(self):
        with self.assertRaisesRegex(ValueError, "location_embed_marker"):
            Qwen3VLModule(loc_mode="loc_embed")

    def test_location_embed_marker_requires_loc_embed_mode(self):
        with self.assertRaisesRegex(ValueError, "loc_mode='loc_embed'"):
            Qwen3VLModule(location_embed_marker="Scene coordinates:")

    def test_location_projection_lr_multiplier_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "location_projection_lr_multiplier"):
            Qwen3VLModule(location_projection_lr_multiplier=0.0)

    def test_validation_trace_sample_ids_require_path(self):
        with self.assertRaisesRegex(ValueError, "validation_trace_path"):
            Qwen3VLModule(validation_trace_sample_ids=["row-a"])

    def test_validation_trace_path_requires_sample_ids(self):
        with self.assertRaisesRegex(ValueError, "validation_trace_sample_ids"):
            Qwen3VLModule(validation_trace_path="trace.jsonl")

    def test_validation_trace_sample_ids_must_be_unique(self):
        with self.assertRaisesRegex(ValueError, "duplicates"):
            Qwen3VLModule(
                validation_trace_sample_ids=["row-a", "row-a"],
                validation_trace_path="trace.jsonl",
            )

    def test_non_rgb_projection_lr_multiplier_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "non_rgb_projection_lr_multiplier"):
            Qwen3VLModule(non_rgb_projection_lr_multiplier=0.0)

    def test_invalid_non_rgb_feature_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported non_rgb_feature_mode"):
            Qwen3VLModule(non_rgb_feature_mode="spatial_tokens")

    def test_invalid_spatial_non_rgb_token_count_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "num_non_rgb_tokens"):
            Qwen3VLModule(non_rgb_spatial_pool_size=4, num_non_rgb_tokens=8)

    def test_validate_loads_saved_adapters_without_wrapping_peft_again(self):
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

    def test_setup_non_rgb_conditioning_rejects_feature_dim_mismatch(self):
        import src.models.bigearthnet_s1s2_encoder as encoder_module
        import src.models.non_rgb_modality_projection as projection_module

        class FakeEncoder(torch.nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
                self.feature_dim = 640

        class FakeProjection(torch.nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()

        original_encoder = encoder_module.BigEarthNetS1S2Encoder
        original_projection = projection_module.NonRGBModalityProjection
        try:
            encoder_module.BigEarthNetS1S2Encoder = FakeEncoder
            projection_module.NonRGBModalityProjection = FakeProjection

            module = object.__new__(Qwen3VLModule)
            module.non_rgb_encoder_dir = "/tmp/encoder"
            module.non_rgb_feature_mode = "spatial_4x4"
            module.non_rgb_spatial_pool_size = 4
            module.non_rgb_encoder_feature_dim = 512
            module.num_non_rgb_tokens = 16
            module.device = torch.device("cpu")
            module._geo_hook_handle = None
            module.print = lambda *args, **kwargs: None

            config = types.SimpleNamespace(text_config=types.SimpleNamespace(hidden_size=8))
            language_model = types.SimpleNamespace(
                register_forward_pre_hook=lambda *args, **kwargs: types.SimpleNamespace(remove=lambda: None)
            )
            qwen_model = types.SimpleNamespace(
                config=config,
                model=types.SimpleNamespace(language_model=language_model),
            )
            module.model = types.SimpleNamespace(base_model=types.SimpleNamespace(model=qwen_model))

            with self.assertRaisesRegex(ValueError, "does not match"):
                module._setup_non_rgb_conditioning()
        finally:
            encoder_module.BigEarthNetS1S2Encoder = original_encoder
            projection_module.NonRGBModalityProjection = original_projection


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

    def test_load_non_rgb_projection_artifacts_restores_saved_weights(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            saved_projection = torch.nn.Linear(3, 4)
            projection_path = Path(tmpdir) / "non_rgb_modality_projection.safetensors"
            save_file(saved_projection.state_dict(), projection_path)

            module = object.__new__(Qwen3VLModule)
            module.adapter_dir = tmpdir
            module.device = torch.device("cpu")
            module.non_rgb_modality_projection = torch.nn.Linear(3, 4)

            for parameter in module.non_rgb_modality_projection.parameters():
                torch.nn.init.zeros_(parameter)

            module._load_non_rgb_projection_artifacts()

            expected = saved_projection.state_dict()
            actual = module.non_rgb_modality_projection.state_dict()
            for key in expected:
                self.assertTrue(torch.equal(actual[key], expected[key]))

    def test_load_non_rgb_projection_artifacts_requires_saved_file(self):
        module = object.__new__(Qwen3VLModule)
        module.adapter_dir = "/tmp/missing-adapter-dir"
        module.device = torch.device("cpu")
        module.non_rgb_modality_projection = torch.nn.Linear(3, 4)

        with self.assertRaises(FileNotFoundError):
            module._load_non_rgb_projection_artifacts()


if __name__ == "__main__":
    unittest.main()
