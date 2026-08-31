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

from src.models.additive_location_projection import AdditiveLocationProjection
from src.models.location_modality_projection import LocationModalityProjection
from src.models.scene_location_encoding import (
    SceneLocationEncoding,
    SceneLocationFeatures,
)


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
        last_init_kwargs = None

        def __init__(self, *args, **kwargs):
            type(self).last_init_kwargs = kwargs

    trainer.UnslothVisionDataCollator = UnslothVisionDataCollator
    sys.modules["unsloth.trainer"] = trainer


_install_qwen3_test_stubs()
qwen3_module = importlib.import_module("src.lightning_modules.qwen3_vl_module")
Qwen3VLModule = qwen3_module.Qwen3VLModule


class _EvalCallable:
    def __init__(self, function):
        self.function = function
        self.training = True

    def eval(self):
        self.training = False
        return self

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)


def _build_encoder_test_module(num_location_tokens: int = 2):
    module = object.__new__(Qwen3VLModule)
    module.loc_mode = "loc_embed"
    module.non_rgb_conditioning = "disabled"
    module.num_location_tokens = num_location_tokens
    module.device = torch.device("cpu")
    module._location_insertion_state = None
    module._non_rgb_insertion_state = None
    module.tokenizer = types.SimpleNamespace(pad_token_id=0)

    config = types.SimpleNamespace(
        image_token_id=999,
        video_token_id=998,
        vision_start_token_id=997,
    )
    language_model = types.SimpleNamespace()
    qwen_model = types.SimpleNamespace(config=config, language_model=language_model)
    wrapper = types.SimpleNamespace(model=qwen_model)
    module.model = types.SimpleNamespace(base_model=types.SimpleNamespace(model=wrapper))
    return module


class InsertTokenHelpersTest(unittest.TestCase):
    def test_write_validation_generations_appends_jsonl_records(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            module = object.__new__(Qwen3VLModule)
            module.validation_generation_path = str(
                Path(tmpdir) / "validation_generations.jsonl"
            )
            module.loc_mode = "loc_text"
            module.model_name_or_path = "qwen-test"
            module.max_new_tokens = 32
            module._trainer_or_none = lambda: types.SimpleNamespace(
                global_step=12,
                is_global_zero=True,
            )

            module._write_validation_generations(
                predictions=["yes"],
                target_texts=[["yes"]],
                lat=torch.tensor([48.1]),
                lon=torch.tensor([12.3]),
                sample_metadata={
                    "input_text": ["Question A"],
                    "sample_id": ["row-a"],
                    "patch_id": ["patch-a"],
                    "task_type": ["binary"],
                    "task_category": ["country"],
                    "split": ["validation"],
                    "country": ["Austria"],
                    "season": ["Spring"],
                    "climate_zone": ["Cfb"],
                },
                batch_idx=0,
            )

            record = json.loads(
                Path(module.validation_generation_path).read_text(encoding="utf-8")
            )
            self.assertEqual(record["global_step"], 12)
            self.assertEqual(record["prediction"], "yes")
            self.assertEqual(record["sample_id"], "row-a")

    def test_write_prediction_export_appends_jsonl_records(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            module = object.__new__(Qwen3VLModule)
            module.prediction_export_path = str(Path(tmpdir) / "predictions.jsonl")
            module.loc_mode = "loc_embed"
            module.model_name_or_path = "qwen-test"
            module.adapter_dir = "/tmp/adapter"
            module.run_label = "loc_embed-2B-full"
            module.model_size = "2B"
            module._prediction_export_count = 0
            module._prediction_export_sample_ids = set()

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
                    "country": ["Austria", "Portugal"],
                    "season": ["Spring", "Summer"],
                    "climate_zone": ["Cfb", "Csa"],
                },
            )

            lines = (Path(tmpdir) / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
            records = [json.loads(line) for line in lines]
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0]["prediction"], "yes")
            self.assertEqual(records[0]["sample_id"], "row-a")
            self.assertEqual(records[0]["location_condition"], "loc_embed")
            self.assertEqual(records[0]["run_label"], "loc_embed-2B-full")
            self.assertEqual(records[0]["model_size"], "2B")
            self.assertEqual(records[0]["country"], "Austria")
            self.assertEqual(records[0]["season"], "Spring")
            self.assertEqual(records[0]["climate_zone"], "Cfb")
            self.assertEqual(module._prediction_export_count, 2)

    def test_prediction_export_generates_without_teacher_forced_forward(self):
        module = object.__new__(Qwen3VLModule)
        module.prediction_export_path = "predictions.jsonl"
        module.max_new_tokens = 32
        module.generation_max_new_tokens_by_bucket = None
        module._prepare_model_inputs = lambda batch: (
            {"input_ids": torch.tensor([[1, 2, 3]])},
            [["reference"]],
            torch.tensor([48.0]),
            torch.tensor([12.0]),
            {},
            {"sample_id": ["row-a"]},
        )
        module._generate_for_batch = lambda batch, **kwargs: ["generated answer"]
        module._print = lambda *args, **kwargs: None
        module._reset_decoder_conditioning_state = lambda: None
        exported = {}
        module._write_prediction_export = lambda **kwargs: exported.update(kwargs)

        result = module.test_step({}, batch_idx=0)

        self.assertEqual(result, {"generated": "generated answer"})
        self.assertEqual(exported["predictions"], ["generated answer"])
        self.assertEqual(exported["target_texts"], [["reference"]])

    def test_task_aware_generation_resolves_bucket_cap(self):
        module = object.__new__(Qwen3VLModule)
        module.max_new_tokens = 512
        module.generation_max_new_tokens_by_bucket = {
            "short_answer": 32,
            "bounding_box": 64,
            "captioning": 512,
        }

        self.assertEqual(
            module._test_generation_settings(["binary", "mcq"]),
            ("short_answer", 32),
        )
        with self.assertRaisesRegex(ValueError, "mixed bucket"):
            module._test_generation_settings(["binary", "captioning"])

    def test_prediction_export_rejects_duplicate_sample_ids(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            module = object.__new__(Qwen3VLModule)
            module.prediction_export_path = str(Path(tmpdir) / "predictions.jsonl")
            module.loc_mode = "no_loc"
            module.model_name_or_path = "qwen-test"
            module.adapter_dir = "/tmp/adapter"
            module.run_label = None
            module.model_size = None
            module._prediction_export_count = 0
            module._prediction_export_sample_ids = set()
            kwargs = {
                "predictions": ["yes"],
                "target_texts": [["yes"]],
                "lat": None,
                "lon": None,
                "sample_metadata": {"sample_id": ["row-a"]},
            }

            module._write_prediction_export(**kwargs)
            with self.assertRaisesRegex(RuntimeError, "Duplicate sample_id"):
                module._write_prediction_export(**kwargs)

    def test_validation_generation_uses_separate_prompt_only_batch(self):
        module = object.__new__(Qwen3VLModule)
        module.device = torch.device("meta")
        prepared_batches = []

        def prepare(batch):
            prepared_batches.append(batch)
            if batch.get("prompt_only"):
                return (
                    {"input_ids": torch.tensor([[4, 5, 6]])},
                    [["reference"]],
                    torch.tensor([48.0]),
                    torch.tensor([12.0]),
                    {},
                    {"sample_id": ["selected"]},
                )
            return (
                {"input_ids": torch.tensor([[1, 2, 3]]), "labels": torch.tensor([[1, 2, 3]])},
                None,
                None,
                None,
                {},
                {},
            )

        class FakeModel:
            def __call__(self, **batch):
                return types.SimpleNamespace(loss=torch.tensor(0.5))

        module._prepare_model_inputs = prepare
        module.model = FakeModel()
        module.log = lambda *args, **kwargs: None
        module._reset_decoder_conditioning_state = lambda: None
        module._generate_for_batch = lambda batch: ["generated answer"]
        written = {}
        module._write_validation_generations = lambda **kwargs: written.update(kwargs)

        result = module.validation_step(
            {
                "supervised": True,
                "validation_generation_batch": {
                    "prompt_only": True,
                    "input_ids": torch.tensor([[4, 5, 6]]),
                },
            },
            batch_idx=7,
        )

        self.assertEqual(prepared_batches[0], {"supervised": True})
        self.assertTrue(prepared_batches[1]["prompt_only"])
        self.assertEqual(prepared_batches[1]["input_ids"].device.type, "meta")
        self.assertEqual(written["predictions"], ["generated answer"])
        self.assertEqual(written["target_texts"], [["reference"]])
        self.assertEqual(written["sample_metadata"], {"sample_id": ["selected"]})
        self.assertEqual(written["batch_idx"], 7)
        self.assertEqual(result, {"loss": torch.tensor(0.5)})

    def test_generation_restores_previous_model_mode(self):
        class FakeModel(torch.nn.Module):
            def generate(self, **kwargs):
                return torch.tensor([[1, 2, 3, 4]])

        module = object.__new__(Qwen3VLModule)
        module.model = FakeModel()
        module.max_new_tokens = 1
        module.tokenizer = types.SimpleNamespace(decode=lambda *args, **kwargs: "answer")

        original_for_inference = qwen3_module.FastVisionModel.for_inference
        original_for_training = qwen3_module.FastVisionModel.for_training
        qwen3_module.FastVisionModel.for_inference = staticmethod(lambda model: model.eval())
        qwen3_module.FastVisionModel.for_training = staticmethod(lambda model: model.train())
        try:
            for initial_mode in (False, True):
                module.model.train(initial_mode)
                module._generate_for_batch({"input_ids": torch.tensor([[1, 2, 3]])})
                self.assertEqual(module.model.training, initial_mode)
        finally:
            qwen3_module.FastVisionModel.for_inference = original_for_inference
            qwen3_module.FastVisionModel.for_training = original_for_training

    def test_generation_preserves_content_special_tokens_but_removes_stop_ids(self):
        box_start_id = 151648
        eos_id = 151645

        class FakeModel(torch.nn.Module):
            generation_config = types.SimpleNamespace(eos_token_id=[eos_id])

            def generate(self, **kwargs):
                return torch.tensor([[1, 2, box_start_id, 42, eos_id, eos_id]])

        decoded = {}
        module = object.__new__(Qwen3VLModule)
        module.model = FakeModel()
        module.max_new_tokens = 4
        module.tokenizer = types.SimpleNamespace(
            eos_token_id=eos_id,
            pad_token_id=151643,
            decode=lambda token_ids, **kwargs: decoded.update(
                token_ids=token_ids,
                kwargs=kwargs,
            )
            or "<|box_start|>answer",
        )

        predictions = module._generate_for_batch(
            {"input_ids": torch.tensor([[1, 2]])}
        )

        self.assertEqual(predictions, ["<|box_start|>answer"])
        self.assertEqual(decoded["token_ids"], [box_start_id, 42])
        self.assertFalse(decoded["kwargs"]["skip_special_tokens"])

    def test_insert_tokens_2d_preserves_order_and_positions(self):
        tensor = torch.tensor([[1, 2, 3, 4], [10, 11, 12, 13]])
        insert = torch.tensor([[90, 91], [80, 81]])
        positions = torch.tensor([2, 0])

        out = Qwen3VLModule._insert_tokens_2d(tensor, insert, positions)

        expected = torch.tensor([[1, 2, 90, 91, 3, 4], [80, 81, 10, 11, 12, 13]])
        self.assertTrue(torch.equal(out, expected))

    def test_replace_projected_token_placeholders_updates_only_embeddings(self):
        module = object.__new__(Qwen3VLModule)
        kwargs = {
            "inputs_embeds": torch.tensor(
                [
                    [[1.0], [0.0], [0.0], [7.0], [2.0], [3.0]],
                    [[10.0], [11.0], [12.0], [0.0], [0.0], [7.0]],
                ]
            ),
            "attention_mask": torch.ones(2, 6, dtype=torch.long),
        }
        tokens = torch.tensor([[[90.0], [91.0]], [[80.0], [81.0]]])
        positions = torch.tensor([1, 3])

        module._replace_projected_token_placeholders(
            kwargs,
            tokens,
            positions,
        )

        expected_embeds = torch.tensor(
            [
                [[1.0], [90.0], [91.0], [7.0], [2.0], [3.0]],
                [[10.0], [11.0], [12.0], [80.0], [81.0], [7.0]],
            ]
        )
        self.assertTrue(torch.equal(kwargs["inputs_embeds"], expected_embeds))
        self.assertTrue(torch.equal(kwargs["attention_mask"], torch.ones(2, 6, dtype=torch.long)))

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
            def eval(self):
                return self

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
                    [[1.0], [0.0], [0.0], [7.0], [2.0], [3.0]],
                    [[10.0], [11.0], [12.0], [0.0], [0.0], [7.0]],
                ]
            ),
            "attention_mask": torch.ones(2, 6, dtype=torch.long),
        }

        module._decoder_input_conditioning_hook(None, (), kwargs)

        expected_embeds = torch.tensor(
            [
                [[1.0], [90.0], [91.0], [7.0], [2.0], [3.0]],
                [[10.0], [11.0], [12.0], [80.0], [81.0], [7.0]],
            ]
        )
        self.assertTrue(torch.equal(kwargs["inputs_embeds"], expected_embeds))
        self.assertTrue(torch.equal(kwargs["attention_mask"], torch.ones(2, 6, dtype=torch.long)))
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
        module.non_rgb_encoder = _EvalCallable(
            lambda imagery, bands: torch.zeros(1, 1, 5)
        )
        module.non_rgb_modality_projection = lambda features: torch.tensor([[[80.0]]])
        module.satclip = _EvalCallable(lambda coords: torch.zeros(1, 3))
        module.location_modality_projection = lambda features: torch.tensor([[[90.0]]])

        kwargs = {
            "inputs_embeds": torch.tensor([[[1.0], [0.0], [0.0], [7.0], [2.0]]])
        }

        module._decoder_input_conditioning_hook(None, (), kwargs)

        expected = torch.tensor([[[1.0], [90.0], [80.0], [7.0], [2.0]]])
        self.assertTrue(torch.equal(kwargs["inputs_embeds"], expected))

    def test_projected_token_hook_runs_with_empty_cache_and_skips_filled_cache(self):
        module = object.__new__(Qwen3VLModule)
        module.device = torch.device("cpu")
        module._non_rgb_insertion_state = None
        module._location_insertion_state = {
            "lat": torch.tensor([1.0], dtype=torch.float64),
            "lon": torch.tensor([2.0], dtype=torch.float64),
            "insert_positions": torch.tensor([1]),
        }
        module.satclip = _EvalCallable(lambda coords: coords.float())
        module.location_modality_projection = lambda features: features[:, :1].unsqueeze(-1)

        class FakeCache:
            def __init__(self, length):
                self.length = length

            def get_seq_length(self):
                return self.length

        prefill = {
            "inputs_embeds": torch.tensor([[[0.0], [0.0], [7.0], [0.0]]]),
            "past_key_values": FakeCache(0),
        }
        module._decoder_input_conditioning_hook(None, (), prefill)
        self.assertEqual(prefill["inputs_embeds"][0, 1, 0].item(), 2.0)
        self.assertEqual(prefill["inputs_embeds"][0, 2, 0].item(), 7.0)

        decode = {
            "inputs_embeds": torch.zeros(1, 1, 1),
            "past_key_values": FakeCache(3),
        }
        module._decoder_input_conditioning_hook(None, (), decode)
        self.assertTrue(torch.equal(decode["inputs_embeds"], torch.zeros(1, 1, 1)))

    def test_projected_location_placeholders_change_with_coordinates(self):
        module = object.__new__(Qwen3VLModule)
        module.device = torch.device("cpu")
        module._non_rgb_insertion_state = None
        module._location_insertion_state = {
            "lat": torch.tensor([1.0, 3.0], dtype=torch.float64),
            "lon": torch.tensor([2.0, 4.0], dtype=torch.float64),
            "insert_positions": torch.tensor([1, 1]),
        }
        module.satclip = _EvalCallable(lambda coords: coords.float())
        module.location_modality_projection = lambda features: features[:, :1].unsqueeze(-1)
        kwargs = {
            "inputs_embeds": torch.tensor(
                [[[0.0], [0.0], [7.0], [0.0]], [[0.0], [0.0], [7.0], [0.0]]]
            )
        }

        module._decoder_input_conditioning_hook(None, (), kwargs)

        self.assertEqual(kwargs["inputs_embeds"][0, 1, 0].item(), 2.0)
        self.assertEqual(kwargs["inputs_embeds"][1, 1, 0].item(), 4.0)
        self.assertEqual(kwargs["inputs_embeds"][0, 2, 0].item(), 7.0)
        self.assertEqual(kwargs["inputs_embeds"][1, 2, 0].item(), 7.0)
        self.assertFalse(
            torch.equal(kwargs["inputs_embeds"][0], kwargs["inputs_embeds"][1])
        )


class PrepareModelInputsTest(unittest.TestCase):
    def test_prepare_and_replace_keep_native_visual_boundary_aligned(self):
        module = _build_encoder_test_module(num_location_tokens=2)
        batch = {
            "input_ids": torch.tensor([[101, 997, 999, 999, 201]]),
            "attention_mask": torch.ones(1, 5, dtype=torch.long),
            "mm_token_type_ids": torch.tensor([[0, 0, 1, 1, 0]]),
            "labels": torch.tensor([[11, 12, 13, 14, 15]]),
            "lat": torch.tensor([48.0], dtype=torch.float64),
            "lon": torch.tensor([12.0], dtype=torch.float64),
        }

        model_batch, _, _, _, _, _ = module._prepare_model_inputs(batch)

        self.assertTrue(
            torch.equal(
                model_batch["input_ids"],
                torch.tensor([[101, 0, 0, 997, 999, 999, 201]]),
            )
        )
        self.assertTrue(
            torch.equal(
                model_batch["mm_token_type_ids"],
                torch.tensor([[0, 0, 0, 0, 1, 1, 0]]),
            )
        )

        module.satclip = _EvalCallable(lambda coords: coords.float())
        module.location_modality_projection = lambda features: torch.tensor(
            [[[90.0], [91.0]]]
        )
        kwargs = {
            "inputs_embeds": model_batch["input_ids"].float().unsqueeze(-1),
        }
        module._decoder_input_conditioning_hook(None, (), kwargs)

        self.assertTrue(
            torch.equal(
                kwargs["inputs_embeds"].squeeze(-1),
                torch.tensor([[101.0, 90.0, 91.0, 997.0, 999.0, 999.0, 201.0]]),
            )
        )

    def test_prepare_model_inputs_inserts_ignore_labels_at_visual_boundary(self):
        module = _build_encoder_test_module(num_location_tokens=2)
        batch = {
            "input_ids": torch.tensor(
                [[101, 102, 997, 999, 999, 201], [301, 997, 999, 302, 303, 304]]
            ),
            "attention_mask": torch.ones(2, 6, dtype=torch.long),
            "mm_token_type_ids": torch.tensor(
                [[0, 0, 0, 1, 1, 0], [0, 0, 1, 0, 0, 0]]
            ),
            "labels": torch.tensor(
                [[11, 12, 13, 14, 15, 16], [21, 22, 23, 24, 25, 26]]
            ),
            "lat": torch.tensor([52.5, -33.9], dtype=torch.float64),
            "lon": torch.tensor([13.4, 151.2], dtype=torch.float64),
            "target_texts": [["a"], ["b"]],
            "sample_id": ["row-1", "row-2"],
            "patch_id": ["patch-1", "patch-2"],
            "task_type": ["captioning", "binary"],
            "task_category": ["caption", "presence"],
            "country": ["Austria", "Portugal"],
            "season": ["Spring", "Summer"],
            "climate_zone": ["Cfb", "Csa"],
        }

        model_batch, target_texts, lat, lon, non_rgb_imagery, metadata = (
            module._prepare_model_inputs(batch)
        )

        expected_labels = torch.tensor(
            [
                [11, 12, -100, -100, 13, 14, 15, 16],
                [21, -100, -100, 22, 23, 24, 25, 26],
            ]
        )
        self.assertTrue(torch.equal(model_batch["labels"], expected_labels))
        self.assertTrue(
            torch.equal(
                model_batch["input_ids"],
                torch.tensor(
                    [
                        [101, 102, 0, 0, 997, 999, 999, 201],
                        [301, 0, 0, 997, 999, 302, 303, 304],
                    ]
                ),
            )
        )
        self.assertTrue(torch.equal(model_batch["attention_mask"], torch.ones(2, 8, dtype=torch.long)))
        self.assertTrue(
            torch.equal(
                model_batch["mm_token_type_ids"],
                torch.tensor(
                    [
                        [0, 0, 0, 0, 0, 1, 1, 0],
                        [0, 0, 0, 0, 1, 0, 0, 0],
                    ]
                ),
            )
        )
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
        self.assertEqual(metadata["country"], ["Austria", "Portugal"])
        self.assertEqual(metadata["season"], ["Spring", "Summer"])
        self.assertEqual(metadata["climate_zone"], ["Cfb", "Csa"])
        self.assertNotIn("task_type", model_batch)
        self.assertNotIn("country", model_batch)

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
        self.assertTrue(
            torch.equal(model_batch["input_ids"], torch.tensor([[101, 102, 103, 0, 0, 0]]))
        )
        self.assertTrue(
            torch.equal(model_batch["attention_mask"], torch.tensor([[1, 1, 1, 1, 0, 0]]))
        )
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
        module.tokenizer = types.SimpleNamespace(pad_token_id=0)
        module._location_insertion_state = None
        module._non_rgb_insertion_state = None

        config = types.SimpleNamespace(
            image_token_id=999,
            video_token_id=998,
            vision_start_token_id=997,
        )
        inner = types.SimpleNamespace(config=config)
        module.model = types.SimpleNamespace(base_model=types.SimpleNamespace(model=types.SimpleNamespace(model=inner)))

        imagery = torch.ones(1, 12, 2, 2)
        batch = {
            "input_ids": torch.tensor([[101, 997, 999, 201]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1]]),
            "mm_token_type_ids": torch.tensor([[0, 0, 1, 0]]),
            "labels": torch.tensor([[11, 12, 13, 14]]),
            "non_rgb_imagery": imagery,
            "non_rgb_bands": ["VV", "VH"],
        }

        model_batch, _, _, _, non_rgb_imagery, _ = module._prepare_model_inputs(batch)

        expected_labels = torch.tensor([[11, -100, -100, 12, 13, 14]])
        self.assertTrue(torch.equal(model_batch["labels"], expected_labels))
        self.assertTrue(
            torch.equal(
                model_batch["input_ids"],
                torch.tensor([[101, 0, 0, 997, 999, 201]]),
            )
        )
        self.assertTrue(
            torch.equal(
                model_batch["mm_token_type_ids"],
                torch.tensor([[0, 0, 0, 0, 1, 0]]),
            )
        )
        self.assertNotIn("non_rgb_imagery", model_batch)
        self.assertNotIn("non_rgb_bands", model_batch)
        self.assertTrue(torch.equal(non_rgb_imagery["tensor"], imagery))
        self.assertEqual(non_rgb_imagery["bands"], ["VV", "VH"])
        self.assertTrue(torch.equal(module._non_rgb_insertion_state["tensor"], imagery))
        self.assertEqual(module._non_rgb_insertion_state["bands"], ["VV", "VH"])
        self.assertEqual(module._non_rgb_insertion_state["insert_positions"].tolist(), [1])
        self.assertIsNone(module._location_insertion_state)

    def test_prepare_model_inputs_aligns_combined_projected_token_metadata(self):
        module = _build_encoder_test_module(num_location_tokens=8)
        module.non_rgb_conditioning = "enabled"
        module.num_non_rgb_tokens = 16
        imagery = torch.ones(1, 12, 2, 2)
        batch = {
            "input_ids": torch.tensor([[101, 997, 999, 201]]),
            "attention_mask": torch.ones(1, 4, dtype=torch.long),
            "mm_token_type_ids": torch.tensor([[0, 0, 1, 0]]),
            "labels": torch.tensor([[11, 12, 13, 14]]),
            "lat": torch.tensor([48.0], dtype=torch.float64),
            "lon": torch.tensor([12.0], dtype=torch.float64),
            "non_rgb_imagery": imagery,
            "non_rgb_bands": ["VV", "VH"],
        }

        model_batch, _, _, _, _, _ = module._prepare_model_inputs(batch)

        self.assertEqual(model_batch["input_ids"].shape[1], 28)
        self.assertEqual(model_batch["attention_mask"].shape[1], 28)
        self.assertEqual(model_batch["mm_token_type_ids"].shape[1], 28)
        self.assertEqual(model_batch["labels"].shape[1], 28)
        self.assertTrue(
            torch.equal(
                model_batch["mm_token_type_ids"][0, 1:25],
                torch.zeros(24, dtype=torch.long),
            )
        )

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

    def test_supervision_invariant_requires_prompt_tokens_to_be_ignored(self):
        module = object.__new__(Qwen3VLModule)
        module._supervision_mask_validated = False
        module.tokenizer = types.SimpleNamespace(
            encode=lambda text, add_special_tokens=False: [20, 21, 22]
        )
        module._print = lambda *args, **kwargs: None
        input_ids = torch.tensor([[10, 999, 20, 21, 22, 30, 151645]])
        labels = torch.tensor([[-100, -100, -100, -100, -100, 30, 151645]])
        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": torch.ones_like(input_ids),
        }

        module._validate_supervision_mask(
            batch,
            insert_positions=None,
            num_inserted_tokens=0,
        )

        self.assertTrue(module._supervision_mask_validated)
        module._supervision_mask_validated = False
        bad_batch = dict(batch, labels=labels.clone())
        bad_batch["labels"][0, 1] = 999
        with self.assertRaisesRegex(ValueError, "Assistant-only loss"):
            module._validate_supervision_mask(
                bad_batch,
                insert_positions=None,
                num_inserted_tokens=0,
            )


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

    def test_compact_location_projection_requires_loc_embed_mode(self):
        with self.assertRaisesRegex(ValueError, "only configurable"):
            Qwen3VLModule(location_projection_architecture="linear")

    def test_invalid_location_projection_architecture_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "architecture"):
            Qwen3VLModule(location_projection_architecture="wide")

    def test_compact_location_projection_is_accepted_for_loc_embed(self):
        module = Qwen3VLModule(
            loc_mode="loc_embed",
            location_embed_marker="Scene coordinates:",
            location_projection_architecture="linear",
        )

        self.assertEqual(module.location_projection_architecture, "linear")

    def test_loc_embed_setup_constructs_selected_projection_architecture(self):
        satclip_module = importlib.import_module("src.models.satclip")
        original_get_satclip = satclip_module.get_satclip
        fake_satclip = torch.nn.Identity()
        fake_satclip.checkpoint_metadata = {"embed_dim": 4}
        satclip_module.get_satclip = lambda *args, **kwargs: fake_satclip
        try:
            module = object.__new__(Qwen3VLModule)
            module.device = torch.device("cpu")
            module.satclip_checkpoint = "/tmp/satclip.ckpt"
            module.satclip_dim = 4
            module.num_location_tokens = 2
            module.location_projection_architecture = "linear"
            module._get_text_hidden_size = lambda: 8
            module._register_decoder_input_hook = lambda: None
            module._print = lambda *args, **kwargs: None

            module._setup_loc_embed()
        finally:
            satclip_module.get_satclip = original_get_satclip

        self.assertEqual(
            module.location_modality_projection.architecture,
            "linear",
        )
        self.assertEqual(
            sum(
                parameter.numel()
                for parameter in module.location_modality_projection.parameters()
            ),
            80,
        )

    def test_loc_encoding_requires_supported_scope(self):
        with self.assertRaisesRegex(ValueError, "location_encoding_scope"):
            Qwen3VLModule(loc_mode="loc_encoding")
        with self.assertRaisesRegex(ValueError, "location_encoding_scope"):
            Qwen3VLModule(
                loc_mode="loc_encoding",
                location_encoding_scope="text",
            )

    def test_location_encoding_scope_requires_loc_encoding_mode(self):
        with self.assertRaisesRegex(ValueError, "loc_mode='loc_encoding'"):
            Qwen3VLModule(location_encoding_scope="all_visual")

    def test_projected_additive_modes_require_fixed_scale(self):
        with self.assertRaisesRegex(ValueError, "learned_scale=false"):
            Qwen3VLModule(
                loc_mode="loc_encoding",
                location_encoding_scope="s1s2",
                location_encoding_projection="linear",
            )

    def test_satclip_additive_mode_requires_linear_projection(self):
        with self.assertRaisesRegex(ValueError, "projection='linear'"):
            Qwen3VLModule(
                loc_mode="loc_additive_satclip",
                location_encoding_scope="s1s2",
            )

    def test_location_projection_lr_multiplier_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "location_projection_lr_multiplier"):
            Qwen3VLModule(location_projection_lr_multiplier=0.0)

    def test_validation_generation_sample_ids_require_path(self):
        with self.assertRaisesRegex(ValueError, "validation_generation_path"):
            Qwen3VLModule(validation_generation_sample_ids=["row-a"])

    def test_validation_generation_path_requires_sample_ids(self):
        with self.assertRaisesRegex(ValueError, "validation_generation_sample_ids"):
            Qwen3VLModule(validation_generation_path="validation_generations.jsonl")

    def test_validation_generation_sample_ids_must_be_unique(self):
        with self.assertRaisesRegex(ValueError, "duplicates"):
            Qwen3VLModule(
                validation_generation_sample_ids=["row-a", "row-a"],
                validation_generation_path="validation_generations.jsonl",
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
            self.assertEqual(
                qwen3_module.UnslothVisionDataCollator.last_init_kwargs,
                {
                    "train_on_responses_only": True,
                    "instruction_part": "<|im_start|>user\n",
                    "response_part": "<|im_start|>assistant\n",
                },
            )
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
            module._decoder_input_hook_handle = None
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
    @staticmethod
    def _build_manifest_module(architecture: str) -> Qwen3VLModule:
        module = object.__new__(Qwen3VLModule)
        module.device = torch.device("cpu")
        module.location_projection_architecture = architecture
        module.location_embed_marker = "Scene coordinates:"
        module.satclip = types.SimpleNamespace(
            checkpoint_metadata={
                "embed_dim": 4,
                "legendre_polys": 40,
            }
        )
        module.location_modality_projection = LocationModalityProjection(
            satclip_dim=4,
            hidden_size=8,
            num_tokens=2,
            architecture=architecture,
        )
        return module

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

    def test_manifest_validates_and_restores_compact_projection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            saved = self._build_manifest_module("linear")
            saved.location_modality_projection.proj.weight.data.fill_(0.25)
            save_file(
                saved.location_modality_projection.state_dict(),
                Path(tmpdir) / "location_modality_projection.safetensors",
            )
            manifest = saved.get_location_projection_manifest()
            (Path(tmpdir) / "location_modality_projection_config.json").write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )

            loaded = self._build_manifest_module("linear")
            loaded.adapter_dir = tmpdir
            loaded.location_modality_projection.proj.weight.data.zero_()
            loaded._load_location_projection_artifacts()

            torch.testing.assert_close(
                loaded.location_modality_projection.proj.weight,
                torch.full_like(
                    loaded.location_modality_projection.proj.weight,
                    0.25,
                ),
            )

    def test_manifest_rejects_projection_architecture_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            saved = self._build_manifest_module("mlp")
            save_file(
                saved.location_modality_projection.state_dict(),
                Path(tmpdir) / "location_modality_projection.safetensors",
            )
            (Path(tmpdir) / "location_modality_projection_config.json").write_text(
                json.dumps(saved.get_location_projection_manifest()),
                encoding="utf-8",
            )

            loaded = self._build_manifest_module("linear")
            loaded.adapter_dir = tmpdir
            with self.assertRaisesRegex(ValueError, "does not match"):
                loaded._load_location_projection_artifacts()

    def test_compact_projection_requires_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            module = self._build_manifest_module("linear")
            module.adapter_dir = tmpdir
            save_file(
                module.location_modality_projection.state_dict(),
                Path(tmpdir) / "location_modality_projection.safetensors",
            )

            with self.assertRaisesRegex(FileNotFoundError, "require a manifest"):
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


class SceneLocationEncodingIntegrationTest(unittest.TestCase):
    @staticmethod
    def _build_module() -> Qwen3VLModule:
        module = object.__new__(Qwen3VLModule)
        module.device = torch.device("cpu")
        module.location_encoding_scope = "all_visual"
        module.location_encoding_scale_init = 0.1
        module.location_encoding_learned_scale = True
        module.num_non_rgb_tokens = 2
        module.scene_location_encoding = SceneLocationEncoding(
            hidden_size=4,
            scale_init=0.1,
            learned_scale=True,
        )
        module._print = lambda *args, **kwargs: None
        module._location_encoding_norm_logged = False
        return module

    def test_all_visual_scope_changes_only_rgb_and_s1s2_content(self):
        module = self._build_module()
        inputs_embeds = torch.zeros(1, 7, 4)
        kwargs = {
            "inputs_embeds": inputs_embeds.clone(),
            "visual_pos_masks": torch.tensor(
                [[False, False, False, False, True, True, False]]
            ),
        }
        encoding_state = {
            "lat": torch.tensor([48.0], dtype=torch.float64),
            "lon": torch.tensor([12.0], dtype=torch.float64),
        }
        non_rgb_state = {
            "insert_positions": torch.tensor([1]),
        }

        module._apply_scene_location_encoding(
            kwargs,
            encoding_state=encoding_state,
            non_rgb_state=non_rgb_state,
        )

        changed = kwargs["inputs_embeds"].ne(inputs_embeds).any(dim=-1)
        expected = torch.tensor(
            [[False, True, True, False, True, True, False]]
        )
        self.assertTrue(torch.equal(changed, expected))
        self.assertTrue(
            torch.equal(
                kwargs["inputs_embeds"][0, 1],
                kwargs["inputs_embeds"][0, 4],
            )
        )

    def test_s1s2_scope_changes_only_s1s2_content(self):
        module = self._build_module()
        module.location_encoding_scope = "s1s2"
        inputs_embeds = torch.zeros(1, 7, 4)
        kwargs = {
            "inputs_embeds": inputs_embeds.clone(),
            "visual_pos_masks": torch.tensor(
                [[False, False, False, False, True, True, False]]
            ),
        }

        module._apply_scene_location_encoding(
            kwargs,
            encoding_state={
                "lat": torch.tensor([48.0], dtype=torch.float64),
                "lon": torch.tensor([12.0], dtype=torch.float64),
            },
            non_rgb_state={"insert_positions": torch.tensor([1])},
        )

        changed = kwargs["inputs_embeds"].ne(inputs_embeds).any(dim=-1)
        expected = torch.tensor(
            [[False, True, True, False, False, False, False]]
        )
        self.assertTrue(torch.equal(changed, expected))

    def test_s1s2_scope_requires_s1s2_conditioning(self):
        module = self._build_module()
        module.location_encoding_scope = "s1s2"

        with self.assertRaisesRegex(ValueError, "requires enabled S1/S2"):
            module._apply_scene_location_encoding(
                {
                    "inputs_embeds": torch.zeros(1, 3, 4),
                    "visual_pos_masks": torch.tensor([[False, True, False]]),
                },
                encoding_state={
                    "lat": torch.tensor([48.0]),
                    "lon": torch.tensor([12.0]),
                },
                non_rgb_state=None,
            )

    def test_loc_encoding_itself_does_not_insert_tokens(self):
        module = _build_encoder_test_module()
        module.loc_mode = "loc_encoding"
        module.non_rgb_conditioning = "disabled"
        original_ids = torch.tensor([[1, 2, 3]])
        original_labels = torch.tensor([[11, 12, 13]])
        batch = {
            "input_ids": original_ids.clone(),
            "labels": original_labels.clone(),
            "lat": torch.tensor([48.0], dtype=torch.float64),
            "lon": torch.tensor([12.0], dtype=torch.float64),
        }

        model_batch, _, _, _, _, _ = module._prepare_model_inputs(batch)

        self.assertTrue(torch.equal(model_batch["input_ids"], original_ids))
        self.assertTrue(torch.equal(model_batch["labels"], original_labels))
        self.assertIsNotNone(module._location_encoding_state)
        self.assertIsNone(module._location_insertion_state)

    def test_satclip_additive_conditioning_does_not_insert_tokens(self):
        module = _build_encoder_test_module()
        module.loc_mode = "loc_additive_satclip"
        module.non_rgb_conditioning = "disabled"
        original_ids = torch.tensor([[1, 2, 3]])
        batch = {
            "input_ids": original_ids.clone(),
            "lat": torch.tensor([48.0], dtype=torch.float64),
            "lon": torch.tensor([12.0], dtype=torch.float64),
        }

        model_batch, _, _, _, _, _ = module._prepare_model_inputs(batch)

        self.assertTrue(torch.equal(model_batch["input_ids"], original_ids))
        self.assertIsNotNone(module._location_encoding_state)
        self.assertIsNone(module._location_insertion_state)

    def test_projected_s1s2_tokens_are_replaced_before_encoding_is_added(self):
        module = self._build_module()
        module._location_insertion_state = None
        module._location_encoding_state = {
            "lat": torch.tensor([48.0], dtype=torch.float64),
            "lon": torch.tensor([12.0], dtype=torch.float64),
        }
        module._non_rgb_insertion_state = {
            "tensor": torch.ones(1, 12, 2, 2),
            "bands": None,
            "insert_positions": torch.tensor([1]),
        }
        module.non_rgb_encoder = _EvalCallable(
            lambda imagery, bands: torch.zeros(1, 2, 5)
        )
        module.non_rgb_modality_projection = lambda features: torch.full(
            (1, 2, 4), 10.0
        )
        kwargs = {
            "inputs_embeds": torch.zeros(1, 6, 4),
            "visual_pos_masks": torch.tensor(
                [[False, False, False, False, True, False]]
            ),
        }
        expected_geo = module.scene_location_encoding(
            module._location_encoding_state["lat"],
            module._location_encoding_state["lon"],
        )[0]

        module._decoder_input_conditioning_hook(None, (), kwargs)

        self.assertTrue(
            torch.allclose(kwargs["inputs_embeds"][0, 1], 10.0 + expected_geo)
        )
        self.assertTrue(
            torch.allclose(kwargs["inputs_embeds"][0, 2], 10.0 + expected_geo)
        )
        self.assertTrue(
            torch.allclose(kwargs["inputs_embeds"][0, 4], expected_geo)
        )

    def test_filled_generation_cache_skips_location_addition(self):
        module = self._build_module()
        module._location_insertion_state = None
        module._non_rgb_insertion_state = None
        module._location_encoding_state = {
            "lat": torch.tensor([48.0], dtype=torch.float64),
            "lon": torch.tensor([12.0], dtype=torch.float64),
        }

        class FakeCache:
            def get_seq_length(self):
                return 3

        kwargs = {
            "inputs_embeds": torch.zeros(1, 1, 4),
            "visual_pos_masks": torch.zeros(1, 1, dtype=torch.bool),
            "past_key_values": FakeCache(),
        }

        module._decoder_input_conditioning_hook(None, (), kwargs)

        self.assertTrue(torch.equal(kwargs["inputs_embeds"], torch.zeros(1, 1, 4)))

    def test_visual_scope_requires_exact_native_visual_mask(self):
        module = self._build_module()

        with self.assertRaisesRegex(ValueError, "visual_pos_masks"):
            module._apply_scene_location_encoding(
                {"inputs_embeds": torch.zeros(1, 3, 4)},
                encoding_state={
                    "lat": torch.tensor([48.0]),
                    "lon": torch.tensor([12.0]),
                },
                non_rgb_state=None,
            )

    def test_artifact_load_validates_manifest_and_restores_scale(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            saved = self._build_module()
            saved.scene_location_encoding.scale.data.fill_(0.35)
            save_file(
                saved.scene_location_encoding.state_dict(),
                Path(tmpdir) / "location_encoding.safetensors",
            )
            manifest = saved.get_scene_location_encoding_manifest()
            (Path(tmpdir) / "location_encoding_config.json").write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )

            loaded = self._build_module()
            loaded.adapter_dir = tmpdir
            loaded.scene_location_encoding.scale.data.zero_()
            loaded._load_scene_location_encoding_artifacts()

            self.assertAlmostEqual(
                float(loaded.scene_location_encoding.scale.detach()),
                0.35,
            )

            loaded.location_encoding_scope = "s1s2"
            with self.assertRaisesRegex(ValueError, "does not match"):
                loaded._load_scene_location_encoding_artifacts()

    def test_scale_uses_base_learning_rate_without_weight_decay(self):
        module = self._build_module()
        module.model = torch.nn.Linear(4, 4)
        module.location_modality_projection = None
        module.non_rgb_modality_projection = None
        module.learning_rate = 2e-4
        module.location_projection_lr_multiplier = 5.0
        module.non_rgb_projection_lr_multiplier = 5.0
        module.weight_decay = 0.01
        module.max_steps = 10
        module.warmup_ratio = 0.1
        module._trainer_or_none = lambda: None

        original_optimizer = qwen3_module.bnb.optim.AdamW8bit
        try:
            qwen3_module.bnb.optim.AdamW8bit = torch.optim.AdamW
            optimizer_config = module.configure_optimizers()
        finally:
            qwen3_module.bnb.optim.AdamW8bit = original_optimizer

        optimizer = optimizer_config["optimizer"]
        location_group = next(
            group
            for group in optimizer.param_groups
            if group["name"] == "location_encoding_no_decay"
        )
        self.assertIs(
            location_group["params"][0],
            module.scene_location_encoding.scale,
        )
        self.assertEqual(location_group["initial_lr"], module.learning_rate)
        self.assertEqual(location_group["weight_decay"], 0.0)


class ProjectedAdditiveLocationIntegrationTest(unittest.TestCase):
    @staticmethod
    def _build_direct_module() -> Qwen3VLModule:
        module = object.__new__(Qwen3VLModule)
        module.device = torch.device("cpu")
        module.loc_mode = "loc_encoding"
        module.location_encoding_scope = "s1s2"
        module.location_encoding_scale_init = 0.1
        module.location_encoding_feature_dim = 4
        module.num_non_rgb_tokens = 2
        module.scene_location_encoding = None
        module.scene_location_features = SceneLocationFeatures(4)
        module.additive_location_projection = AdditiveLocationProjection(4, 4)
        module._print = lambda *args, **kwargs: None
        module._location_encoding_norm_logged = False
        return module

    def test_direct_projection_changes_only_s1s2_tokens(self):
        module = self._build_direct_module()
        kwargs = {
            "inputs_embeds": torch.zeros(1, 7, 4),
            "visual_pos_masks": torch.tensor(
                [[False, False, False, False, True, True, False]]
            ),
        }

        module._apply_scene_location_encoding(
            kwargs,
            encoding_state={
                "lat": torch.tensor([48.0]),
                "lon": torch.tensor([12.0]),
            },
            non_rgb_state={"insert_positions": torch.tensor([1])},
        )

        changed = kwargs["inputs_embeds"].ne(0).any(dim=-1)
        self.assertTrue(
            torch.equal(
                changed,
                torch.tensor(
                    [[False, True, True, False, False, False, False]]
                ),
            )
        )

    def test_satclip_receives_longitude_then_latitude(self):
        module = self._build_direct_module()
        module.loc_mode = "loc_additive_satclip"
        module.scene_location_features = None
        captured = {}

        class FakeSatclip:
            checkpoint_metadata = {"embed_dim": 4}

            def eval(self):
                return self

            def __call__(self, coordinates):
                captured["coordinates"] = coordinates.clone()
                return torch.cat([coordinates, coordinates], dim=-1)

        module.satclip = FakeSatclip()
        kwargs = {
            "inputs_embeds": torch.zeros(1, 5, 4),
            "visual_pos_masks": torch.tensor(
                [[False, False, False, True, False]]
            ),
        }

        module._apply_scene_location_encoding(
            kwargs,
            encoding_state={
                "lat": torch.tensor([48.0]),
                "lon": torch.tensor([12.0]),
            },
            non_rgb_state={"insert_positions": torch.tensor([1])},
        )

        torch.testing.assert_close(
            captured["coordinates"],
            torch.tensor([[12.0, 48.0]], dtype=torch.float64),
        )
        changed = kwargs["inputs_embeds"].ne(0).any(dim=-1)
        self.assertTrue(
            torch.equal(
                changed,
                torch.tensor(
                    [[False, True, True, False, False]],
                ),
            )
        )

    def test_frozen_side_encoders_are_forced_to_eval_mode(self):
        module = self._build_direct_module()

        class DropoutSatclip(torch.nn.Module):
            def forward(self, coordinates):
                features = torch.cat([coordinates, coordinates], dim=-1).float()
                return torch.nn.functional.dropout(
                    features,
                    p=0.9,
                    training=self.training,
                )

        class DropoutNonRGB(torch.nn.Module):
            def forward(self, imagery, bands):
                return torch.nn.functional.dropout(
                    imagery,
                    p=0.9,
                    training=self.training,
                )

        module.satclip = DropoutSatclip().train()
        module.non_rgb_encoder = DropoutNonRGB().train()
        lat = torch.tensor([48.0])
        lon = torch.tensor([12.0])
        imagery = torch.ones(1, 2, 2)

        first_location = module._encode_satclip_coordinates(lat, lon)
        second_location = module._encode_satclip_coordinates(lat, lon)
        first_imagery = module._encode_non_rgb_imagery(imagery, None)
        second_imagery = module._encode_non_rgb_imagery(imagery, None)

        self.assertFalse(module.satclip.training)
        self.assertFalse(module.non_rgb_encoder.training)
        self.assertTrue(torch.equal(first_location, second_location))
        self.assertTrue(torch.equal(first_imagery, second_imagery))

    def test_artifact_manifest_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            saved = self._build_direct_module()
            saved.additive_location_projection.projection.weight.data.fill_(0.25)
            save_file(
                saved.additive_location_projection.state_dict(),
                Path(tmpdir) / "additive_location_projection.safetensors",
            )
            manifest = saved.get_additive_location_projection_manifest()
            (Path(tmpdir) / "additive_location_projection_config.json").write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )

            loaded = self._build_direct_module()
            loaded.adapter_dir = tmpdir
            loaded.additive_location_projection.projection.weight.data.zero_()
            loaded._load_additive_location_projection_artifacts()
            torch.testing.assert_close(
                loaded.additive_location_projection.projection.weight,
                torch.full_like(
                    loaded.additive_location_projection.projection.weight,
                    0.25,
                ),
            )

            loaded.location_encoding_scope = "all_visual"
            with self.assertRaisesRegex(ValueError, "does not match"):
                loaded._load_additive_location_projection_artifacts()

    def test_bridge_uses_location_projection_decay_group(self):
        module = self._build_direct_module()
        module.model = torch.nn.Linear(4, 4)
        module.location_modality_projection = None
        module.non_rgb_modality_projection = None
        module.learning_rate = 2e-4
        module.location_projection_lr_multiplier = 5.0
        module.non_rgb_projection_lr_multiplier = 5.0
        module.weight_decay = 0.01
        module.max_steps = 10
        module.warmup_ratio = 0.1
        module._trainer_or_none = lambda: None

        original_optimizer = qwen3_module.bnb.optim.AdamW8bit
        try:
            qwen3_module.bnb.optim.AdamW8bit = torch.optim.AdamW
            optimizer_config = module.configure_optimizers()
        finally:
            qwen3_module.bnb.optim.AdamW8bit = original_optimizer

        location_group = next(
            group
            for group in optimizer_config["optimizer"].param_groups
            if group["name"] == "location_projection_decay"
        )
        self.assertEqual(location_group["initial_lr"], 1e-3)
        self.assertEqual(location_group["weight_decay"], 0.01)
        self.assertIn(
            module.additive_location_projection.projection.weight,
            location_group["params"],
        )


if __name__ == "__main__":
    unittest.main()
