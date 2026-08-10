import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path

import torch

from src.models.scene_location_encoding import SceneLocationEncoding


def _install_callback_test_stubs():
    lightning = types.ModuleType("lightning")

    class LightningModule:
        def print(self, *args, **kwargs):
            return None

    class Trainer:
        pass

    lightning.LightningModule = LightningModule
    lightning.Trainer = Trainer
    sys.modules["lightning"] = lightning

    pytorch = types.ModuleType("lightning.pytorch")
    callbacks = types.ModuleType("lightning.pytorch.callbacks")

    class Callback:
        pass

    callbacks.Callback = Callback
    sys.modules["lightning.pytorch"] = pytorch
    sys.modules["lightning.pytorch.callbacks"] = callbacks


_install_callback_test_stubs()
callback_module = importlib.import_module("src.callbacks.save_qlora_adapters")
SaveQLoRAAdaptersCallback = callback_module.SaveQLoRAAdaptersCallback


class _FakeModel:
    def save_pretrained(self, save_dir):
        Path(save_dir, "adapter_model.safetensors").write_text("adapter")


class _FakeTokenizer:
    def save_pretrained(self, save_dir):
        Path(save_dir, "tokenizer.json").write_text("{}")


class SaveQLoRAAdaptersCallbackTest(unittest.TestCase):
    def test_validation_save_writes_adapter_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            callback = SaveQLoRAAdaptersCallback(dirpath="unused", best_dirpath=tmpdir)
            trainer = types.SimpleNamespace(
                callback_metrics={"val/loss": torch.tensor(0.5)},
                sanity_checking=False,
            )
            pl_module = types.SimpleNamespace(
                model=_FakeModel(),
                tokenizer=_FakeTokenizer(),
                location_modality_projection=torch.nn.Linear(2, 3),
                non_rgb_modality_projection=torch.nn.Linear(4, 5),
                scene_location_encoding=SceneLocationEncoding(8),
                additive_location_projection=torch.nn.Linear(6, 7, bias=False),
                get_scene_location_encoding_manifest=lambda: {
                    "version": 1,
                    "scope": "all_visual",
                },
                get_additive_location_projection_manifest=lambda: {
                    "version": 1,
                    "feature_source": "direct",
                    "scope": "s1s2",
                },
                print=lambda *args, **kwargs: None,
            )

            callback.on_validation_epoch_end(trainer, pl_module)

            self.assertTrue((Path(tmpdir) / "adapter_model.safetensors").exists())
            self.assertTrue((Path(tmpdir) / "tokenizer.json").exists())
            self.assertTrue((Path(tmpdir) / "location_modality_projection.safetensors").exists())
            self.assertTrue((Path(tmpdir) / "non_rgb_modality_projection.safetensors").exists())
            self.assertTrue((Path(tmpdir) / "location_encoding.safetensors").exists())
            self.assertTrue((Path(tmpdir) / "location_encoding_config.json").exists())
            self.assertTrue(
                (Path(tmpdir) / "additive_location_projection.safetensors").exists()
            )
            self.assertTrue(
                (Path(tmpdir) / "additive_location_projection_config.json").exists()
            )
            self.assertEqual(callback.best_score, 0.5)

    def test_train_end_saves_when_no_validation_metric_was_seen(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            callback = SaveQLoRAAdaptersCallback(dirpath=tmpdir)
            stale_paths = (
                "location_modality_projection.safetensors",
                "non_rgb_modality_projection.safetensors",
                "location_encoding.safetensors",
                "location_encoding_config.json",
                "additive_location_projection.safetensors",
                "additive_location_projection_config.json",
            )
            for filename in stale_paths:
                Path(tmpdir, filename).write_text("stale", encoding="utf-8")
            trainer = types.SimpleNamespace()
            pl_module = types.SimpleNamespace(
                model=_FakeModel(),
                tokenizer=_FakeTokenizer(),
                location_modality_projection=None,
                non_rgb_modality_projection=None,
                scene_location_encoding=None,
                additive_location_projection=None,
                print=lambda *args, **kwargs: None,
            )

            callback.on_train_end(trainer, pl_module)

            self.assertTrue((Path(tmpdir) / "adapter_model.safetensors").exists())
            self.assertTrue((Path(tmpdir) / "tokenizer.json").exists())
            for filename in stale_paths:
                self.assertFalse(Path(tmpdir, filename).exists())


if __name__ == "__main__":
    unittest.main()
