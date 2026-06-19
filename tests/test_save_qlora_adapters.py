import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path

import torch


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
                print=lambda *args, **kwargs: None,
            )

            callback.on_validation_epoch_end(trainer, pl_module)

            self.assertTrue((Path(tmpdir) / "adapter_model.safetensors").exists())
            self.assertTrue((Path(tmpdir) / "tokenizer.json").exists())
            self.assertTrue((Path(tmpdir) / "location_modality_projection.safetensors").exists())
            self.assertTrue((Path(tmpdir) / "non_rgb_modality_projection.safetensors").exists())
            self.assertEqual(callback.best_score, 0.5)

    def test_train_end_saves_when_no_validation_metric_was_seen(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            callback = SaveQLoRAAdaptersCallback(dirpath=tmpdir)
            trainer = types.SimpleNamespace()
            pl_module = types.SimpleNamespace(
                model=_FakeModel(),
                tokenizer=_FakeTokenizer(),
                location_modality_projection=None,
                non_rgb_modality_projection=None,
                print=lambda *args, **kwargs: None,
            )

            callback.on_train_end(trainer, pl_module)

            self.assertTrue((Path(tmpdir) / "adapter_model.safetensors").exists())
            self.assertTrue((Path(tmpdir) / "tokenizer.json").exists())
            self.assertFalse((Path(tmpdir) / "location_modality_projection.safetensors").exists())
            self.assertFalse((Path(tmpdir) / "non_rgb_modality_projection.safetensors").exists())


if __name__ == "__main__":
    unittest.main()
