import json
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import Callback
from safetensors.torch import save_file


class SaveQLoRAAdaptersCallback(Callback):
    """Save the QLoRA adapter bundle and optional projection modules."""

    def __init__(
        self,
        dirpath: str,
        best_dirpath: str | None = None,
        monitor: str = "val/loss",
        mode: str = "min",
    ) -> None:
        super().__init__()
        self.dirpath = Path(dirpath)
        self.best_dirpath = Path(best_dirpath) if best_dirpath is not None else None
        self.monitor = monitor
        self.mode = mode
        if self.mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")
        self.best_score: float | None = None

    def _is_better(self, current: float) -> bool:
        if self.best_score is None:
            return True
        if self.mode == "min":
            return current < self.best_score
        return current > self.best_score

    def _save_adapters(self, pl_module: L.LightningModule, dirpath: Path) -> None:
        dirpath.mkdir(parents=True, exist_ok=True)
        pl_module.model.save_pretrained(dirpath)
        pl_module.tokenizer.save_pretrained(dirpath)

        projection_path = dirpath / "location_modality_projection.safetensors"
        location_projection = getattr(pl_module, "location_modality_projection", None)
        if location_projection is not None:
            save_file(location_projection.state_dict(), projection_path)
        elif projection_path.exists():
            projection_path.unlink()

        non_rgb_projection_path = dirpath / "non_rgb_modality_projection.safetensors"
        non_rgb_projection = getattr(pl_module, "non_rgb_modality_projection", None)
        if non_rgb_projection is not None:
            save_file(non_rgb_projection.state_dict(), non_rgb_projection_path)
        elif non_rgb_projection_path.exists():
            non_rgb_projection_path.unlink()

        location_encoding_path = dirpath / "location_encoding.safetensors"
        location_encoding_manifest_path = (
            dirpath / "location_encoding_config.json"
        )
        location_encoding = getattr(pl_module, "scene_location_encoding", None)
        if location_encoding is not None:
            manifest = pl_module.get_scene_location_encoding_manifest()
            if manifest is None:
                raise RuntimeError(
                    "Scene-location encoding is active but its manifest is missing"
                )
            save_file(location_encoding.state_dict(), location_encoding_path)
            location_encoding_manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        else:
            if location_encoding_path.exists():
                location_encoding_path.unlink()
            if location_encoding_manifest_path.exists():
                location_encoding_manifest_path.unlink()

        additive_projection_path = (
            dirpath / "additive_location_projection.safetensors"
        )
        additive_manifest_path = (
            dirpath / "additive_location_projection_config.json"
        )
        additive_projection = getattr(
            pl_module,
            "additive_location_projection",
            None,
        )
        if additive_projection is not None:
            manifest = pl_module.get_additive_location_projection_manifest()
            if manifest is None:
                raise RuntimeError(
                    "Additive location projection is active but its manifest "
                    "is missing"
                )
            save_file(additive_projection.state_dict(), additive_projection_path)
            additive_manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        else:
            if additive_projection_path.exists():
                additive_projection_path.unlink()
            if additive_manifest_path.exists():
                additive_manifest_path.unlink()

        pl_module.print(f"Saved QLoRA adapter bundle to {dirpath}")

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if self.best_dirpath is None:
            return
        if trainer.sanity_checking:
            return

        current = trainer.callback_metrics.get(self.monitor)
        if current is None:
            return

        current_value = float(current)
        if self._is_better(current_value):
            self.best_score = current_value
            self._save_adapters(pl_module, self.best_dirpath)
            pl_module.print(f"Updated best {self.monitor}: {current_value:.4f}")

    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._save_adapters(pl_module, self.dirpath)
