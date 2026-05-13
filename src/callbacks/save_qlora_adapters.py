from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import Callback
from safetensors.torch import save_file


class SaveQLoRAAdaptersCallback(Callback):
    """Save the adapter directory and optional projection modules for the best run."""

    def __init__(
        self,
        dirpath: str,
        monitor: str = "val/loss",
        mode: str = "min",
    ) -> None:
        super().__init__()
        self.dirpath = Path(dirpath)
        self.monitor = monitor
        self.mode = mode
        self.best_score: float | None = None
        self._saved_once = False

    def _is_better(self, current: float) -> bool:
        if self.best_score is None:
            return True
        if self.mode == "min":
            return current < self.best_score
        if self.mode == "max":
            return current > self.best_score
        raise ValueError(f"Unsupported mode: {self.mode}")

    def _save_adapters(self, pl_module: L.LightningModule) -> None:
        self.dirpath.mkdir(parents=True, exist_ok=True)
        pl_module.model.save_pretrained(self.dirpath)
        pl_module.tokenizer.save_pretrained(self.dirpath)

        projection_path = self.dirpath / "location_modality_projection.safetensors"
        location_projection = getattr(pl_module, "location_modality_projection", None)
        if location_projection is not None:
            save_file(location_projection.state_dict(), projection_path)
        elif projection_path.exists():
            projection_path.unlink()

        non_rgb_projection_path = self.dirpath / "non_rgb_modality_projection.safetensors"
        non_rgb_projection = getattr(pl_module, "non_rgb_modality_projection", None)
        if non_rgb_projection is not None:
            save_file(non_rgb_projection.state_dict(), non_rgb_projection_path)
        elif non_rgb_projection_path.exists():
            non_rgb_projection_path.unlink()

        self._saved_once = True
        pl_module.print(f"Saved QLoRA adapters to {self.dirpath}")

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if trainer.sanity_checking:
            return

        current = trainer.callback_metrics.get(self.monitor)
        if current is None:
            return

        current_value = float(current)
        if self._is_better(current_value):
            self.best_score = current_value
            self._save_adapters(pl_module)
            pl_module.print(f"Updated best {self.monitor}: {current_value:.4f}")

    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if not self._saved_once:
            self._save_adapters(pl_module)
