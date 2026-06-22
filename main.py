#!/usr/bin/env python3
"""Lightning CLI entrypoint for finetuning and evaluation."""

import os
from datetime import datetime
from pathlib import Path

import unsloth  # Must be imported before transformers for Unsloth optimizations
import lightning as L
import torch
from lightning.pytorch.cli import LightningArgumentParser, LightningCLI

# Import concrete classes so Lightning CLI can discover them via class_path.
from src.data_modules import BENTxTDataModule, GAIADataModule  # noqa: F401
from src.lightning_modules import Qwen3VLModule


class FinetuningCLI(LightningCLI):
    """Custom Lightning CLI for finetuning runs."""

    PATH_ARGUMENTS = (
        "output_dir",
        "adapter_dir",
        "bigearthnet_v2_lmdb_root",
        "bigearthnet_txt_parquet_path",
        "bigearthnet_encoder_dir",
        "location_redacted_caption_file",
        "satclip_checkpoint_path",
        "gaia_root",
    )

    LOC_MODE_TO_RUN_LABEL = {
        "no_loc": "no_loc",
        "loc_text": "loc_text",
        "loc_embed": "loc_embed",
    }

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        """Register config-only path aliases used by YAML interpolation."""
        for path_name in self.PATH_ARGUMENTS:
            parser.add_argument(f"--paths.{path_name}", type=str, default=None)

    @staticmethod
    def _uses_default_outputs_root(path_value: object) -> bool:
        if path_value is None:
            return False
        normalized = Path(str(path_value)).as_posix().rstrip("/")
        return normalized in {"outputs", "./outputs"}

    @staticmethod
    def _retarget_path(path_value: object, old_root: str, new_root: str) -> str:
        path = Path(str(path_value))
        old = Path(old_root)
        try:
            relative = path.relative_to(old)
        except ValueError:
            return str(path_value)
        return str(Path(new_root) / relative)

    def before_instantiate_classes(self) -> None:
        """Normalize generic output roots into dedicated per-run directories."""
        config = self.config.get(self.subcommand, self.config)
        trainer = getattr(config, "trainer", None)
        if trainer is None:
            return

        default_root_dir = getattr(trainer, "default_root_dir", None)
        if not self._uses_default_outputs_root(default_root_dir):
            return

        model_cfg = getattr(config, "model", None)
        model_args = getattr(model_cfg, "init_args", None)
        loc_mode = getattr(model_args, "loc_mode", "no_loc") if model_args is not None else "no_loc"
        run_label = self.LOC_MODE_TO_RUN_LABEL.get(str(loc_mode), str(loc_mode))
        non_rgb_conditioning = (
            getattr(model_args, "non_rgb_conditioning", "disabled")
            if model_args is not None
            else "disabled"
        )
        if str(non_rgb_conditioning) == "enabled":
            run_label = f"{run_label}_non_rgb"
        model_name = getattr(model_args, "model_name_or_path", "model") if model_args is not None else "model"
        model_slug = str(model_name).split("/")[-1].lower().replace(".", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path("outputs") / "runs" / f"{run_label}_{model_slug}_{timestamp}"
        run_dir_str = str(run_dir)

        old_root = str(default_root_dir)
        trainer.default_root_dir = run_dir_str

        callbacks = getattr(trainer, "callbacks", None)
        if callbacks:
            for callback in callbacks:
                init_args = getattr(callback, "init_args", None)
                dirpath = getattr(init_args, "dirpath", None) if init_args is not None else None
                if dirpath and self._uses_default_outputs_root(Path(dirpath).parent):
                    init_args.dirpath = self._retarget_path(dirpath, old_root, run_dir_str)

        loggers = getattr(trainer, "logger", None)
        if loggers:
            for logger in loggers:
                init_args = getattr(logger, "init_args", None)
                save_dir = getattr(init_args, "save_dir", None) if init_args is not None else None
                if save_dir and self._uses_default_outputs_root(save_dir):
                    init_args.save_dir = run_dir_str

    def before_fit(self) -> None:
        """Called before fit starts."""
        # Log configuration
        if self.trainer.is_global_zero:
            print("\n" + "=" * 60)
            print("Finetuning Configuration")
            print("=" * 60)

            # Model info (handle different model types)
            if hasattr(self.model, "model_name_or_path"):
                print(f"Model: {self.model.model_name_or_path}")
            if hasattr(self.model, "load_in_4bit"):
                print(f"4-bit Quantization: {self.model.load_in_4bit}")
            if hasattr(self.model, "lora_r"):
                print(f"LoRA rank: {self.model.lora_r}")
            if hasattr(self.model, "learning_rate"):
                print(f"Learning rate: {self.model.learning_rate}")

            # Trainer info
            print(f"Max steps: {self.trainer.max_steps}")
            print(f"Max epochs: {self.trainer.max_epochs}")
            print(f"Devices: {self.trainer.num_devices}")
            print(f"Precision: {self.trainer.precision}")
            print("=" * 60 + "\n")

def cli_main() -> None:
    """Run the CLI."""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_float32_matmul_precision("high")
    cli = FinetuningCLI(
        model_class=L.LightningModule,
        datamodule_class=L.LightningDataModule,
        subclass_mode_model=True,
        subclass_mode_data=True,
        seed_everything_default=42,
        parser_kwargs={
            "default_config_files": [],
            "parser_mode": "omegaconf",  # Support OmegaConf syntax in configs
        },
        save_config_kwargs={
            "overwrite": True,
        },
        run=True,
    )


def main() -> None:
    """Run the training CLI."""
    cli_main()


if __name__ == "__main__":
    main()
