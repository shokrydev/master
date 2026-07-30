#!/usr/bin/env python3
# ruff: noqa: I001
"""Lightning CLI entrypoint for finetuning and evaluation."""

import os

import unsloth  # noqa: F401  # Must precede Lightning/Transformers for patching
import lightning as L
import torch
from lightning.pytorch.cli import LightningArgumentParser, LightningCLI

# Import concrete classes so Lightning CLI can discover them via class_path.
from src.data_modules import BENTxTDataModule  # noqa: F401
from src.lightning_modules import Qwen3VLModule  # noqa: F401


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
    )

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        """Register config-only path aliases used by YAML interpolation."""
        for path_name in self.PATH_ARGUMENTS:
            parser.add_argument(f"--paths.{path_name}", type=str, default=None)

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
    FinetuningCLI(
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
