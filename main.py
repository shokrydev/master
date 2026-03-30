#!/usr/bin/env python3
"""Lightning CLI entrypoint for VLM training, evaluation, and export."""

import os
from datetime import datetime
from pathlib import Path

import unsloth  # Must be imported before transformers for Unsloth optimizations

import lightning as L
from lightning.pytorch.cli import LightningCLI, LightningArgumentParser

# Import modules for CLI auto-discovery
from src.lightning_modules import Qwen3VLModule
from src.data_modules import VLMDataModule


class VLMTrainingCLI(LightningCLI):
    """Custom Lightning CLI for VLM training."""

    LOC_MODE_TO_RUN_LABEL = {
        "none": "baseline",
        "text": "loc_text",
        "encoder": "loc_embed",
    }

    @staticmethod
    def _is_generic_outputs_root(path_value: object) -> bool:
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
        if not self._is_generic_outputs_root(default_root_dir):
            return

        model_cfg = getattr(config, "model", None)
        model_args = getattr(model_cfg, "init_args", None)
        loc_mode = getattr(model_args, "loc_mode", "run") if model_args is not None else "run"
        run_label = self.LOC_MODE_TO_RUN_LABEL.get(str(loc_mode), str(loc_mode))
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
                if dirpath and self._is_generic_outputs_root(Path(dirpath).parent):
                    init_args.dirpath = self._retarget_path(dirpath, old_root, run_dir_str)

        loggers = getattr(trainer, "logger", None)
        if loggers:
            for logger in loggers:
                init_args = getattr(logger, "init_args", None)
                save_dir = getattr(init_args, "save_dir", None) if init_args is not None else None
                if save_dir and self._is_generic_outputs_root(save_dir):
                    init_args.save_dir = run_dir_str

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        """Register project-specific export arguments."""
        parser.add_argument(
            "--export_gguf",
            type=str,
            default=None,
            help="Export trained model to GGUF format at specified path",
        )
        parser.add_argument(
            "--export_gguf_quantization",
            type=str,
            default="q4_k_m",
            choices=["f16", "f32", "q8_0", "q4_k_m", "q5_k_m", "q2_k"],
            help="GGUF quantization method",
        )
        parser.add_argument(
            "--export_merged",
            type=str,
            default=None,
            help="Export merged model (LoRA merged into base) at specified path",
        )
        parser.add_argument(
            "--push_to_hub",
            type=str,
            default=None,
            help="Push model to HuggingFace Hub (provide repo_id)",
        )

    def before_fit(self) -> None:
        """Called before fit starts."""
        # Log configuration
        if self.trainer.is_global_zero:
            print("\n" + "=" * 60)
            print("VLM Training Configuration")
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

    def after_fit(self) -> None:
        """Called after fit completes."""
        if not self.trainer.is_global_zero:
            return

        # Handle exports — self.model is the LightningModule (Qwen3VLModule),
        # the Unsloth model and tokenizer live on it as attributes.
        config = self.config.get(self.subcommand, self.config)
        unsloth_model = self.model.model
        tokenizer = self.model.tokenizer

        # Export to GGUF
        export_gguf = getattr(config, "export_gguf", None)
        if export_gguf and hasattr(unsloth_model, "save_pretrained_gguf"):
            quantization = getattr(config, "export_gguf_quantization", "q4_k_m")
            print(f"\nExporting model to GGUF: {export_gguf}")
            unsloth_model.save_pretrained_gguf(export_gguf, tokenizer, quantization_method=quantization)

        # Export merged model
        export_merged = getattr(config, "export_merged", None)
        if export_merged and hasattr(unsloth_model, "save_pretrained_merged"):
            print(f"\nExporting merged model: {export_merged}")
            unsloth_model.save_pretrained_merged(export_merged, tokenizer)

        # Push to hub (merged model)
        push_to_hub = getattr(config, "push_to_hub", None)
        if push_to_hub and hasattr(unsloth_model, "push_to_hub_merged"):
            print(f"\nPushing merged model to HuggingFace Hub: {push_to_hub}")
            unsloth_model.push_to_hub_merged(push_to_hub, tokenizer=tokenizer)


def cli_main() -> None:
    """Run the CLI."""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    cli = VLMTrainingCLI(
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
