# LightningModule for Qwen3-VL Vision-Language Model Finetuning with Unsloth
# Docs: https://lightning.ai/docs/pytorch/stable/common/lightning_module.html

import json
import math
from pathlib import Path
from typing import Any, Literal

import bitsandbytes as bnb
import lightning as L
import torch
from safetensors.torch import load_file
from torch.optim.lr_scheduler import LambdaLR
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator

from src.data_modules.geo_aware_collator import (
    GeoAwareCollator,
    ValidationGenerationCollator,
)

QWEN_MODALITY_PROJECTION_MODULES = [
    "merger",
    "deepstack_merger_list.0",
    "deepstack_merger_list.1",
    "deepstack_merger_list.2",
]

ADDITIVE_LOCATION_MODES = {"loc_encoding", "loc_additive_satclip"}


class Qwen3VLModule(L.LightningModule):
    """
    PyTorch Lightning Module for finetuning Qwen3-VL using Unsloth.

    Uses Unsloth's FastVisionModel for:
    - 4-bit quantization (QLoRA)
    - Optimized gradient checkpointing
    - Memory-efficient training

    Training saves adapter artifacts through a Lightning callback.
    """

    def __init__(
        self,
        model_name_or_path: str = "unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit",
        adapter_dir: str | None = None,
        max_seq_length: int = 2048,
        lora_r: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        lora_target_modules: list[str] | None = None,
        modules_to_save: list[str] | None = None,
        finetune_vision_layers: bool = False,
        finetune_language_layers: bool = True,
        finetune_attention_modules: bool = True,
        finetune_mlp_modules: bool = True,
        learning_rate: float = 2e-4,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        max_steps: int | None = None,
        max_new_tokens: int = 256,
        validation_generation_sample_ids: list[str] | None = None,
        validation_generation_path: str | None = None,
        system_prompt: str | None = "You are a remote sensing image analysis assistant.",
        loc_mode: Literal[
            "no_loc",
            "loc_text",
            "loc_embed",
            "loc_encoding",
            "loc_additive_satclip",
        ] = "no_loc",
        location_text_template: str | None = None,
        coordinates_decimal_places: int = 0,
        location_embed_marker: str | None = None,
        location_encoding_scope: Literal["all_visual", "s1s2"] | None = None,
        location_encoding_projection: Literal["none", "linear"] = "none",
        location_encoding_feature_dim: int = 256,
        location_encoding_scale_init: float = 0.1,
        location_encoding_learned_scale: bool = True,
        non_rgb_conditioning: Literal["disabled", "enabled"] = "disabled",
        non_rgb_encoder_dir: str | None = None,
        non_rgb_encoder_feature_dim: int | None = None,
        non_rgb_feature_mode: Literal["spatial_4x4", "pooled_prelogit"] = "spatial_4x4",
        non_rgb_spatial_pool_size: int = 4,
        num_non_rgb_tokens: int = 16,
        non_rgb_projection_lr_multiplier: float = 1.0,
        satclip_checkpoint: str | None = None,
        satclip_dim: int = 256,
        num_location_tokens: int = 1,
        location_projection_architecture: Literal["mlp", "linear"] = "mlp",
        location_projection_lr_multiplier: float = 1.0,
        prediction_export_path: str | None = None,
        run_label: str | None = None,
        model_size: str | None = None,
    ):
        """
        Initialize Qwen3-VL finetuning module.

        Args:
            model_name_or_path: Unsloth model ID (e.g., "unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit")
            adapter_dir: Saved adapter bundle directory for validate/test/predict
            max_seq_length: Maximum sequence length for training
            lora_r: LoRA rank
            lora_alpha: LoRA alpha scaling factor
            lora_dropout: LoRA dropout rate
            lora_target_modules: Target modules for LoRA (None = Unsloth native selection)
            modules_to_save: Modules to train in full bf16 (no LoRA). Defaults to
                Qwen's modality projection (`merger`) modules so the modality
                projection remains trainable.
            finetune_vision_layers: Whether to apply QLoRA to vision encoder
            finetune_language_layers: Whether to finetune language model
            finetune_attention_modules: Whether to finetune attention modules
            finetune_mlp_modules: Whether to finetune MLP modules
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            warmup_ratio: Warmup ratio for scheduler
            max_steps: Total training steps (for scheduler)
            max_new_tokens: Maximum tokens generated for validation examples and
                test predictions.
            validation_generation_sample_ids: Explicit sample IDs for qualitative
                validation generations.
            validation_generation_path: JSONL file for qualitative validation
                generations.
            system_prompt: Optional system message injected during chat formatting.
            loc_mode: Location conditioning mode ("no_loc", "loc_text",
                "loc_embed", "loc_encoding", "loc_additive_satclip").
            location_text_template: Format string appended to the user prompt when
                `loc_mode="loc_text"`. Coordinate formatting is handled by the
                shared collator.
            coordinates_decimal_places: Decimal places used for the compact
                `{location}` field in `location_text_template`.
            location_embed_marker: Text marker appended to the prompt before
                projected SatCLIP tokens when `loc_mode="loc_embed"`.
            location_encoding_scope: Existing embeddings that receive direct
                scene-coordinate encoding for additive location modes.
            location_encoding_projection: Whether direct `loc_encoding` uses
                its original fixed hidden-size basis or the shared linear
                alignment bridge.
            location_encoding_feature_dim: Input width of the shared additive
                location bridge.
            location_encoding_scale_init: Additive residual scale; it is the
                initialization when the legacy direct scale is learned and a
                fixed scale when a linear alignment bridge is used.
            location_encoding_learned_scale: Whether the encoding amplitude is
                trainable.
            non_rgb_conditioning: Whether non-RGB imagery conditions Qwen.
                "disabled" strips non-RGB imagery before Qwen; "enabled" activates
                the non-RGB encoder/projection path.
            non_rgb_encoder_dir: Local Hugging Face-style directory for the frozen
                non-RGB encoder. Expected to contain config.json and model.safetensors.
            non_rgb_encoder_feature_dim: Feature dimension returned by the frozen
                non-RGB encoder. Discovered from the encoder API rather than assumed.
            non_rgb_feature_mode: Feature extraction mode for the non-RGB encoder.
                "spatial_4x4" preserves a fixed 4x4 feature grid; "pooled_prelogit"
                uses the pooled MobileViT embedding before the classifier.
            non_rgb_spatial_pool_size: Spatial grid size for "spatial_4x4" mode.
            num_non_rgb_tokens: Number of projected non-RGB imagery tokens to insert.
            non_rgb_projection_lr_multiplier: Learning-rate multiplier for
                the randomly initialized S1/S2-to-Qwen projection.
            satclip_checkpoint: Path to the SatCLIP checkpoint required by
                SatCLIP-based location modes.
            satclip_dim: SatCLIP embedding dimension
            num_location_tokens: Number of location tokens to insert before the visual block (encoder mode)
            location_projection_architecture: SatCLIP-to-token projection:
                original ``mlp`` or compact ``linear``.
            location_projection_lr_multiplier: Learning-rate multiplier for
                the randomly initialized SatCLIP-to-Qwen projection.
            prediction_export_path: If set, stream per-sample test predictions to this JSONL path.
            run_label: Human-readable run label exported with each prediction.
            model_size: Model size label exported with each prediction, e.g. "2B".
        """
        super().__init__()

        if loc_mode not in {
            "no_loc",
            "loc_text",
            "loc_embed",
            "loc_encoding",
            "loc_additive_satclip",
        }:
            raise ValueError(f"Unsupported loc_mode: {loc_mode}")
        if non_rgb_conditioning not in {"disabled", "enabled"}:
            raise ValueError(f"Unsupported non_rgb_conditioning: {non_rgb_conditioning}")
        if loc_mode == "loc_text" and not location_text_template:
            raise ValueError("loc_mode='loc_text' requires location_text_template")
        if loc_mode != "loc_text" and location_text_template is not None:
            raise ValueError("location_text_template is only used when loc_mode='loc_text'")
        if coordinates_decimal_places < 0:
            raise ValueError("coordinates_decimal_places must be non-negative")
        if loc_mode != "loc_text" and coordinates_decimal_places != 0:
            raise ValueError("coordinates_decimal_places is only used when loc_mode='loc_text'")
        if loc_mode == "loc_embed" and not location_embed_marker:
            raise ValueError("loc_mode='loc_embed' requires location_embed_marker")
        if loc_mode != "loc_embed" and location_embed_marker is not None:
            raise ValueError("location_embed_marker is only used when loc_mode='loc_embed'")
        if location_projection_architecture not in {"mlp", "linear"}:
            raise ValueError(
                "location_projection_architecture must be 'mlp' or 'linear'"
            )
        if loc_mode != "loc_embed" and location_projection_architecture != "mlp":
            raise ValueError(
                "location_projection_architecture is only configurable when "
                "loc_mode='loc_embed'"
            )
        if (
            loc_mode in ADDITIVE_LOCATION_MODES
            and location_encoding_scope not in {"all_visual", "s1s2"}
        ):
            raise ValueError(
                "Additive location modes require location_encoding_scope to "
                "be 'all_visual' or 's1s2'"
            )
        if (
            loc_mode not in ADDITIVE_LOCATION_MODES
            and location_encoding_scope is not None
        ):
            raise ValueError(
                "location_encoding_scope is only used when "
                "loc_mode='loc_encoding' or 'loc_additive_satclip'"
            )
        if location_encoding_projection not in {"none", "linear"}:
            raise ValueError(
                "location_encoding_projection must be 'none' or 'linear'"
            )
        if (
            loc_mode == "loc_additive_satclip"
            and location_encoding_projection != "linear"
        ):
            raise ValueError(
                "loc_mode='loc_additive_satclip' requires "
                "location_encoding_projection='linear'"
            )
        if (
            loc_mode not in ADDITIVE_LOCATION_MODES
            and location_encoding_projection != "none"
        ):
            raise ValueError(
                "location_encoding_projection is only used by additive "
                "location modes"
            )
        if location_encoding_feature_dim <= 0:
            raise ValueError("location_encoding_feature_dim must be positive")
        if (
            loc_mode == "loc_encoding"
            and location_encoding_projection == "linear"
            and location_encoding_feature_dim % 4 != 0
        ):
            raise ValueError(
                "Projected direct location encoding requires "
                "location_encoding_feature_dim to be divisible by four"
            )
        if (
            location_encoding_projection == "linear"
            and location_encoding_learned_scale
        ):
            raise ValueError(
                "Projected additive location conditioning requires "
                "location_encoding_learned_scale=false because the trainable "
                "projection already controls amplitude"
            )
        if (
            loc_mode == "loc_additive_satclip"
            and satclip_dim != location_encoding_feature_dim
        ):
            raise ValueError(
                "loc_additive_satclip requires satclip_dim to equal "
                "location_encoding_feature_dim"
            )
        if (
            not math.isfinite(location_encoding_scale_init)
            or location_encoding_scale_init <= 0
        ):
            raise ValueError(
                "location_encoding_scale_init must be a finite positive number"
            )
        if location_projection_lr_multiplier <= 0:
            raise ValueError("location_projection_lr_multiplier must be positive")
        if non_rgb_projection_lr_multiplier <= 0:
            raise ValueError("non_rgb_projection_lr_multiplier must be positive")
        validation_generation_sample_ids = validation_generation_sample_ids or []
        if len(set(validation_generation_sample_ids)) != len(
            validation_generation_sample_ids
        ):
            raise ValueError(
                "validation_generation_sample_ids must not contain duplicates"
            )
        if validation_generation_sample_ids and not validation_generation_path:
            raise ValueError(
                "validation_generation_path is required when validation generation is enabled"
            )
        if validation_generation_path and not validation_generation_sample_ids:
            raise ValueError(
                "validation_generation_sample_ids is required when "
                "validation_generation_path is set"
            )
        if not 0.0 <= warmup_ratio < 1.0:
            raise ValueError("warmup_ratio must be in the interval [0, 1)")
        if non_rgb_feature_mode not in {"spatial_4x4", "pooled_prelogit"}:
            raise ValueError(f"Unsupported non_rgb_feature_mode: {non_rgb_feature_mode}")
        if non_rgb_spatial_pool_size <= 0:
            raise ValueError("non_rgb_spatial_pool_size must be positive")
        if (
            non_rgb_feature_mode == "spatial_4x4"
            and num_non_rgb_tokens != non_rgb_spatial_pool_size * non_rgb_spatial_pool_size
        ):
            raise ValueError(
                "num_non_rgb_tokens must equal non_rgb_spatial_pool_size ** 2 "
                "for spatial_4x4 mode"
            )

        self.save_hyperparameters()

        self.model_name_or_path = model_name_or_path
        self.adapter_dir = str(adapter_dir) if adapter_dir else None
        self.max_seq_length = max_seq_length
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules
        self.modules_to_save = modules_to_save
        self.finetune_vision_layers = finetune_vision_layers
        self.finetune_language_layers = finetune_language_layers
        self.finetune_attention_modules = finetune_attention_modules
        self.finetune_mlp_modules = finetune_mlp_modules
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.max_steps = max_steps
        self.max_new_tokens = max_new_tokens
        self.validation_generation_sample_ids = tuple(
            str(sample_id) for sample_id in validation_generation_sample_ids
        )
        self.validation_generation_path = (
            str(validation_generation_path) if validation_generation_path else None
        )
        self.system_prompt = system_prompt
        self.loc_mode = loc_mode
        self.location_text_template = location_text_template
        self.coordinates_decimal_places = coordinates_decimal_places
        self.location_embed_marker = location_embed_marker
        self.location_encoding_scope = location_encoding_scope
        self.location_encoding_projection = location_encoding_projection
        self.location_encoding_feature_dim = location_encoding_feature_dim
        self.location_encoding_scale_init = float(location_encoding_scale_init)
        self.location_encoding_learned_scale = location_encoding_learned_scale
        self.non_rgb_conditioning = non_rgb_conditioning
        self.non_rgb_encoder_dir = str(non_rgb_encoder_dir) if non_rgb_encoder_dir else None
        self.non_rgb_encoder_feature_dim = non_rgb_encoder_feature_dim
        self.non_rgb_feature_mode = non_rgb_feature_mode
        self.non_rgb_spatial_pool_size = non_rgb_spatial_pool_size
        self.num_non_rgb_tokens = num_non_rgb_tokens
        self.non_rgb_projection_lr_multiplier = non_rgb_projection_lr_multiplier
        self.satclip_checkpoint = satclip_checkpoint
        self.satclip_dim = satclip_dim
        self.num_location_tokens = num_location_tokens
        self.location_projection_architecture = location_projection_architecture
        self.location_projection_lr_multiplier = location_projection_lr_multiplier
        self.prediction_export_path = str(prediction_export_path) if prediction_export_path else None
        self.run_label = run_label
        self.model_size = model_size

        self.model = None
        self.tokenizer = None
        self._collator = None
        self._validation_collator = None
        self._test_collator = None

        # Projected side-modality components, initialized in setup when enabled.
        self.satclip = None
        self.location_modality_projection = None
        self.scene_location_encoding = None
        self.scene_location_features = None
        self.additive_location_projection = None
        self.non_rgb_encoder = None
        self.non_rgb_modality_projection = None
        self._decoder_input_hook_handle = None
        self._location_insertion_state = None
        self._location_encoding_state = None
        self._non_rgb_insertion_state = None
        self._location_encoding_norm_logged = False
        self._supervision_mask_validated = False

        self._prediction_export_count = 0

    def _trainer_or_none(self) -> Any | None:
        """Return the attached Trainer, or None for direct utility execution."""
        try:
            return self.trainer
        except RuntimeError as error:
            if "not attached to a `Trainer`" not in str(error):
                raise
            return None

    def _print(self, *args: Any, **kwargs: Any) -> None:
        """Print inside or outside a Lightning Trainer."""
        if self._trainer_or_none() is None:
            print(*args, **kwargs)
            return
        self.print(*args, **kwargs)

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            if self.satclip is not None:
                self.satclip.eval()
            if self.non_rgb_encoder is not None:
                self.non_rgb_encoder.eval()
        return self

    def setup(self, stage: str):
        """Load model with Unsloth and configure QLoRA."""
        if self.model is not None:
            self._set_datamodule_collator()
            return

        if stage == "fit" and self.adapter_dir:
            raise ValueError("adapter_dir cannot be set for fit; training starts from a base model")
        if stage in {"validate", "test", "predict"} and not self.adapter_dir:
            raise ValueError(
                f"{stage} requires model.init_args.adapter_dir to point at a saved adapter bundle"
            )

        model_source = self.adapter_dir or self.model_name_or_path

        self.model, self.tokenizer = FastVisionModel.from_pretrained(
            model_name=model_source,
            max_seq_length=self.max_seq_length,
            dtype=None,
            load_in_4bit=True,
            use_gradient_checkpointing="unsloth",
        )
        if self.adapter_dir is None:
            # These are Qwen's internal module names for the modality projection.
            # Keep the names unchanged so PEFT/Unsloth can find the pretrained modules.
            modules_to_save = self.modules_to_save
            if modules_to_save is None:
                modules_to_save = list(QWEN_MODALITY_PROJECTION_MODULES)

            self.model = FastVisionModel.get_peft_model(
                self.model,
                r=self.lora_r,
                target_modules=self.lora_target_modules,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                bias="none",
                use_gradient_checkpointing="unsloth",
                finetune_vision_layers=self.finetune_vision_layers,
                finetune_language_layers=self.finetune_language_layers,
                finetune_attention_modules=self.finetune_attention_modules,
                finetune_mlp_modules=self.finetune_mlp_modules,
                modules_to_save=modules_to_save if modules_to_save else None,
            )

        FastVisionModel.for_training(self.model)

        # Wrap collator with GeoAwareCollator
        base_collator = UnslothVisionDataCollator(
            self.model,
            self.tokenizer,
            train_on_responses_only=True,
            instruction_part="<|im_start|>user\n",
            response_part="<|im_start|>assistant\n",
        )
        location_prompt_template = None
        if self.loc_mode == "loc_text":
            location_prompt_template = self.location_text_template
        elif self.loc_mode == "loc_embed":
            location_prompt_template = self.location_embed_marker

        self._collator = GeoAwareCollator(
            base_collator,
            system_prompt=self.system_prompt,
            location_text_template=location_prompt_template,
            coordinates_decimal_places=self.coordinates_decimal_places,
        )
        if self.validation_generation_sample_ids:
            validation_generation_collator = GeoAwareCollator(
                base_collator,
                system_prompt=self.system_prompt,
                location_text_template=location_prompt_template,
                coordinates_decimal_places=self.coordinates_decimal_places,
                generation_prompt=True,
            )
            self._validation_collator = ValidationGenerationCollator(
                self._collator,
                validation_generation_collator,
                self.validation_generation_sample_ids,
            )
        if self.prediction_export_path:
            self._test_collator = GeoAwareCollator(
                base_collator,
                system_prompt=self.system_prompt,
                location_text_template=location_prompt_template,
                coordinates_decimal_places=self.coordinates_decimal_places,
                generation_prompt=True,
            )
        self._set_datamodule_collator()

        projected_additive_location = (
            self.loc_mode in ADDITIVE_LOCATION_MODES
            and self.location_encoding_projection == "linear"
        )
        if (
            projected_additive_location
            and self.non_rgb_conditioning == "enabled"
        ):
            # Keep the paired direct/SatCLIP experiment's randomly initialized
            # S1/S2 projection identical. SatCLIP construction consumes random
            # numbers before its checkpoint is loaded.
            self._setup_non_rgb_conditioning()
            if self.adapter_dir is not None:
                self._load_non_rgb_projection_artifacts()

        if self.loc_mode == "loc_embed":
            self._setup_loc_embed()
            if self.adapter_dir is not None:
                self._load_location_projection_artifacts()
        elif self.loc_mode in ADDITIVE_LOCATION_MODES:
            if self.location_encoding_projection == "linear":
                self._setup_projected_additive_location_conditioning()
                if self.adapter_dir is not None:
                    self._load_additive_location_projection_artifacts()
            else:
                self._setup_scene_location_encoding()
                if self.adapter_dir is not None:
                    self._load_scene_location_encoding_artifacts()
        if (
            self.non_rgb_conditioning == "enabled"
            and not projected_additive_location
        ):
            self._setup_non_rgb_conditioning()
            if self.adapter_dir is not None:
                self._load_non_rgb_projection_artifacts()

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        location_proj_params = 0
        location_encoding_params = 0
        additive_location_proj_params = 0
        non_rgb_proj_params = 0
        if self.location_modality_projection is not None:
            location_proj_params = sum(p.numel() for p in self.location_modality_projection.parameters())
            trainable_params += location_proj_params
            total_params += location_proj_params
        if getattr(self, "scene_location_encoding", None) is not None:
            location_encoding_params = sum(
                p.numel() for p in self.scene_location_encoding.parameters()
            )
            trainable_params += location_encoding_params
            total_params += location_encoding_params
        if getattr(self, "additive_location_projection", None) is not None:
            additive_location_proj_params = sum(
                p.numel() for p in self.additive_location_projection.parameters()
            )
            trainable_params += additive_location_proj_params
            total_params += additive_location_proj_params
        if self.non_rgb_modality_projection is not None:
            non_rgb_proj_params = sum(p.numel() for p in self.non_rgb_modality_projection.parameters())
            trainable_params += non_rgb_proj_params
            total_params += non_rgb_proj_params
        self._print(
            f"Trainable params: {trainable_params:,} / {total_params:,} "
            f"({100 * trainable_params / total_params:.2f}%)"
        )
        if self.location_modality_projection is not None:
            self._print(f"LocationModalityProjection params: {location_proj_params:,}")
            if self.location_projection_lr_multiplier != 1.0:
                location_lr = self.learning_rate * self.location_projection_lr_multiplier
                self._print(
                    "LocationModalityProjection LR: "
                    f"{location_lr:g} ({self.location_projection_lr_multiplier:g}x)"
                )
        if self.scene_location_encoding is not None:
            self._print(
                "SceneLocationEncoding params: "
                f"{location_encoding_params:,}; "
                f"scope={self.location_encoding_scope}; "
                f"scale_init={self.location_encoding_scale_init:g}; "
                f"learned_scale={self.location_encoding_learned_scale}"
            )
        if getattr(self, "additive_location_projection", None) is not None:
            self._print(
                "AdditiveLocationProjection params: "
                f"{additive_location_proj_params:,}; "
                f"source={self._additive_location_feature_source()}; "
                f"scope={self.location_encoding_scope}; "
                f"scale={self.location_encoding_scale_init:g}"
            )
            if self.location_projection_lr_multiplier != 1.0:
                location_lr = self.learning_rate * self.location_projection_lr_multiplier
                self._print(
                    "AdditiveLocationProjection LR: "
                    f"{location_lr:g} "
                    f"({self.location_projection_lr_multiplier:g}x)"
                )
        if self.non_rgb_modality_projection is not None:
            self._print(f"NonRGBModalityProjection params: {non_rgb_proj_params:,}")
            if self.non_rgb_projection_lr_multiplier != 1.0:
                non_rgb_lr = self.learning_rate * self.non_rgb_projection_lr_multiplier
                self._print(
                    "NonRGBModalityProjection LR: "
                    f"{non_rgb_lr:g} ({self.non_rgb_projection_lr_multiplier:g}x)"
                )

    def _get_text_hidden_size(self) -> int:
        """Return the Qwen text hidden size behind the PEFT wrapper."""
        inner_model = self.model
        if hasattr(inner_model, "base_model"):
            inner_model = inner_model.base_model
        if hasattr(inner_model, "model"):
            inner_model = inner_model.model
        config = inner_model.config
        if hasattr(config, "text_config") and hasattr(config.text_config, "hidden_size"):
            return int(config.text_config.hidden_size)
        return int(config.hidden_size)

    def _register_decoder_input_hook(self) -> None:
        """Register the shared decoder-input conditioning hook."""
        if self._decoder_input_hook_handle is not None:
            return
        language_model = self.model.base_model.model.model.language_model
        self._decoder_input_hook_handle = language_model.register_forward_pre_hook(
            self._decoder_input_conditioning_hook, with_kwargs=True
        )
        self._print(f"Registered projected token hook on {type(language_model).__name__}")

    def _load_location_projection_artifacts(self) -> None:
        """Load the saved location projection that lives outside the PEFT adapter package."""
        projection_path = Path(self.adapter_dir) / "location_modality_projection.safetensors"
        manifest_path = Path(self.adapter_dir) / "location_modality_projection_config.json"
        if not projection_path.is_file():
            raise FileNotFoundError(f"Missing location projection artifacts: {projection_path}")

        if manifest_path.is_file():
            actual_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            expected_manifest = self.get_location_projection_manifest()
            if actual_manifest != expected_manifest:
                raise ValueError(
                    "Location projection config does not match the saved adapter: "
                    f"expected {expected_manifest}, found {actual_manifest}"
                )
        elif getattr(self, "location_projection_architecture", "mlp") != "mlp":
            raise FileNotFoundError(
                "Compact location projection adapters require a manifest: "
                f"{manifest_path}"
            )

        state_dict = load_file(projection_path, device=str(self.device))
        self.location_modality_projection.load_state_dict(state_dict)

    def get_location_projection_manifest(self) -> dict[str, object] | None:
        if self.location_modality_projection is None:
            return None
        metadata = getattr(self.satclip, "checkpoint_metadata", None)
        if metadata is None:
            raise RuntimeError("Loaded SatCLIP encoder has no checkpoint metadata")
        manifest = self.location_modality_projection.manifest()
        manifest.update(
            {
                "feature_source": "satclip",
                "satclip": metadata,
                "coordinate_order": ["longitude", "latitude"],
                "location_embed_marker": self.location_embed_marker,
                "token_placement": "before_vision_start",
            }
        )
        return manifest

    def _load_non_rgb_projection_artifacts(self) -> None:
        """Load the saved non-RGB projection that lives outside the PEFT adapter package."""
        projection_path = Path(self.adapter_dir) / "non_rgb_modality_projection.safetensors"
        if not projection_path.is_file():
            raise FileNotFoundError(f"Missing non-RGB projection artifacts: {projection_path}")

        state_dict = load_file(projection_path, device=str(self.device))
        self.non_rgb_modality_projection.load_state_dict(state_dict)

    def get_scene_location_encoding_manifest(self) -> dict[str, object] | None:
        if self.scene_location_encoding is None:
            return None
        return self.scene_location_encoding.manifest(
            scope=self.location_encoding_scope,
        )

    def _load_scene_location_encoding_artifacts(self) -> None:
        encoding_path = Path(self.adapter_dir) / "location_encoding.safetensors"
        manifest_path = Path(self.adapter_dir) / "location_encoding_config.json"
        if not encoding_path.is_file():
            raise FileNotFoundError(
                f"Missing scene-location encoding artifact: {encoding_path}"
            )
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Missing scene-location encoding manifest: {manifest_path}"
            )

        actual_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_manifest = self.get_scene_location_encoding_manifest()
        if actual_manifest != expected_manifest:
            raise ValueError(
                "Scene-location encoding config does not match the saved adapter: "
                f"expected {expected_manifest}, found {actual_manifest}"
            )

        state_dict = load_file(encoding_path, device=str(self.device))
        self.scene_location_encoding.load_state_dict(state_dict)

    def _setup_scene_location_encoding(self) -> None:
        from src.models.scene_location_encoding import SceneLocationEncoding

        self.scene_location_encoding = SceneLocationEncoding(
            hidden_size=self._get_text_hidden_size(),
            scale_init=self.location_encoding_scale_init,
            learned_scale=self.location_encoding_learned_scale,
        ).to(self.device)
        self._register_decoder_input_hook()

    def _additive_location_feature_source(self) -> str:
        if self.loc_mode == "loc_encoding":
            return "direct"
        if self.loc_mode == "loc_additive_satclip":
            return "satclip"
        raise ValueError(
            f"Unsupported projected additive location mode: {self.loc_mode}"
        )

    def _additive_location_source_config(self) -> dict[str, object]:
        source = self._additive_location_feature_source()
        if source == "direct":
            return {
                "encoding_type": self.scene_location_features.encoding_type,
                "coordinate_order": ["latitude", "longitude"],
                "coordinate_units": "degrees",
                "coordinate_ranges": {
                    "latitude": [-90.0, 90.0],
                    "longitude": [-180.0, 180.0],
                },
            }
        metadata = getattr(self.satclip, "checkpoint_metadata", None)
        if metadata is None:
            raise RuntimeError("Loaded SatCLIP encoder has no checkpoint metadata")
        return {
            "coordinate_order": ["longitude", "latitude"],
            "coordinate_units": "degrees",
            "satclip": metadata,
        }

    def get_additive_location_projection_manifest(self) -> dict[str, object] | None:
        if self.additive_location_projection is None:
            return None
        return self.additive_location_projection.manifest(
            feature_source=self._additive_location_feature_source(),
            scope=self.location_encoding_scope,
            source_config=self._additive_location_source_config(),
        )

    def _setup_projected_additive_location_conditioning(self) -> None:
        from src.models.additive_location_projection import (
            AdditiveLocationProjection,
        )

        self.additive_location_projection = AdditiveLocationProjection(
            feature_dim=self.location_encoding_feature_dim,
            hidden_size=self._get_text_hidden_size(),
            scale=self.location_encoding_scale_init,
        ).to(self.device)

        if self.loc_mode == "loc_encoding":
            from src.models.scene_location_encoding import SceneLocationFeatures

            self.scene_location_features = SceneLocationFeatures(
                self.location_encoding_feature_dim
            ).to(self.device)
        elif self.loc_mode == "loc_additive_satclip":
            from src.models.satclip import get_satclip

            if not self.satclip_checkpoint:
                raise ValueError(
                    "satclip_checkpoint is required when "
                    "loc_mode='loc_additive_satclip'"
                )
            self.satclip = get_satclip(
                self.satclip_checkpoint,
                device=self.device,
            )
            checkpoint_dim = int(self.satclip.checkpoint_metadata["embed_dim"])
            if checkpoint_dim != self.location_encoding_feature_dim:
                raise ValueError(
                    "SatCLIP checkpoint output dimension does not match the "
                    "additive bridge: "
                    f"{checkpoint_dim} != {self.location_encoding_feature_dim}"
                )
            self.satclip.eval()
            for parameter in self.satclip.parameters():
                parameter.requires_grad = False
        else:
            raise ValueError(
                f"Unsupported projected additive location mode: {self.loc_mode}"
            )

        self._register_decoder_input_hook()

    def _load_additive_location_projection_artifacts(self) -> None:
        projection_path = (
            Path(self.adapter_dir) / "additive_location_projection.safetensors"
        )
        manifest_path = (
            Path(self.adapter_dir) / "additive_location_projection_config.json"
        )
        if not projection_path.is_file():
            raise FileNotFoundError(
                f"Missing additive location projection artifact: {projection_path}"
            )
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Missing additive location projection manifest: {manifest_path}"
            )

        actual_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_manifest = self.get_additive_location_projection_manifest()
        if actual_manifest != expected_manifest:
            raise ValueError(
                "Additive location projection config does not match the saved "
                f"adapter: expected {expected_manifest}, found {actual_manifest}"
            )

        state_dict = load_file(projection_path, device=str(self.device))
        self.additive_location_projection.load_state_dict(state_dict)

    def _setup_non_rgb_conditioning(self) -> None:
        """Initialize frozen BigEarthNet encoder and trainable non-RGB projection."""
        from src.models.bigearthnet_s1s2_encoder import BigEarthNetS1S2Encoder
        from src.models.non_rgb_modality_projection import NonRGBModalityProjection

        if not self.non_rgb_encoder_dir:
            raise ValueError("non_rgb_encoder_dir is required when non_rgb_conditioning='enabled'")

        self.non_rgb_encoder = BigEarthNetS1S2Encoder(
            model_dir=self.non_rgb_encoder_dir,
            feature_mode=self.non_rgb_feature_mode,
            spatial_pool_size=self.non_rgb_spatial_pool_size,
        ).to(self.device)
        self.non_rgb_encoder.eval()
        for p in self.non_rgb_encoder.parameters():
            p.requires_grad = False

        inferred_encoder_dim = self.non_rgb_encoder.feature_dim
        if self.non_rgb_encoder_feature_dim is not None:
            if (
                inferred_encoder_dim is not None
                and int(self.non_rgb_encoder_feature_dim) != int(inferred_encoder_dim)
            ):
                raise ValueError(
                    "non_rgb_encoder_feature_dim does not match the loaded encoder: "
                    f"{self.non_rgb_encoder_feature_dim} != {inferred_encoder_dim}"
                )
            encoder_dim = int(self.non_rgb_encoder_feature_dim)
        else:
            encoder_dim = inferred_encoder_dim
        if encoder_dim is None:
            raise ValueError("Could not infer non-RGB encoder feature dimension")

        self.non_rgb_modality_projection = NonRGBModalityProjection(
            encoder_dim=int(encoder_dim),
            hidden_size=self._get_text_hidden_size(),
            num_tokens=self.num_non_rgb_tokens,
        ).to(self.device)
        self._register_decoder_input_hook()
        self._print(f"NonRGBModalityProjection: {encoder_dim} -> hidden x {self.num_non_rgb_tokens}")

    def _setup_loc_embed(self):
        """Initialize SatCLIP encoder and LocationModalityProjection for loc_embed mode."""
        from src.models.location_modality_projection import LocationModalityProjection
        from src.models.satclip import get_satclip

        if not self.satclip_checkpoint:
            raise ValueError("satclip_checkpoint is required when loc_mode='loc_embed'")

        # Load frozen SatCLIP
        self.satclip = get_satclip(self.satclip_checkpoint, device=self.device)
        self.satclip.eval()
        for p in self.satclip.parameters():
            p.requires_grad = False

        hidden_size = self._get_text_hidden_size()

        # Trainable projection
        self.location_modality_projection = LocationModalityProjection(
            satclip_dim=self.satclip_dim,
            hidden_size=hidden_size,
            num_tokens=self.num_location_tokens,
            architecture=self.location_projection_architecture,
        ).to(self.device)

        self._register_decoder_input_hook()
        self._print(
            f"LocationModalityProjection: "
            f"{self.satclip_dim} -> {hidden_size} x {self.num_location_tokens}; "
            f"architecture={self.location_projection_architecture}"
        )

    def _encode_satclip_coordinates(
        self,
        lat: torch.Tensor,
        lon: torch.Tensor,
    ) -> torch.Tensor:
        """Encode coordinates with frozen SatCLIP in deterministic eval mode."""
        self.satclip.eval()
        coords = torch.stack([lon, lat], dim=-1).double()
        with torch.no_grad():
            return self.satclip(coords).float()

    def _encode_non_rgb_imagery(
        self,
        imagery: torch.Tensor,
        bands: Any,
    ) -> torch.Tensor:
        """Encode S1/S2 imagery with the frozen encoder in eval mode."""
        self.non_rgb_encoder.eval()
        with torch.no_grad():
            return self.non_rgb_encoder(imagery, bands).float()

    @staticmethod
    def _insert_tokens_2d(
        tensor: torch.Tensor, insert: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        """Insert `(B, n)` tokens into a `(B, L)` tensor at per-sample positions."""
        B, L = tensor.shape
        n = insert.shape[1]
        out = tensor.new_empty(B, L + n)
        for b in range(B):
            p = int(positions[b].item())
            out[b, :p] = tensor[b, :p]
            out[b, p : p + n] = insert[b]
            out[b, p + n :] = tensor[b, p:]
        return out

    def _replace_projected_token_placeholders(
        self,
        kwargs: dict[str, Any],
        tokens: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        """Replace prefix placeholders without modifying Qwen's visual block."""
        if "inputs_embeds" not in kwargs or kwargs["inputs_embeds"] is None:
            return

        inputs_embeds = kwargs["inputs_embeds"]
        tokens = tokens.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        positions = positions.to(inputs_embeds.device)
        out = inputs_embeds.clone()
        num_tokens = tokens.shape[1]
        for batch_index in range(inputs_embeds.shape[0]):
            position = int(positions[batch_index].item())
            out[batch_index, position : position + num_tokens] = tokens[batch_index]
        kwargs["inputs_embeds"] = out

    @staticmethod
    def _cache_has_tokens(past_key_values: Any) -> bool:
        """Return whether a generation cache already contains prompt tokens."""
        if past_key_values is None:
            return False
        get_seq_length = getattr(past_key_values, "get_seq_length", None)
        if callable(get_seq_length):
            return int(get_seq_length()) > 0
        if isinstance(past_key_values, (tuple, list)):
            if not past_key_values:
                return False
            first_layer = past_key_values[0]
            if isinstance(first_layer, (tuple, list)) and first_layer:
                return int(first_layer[0].shape[-2]) > 0
        return True

    def _decoder_input_conditioning_hook(self, module, args, kwargs):
        """Replace projected tokens and apply optional scene-location encoding."""
        location_state = getattr(self, "_location_insertion_state", None)
        encoding_state = getattr(self, "_location_encoding_state", None)
        non_rgb_state = getattr(self, "_non_rgb_insertion_state", None)
        if location_state is None and encoding_state is None and non_rgb_state is None:
            return args, kwargs

        # Replace placeholders during training and generation prefill, but not
        # during later one-token decode steps.
        if self._cache_has_tokens(kwargs.get("past_key_values")):
            return args, kwargs

        projected_tokens = []
        insert_positions = None

        if location_state is not None:
            insert_positions = location_state["insert_positions"]
            lat = location_state["lat"]
            lon = location_state["lon"]

            loc_embed = self._encode_satclip_coordinates(lat, lon)

            loc_tokens = self.location_modality_projection(loc_embed)
            projected_tokens.append(loc_tokens)

        if non_rgb_state is not None:
            insert_positions = non_rgb_state["insert_positions"]
            imagery = non_rgb_state["tensor"].to(self.device)
            bands = non_rgb_state["bands"]
            non_rgb_features = self._encode_non_rgb_imagery(imagery, bands)
            non_rgb_tokens = self.non_rgb_modality_projection(non_rgb_features)
            projected_tokens.append(non_rgb_tokens)

        if projected_tokens:
            tokens = torch.cat(projected_tokens, dim=1)
            self._replace_projected_token_placeholders(
                kwargs,
                tokens,
                insert_positions,
            )

        if encoding_state is not None:
            self._apply_scene_location_encoding(
                kwargs,
                encoding_state=encoding_state,
                non_rgb_state=non_rgb_state,
            )

        return args, kwargs

    def _apply_scene_location_encoding(
        self,
        kwargs: dict[str, Any],
        *,
        encoding_state: dict[str, torch.Tensor],
        non_rgb_state: dict[str, Any] | None,
    ) -> None:
        inputs_embeds = kwargs.get("inputs_embeds")
        if inputs_embeds is None:
            raise ValueError(
                "Additive location conditioning requires language-model "
                "inputs_embeds"
            )
        if self.location_encoding_scope not in {"all_visual", "s1s2"}:
            raise ValueError(
                f"Unsupported location encoding scope: {self.location_encoding_scope}"
            )

        native_visual_mask = kwargs.get("visual_pos_masks")
        if native_visual_mask is None:
            raise ValueError(
                "Visual location encoding requires Qwen visual_pos_masks"
            )
        if native_visual_mask.shape != inputs_embeds.shape[:2]:
            raise ValueError(
                "visual_pos_masks shape does not match inputs_embeds: "
                f"{tuple(native_visual_mask.shape)} != "
                f"{tuple(inputs_embeds.shape[:2])}"
            )

        native_visual_mask = native_visual_mask.to(
            device=inputs_embeds.device,
            dtype=torch.bool,
        )
        if not native_visual_mask.any(dim=1).all():
            raise ValueError(
                "Visual location encoding requires native visual "
                "content positions for every sample"
            )
        target_mask = (
            native_visual_mask.clone()
            if self.location_encoding_scope == "all_visual"
            else torch.zeros_like(native_visual_mask)
        )
        non_rgb_mask = torch.zeros_like(target_mask)
        if self.location_encoding_scope == "s1s2" and non_rgb_state is None:
            raise ValueError(
                "location_encoding_scope='s1s2' requires enabled S1/S2 conditioning"
            )
        if non_rgb_state is not None:
            positions = non_rgb_state["insert_positions"].to(inputs_embeds.device)
            for batch_index in range(inputs_embeds.shape[0]):
                position = int(positions[batch_index].item())
                end = position + self.num_non_rgb_tokens
                if position < 0 or end > inputs_embeds.shape[1]:
                    raise ValueError(
                        "S1/S2 token range falls outside inputs_embeds: "
                        f"[{position}, {end}) for length {inputs_embeds.shape[1]}"
                    )
                if native_visual_mask[batch_index, position:end].any():
                    raise ValueError(
                        "S1/S2 token range overlaps native visual content positions"
                    )
                target_mask[batch_index, position:end] = True
                non_rgb_mask[batch_index, position:end] = True

        lat = encoding_state["lat"].to(inputs_embeds.device)
        lon = encoding_state["lon"].to(inputs_embeds.device)
        if getattr(self, "additive_location_projection", None) is not None:
            if self.loc_mode == "loc_encoding":
                location_features = self.scene_location_features(lat, lon)
            elif self.loc_mode == "loc_additive_satclip":
                location_features = self._encode_satclip_coordinates(lat, lon)
            else:
                raise ValueError(
                    f"Unsupported projected additive location mode: {self.loc_mode}"
                )
            geo_encoding = self.additive_location_projection(location_features)
            scale = self.additive_location_projection.scale.detach().float()
        else:
            geo_encoding = self.scene_location_encoding(lat, lon)
            scale = self.scene_location_encoding.scale.detach().float()
        geo_encoding = geo_encoding.to(dtype=inputs_embeds.dtype)
        if geo_encoding.shape != (
            inputs_embeds.shape[0],
            inputs_embeds.shape[2],
        ):
            raise ValueError(
                "Scene-location encoding shape does not match inputs_embeds: "
                f"{tuple(geo_encoding.shape)} != "
                f"{(inputs_embeds.shape[0], inputs_embeds.shape[2])}"
            )

        if not getattr(self, "_location_encoding_norm_logged", False):
            with torch.no_grad():
                rgb_rms = (
                    inputs_embeds[native_visual_mask].float().square().mean().sqrt()
                )
                non_rgb_rms = (
                    inputs_embeds[non_rgb_mask].float().square().mean().sqrt()
                    if non_rgb_mask.any()
                    else None
                )
                encoding_rms = geo_encoding.float().square().mean().sqrt()
            non_rgb_rms_text = (
                f"{float(non_rgb_rms):.6g}" if non_rgb_rms is not None else "n/a"
            )
            message = (
                "Additive location conditioning first-batch RMS: "
                f"rgb={float(rgb_rms):.6g}, "
                f"s1s2={non_rgb_rms_text}, "
                f"scaled_encoding={float(encoding_rms):.6g}, "
                f"scale={float(scale):.6g}, "
                f"rgb_tokens={int(native_visual_mask.sum())}, "
                f"s1s2_tokens={int(non_rgb_mask.sum())}"
            )
            self._print(message)
            self._location_encoding_norm_logged = True

        kwargs["inputs_embeds"] = inputs_embeds + (
            target_mask.unsqueeze(-1).to(inputs_embeds.dtype)
            * geo_encoding.unsqueeze(1)
        )

    def _compute_visual_boundary(self, batch: dict[str, Any]) -> torch.Tensor:
        """Return the first visual-start position, or the attended sequence end."""
        input_ids = batch.get("input_ids")
        attention_mask = batch.get("attention_mask")
        if input_ids is None:
            raise ValueError("input_ids are required for projected token insertion")

        config = self.model.base_model.model.model.config
        visual_start_token_id = config.vision_start_token_id
        visual_mask = input_ids.eq(visual_start_token_id)

        has_visual = visual_mask.any(dim=1)
        first_visual = visual_mask.int().argmax(dim=1)
        if attention_mask is not None:
            fallback = attention_mask.sum(dim=1)
        else:
            fallback = torch.full_like(first_visual, input_ids.shape[1])
        positions = torch.where(has_visual, first_visual, fallback)
        return positions

    def _prepare_model_inputs(self, batch: dict[str, Any]):
        """Strip non-model fields and set up projected token insertion state."""
        self._reset_decoder_conditioning_state()
        lat = batch.pop("lat", None)
        lon = batch.pop("lon", None)
        target_texts = batch.pop("target_texts", None)
        sample_metadata = {
            "input_text": batch.pop("input_text", None),
            "sample_id": batch.pop("sample_id", None),
            "patch_id": batch.pop("patch_id", None),
            "task_type": batch.pop("task_type", None),
            "task_category": batch.pop("task_category", None),
            "split": batch.pop("split", None),
            "country": batch.pop("country", None),
            "season": batch.pop("season", None),
            "climate_zone": batch.pop("climate_zone", None),
            "grounding_format": batch.pop("grounding_format", None),
        }
        non_rgb_imagery = {
            "tensor": batch.pop("non_rgb_imagery", None),
            "bands": batch.pop("non_rgb_bands", None),
        }
        if self.non_rgb_conditioning == "enabled":
            if non_rgb_imagery["tensor"] is None:
                raise ValueError(
                    "non_rgb_conditioning='enabled' requires non_rgb_imagery in the batch"
                )
        elif self.non_rgb_conditioning != "disabled":
            raise ValueError(f"Unsupported non_rgb_conditioning: {self.non_rgb_conditioning}")

        uses_projected_tokens = self.loc_mode == "loc_embed" or self.non_rgb_conditioning == "enabled"
        insert_positions = None
        if uses_projected_tokens:
            insert_positions = self._compute_visual_boundary(batch)

        num_inserted_tokens = 0
        if self.non_rgb_conditioning == "enabled":
            self._non_rgb_insertion_state = {
                "tensor": non_rgb_imagery["tensor"].to(self.device),
                "bands": non_rgb_imagery["bands"],
                "insert_positions": insert_positions.to(self.device),
            }
            num_inserted_tokens += self.num_non_rgb_tokens

        if self.loc_mode == "loc_embed":
            if lat is None or lon is None:
                raise ValueError("loc_mode='loc_embed' requires both lat and lon in the batch")

            self._location_insertion_state = {
                "lat": lat.to(self.device),
                "lon": lon.to(self.device),
                "insert_positions": insert_positions.to(self.device),
            }
            num_inserted_tokens += self.num_location_tokens
        elif self.loc_mode in ADDITIVE_LOCATION_MODES:
            if lat is None or lon is None:
                raise ValueError(
                    "Additive location modes require both lat and lon in the batch"
                )
            self._location_encoding_state = {
                "lat": lat.to(self.device),
                "lon": lon.to(self.device),
            }

        if num_inserted_tokens > 0 and "labels" in batch:
            B = batch["labels"].shape[0]
            ignore = torch.full(
                (B, num_inserted_tokens),
                -100,
                device=batch["labels"].device,
                dtype=batch["labels"].dtype,
            )
            batch["labels"] = self._insert_tokens_2d(batch["labels"], ignore, insert_positions)

        if num_inserted_tokens > 0:
            batch_size = batch["input_ids"].shape[0]
            placeholder_token_id = self.tokenizer.pad_token_id
            if placeholder_token_id is None:
                raise ValueError("Projected token conditioning requires a tokenizer pad token")
            placeholders = torch.full(
                (batch_size, num_inserted_tokens),
                placeholder_token_id,
                device=batch["input_ids"].device,
                dtype=batch["input_ids"].dtype,
            )
            batch["input_ids"] = self._insert_tokens_2d(
                batch["input_ids"], placeholders, insert_positions
            )
            if "attention_mask" in batch:
                attended = torch.ones(
                    batch_size,
                    num_inserted_tokens,
                    device=batch["attention_mask"].device,
                    dtype=batch["attention_mask"].dtype,
                )
                batch["attention_mask"] = self._insert_tokens_2d(
                    batch["attention_mask"], attended, insert_positions
                )
            if "mm_token_type_ids" in batch:
                text_token_types = torch.zeros(
                    batch_size,
                    num_inserted_tokens,
                    device=batch["mm_token_type_ids"].device,
                    dtype=batch["mm_token_type_ids"].dtype,
                )
                batch["mm_token_type_ids"] = self._insert_tokens_2d(
                    batch["mm_token_type_ids"], text_token_types, insert_positions
                )

            sequence_length = batch["input_ids"].shape[1]
            for key in ("attention_mask", "labels", "mm_token_type_ids"):
                if key in batch and batch[key].shape[-1] != sequence_length:
                    raise ValueError(
                        f"{key} length {batch[key].shape[-1]} does not match "
                        f"input_ids length {sequence_length} after projected-token insertion"
                    )

        self._validate_supervision_mask(
            batch,
            insert_positions=insert_positions,
            num_inserted_tokens=num_inserted_tokens,
        )

        return batch, target_texts, lat, lon, non_rgb_imagery, sample_metadata

    @staticmethod
    def _find_token_subsequence(sequence: list[int], subsequence: list[int]) -> int | None:
        if not subsequence:
            return None
        limit = len(sequence) - len(subsequence) + 1
        for index in range(max(limit, 0)):
            if sequence[index : index + len(subsequence)] == subsequence:
                return index
        return None

    def _validate_supervision_mask(
        self,
        batch: dict[str, Any],
        *,
        insert_positions: torch.Tensor | None,
        num_inserted_tokens: int,
    ) -> None:
        """Validate assistant-only supervision and masked multimodal token positions once."""
        if (
            not hasattr(self, "_supervision_mask_validated")
            or self._supervision_mask_validated
            or "labels" not in batch
        ):
            return
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        if input_ids.shape != labels.shape:
            raise ValueError("labels must align with input_ids")
        if not labels.ne(-100).any(dim=1).all():
            raise ValueError("Every supervised sample must expose assistant answer tokens to loss")

        attention_mask = batch.get("attention_mask")
        if attention_mask is not None and labels[attention_mask.eq(0)].ne(-100).any():
            raise ValueError("Padding tokens must be ignored by the training loss")

        tokenizer = getattr(self.tokenizer, "tokenizer", self.tokenizer)
        assistant_marker_ids = tokenizer.encode(
            "<|im_start|>assistant\n",
            add_special_tokens=False,
        )
        for row_index in range(input_ids.shape[0]):
            row_ids = input_ids[row_index].tolist()
            marker_index = self._find_token_subsequence(row_ids, assistant_marker_ids)
            if marker_index is None:
                raise ValueError("Could not find Qwen assistant header in supervised input")
            response_start = marker_index + len(assistant_marker_ids)
            if labels[row_index, :response_start].ne(-100).any():
                raise ValueError(
                    "Assistant-only loss requires system, user, RGB and side-token "
                    "positions before the response to use label -100"
                )

        if num_inserted_tokens > 0:
            if insert_positions is None:
                raise ValueError("Projected token insertion positions are missing")
            for row_index, position in enumerate(insert_positions.tolist()):
                inserted_labels = labels[
                    row_index,
                    position : position + num_inserted_tokens,
                ]
                if inserted_labels.ne(-100).any():
                    raise ValueError("Projected S1/S2/location tokens must be ignored by loss")

        supervised_count = int(labels.ne(-100).sum())
        ignored_count = int(labels.eq(-100).sum())
        self._print(
            "Verified assistant-only loss mask on the first supervised batch: "
            f"supervised_answer_tokens={supervised_count}, "
            f"ignored_prompt_or_multimodal_tokens={ignored_count}."
        )
        self._supervision_mask_validated = True

    def _reset_decoder_conditioning_state(self) -> None:
        self._location_insertion_state = None
        self._location_encoding_state = None
        self._non_rgb_insertion_state = None

    def _set_datamodule_collator(self):
        """Attach supervised and optional generation collators to the active datamodule."""
        if self._collator is None:
            return
        trainer = self._trainer_or_none()
        datamodule = getattr(trainer, "datamodule", None) if trainer is not None else None
        if datamodule is not None and hasattr(datamodule, "set_collator"):
            datamodule.set_collator(self._collator)
        if datamodule is not None and hasattr(datamodule, "set_validation_collator"):
            datamodule.set_validation_collator(self._validation_collator)
        if datamodule is not None and hasattr(datamodule, "set_test_collator"):
            datamodule.set_test_collator(self._test_collator)

    def forward(self, **inputs) -> Any:
        """Forward pass through the model."""
        return self.model(**inputs)

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Training step."""
        batch, _, _, _, _, _ = self._prepare_model_inputs(batch)
        batch_size = batch["input_ids"].shape[0]
        outputs = self.model(**batch)
        self._reset_decoder_conditioning_state()
        self.log(
            "train/loss",
            outputs.loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        if getattr(self, "scene_location_encoding", None) is not None:
            self.log(
                "train/location_encoding_scale",
                self.scene_location_encoding.scale,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                batch_size=batch_size,
            )
        return outputs.loss

    def _generation_stop_token_ids(self) -> set[int]:
        """Return terminal generation IDs that are not assistant response content."""
        token_ids: set[int] = set()
        generation_config = getattr(self.model, "generation_config", None)
        eos_ids = getattr(generation_config, "eos_token_id", None)
        if isinstance(eos_ids, int):
            token_ids.add(eos_ids)
        elif eos_ids is not None:
            token_ids.update(int(token_id) for token_id in eos_ids)
        for token_id in (
            getattr(self.tokenizer, "eos_token_id", None),
            getattr(self.tokenizer, "pad_token_id", None),
        ):
            if token_id is not None:
                token_ids.add(int(token_id))
        return token_ids

    def _generate_for_batch(self, batch: dict[str, Any]) -> list[str]:
        """Run greedy generation and decode the assistant response without stripping content tokens."""
        gen_batch = {k: v for k, v in batch.items() if k != "labels"}
        was_training = self.model.training
        FastVisionModel.for_inference(self.model)
        try:
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **gen_batch,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                )
        finally:
            FastVisionModel.for_training(self.model)
            self.model.train(was_training)
        input_len = gen_batch["input_ids"].shape[-1]
        stop_token_ids = self._generation_stop_token_ids()
        predictions = []
        for i in range(generated_ids.shape[0]):
            response_ids = generated_ids[i, input_len:].tolist()
            while response_ids and response_ids[-1] in stop_token_ids:
                response_ids.pop()
            text = self.tokenizer.decode(response_ids, skip_special_tokens=False)
            predictions.append(text)
        return predictions

    def on_fit_start(self) -> None:
        """Initialize the optional qualitative validation generation file."""
        if not self.validation_generation_path:
            return
        trainer = self._trainer_or_none()
        if trainer is not None and not trainer.is_global_zero:
            return
        output_path = Path(self.validation_generation_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("", encoding="utf-8")

    def _write_validation_generations(
        self,
        *,
        predictions: list[str],
        target_texts: list[list[str]] | None,
        lat: torch.Tensor | None,
        lon: torch.Tensor | None,
        sample_metadata: dict[str, Any],
        batch_idx: int,
    ) -> None:
        if not self.validation_generation_path:
            return
        trainer = self._trainer_or_none()
        if trainer is not None and not trainer.is_global_zero:
            return
        output_path = Path(self.validation_generation_path)
        global_step = int(trainer.global_step) if trainer is not None else 0
        with output_path.open("a", encoding="utf-8") as handle:
            for index, prediction in enumerate(predictions):
                entry = {
                    "global_step": global_step,
                    "validation_batch_idx": batch_idx,
                    "prediction": prediction,
                    "target_texts": target_texts[index] if target_texts else [],
                    "location_condition": self.loc_mode,
                    "model_name_or_path": self.model_name_or_path,
                    "max_new_tokens": self.max_new_tokens,
                }
                location_encoding_scope = getattr(
                    self, "location_encoding_scope", None
                )
                if location_encoding_scope is not None:
                    entry["location_encoding_scope"] = location_encoding_scope
                for key, values in sample_metadata.items():
                    if values is not None:
                        entry[key] = values[index]
                if lat is not None and lon is not None:
                    entry["lat"] = float(lat[index])
                    entry["lon"] = float(lon[index])
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> dict[str, Any]:
        """Compute teacher-forced loss and optional prompt-only generations."""
        generation_batch = batch.pop("validation_generation_batch", None)
        batch, _, _, _, _, _ = self._prepare_model_inputs(batch)
        batch_size = batch["input_ids"].shape[0]
        with torch.no_grad():
            outputs = self.model(**batch)

        self.log(
            "val/loss",
            outputs.loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        self._reset_decoder_conditioning_state()

        if generation_batch is not None:
            generation_batch = {
                key: value.to(self.device) if isinstance(value, torch.Tensor) else value
                for key, value in generation_batch.items()
            }
            generation_batch, target_texts, lat, lon, _, sample_metadata = (
                self._prepare_model_inputs(generation_batch)
            )
            predictions = self._generate_for_batch(generation_batch)
            self._write_validation_generations(
                predictions=predictions,
                target_texts=target_texts,
                lat=lat,
                lon=lon,
                sample_metadata=sample_metadata,
                batch_idx=batch_idx,
            )
            self._reset_decoder_conditioning_state()

        return {"loss": outputs.loss}

    def _write_prediction_export(
        self,
        *,
        predictions: list[str],
        target_texts: list[list[str]] | None,
        lat: torch.Tensor | None,
        lon: torch.Tensor | None,
        sample_metadata: dict[str, Any],
    ) -> None:
        if not self.prediction_export_path:
            return
        output_path = Path(self.prediction_export_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("a", encoding="utf-8") as handle:
            for index, prediction in enumerate(predictions):
                entry = {
                    "prediction": prediction,
                    "target_texts": target_texts[index] if target_texts else [],
                    "location_condition": self.loc_mode,
                    "model_name_or_path": self.model_name_or_path,
                    "adapter_dir": self.adapter_dir,
                }
                location_encoding_scope = getattr(
                    self, "location_encoding_scope", None
                )
                if location_encoding_scope is not None:
                    entry["location_encoding_scope"] = location_encoding_scope
                if self.run_label is not None:
                    entry["run_label"] = self.run_label
                if self.model_size is not None:
                    entry["model_size"] = self.model_size
                for key, values in sample_metadata.items():
                    if values is not None:
                        entry[key] = values[index]
                if lat is not None and lon is not None:
                    entry["lat"] = float(lat[index])
                    entry["lon"] = float(lon[index])
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
                self._prediction_export_count += 1

    def on_test_start(self) -> None:
        """Initialize streaming prediction export."""
        if not self.prediction_export_path:
            return
        output_path = Path(self.prediction_export_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("", encoding="utf-8")
        self._prediction_export_count = 0

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> dict[str, Any]:
        """Test step with loss logging and optional raw prediction export."""
        batch, target_texts, lat, lon, _, sample_metadata = self._prepare_model_inputs(batch)
        if self.prediction_export_path:
            if self.max_new_tokens <= 0:
                raise ValueError("prediction_export_path requires max_new_tokens to be positive")
            predictions = self._generate_for_batch(batch)

            if batch_idx == 0 and predictions:
                self._print(f"\n[Test Sample] Generated: {predictions[0][:500]}...")

            self._write_prediction_export(
                predictions=predictions,
                target_texts=target_texts,
                lat=lat,
                lon=lon,
                sample_metadata=sample_metadata,
            )
            result = {"generated": predictions[0] if predictions else ""}
        else:
            batch_size = batch["input_ids"].shape[0]
            with torch.no_grad():
                outputs = self.model(**batch)
            self.log(
                "test/loss",
                outputs.loss,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
            )
            result = {"loss": outputs.loss}

        self._reset_decoder_conditioning_state()
        return result

    def on_test_epoch_end(self) -> None:
        """Report prediction export completion."""
        if self.prediction_export_path and self._prediction_export_count:
            self._print(
                f"Saved {self._prediction_export_count} predictions to "
                f"{self.prediction_export_path}"
            )

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        decay_params: list[torch.nn.Parameter] = []
        no_decay_params: list[torch.nn.Parameter] = []
        location_decay_params: list[torch.nn.Parameter] = []
        location_no_decay_params: list[torch.nn.Parameter] = []
        location_encoding_no_decay_params: list[torch.nn.Parameter] = []
        non_rgb_decay_params: list[torch.nn.Parameter] = []
        non_rgb_no_decay_params: list[torch.nn.Parameter] = []

        def add_param(
            name: str,
            param: torch.nn.Parameter,
            *,
            location_projection: bool = False,
            location_encoding: bool = False,
            non_rgb_projection: bool = False,
        ) -> None:
            if not param.requires_grad:
                return
            if location_encoding:
                location_encoding_no_decay_params.append(param)
                return
            if "bias" in name or "LayerNorm" in name or "layer_norm" in name:
                if location_projection:
                    target = location_no_decay_params
                elif non_rgb_projection:
                    target = non_rgb_no_decay_params
                else:
                    target = no_decay_params
            else:
                if location_projection:
                    target = location_decay_params
                elif non_rgb_projection:
                    target = non_rgb_decay_params
                else:
                    target = decay_params
            target.append(param)

        for name, param in self.model.named_parameters():
            add_param(name, param)
        if self.location_modality_projection is not None:
            for name, param in self.location_modality_projection.named_parameters():
                add_param(
                    f"location_modality_projection.{name}",
                    param,
                    location_projection=True,
                )
        if getattr(self, "additive_location_projection", None) is not None:
            for name, param in self.additive_location_projection.named_parameters():
                add_param(
                    f"additive_location_projection.{name}",
                    param,
                    location_projection=True,
                )
        if self.scene_location_encoding is not None:
            for name, param in self.scene_location_encoding.named_parameters():
                add_param(
                    f"scene_location_encoding.{name}",
                    param,
                    location_encoding=True,
                )
        if self.non_rgb_modality_projection is not None:
            for name, param in self.non_rgb_modality_projection.named_parameters():
                add_param(
                    f"non_rgb_modality_projection.{name}",
                    param,
                    non_rgb_projection=True,
                )

        location_lr = self.learning_rate * self.location_projection_lr_multiplier
        non_rgb_lr = self.learning_rate * self.non_rgb_projection_lr_multiplier
        optimizer_groups = [
            {
                "name": "base_decay",
                "params": decay_params,
                "weight_decay": self.weight_decay,
                "lr": self.learning_rate,
            },
            {
                "name": "base_no_decay",
                "params": no_decay_params,
                "weight_decay": 0.0,
                "lr": self.learning_rate,
            },
            {
                "name": "location_projection_decay",
                "params": location_decay_params,
                "weight_decay": self.weight_decay,
                "lr": location_lr,
            },
            {
                "name": "location_projection_no_decay",
                "params": location_no_decay_params,
                "weight_decay": 0.0,
                "lr": location_lr,
            },
            {
                "name": "location_encoding_no_decay",
                "params": location_encoding_no_decay_params,
                "weight_decay": 0.0,
                "lr": self.learning_rate,
            },
            {
                "name": "non_rgb_projection_decay",
                "params": non_rgb_decay_params,
                "weight_decay": self.weight_decay,
                "lr": non_rgb_lr,
            },
            {
                "name": "non_rgb_projection_no_decay",
                "params": non_rgb_no_decay_params,
                "weight_decay": 0.0,
                "lr": non_rgb_lr,
            },
        ]
        optimizer_groups = [group for group in optimizer_groups if group["params"]]

        optimizer = bnb.optim.AdamW8bit(
            optimizer_groups,
            lr=self.learning_rate,
        )

        trainer = self._trainer_or_none()

        if self.max_steps is not None:
            total_steps = self.max_steps
        elif trainer is not None and trainer.max_steps > 0:
            total_steps = trainer.max_steps
        elif trainer is not None and hasattr(trainer, "estimated_stepping_batches"):
            total_steps = trainer.estimated_stepping_batches
        else:
            total_steps = 10000
            self._print(
                "WARNING: Could not determine total training steps. "
                "Defaulting to 10000 for LR schedule. Set max_steps or trainer.max_steps explicitly."
            )

        warmup_steps = int(total_steps * self.warmup_ratio)
        decay_steps = max(1, total_steps - warmup_steps)
        min_lr_factor = 0.1

        def lr_factor(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return min_lr_factor + (1.0 - min_lr_factor) * step / warmup_steps
            progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_factor + (1.0 - min_lr_factor) * cosine

        scheduler = LambdaLR(optimizer, lr_lambda=lr_factor)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }
