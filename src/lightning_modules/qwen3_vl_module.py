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
        loc_mode: Literal["no_loc", "loc_text", "loc_embed"] = "no_loc",
        location_text_template: str | None = None,
        coordinates_decimal_places: int = 0,
        location_embed_marker: str | None = None,
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
            loc_mode: Location conditioning mode ("no_loc", "loc_text", "loc_embed")
            location_text_template: Format string appended to the user prompt when
                `loc_mode="loc_text"`. Coordinate formatting is handled by the
                shared collator.
            coordinates_decimal_places: Decimal places used for the compact
                `{location}` field in `location_text_template`.
            location_embed_marker: Text marker appended to the prompt before
                projected SatCLIP tokens when `loc_mode="loc_embed"`.
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
            satclip_checkpoint: Path to SatCLIP checkpoint (required for encoder mode)
            satclip_dim: SatCLIP embedding dimension
            num_location_tokens: Number of location tokens to insert before the visual block (encoder mode)
            location_projection_lr_multiplier: Learning-rate multiplier for
                the randomly initialized SatCLIP-to-Qwen projection.
            prediction_export_path: If set, stream per-sample test predictions to this JSONL path.
            run_label: Human-readable run label exported with each prediction.
            model_size: Model size label exported with each prediction, e.g. "2B".
        """
        super().__init__()

        if loc_mode not in {"no_loc", "loc_text", "loc_embed"}:
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
        self.non_rgb_encoder = None
        self.non_rgb_modality_projection = None
        self._geo_hook_handle = None
        self._location_insertion_state = None
        self._non_rgb_insertion_state = None

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
        base_collator = UnslothVisionDataCollator(self.model, self.tokenizer)
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

        # Set up loc_embed components
        if self.loc_mode == "loc_embed":
            self._setup_loc_embed()
            if self.adapter_dir is not None:
                self._load_location_projection_artifacts()
        if self.non_rgb_conditioning == "enabled":
            self._setup_non_rgb_conditioning()
            if self.adapter_dir is not None:
                self._load_non_rgb_projection_artifacts()

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        location_proj_params = 0
        non_rgb_proj_params = 0
        if self.location_modality_projection is not None:
            location_proj_params = sum(p.numel() for p in self.location_modality_projection.parameters())
            trainable_params += location_proj_params
            total_params += location_proj_params
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

    def _register_projected_token_hook(self) -> None:
        """Register the shared decoder hook used by projected side modalities."""
        if self._geo_hook_handle is not None:
            return
        language_model = self.model.base_model.model.model.language_model
        self._geo_hook_handle = language_model.register_forward_pre_hook(
            self._projected_token_insertion_hook, with_kwargs=True
        )
        self._print(f"Registered projected token hook on {type(language_model).__name__}")

    def _load_location_projection_artifacts(self) -> None:
        """Load the saved location projection that lives outside the PEFT adapter package."""
        projection_path = Path(self.adapter_dir) / "location_modality_projection.safetensors"
        if not projection_path.is_file():
            raise FileNotFoundError(f"Missing location projection artifacts: {projection_path}")

        state_dict = load_file(projection_path, device=str(self.device))
        self.location_modality_projection.load_state_dict(state_dict)

    def _load_non_rgb_projection_artifacts(self) -> None:
        """Load the saved non-RGB projection that lives outside the PEFT adapter package."""
        projection_path = Path(self.adapter_dir) / "non_rgb_modality_projection.safetensors"
        if not projection_path.is_file():
            raise FileNotFoundError(f"Missing non-RGB projection artifacts: {projection_path}")

        state_dict = load_file(projection_path, device=str(self.device))
        self.non_rgb_modality_projection.load_state_dict(state_dict)

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
        self._register_projected_token_hook()
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
        ).to(self.device)

        self._register_projected_token_hook()
        self._print(
            f"LocationModalityProjection: "
            f"{self.satclip_dim} -> {hidden_size} x {self.num_location_tokens}"
        )

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

    def _projected_token_insertion_hook(self, module, args, kwargs):
        """Forward pre-hook that inserts projected side-modality tokens."""
        location_state = self._location_insertion_state
        non_rgb_state = self._non_rgb_insertion_state
        if location_state is None and non_rgb_state is None:
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

            # SatCLIP expects (B, 2) of (lon, lat) as float64
            coords = torch.stack([lon, lat], dim=-1).double()
            with torch.no_grad():
                loc_embed = self.satclip(coords).float()  # (B, satclip_dim)

            loc_tokens = self.location_modality_projection(loc_embed)
            projected_tokens.append(loc_tokens)

        if non_rgb_state is not None:
            insert_positions = non_rgb_state["insert_positions"]
            imagery = non_rgb_state["tensor"].to(self.device)
            bands = non_rgb_state["bands"]
            with torch.no_grad():
                non_rgb_features = self.non_rgb_encoder(imagery, bands).float()
            non_rgb_tokens = self.non_rgb_modality_projection(non_rgb_features)
            projected_tokens.append(non_rgb_tokens)

        if not projected_tokens:
            return args, kwargs

        tokens = torch.cat(projected_tokens, dim=1)
        self._replace_projected_token_placeholders(
            kwargs,
            tokens,
            insert_positions,
        )

        return args, kwargs

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
        self._reset_projected_token_state()
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

        return batch, target_texts, lat, lon, non_rgb_imagery, sample_metadata

    def _reset_projected_token_state(self) -> None:
        self._location_insertion_state = None
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
        self._reset_projected_token_state()
        self.log(
            "train/loss",
            outputs.loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        return outputs.loss

    def _generate_for_batch(self, batch: dict[str, Any]) -> list[str]:
        """Run greedy generation on a batch and return decoded predictions."""
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
        predictions = []
        for i in range(generated_ids.shape[0]):
            text = self.tokenizer.decode(
                generated_ids[i, input_len:], skip_special_tokens=True
            )
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
        self._reset_projected_token_state()

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
            self._reset_projected_token_state()

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

        self._reset_projected_token_state()
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
        non_rgb_decay_params: list[torch.nn.Parameter] = []
        non_rgb_no_decay_params: list[torch.nn.Parameter] = []

        def add_param(
            name: str,
            param: torch.nn.Parameter,
            *,
            location_projection: bool = False,
            non_rgb_projection: bool = False,
        ) -> None:
            if not param.requires_grad:
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
