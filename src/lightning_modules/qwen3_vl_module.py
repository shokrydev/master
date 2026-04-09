# LightningModule for Qwen3-VL Vision-Language Model Finetuning with Unsloth
# Docs: https://lightning.ai/docs/pytorch/stable/common/lightning_module.html

import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import bitsandbytes as bnb
import lightning as L
import torch
from safetensors.torch import load_file
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator

from src.data_modules.geo_aware_collator import GeoAwareCollator

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
        adapter_dir: Optional[str] = None,
        max_seq_length: int = 2048,
        lora_r: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        lora_target_modules: Optional[List[str]] = None,
        modules_to_save: Optional[List[str]] = None,
        finetune_vision_layers: bool = False,
        finetune_language_layers: bool = True,
        finetune_attention_modules: bool = True,
        finetune_mlp_modules: bool = True,
        learning_rate: float = 2e-4,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        max_steps: Optional[int] = None,
        max_new_tokens: int = 256,
        val_generate_batches: int = 0,
        loc_mode: Literal["no_loc", "loc_text", "loc_embed"] = "no_loc",
        satclip_checkpoint: Optional[str] = None,
        satclip_dim: int = 256,
        num_location_tokens: int = 1,
        test_predictions_path: Optional[str] = None,
    ):
        """
        Initialize Qwen3-VL finetuning module.

        Args:
            model_name_or_path: Unsloth model ID (e.g., "unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit")
            adapter_dir: Saved adapter directory for validate/test/predict
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
            max_new_tokens: Max tokens to generate during validation
            val_generate_batches: How many val batches to run generation on
                (0 = no generation metrics, -1 = all batches). Test always generates.
            loc_mode: Location conditioning mode ("no_loc", "loc_text", "loc_embed")
            satclip_checkpoint: Path to SatCLIP checkpoint (required for encoder mode)
            satclip_dim: SatCLIP embedding dimension
            num_location_tokens: Number of location tokens to insert before the visual block (encoder mode)
            test_predictions_path: If set, save per-sample predictions to this JSON path during test
        """
        super().__init__()

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
        self.val_generate_batches = val_generate_batches
        self.loc_mode = loc_mode
        self.satclip_checkpoint = satclip_checkpoint
        self.satclip_dim = satclip_dim
        self.num_location_tokens = num_location_tokens
        self.test_predictions_path = test_predictions_path

        self.model = None
        self.tokenizer = None
        self._collator = None

        # loc_embed components (initialized in setup)
        self.satclip = None
        self.location_modality_projection = None
        self._geo_hook_handle = None
        self._location_insertion_state = None

        # Test prediction accumulator (for saving per-sample predictions to JSON)
        self._test_predictions: List[Dict[str, Any]] = []

        # Captioning metrics (initialized in setup)
        self.val_captioning_metrics = None
        self.test_captioning_metrics = None

    def setup(self, stage: str):
        """Load model with Unsloth and configure QLoRA."""
        if self.model is not None:
            self._set_datamodule_collator()
            return

        if stage == "fit" and self.adapter_dir:
            raise ValueError("adapter_dir cannot be set for fit; training starts from a base model")
        if stage in {"validate", "test", "predict"} and not self.adapter_dir:
            raise ValueError(
                f"{stage} requires model.init_args.adapter_dir to point at saved adapters"
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
        self._collator = GeoAwareCollator(
            base_collator, include_coordinates=self.loc_mode in {"loc_text", "loc_embed"}
        )
        self._set_datamodule_collator()

        # Set up loc_embed components
        if self.loc_mode == "loc_embed":
            self._setup_loc_embed()
            if self.adapter_dir is not None:
                self._load_location_projection_artifacts()

        # Initialize captioning metrics
        from src.metrics.captioning import CaptioningMetrics
        if self.val_generate_batches != 0:
            self.val_captioning_metrics = CaptioningMetrics()
        self.test_captioning_metrics = CaptioningMetrics()

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        proj_params = 0
        if self.location_modality_projection is not None:
            proj_params = sum(p.numel() for p in self.location_modality_projection.parameters())
            trainable_params += proj_params
            total_params += proj_params
        self.print(f"Trainable params: {trainable_params:,} / {total_params:,} "
                   f"({100 * trainable_params / total_params:.2f}%)")
        if proj_params:
            self.print(f"LocationModalityProjection params: {proj_params:,}")

    def _load_location_projection_artifacts(self) -> None:
        """Load the saved location projection that lives outside the PEFT adapter package."""
        projection_path = Path(self.adapter_dir) / "location_modality_projection.safetensors"
        if not projection_path.is_file():
            raise FileNotFoundError(f"Missing location projection artifacts: {projection_path}")

        state_dict = load_file(projection_path, device=str(self.device))
        self.location_modality_projection.load_state_dict(state_dict)

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

        # Get hidden size from the underlying model config
        # Navigate through PEFT wrapper: PeftModel → base_model → model → config
        inner_model = self.model
        if hasattr(inner_model, "base_model"):
            inner_model = inner_model.base_model
        if hasattr(inner_model, "model"):
            inner_model = inner_model.model
        config = inner_model.config
        # Qwen3VL uses nested text_config; fall back to top-level for other architectures
        if hasattr(config, "text_config") and hasattr(config.text_config, "hidden_size"):
            hidden_size = config.text_config.hidden_size
        else:
            hidden_size = config.hidden_size

        # Trainable projection
        self.location_modality_projection = LocationModalityProjection(
            satclip_dim=self.satclip_dim,
            hidden_size=hidden_size,
            num_tokens=self.num_location_tokens,
        ).to(self.device)

        # Register forward pre-hook on the language model (text decoder), NOT the
        # vision-language wrapper. Qwen3VLModel receives input_ids, computes
        # inputs_embeds internally (including visual token scatter), then passes
        # inputs_embeds to language_model. Our hook must fire at that point.
        # Path: PeftModel → .base_model → .model (Qwen3VLForConditionalGeneration)
        #        → .model (Qwen3VLModel) → .language_model (Qwen3VLTextModel)
        language_model = self.model.base_model.model.model.language_model
        self._geo_hook_handle = language_model.register_forward_pre_hook(
            self._location_embed_insertion_hook, with_kwargs=True
        )
        self.print(f"Registered geo embedding hook on {type(language_model).__name__}")
        self.print(f"LocationModalityProjection: {self.satclip_dim} → {hidden_size} × {self.num_location_tokens}")

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

    @staticmethod
    def _insert_tokens_3d(
        tensor: torch.Tensor, insert: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        """Insert `(B, n, H)` tokens into a `(B, L, H)` tensor at per-sample positions."""
        B, L, H = tensor.shape
        n = insert.shape[1]
        out = tensor.new_empty(B, L + n, H)
        for b in range(B):
            p = int(positions[b].item())
            out[b, :p] = tensor[b, :p]
            out[b, p : p + n] = insert[b]
            out[b, p + n :] = tensor[b, p:]
        return out

    @staticmethod
    def _insert_position_ids(
        position_ids: torch.Tensor, positions: torch.Tensor, n: int
    ) -> torch.Tensor:
        """Insert contiguous position ids before the visual block and shift later ids."""
        _, B, L = position_ids.shape
        out = position_ids.new_empty(3, B, L + n)
        for b in range(B):
            p = int(positions[b].item())
            out[:, b, :p] = position_ids[:, b, :p]
            if p > 0:
                loc_start = position_ids[:, b, p - 1 : p] + 1
            else:
                loc_start = torch.zeros(3, 1, device=position_ids.device, dtype=position_ids.dtype)
            loc_offsets = torch.arange(n, device=position_ids.device, dtype=position_ids.dtype).view(1, n)
            out[:, b, p : p + n] = loc_start + loc_offsets
            out[:, b, p + n :] = position_ids[:, b, p:] + n
        return out

    def _location_embed_insertion_hook(self, module, args, kwargs):
        """Forward pre-hook that inserts location embeddings before visual tokens."""
        location_state = self._location_insertion_state
        if location_state is None:
            return args, kwargs

        # Skip during KV-cache steps (autoregressive generation after the first forward)
        if kwargs.get("past_key_values") is not None:
            return args, kwargs

        lat = location_state["lat"]
        lon = location_state["lon"]
        insert_positions = location_state["insert_positions"]
        device = lat.device

        # SatCLIP expects (B, 2) of (lon, lat) as float64
        coords = torch.stack([lon, lat], dim=-1).double()
        with torch.no_grad():
            loc_embed = self.satclip(coords).float()  # (B, satclip_dim)

        # Project to token space: (B, num_tokens, hidden_size)
        loc_tokens = self.location_modality_projection(loc_embed)

        # Insert location tokens immediately before the visual block.
        if "inputs_embeds" in kwargs and kwargs["inputs_embeds"] is not None:
            inputs_embeds = kwargs["inputs_embeds"]
            kwargs["inputs_embeds"] = self._insert_tokens_3d(inputs_embeds, loc_tokens, insert_positions)

            # Extend attention_mask
            if "attention_mask" in kwargs and kwargs["attention_mask"] is not None:
                B = inputs_embeds.shape[0]
                n = self.num_location_tokens
                ones = torch.ones(B, n, device=device, dtype=kwargs["attention_mask"].dtype)
                kwargs["attention_mask"] = self._insert_tokens_2d(
                    kwargs["attention_mask"], ones, insert_positions
                )

            # Extend position_ids (M-RoPE: shape [3, B, seq_len])
            if "position_ids" in kwargs and kwargs["position_ids"] is not None:
                kwargs["position_ids"] = self._insert_position_ids(
                    kwargs["position_ids"], insert_positions, self.num_location_tokens
                )

            # Extend visual_pos_masks (bool mask for deepstack visual processing)
            if kwargs.get("visual_pos_masks") is not None:
                B = inputs_embeds.shape[0]
                n = self.num_location_tokens
                pad = torch.zeros(B, n, device=device, dtype=kwargs["visual_pos_masks"].dtype)
                kwargs["visual_pos_masks"] = self._insert_tokens_2d(
                    kwargs["visual_pos_masks"], pad, insert_positions
                )

        return args, kwargs

    def _prepare_model_inputs(self, batch: Dict[str, Any]):
        """Strip non-model fields from batch and set up location state for loc_embed mode.

        Returns:
            Tuple of (batch, references, image_ids, lat, lon) where the last three
            may be None. lat/lon are raw tensors (not moved to device yet).
        """
        lat = batch.pop("lat", None)
        lon = batch.pop("lon", None)
        references = batch.pop("references", None)
        image_ids = batch.pop("image_ids", None)

        if self.loc_mode == "loc_embed":
            if lat is None or lon is None:
                raise ValueError("loc_mode='loc_embed' requires both lat and lon in the batch")
            input_ids = batch.get("input_ids")
            attention_mask = batch.get("attention_mask")
            if input_ids is None:
                raise ValueError("input_ids are required for loc_mode='loc_embed'")

            config = self.model.base_model.model.model.config
            image_token_id = config.image_token_id
            video_token_id = getattr(config, "video_token_id", None)
            visual_mask = input_ids.eq(image_token_id)
            if video_token_id is not None:
                visual_mask = visual_mask | input_ids.eq(video_token_id)

            has_visual = visual_mask.any(dim=1)
            first_visual = visual_mask.int().argmax(dim=1)
            if attention_mask is not None:
                fallback = attention_mask.sum(dim=1)
            else:
                fallback = torch.full_like(first_visual, input_ids.shape[1])
            insert_positions = torch.where(has_visual, first_visual, fallback)

            self._location_insertion_state = {
                "lat": lat.to(self.device),
                "lon": lon.to(self.device),
                "insert_positions": insert_positions.to(self.device),
            }

            # Insert ignore tokens at the location-token boundary.
            if "labels" in batch:
                B = batch["labels"].shape[0]
                n = self.num_location_tokens
                ignore = torch.full(
                    (B, n), -100, device=batch["labels"].device, dtype=batch["labels"].dtype
                )
                batch["labels"] = self._insert_tokens_2d(batch["labels"], ignore, insert_positions)

        return batch, references, image_ids, lat, lon

    def _set_datamodule_collator(self):
        """Attach the collator to the active datamodule once it exists."""
        if self._collator is None:
            return
        datamodule = getattr(self.trainer, "datamodule", None)
        if datamodule is not None and hasattr(datamodule, "set_collator"):
            datamodule.set_collator(self._collator)

    def forward(self, **inputs) -> Any:
        """Forward pass through the model."""
        return self.model(**inputs)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Training step."""
        batch, _, _, _, _ = self._prepare_model_inputs(batch)
        outputs = self.model(**batch)
        self._location_insertion_state = None
        self.log("train/loss", outputs.loss, on_step=True, on_epoch=True, prog_bar=True)
        return outputs.loss

    def _should_generate_val(self, batch_idx: int) -> bool:
        """Check whether to run generation for this validation batch."""
        if self.val_generate_batches == 0:
            return False
        if self.val_generate_batches == -1:
            return True
        return batch_idx < self.val_generate_batches

    def _generate_for_batch(self, batch: Dict[str, Any]) -> List[str]:
        """Run greedy generation on a batch and return decoded predictions."""
        gen_batch = {k: v for k, v in batch.items() if k != "labels"}
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
        input_len = gen_batch["input_ids"].shape[-1]
        predictions = []
        for i in range(generated_ids.shape[0]):
            text = self.tokenizer.decode(
                generated_ids[i, input_len:], skip_special_tokens=True
            )
            predictions.append(text)
        return predictions

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:
        """Validation step with loss computation and optional generation metrics."""
        batch, references, _, _, _ = self._prepare_model_inputs(batch)
        with torch.no_grad():
            outputs = self.model(**batch)

        self.log("val/loss", outputs.loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        result = {"loss": outputs.loss}

        if self._should_generate_val(batch_idx) and self.max_new_tokens > 0:
            try:
                predictions = self._generate_for_batch(batch)

                # Log a sample from the first batch
                if batch_idx == 0 and predictions:
                    self.print(f"\n[Val Sample] Generated: {predictions[0][:500]}...")

                # Accumulate for captioning metrics
                if self.val_captioning_metrics is not None and references is not None:
                    self.val_captioning_metrics.update(predictions, references)

                result["generated"] = predictions[0] if predictions else ""
            except Exception as e:
                self.print(f"[Val] Generation failed: {e}")

        self._location_insertion_state = None
        return result

    def on_validation_epoch_end(self) -> None:
        """Compute and log captioning metrics at the end of validation."""
        if self.val_captioning_metrics is not None and len(self.val_captioning_metrics.predictions) > 0:
            scores = self.val_captioning_metrics.compute()
            for name, value in scores.items():
                self.log(f"val/{name}", value, prog_bar=(name in ("bleu4", "cider")), sync_dist=True)
            self.val_captioning_metrics.reset()

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:
        """Test step — always generates and accumulates captioning metrics."""
        batch, references, image_ids, lat, lon = self._prepare_model_inputs(batch)
        with torch.no_grad():
            outputs = self.model(**batch)

        self.log("test/loss", outputs.loss, on_step=False, on_epoch=True, sync_dist=True)
        result = {"loss": outputs.loss}

        if self.max_new_tokens > 0:
            try:
                predictions = self._generate_for_batch(batch)

                if batch_idx == 0 and predictions:
                    self.print(f"\n[Test Sample] Generated: {predictions[0][:500]}...")

                if self.test_captioning_metrics is not None and references is not None:
                    self.test_captioning_metrics.update(predictions, references)

                # Accumulate per-sample predictions for JSON export
                if self.test_predictions_path and image_ids is not None:
                    for i, pred in enumerate(predictions):
                        entry = {
                            "image_id": image_ids[i],
                            "prediction": pred,
                            "references": references[i] if references else [],
                        }
                        if lat is not None:
                            entry["lat"] = float(lat[i])
                            entry["lon"] = float(lon[i])
                        self._test_predictions.append(entry)

                result["generated"] = predictions[0] if predictions else ""
            except Exception as e:
                self.print(f"[Test] Generation failed: {e}")

        self._location_insertion_state = None
        return result

    def on_test_epoch_end(self) -> None:
        """Compute and log captioning metrics at the end of testing."""
        if self.test_captioning_metrics is not None and len(self.test_captioning_metrics.predictions) > 0:
            scores = self.test_captioning_metrics.compute()
            for name, value in scores.items():
                self.log(f"test/{name}", value, sync_dist=True)
            self.test_captioning_metrics.reset()

        # Save per-sample predictions to JSON
        if self.test_predictions_path and self._test_predictions:
            Path(self.test_predictions_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.test_predictions_path, "w") as f:
                json.dump(self._test_predictions, f, indent=2)
            self.print(f"Saved {len(self._test_predictions)} predictions to {self.test_predictions_path}")
            self._test_predictions = []

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        decay_params = []
        no_decay_params = []

        # Collect params from model + location_modality_projection (if present)
        all_named_params = list(self.model.named_parameters())
        if self.location_modality_projection is not None:
            all_named_params.extend(
                (f"location_modality_projection.{n}", p)
                for n, p in self.location_modality_projection.named_parameters()
            )

        for name, param in all_named_params:
            if not param.requires_grad:
                continue
            if "bias" in name or "LayerNorm" in name or "layer_norm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer = bnb.optim.AdamW8bit(
            [
                {"params": decay_params, "weight_decay": self.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=self.learning_rate,
        )

        if self.max_steps is not None:
            total_steps = self.max_steps
        elif self.trainer is not None and self.trainer.max_steps > 0:
            total_steps = self.trainer.max_steps
        elif self.trainer is not None and hasattr(self.trainer, "estimated_stepping_batches"):
            total_steps = self.trainer.estimated_stepping_batches
        else:
            total_steps = 10000
            self.print(
                "WARNING: Could not determine total training steps. "
                "Defaulting to 10000 for LR schedule. Set max_steps or trainer.max_steps explicitly."
            )

        warmup_steps = int(total_steps * self.warmup_ratio)

        scheduler = SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps),
                CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=self.learning_rate * 0.1),
            ],
            milestones=[warmup_steps],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }
