# LightningDataModule for Vision-Language Model Finetuning (File-based variant)
# Compatible with Unsloth's FastVisionModel and UnslothVisionDataCollator
# Docs: https://lightning.ai/docs/pytorch/stable/data/datamodule.html

import os
import random
from pathlib import Path
from typing import Any, Callable, Dict, Literal, Optional

import lightning as L
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class VLMDataset(Dataset):
    """
    Dataset for Vision-Language Model finetuning using file-based image loading.

    Returns data in Unsloth's expected conversation format:
        {"messages": [
            {"role": "user", "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image", "image": PIL.Image}
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": caption}
            ]}
        ]}

    Directory Structure:
        image_dir/
            image1.jpg
            image2.png
            ...

    Metadata Structure (CSV or Parquet):
        - image_id: filename or path relative to image_dir
        - caption: text description of the image
        - split: one of 'train', 'validation', 'test'
    """

    def __init__(
        self,
        image_dir: str,
        metadata_path: Optional[str] = None,
        split: Optional[Literal['train', 'validation', 'test']] = None,
        id_column: str = 'image_id',
        caption_column: str = 'caption',
        file_extension: Optional[str] = None,
        system_prompt: Optional[str] = None,
        user_prompt: str = "Describe this image in detail.",
        multi_caption: bool = False,
        lat_column: Optional[str] = None,
        lon_column: Optional[str] = None,
        metadata: Optional[pd.DataFrame] = None,
    ):
        """
        Initialize VLM Dataset with file-based image loading.

        Args:
            image_dir: Path to directory containing images
            metadata_path: Path to metadata file (CSV or Parquet)
            split: Dataset split to use ('train', 'validation', 'test', or None for all)
            id_column: Column name for image IDs/filenames
            caption_column: Column name for text captions
            file_extension: File extension to append to image_id if not already present
            system_prompt: Optional system prompt for the conversation
            user_prompt: User prompt template for the image description task
            multi_caption: If True, caption_column contains a list; pick random per access
            lat_column: Column name for latitude (enables geo features)
            lon_column: Column name for longitude (enables geo features)
            metadata: Pre-loaded DataFrame (skips reading metadata_path if provided)
        """
        self.image_dir = Path(image_dir)
        self.id_column = id_column
        self.caption_column = caption_column
        self.file_extension = file_extension
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.multi_caption = multi_caption
        self.lat_column = lat_column
        self.lon_column = lon_column

        # Use pre-loaded metadata or read from file
        if metadata is not None:
            self.metadata = metadata
        elif metadata_path is not None:
            if metadata_path.endswith('.parquet'):
                self.metadata = pd.read_parquet(metadata_path)
            else:
                self.metadata = pd.read_csv(metadata_path)
        else:
            raise ValueError("Either metadata_path or metadata must be provided")

        if split is not None:
            self.metadata = self.metadata[self.metadata['split'] == split].reset_index(drop=True)

        self.keys = self.metadata[id_column].tolist()
        self.keys.sort()  # Ensure reproducibility

        # Create mapping for fast lookup
        self._key_to_idx = {k: i for i, k in enumerate(self.metadata[id_column].tolist())}

        # Resolve all image paths once at init
        self._image_paths = {k: self._resolve_image_path(k) for k in self.keys}

    def __len__(self) -> int:
        return len(self.keys)

    def _resolve_image_path(self, image_id: str) -> Path:
        """Resolve full image path from image_id (called once per image at init)."""
        image_path = self.image_dir / image_id
        if image_path.exists():
            return image_path

        if self.file_extension:
            image_path = self.image_dir / f"{image_id}{self.file_extension}"
            if image_path.exists():
                return image_path

        for ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp']:
            candidate = self.image_dir / f"{image_id}{ext}"
            if candidate.exists():
                return candidate

        raise FileNotFoundError(
            f"Image not found for id '{image_id}' in {self.image_dir}. "
            f"Tried: {image_id}, {image_id}.jpg, {image_id}.png, etc."
        )

    def _load_image(self, image_id: str) -> Image.Image:
        """Load image as PIL Image (required by Unsloth)."""
        image_path = self._image_paths[image_id]

        # Load image with PIL
        img = Image.open(image_path)

        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')

        return img

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get an item from the dataset in Unsloth conversation format.

        Returns:
            Dict containing:
                - 'messages': List of conversation messages with image and text
                - 'image_id': Image identifier (for reference)
                - 'lat', 'lon': float values if lat_column/lon_column are set
        """
        image_id = self.keys[idx]

        # Load PIL image (Unsloth handles preprocessing)
        image = self._load_image(image_id)

        # Get caption from metadata
        metadata_idx = self._key_to_idx[image_id]
        row = self.metadata.iloc[metadata_idx]

        if self.multi_caption:
            captions = row[self.caption_column]
            caption = random.choice(captions) if isinstance(captions, list) else str(captions)
        else:
            caption = str(row[self.caption_column])

        # Resolve user prompt — format with lat/lon if placeholders exist
        user_prompt = self.user_prompt
        lat, lon = None, None
        if self.lat_column and self.lon_column:
            lat = float(row[self.lat_column])
            lon = float(row[self.lon_column])
            try:
                user_prompt = user_prompt.format(lat=lat, lon=lon)
            except (KeyError, IndexError):
                pass  # no placeholders in prompt — that's fine (baseline mode)

        # Build conversation in Unsloth format
        messages = []

        # Optional system message
        if self.system_prompt:
            messages.append({
                "role": "system",
                "content": self.system_prompt
            })

        # User message with image
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image", "image": image},
            ]
        })

        # Assistant response
        messages.append({
            "role": "assistant",
            "content": [
                {"type": "text", "text": caption}
            ]
        })

        # Collect all reference captions for evaluation
        if self.multi_caption:
            captions_raw = row[self.caption_column]
            references = list(captions_raw) if isinstance(captions_raw, list) else [str(captions_raw)]
        else:
            references = [caption]

        item = {
            "messages": messages,
            "image_id": image_id,
            "references": references,
        }

        if lat is not None:
            item["lat"] = lat
            item["lon"] = lon

        return item



class VLMDataModule(L.LightningDataModule):
    """
    PyTorch Lightning DataModule for Vision-Language Model finetuning.
    Compatible with Unsloth's FastVisionModel and UnslothVisionDataCollator.

    Note: The collate_fn must be set externally after model initialization,
    as UnslothVisionDataCollator requires the model and tokenizer.
    Use set_collate_fn() method after model setup.
    """

    def __init__(
        self,
        image_dir: str,
        metadata_path: str,
        batch_size: int = 1,
        num_workers: int = 4,
        id_column: str = 'image_id',
        caption_column: str = 'caption',
        file_extension: Optional[str] = None,
        system_prompt: Optional[str] = None,
        user_prompt: str = "Describe this image in detail.",
        pin_memory: bool = True,
        persistent_workers: bool = True,
        multi_caption: bool = False,
        lat_column: Optional[str] = None,
        lon_column: Optional[str] = None,
    ):
        """
        Initialize VLM DataModule.

        Args:
            image_dir: Path to directory containing images
            metadata_path: Path to metadata file (CSV or Parquet)
            batch_size: Batch size for dataloaders (recommend 1-2 for VLMs)
            num_workers: Number of workers for dataloaders
            id_column: Column name for image IDs/filenames
            caption_column: Column name for text captions
            file_extension: File extension to append to image_id if not present
            system_prompt: Optional system prompt for conversations
            user_prompt: User prompt template for image description
            pin_memory: Whether to pin memory in dataloaders
            persistent_workers: Whether to keep workers alive between epochs
            multi_caption: If True, caption_column is a list; pick random per access
            lat_column: Column name for latitude (enables geo features)
            lon_column: Column name for longitude (enables geo features)
        """
        super().__init__()
        self.save_hyperparameters()

        self.image_dir = image_dir
        self.metadata_path = metadata_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.id_column = id_column
        self.caption_column = caption_column
        self.file_extension = file_extension
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        self.multi_caption = multi_caption
        self.lat_column = lat_column
        self.lon_column = lon_column

        # Collate function (set by lightning module after model init)
        self._collate_fn: Optional[Callable] = None

        # Dataset instances (initialized in setup)
        self.train_dataset: Optional[VLMDataset] = None
        self.val_dataset: Optional[VLMDataset] = None
        self.test_dataset: Optional[VLMDataset] = None
        self.predict_dataset: Optional[VLMDataset] = None

    def set_collate_fn(self, collate_fn: Callable):
        """
        Set the collate function for dataloaders.

        This should be called with UnslothVisionDataCollator after model setup:
            from unsloth.trainer import UnslothVisionDataCollator
            data_module.set_collate_fn(UnslothVisionDataCollator(model, tokenizer))
        """
        self._collate_fn = collate_fn

    def prepare_data(self):
        """
        Prepare data (download, verify files exist, etc.).
        Called only on rank 0 in distributed settings.
        """
        if not os.path.exists(self.image_dir):
            raise FileNotFoundError(f"Image directory not found at: {self.image_dir}")
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata file not found at: {self.metadata_path}")

    def _load_metadata(self) -> pd.DataFrame:
        """Load metadata file once (CSV or Parquet)."""
        if self.metadata_path.endswith('.parquet'):
            return pd.read_parquet(self.metadata_path)
        return pd.read_csv(self.metadata_path)

    def setup(self, stage: Optional[str] = None):
        """
        Setup datasets for each stage.
        Called on every GPU in distributed settings.
        """
        metadata = self._load_metadata()

        common_kwargs = {
            'image_dir': self.image_dir,
            'metadata': metadata,
            'id_column': self.id_column,
            'caption_column': self.caption_column,
            'file_extension': self.file_extension,
            'system_prompt': self.system_prompt,
            'user_prompt': self.user_prompt,
            'multi_caption': self.multi_caption,
            'lat_column': self.lat_column,
            'lon_column': self.lon_column,
        }

        if stage == 'fit' or stage is None:
            self.train_dataset = VLMDataset(
                split='train',
                **common_kwargs
            )
            self.val_dataset = VLMDataset(
                split='validation',
                **common_kwargs
            )

        if stage == 'validate':
            self.val_dataset = VLMDataset(
                split='validation',
                **common_kwargs
            )

        if stage == 'test' or stage is None:
            self.test_dataset = VLMDataset(
                split='test',
                **common_kwargs
            )

        if stage == 'predict':
            self.predict_dataset = VLMDataset(
                split='test',
                **common_kwargs
            )

    def _create_dataloader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        """Create a dataloader with common settings."""
        if self._collate_fn is None:
            raise RuntimeError(
                "Collate function not set. Call set_collate_fn() with "
                "UnslothVisionDataCollator after model initialization."
            )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=self._collate_fn,
        )

    def train_dataloader(self) -> DataLoader:
        return self._create_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._create_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._create_dataloader(self.test_dataset, shuffle=False)

    def predict_dataloader(self) -> DataLoader:
        return self._create_dataloader(self.predict_dataset, shuffle=False)
