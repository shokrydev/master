# LightningDataModule for GAIA captioning with optional geolocation conditioning

import os
import random
from pathlib import Path
from typing import Any, Callable, Dict, Literal, Optional

import lightning as L
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class GAIADataset(Dataset):
    """Dataset for GAIA remote-sensing captioning.

    Returns data in Unsloth's expected conversation format and re-attaches GAIA
    metadata needed for evaluation and optional geolocation conditioning.
    """

    def __init__(
        self,
        image_dir: str,
        metadata_path: Optional[str] = None,
        split: Optional[Literal["train", "validation", "test"]] = None,
        id_column: str = "image_id",
        caption_column: str = "caption",
        file_extension: Optional[str] = None,
        system_prompt: Optional[str] = None,
        user_prompt: str = "Describe this image in detail.",
        multi_caption: bool = False,
        lat_column: Optional[str] = None,
        lon_column: Optional[str] = None,
        coordinate_perturbation: Optional[Literal["shuffled", "antipodal"]] = None,
        metadata: Optional[pd.DataFrame] = None,
    ):
        self.image_dir = Path(image_dir)
        self.id_column = id_column
        self.caption_column = caption_column
        self.file_extension = file_extension
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.multi_caption = multi_caption
        self.lat_column = lat_column
        self.lon_column = lon_column

        if metadata is not None:
            self.metadata = metadata
        elif metadata_path is not None:
            if metadata_path.endswith(".parquet"):
                self.metadata = pd.read_parquet(metadata_path)
            else:
                self.metadata = pd.read_csv(metadata_path)
        else:
            raise ValueError("Either metadata_path or metadata must be provided")

        if split is not None:
            self.metadata = self.metadata[self.metadata["split"] == split].reset_index(drop=True)

        if coordinate_perturbation and lat_column and lon_column:
            if coordinate_perturbation == "shuffled":
                rng = np.random.RandomState(42)
                perm = rng.permutation(len(self.metadata))
                self.metadata[lat_column] = self.metadata[lat_column].values[perm]
                self.metadata[lon_column] = self.metadata[lon_column].values[perm]
            elif coordinate_perturbation == "antipodal":
                self.metadata[lat_column] = -self.metadata[lat_column]
                lon = self.metadata[lon_column].values
                self.metadata[lon_column] = np.where(lon <= 0, lon + 180, lon - 180)

        self.keys = self.metadata[id_column].tolist()
        self.keys.sort()
        self._key_to_idx = {k: i for i, k in enumerate(self.metadata[id_column].tolist())}
        self._image_paths = {k: self._resolve_image_path(k) for k in self.keys}

    def __len__(self) -> int:
        return len(self.keys)

    def _resolve_image_path(self, image_id: str) -> Path:
        image_path = self.image_dir / image_id
        if image_path.exists():
            return image_path

        if self.file_extension:
            image_path = self.image_dir / f"{image_id}{self.file_extension}"
            if image_path.exists():
                return image_path

        for ext in [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"]:
            candidate = self.image_dir / f"{image_id}{ext}"
            if candidate.exists():
                return candidate

        raise FileNotFoundError(
            f"Image not found for id '{image_id}' in {self.image_dir}. "
            f"Tried: {image_id}, {image_id}.jpg, {image_id}.png, etc."
        )

    def _load_image(self, image_id: str) -> Image.Image:
        image = Image.open(self._image_paths[image_id])
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image_id = self.keys[idx]
        image = self._load_image(image_id)

        metadata_idx = self._key_to_idx[image_id]
        row = self.metadata.iloc[metadata_idx]

        if self.multi_caption:
            captions = row[self.caption_column]
            caption = random.choice(captions) if isinstance(captions, list) else str(captions)
        else:
            caption = str(row[self.caption_column])

        user_prompt = self.user_prompt
        lat, lon = None, None
        if self.lat_column and self.lon_column:
            lat = float(row[self.lat_column])
            lon = float(row[self.lon_column])
            try:
                user_prompt = user_prompt.format(lat=lat, lon=lon)
            except (KeyError, IndexError):
                pass

        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image", "image": image},
                ],
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": caption}],
            }
        )

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


class GAIADataModule(L.LightningDataModule):
    """LightningDataModule for GAIA captioning experiments."""

    def __init__(
        self,
        image_dir: str,
        metadata_path: str,
        batch_size: int = 1,
        num_workers: int = 4,
        id_column: str = "image_id",
        caption_column: str = "caption",
        file_extension: Optional[str] = None,
        system_prompt: Optional[str] = None,
        user_prompt: str = "Describe this image in detail.",
        pin_memory: bool = True,
        persistent_workers: bool = True,
        multi_caption: bool = False,
        lat_column: Optional[str] = None,
        lon_column: Optional[str] = None,
        coordinate_perturbation: Optional[Literal["shuffled", "antipodal"]] = None,
    ):
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
        self.coordinate_perturbation = coordinate_perturbation

        self._collate_fn: Optional[Callable] = None
        self.train_dataset: Optional[GAIADataset] = None
        self.val_dataset: Optional[GAIADataset] = None
        self.test_dataset: Optional[GAIADataset] = None
        self.predict_dataset: Optional[GAIADataset] = None

    def set_collate_fn(self, collate_fn: Callable):
        self._collate_fn = collate_fn

    def prepare_data(self):
        if not os.path.exists(self.image_dir):
            raise FileNotFoundError(f"Image directory not found at: {self.image_dir}")
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata file not found at: {self.metadata_path}")

    def _load_metadata(self) -> pd.DataFrame:
        if self.metadata_path.endswith(".parquet"):
            return pd.read_parquet(self.metadata_path)
        return pd.read_csv(self.metadata_path)

    def setup(self, stage: Optional[str] = None):
        metadata = self._load_metadata()
        common_kwargs = {
            "image_dir": self.image_dir,
            "metadata": metadata,
            "id_column": self.id_column,
            "caption_column": self.caption_column,
            "file_extension": self.file_extension,
            "system_prompt": self.system_prompt,
            "user_prompt": self.user_prompt,
            "multi_caption": self.multi_caption,
            "lat_column": self.lat_column,
            "lon_column": self.lon_column,
            "coordinate_perturbation": self.coordinate_perturbation,
        }

        if stage == "fit" or stage is None:
            self.train_dataset = GAIADataset(split="train", **common_kwargs)
            self.val_dataset = GAIADataset(split="validation", **common_kwargs)
        if stage == "validate":
            self.val_dataset = GAIADataset(split="validation", **common_kwargs)
        if stage == "test" or stage is None:
            self.test_dataset = GAIADataset(split="test", **common_kwargs)
        if stage == "predict":
            self.predict_dataset = GAIADataset(split="test", **common_kwargs)

    def _create_dataloader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
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
