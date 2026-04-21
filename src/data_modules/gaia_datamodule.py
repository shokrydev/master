# LightningDataModule for GAIA captioning with optional geolocation conditioning

import json
import random
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Literal

import lightning as L
import numpy as np
import pandas as pd
import torch
import webdataset as wds
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

SPLIT_LAYOUT: dict[str, tuple[str, str]] = {
    "train": ("train", "train_data.json"),
    "validation": ("val", "val_data.json"),
    "test": ("test", "test_data.json"),
}

IMAGE_KEY = "png"

ImageFile.LOAD_TRUNCATED_IMAGES = True


def _load_manifest_records(path: Path) -> list[dict[str, object]]:
    records = json.loads(path.read_text())
    if not isinstance(records, list):
        raise ValueError(f"Expected GAIA manifest to be a JSON list: {path}")
    return records


def _normalize_captions(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        return [value]
    if value is None or pd.isna(value):
        return []
    return [str(value)]


def _build_target_texts(
    references: list[str], *, multi_caption: bool, rng: random.Random
) -> list[str]:
    """Return target texts with the first entry used as supervised target.

    For multi-caption mode we randomize which caption appears first, while still
    keeping the full set available for multi-reference evaluation.
    """
    if not references:
        raise ValueError("No caption references available")

    if not multi_caption or len(references) == 1:
        return [references[0]]

    targets = list(references)
    selected_idx = rng.randrange(len(targets))
    targets[0], targets[selected_idx] = targets[selected_idx], targets[0]
    return targets


class GAIADataset(IterableDataset):
    """Iterable dataset for the official shard-based GAIA layout."""

    def __init__(
        self,
        gaia_root: str,
        split: Literal["train", "validation", "test"],
        id_column: str = "id",
        caption_column: str = "captions",
        system_prompt: str | None = None,
        user_prompt: str = "Describe this image in detail.",
        multi_caption: bool = False,
        lat_column: str | None = None,
        lon_column: str | None = None,
        coordinate_perturbation: Literal["shuffled", "antipodal"] | None = None,
        shuffle_samples: bool = False,
    ):
        self.gaia_root = Path(gaia_root)
        self.split = split
        self.id_column = id_column
        self.caption_column = caption_column
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.multi_caption = multi_caption
        self.lat_column = lat_column
        self.lon_column = lon_column
        self.coordinate_perturbation = coordinate_perturbation
        self.shuffle_samples = shuffle_samples

        split_dir_name, manifest_name = SPLIT_LAYOUT[split]
        self.shards_dir = self.gaia_root / split_dir_name
        self.manifest_path = self.gaia_root / manifest_name

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"GAIA manifest not found at: {self.manifest_path}")
        if not self.shards_dir.exists():
            raise FileNotFoundError(f"GAIA shard directory not found at: {self.shards_dir}")

        self._shard_paths = sorted(self.shards_dir.glob("*.tar"))
        if not self._shard_paths:
            raise FileNotFoundError(f"No GAIA shard tar files found in: {self.shards_dir}")

        self.metadata = pd.DataFrame(_load_manifest_records(self.manifest_path))
        if self.metadata.empty:
            raise ValueError(f"No records found in GAIA manifest: {self.manifest_path}")
        if self.id_column not in self.metadata.columns:
            raise KeyError(f"Expected id column '{self.id_column}' in {self.manifest_path}")
        if self.caption_column not in self.metadata.columns:
            raise KeyError(f"Expected caption column '{self.caption_column}' in {self.manifest_path}")

        if not lat_column or not lon_column:
            raise ValueError("GAIADataset requires both lat_column and lon_column")

        if coordinate_perturbation:
            if coordinate_perturbation == "shuffled":
                rng = np.random.RandomState(42)
                perm = rng.permutation(len(self.metadata))
                self.metadata[lat_column] = self.metadata[lat_column].values[perm]
                self.metadata[lon_column] = self.metadata[lon_column].values[perm]
            elif coordinate_perturbation == "antipodal":
                self.metadata[lat_column] = -self.metadata[lat_column]
                lon = self.metadata[lon_column].values
                self.metadata[lon_column] = np.where(lon <= 0, lon + 180, lon - 180)

        self.keys = self.metadata[self.id_column].tolist()
        self._metadata_by_id = {
            row[self.id_column]: row.to_dict() for _, row in self.metadata.iterrows()
        }

    def __len__(self) -> int:
        return len(self.keys)

    def _build_pipeline(self):
        shard_paths = [str(path) for path in self._shard_paths]
        dataset = wds.WebDataset(
            shard_paths,
            shardshuffle=100 if self.shuffle_samples else 0,
            workersplitter=wds.split_by_worker,
        ).decode("pil")
        if self.shuffle_samples:
            dataset = dataset.shuffle(1024)
        return dataset

    def _read_sample_json(self, sample: dict[str, object]) -> dict[str, object]:
        value = sample.get("json")
        if value is None:
            return {}
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            return json.loads(value)
        if isinstance(value, bytes):
            return json.loads(value.decode("utf-8"))
        raise TypeError(f"Unsupported GAIA shard json payload type: {type(value)!r}")

    def _read_sample_text(self, sample: dict[str, object]) -> str | None:
        value = sample.get("txt")
        if value is None:
            return None
        if isinstance(value, bytes):
            return value.decode("utf-8").strip()
        return str(value).strip()

    def _read_sample_image(self, sample: dict[str, object]) -> Image.Image:
        image = sample.get(IMAGE_KEY)
        if image is None:
            raise FileNotFoundError(f"GAIA shard sample is missing '{IMAGE_KEY}' image data")
        if not isinstance(image, Image.Image):
            raise TypeError(f"Expected PIL image for key '{IMAGE_KEY}', got {type(image)!r}")
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image

    def _select_targets(self, references: list[str], text_fallback: str | None, rng: random.Random) -> list[str]:
        if not references and text_fallback:
            references = [text_fallback]
        if not references:
            raise ValueError(f"No caption references available for GAIA split '{self.split}'")
        return _build_target_texts(references, multi_caption=self.multi_caption, rng=rng)

    def _build_item(
        self,
        image: Image.Image,
        sample_json: dict[str, object],
        text_fallback: str | None,
        rng: random.Random,
    ) -> dict[str, object]:
        sample_id = sample_json.get(self.id_column)
        if sample_id is None:
            raise KeyError(
                f"GAIA shard sample is missing id column '{self.id_column}' in its JSON sidecar"
            )

        row = self._metadata_by_id.get(sample_id)
        if row is None:
            raise KeyError(f"Image id '{sample_id}' not found in GAIA manifest {self.manifest_path}")

        references = _normalize_captions(row.get(self.caption_column))
        target_texts = self._select_targets(references, text_fallback, rng)

        user_prompt = self.user_prompt
        lat_value = row.get(self.lat_column)
        lon_value = row.get(self.lon_column)
        if lat_value is None or lon_value is None:
            raise ValueError(
                f"GAIA sample '{sample_id}' is missing required coordinates "
                f"('{self.lat_column}', '{self.lon_column}')"
            )
        lat = float(lat_value)
        lon = float(lon_value)

        user_prompt = user_prompt.format(lat=lat, lon=lon)
        if self.system_prompt:
            # Keep the configured system instruction while using the normalized schema.
            user_prompt = f"{self.system_prompt}\n\n{user_prompt}"

        return {
            "image": image,
            "input_text": user_prompt,
            "target_texts": target_texts,
            "lat": lat,
            "lon": lon,
        }

    def __iter__(self) -> Iterator[dict[str, object]]:
        worker_info = get_worker_info()
        seed = worker_info.seed if worker_info is not None else torch.initial_seed()
        rng = random.Random(seed)

        for sample in self._build_pipeline():
            sample_json = self._read_sample_json(sample)
            image = self._read_sample_image(sample)
            text_fallback = self._read_sample_text(sample)
            yield self._build_item(image, sample_json, text_fallback, rng)


class GAIADataModule(L.LightningDataModule):
    """LightningDataModule for GAIA captioning experiments."""

    def __init__(
        self,
        gaia_root: str,
        batch_size: int = 1,
        num_workers: int = 4,
        id_column: str = "id",
        caption_column: str = "captions",
        system_prompt: str | None = None,
        user_prompt: str = "Describe this image in detail.",
        pin_memory: bool = True,
        persistent_workers: bool = True,
        multi_caption: bool = False,
        lat_column: str | None = None,
        lon_column: str | None = None,
        coordinate_perturbation: Literal["shuffled", "antipodal"] | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.gaia_root = gaia_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.id_column = id_column
        self.caption_column = caption_column
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        self.multi_caption = multi_caption
        self.lat_column = lat_column
        self.lon_column = lon_column
        self.coordinate_perturbation = coordinate_perturbation

        self._collator: Callable | None = None
        self.train_dataset: GAIADataset | None = None
        self.val_dataset: GAIADataset | None = None
        self.test_dataset: GAIADataset | None = None
        self.predict_dataset: GAIADataset | None = None

    def set_collator(self, collator: Callable):
        self._collator = collator

    def _build_dataset(
        self,
        split: Literal["train", "validation", "test"],
        *,
        shuffle_samples: bool,
    ) -> GAIADataset:
        return GAIADataset(
            gaia_root=self.gaia_root,
            split=split,
            id_column=self.id_column,
            caption_column=self.caption_column,
            system_prompt=self.system_prompt,
            user_prompt=self.user_prompt,
            multi_caption=self.multi_caption,
            lat_column=self.lat_column,
            lon_column=self.lon_column,
            coordinate_perturbation=self.coordinate_perturbation,
            shuffle_samples=shuffle_samples,
        )

    def setup(self, stage: str | None = None):
        if stage in ("fit", None):
            self.train_dataset = self._build_dataset("train", shuffle_samples=True)
            self.val_dataset = self._build_dataset("validation", shuffle_samples=False)
        elif stage == "validate":
            self.val_dataset = self._build_dataset("validation", shuffle_samples=False)
        elif stage == "test":
            self.test_dataset = self._build_dataset("test", shuffle_samples=False)
        elif stage == "predict":
            self.predict_dataset = self._build_dataset("test", shuffle_samples=False)

    def _create_dataloader(self, dataset: IterableDataset) -> DataLoader:
        if self._collator is None:
            raise RuntimeError(
                "Collator not set. Call set_collator() with "
                "UnslothVisionDataCollator after model initialization."
            )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=self._collator,
        )

    def train_dataloader(self) -> DataLoader:
        return self._create_dataloader(self.train_dataset)

    def val_dataloader(self) -> DataLoader:
        return self._create_dataloader(self.val_dataset)

    def test_dataloader(self) -> DataLoader:
        return self._create_dataloader(self.test_dataset)

    def predict_dataloader(self) -> DataLoader:
        return self._create_dataloader(self.predict_dataset)
