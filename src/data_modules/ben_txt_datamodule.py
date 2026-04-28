"""BigEarthNet.txt datamodule, based on the dataset publication implementation.

Upstream source:
https://huggingface.co/datasets/BIFOLD-BigEarthNetv2-0/BigEarthNet.txt/blob/main/ben_txt_datamodule.py
"""

from collections.abc import Callable, Iterable
from pathlib import Path

import lmdb
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.numpy import load as safetensor_load
from torch.utils.data import DataLoader, Dataset

try:
    import lightning.pytorch as pl
except ImportError:
    import pytorch_lightning as pl

_s1_bandnames = ["VV", "VH"]
_s2_bandnames = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
_predefined_bandcombinations = {
    "RGB": ["B04", "B03", "B02"],
    "S2-10m20m": ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"],
    "S1S2-10m20m": ["VV", "VH", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"],
    "all":  _s1_bandnames + _s2_bandnames,
}
_rgb_band_order = ["B04", "B03", "B02"]

def collate_normalized(batch):
    images = []
    input_texts = []
    target_texts = []
    latitudes = []
    longitudes = []

    for item in batch:
        images.append(item["image"])
        input_texts.append(item["input_text"])
        target_texts.append(item["target_texts"])
        latitudes.append(item["lat"])
        longitudes.append(item["lon"])

    return {
        "image": images,
        "input_text": input_texts,
        "target_texts": target_texts,
        "lat": torch.tensor(latitudes, dtype=torch.float64),
        "lon": torch.tensor(longitudes, dtype=torch.float64),
    }


def _sentinel2_rgb_tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    """Render Sentinel-2 RGB bands into a deterministic PIL image for VLM input.

    BigEarthNet reflectance values are not stored as byte RGB images. For the
    shared Qwen/Unsloth path we apply a fixed 0..3000 stretch so the dataset
    emits an actual RGB image rather than classifier-style normalized tensors.
    """
    if image_tensor.ndim != 3 or image_tensor.shape[0] != 3:
        raise ValueError(
            "Expected Sentinel-2 RGB tensor with shape (3, H, W) for VLM image rendering"
        )

    image_array = image_tensor.detach().cpu().numpy().astype(np.float32)
    image_array = np.clip(image_array, 0.0, 3000.0) / 3000.0
    image_array = (image_array * 255.0).round().astype(np.uint8)
    image_array = np.transpose(image_array, (1, 2, 0))
    return Image.fromarray(image_array, mode="RGB")


class BENImageReader:
    def __init__(
            self,
            image_lmdb_file: str | Path,
            metadata_file: str | Path,
            bands: Iterable[str],
            img_size: int = 120,
            upsample_mode: str = "nearest",
            info_fn: Callable | None = lambda x: x,
    ):
        self.img_size = img_size
        self.upsample_mode = upsample_mode
        self.image_lmdb_file = image_lmdb_file
        self.bands = bands
        self.env = None

        info_fn(f"Using bandorder {self.bands}")
        self.uses_s1 = any(x in _s1_bandnames for x in self.bands)
        self.uses_s2 = any(x in _s2_bandnames for x in self.bands)

        metadata = pd.read_parquet(metadata_file)
        self.mapping = dict(zip(metadata["patch_id"], metadata["s1_name"], strict=True))
        info_fn("S1-S2 mapping created")

    def stack_and_interpolate(
            self,
            data: dict[str, np.ndarray],
    ) -> np.array:
        def _interpolate(img_data):
            if not img_data.shape[-2:] == (self.img_size, self.img_size):
                return F.interpolate(
                    torch.Tensor(np.float32(img_data)).unsqueeze(0).unsqueeze(0),
                    (self.img_size, self.img_size),
                    mode=self.upsample_mode,
                    align_corners=True if self.upsample_mode in ["bilinear", "bicubic"] else None,
                ).squeeze()
            else:
                return torch.Tensor(np.float32(img_data))

        return torch.stack([_interpolate(data[x]) for x in self.bands])

    def open_env(self):
        if self.env is None:
            print("Opening LMDB environment ...")
            self.env = lmdb.open(
                str(self.image_lmdb_file),
                readonly=True,
                lock=False,
                meminit=False,
                readahead=True,
                map_size=8 * 1024 ** 3,  # 8GB blocked for caching
                max_spare_txns=16,  # expected number of concurrent transactions (e.g. threads/workers)
            )

    def __getitem__(self, key: str):
        # open lmdb file if not opened yet
        self.open_env()
        img_data_dict: dict = {}
        if self.uses_s2:
            assert self.env is not None, "Environment not opened yet"
            # read image data for S2v2
            with self.env.begin(write=False, buffers=True) as txn:
                byte_data = txn.get(key.encode())
            img_data_dict.update(safetensor_load(bytes(byte_data)))

        if self.uses_s1:
            # read image data for S1
            assert self.mapping is not None, "S1 bands are used, but no mapping is provided"
            s1_key = self.mapping[key]
            assert self.env is not None, "Environment not opened yet"
            with self.env.begin(write=False, buffers=True) as txn:
                byte_data = txn.get(s1_key.encode())
            img_data_dict.update(safetensor_load(bytes(byte_data)))

        img_data_dict = {k: v for k, v in img_data_dict.items() if k in self.bands}

        img_data = self.stack_and_interpolate(img_data_dict)
        return img_data


class BENTxTDataset(Dataset):
    """
    PyTorch Dataset for BigEarthNet.txt.

    This dataset class loads the textual annotations from BigEarthNet.txt
    together with RGB Sentinel-2 imagery from BigEarthNet-v2.0 (converted to
    LMDB format) and emits the shared repo sample schema for the VLM path. It
    supports various filtering options to create custom dataset splits based on
    textual annotation metadata, such as type or category, or image metadata
    like country, season, and climate zone.
    """
    _expected_columns = {'s1_name', 'output', 'longitude', 'country', 'climate_zone', 'type', 'input', 'split', 'latitude', 'ID', 'patch_id', 'category', 'season'}

    def __init__(
            self,
            lmdb_file: str | Path,
            metadata_file: str | Path,
            bands: Iterable[str] | str | None = "RGB",
            img_size: int = 120,
            upsample_mode: str = "nearest",
            types: Iterable[str] | None = None,
            categories: Iterable[str] | None = None,
            countries: Iterable[str] | None = None,
            seasons: Iterable[str] | None = None,
            climate_zones: Iterable[str] | None = None,
            transform: Callable | None = None,
            splits: Iterable[str] | None = None,
            point_token: str | None = None,
            ref_token: str | None = None,
            info_fn: Callable = lambda x: x,
    ):
        """
        Initialize the BigEarthNet.txt Dataset.

        Args:
            lmdb_file: Path to the LMDB file containing the BigEarthNet-v2.0 image data.
            metadata_file: Path to the BigEarthNet.txt Parquet file.
            bands: Band names to load. The current shared VLM path requires RGB
                Sentinel-2 bands ('B04', 'B03', 'B02'); defaults to 'RGB'.
            img_size: Target image size for interpolation (default: 120).
            upsample_mode: Interpolation mode for resizing ('nearest', 'bilinear', 'bicubic', etc.).
                Default: 'nearest'.
            types: Optional filter for annotation types (e.g., 'binary', 'mcq', 'captioning', 'bounding box').
            categories: Optional filter for annotation categories. See [here](https://huggingface.co/datasets/BIFOLD-BigEarthNetv2-0/BigEarthNet.txt/sql-console/8okbuKf) for possible type-category combinations or retrieve them by yourself using some kind of database tool on the Parquet file.
            countries: Optional filter for acquisition countries (e.g., 'Austria', 'Belgium', 'Finland', 'Ireland', 'Kosovo', 'Lithuania', 'Luxembourg', 'Portugal', 'Serbia', 'Switzerland').
            seasons: Optional filter for seasons (e.g., 'Spring', 'Summer', 'Fall', 'Winter').
            climate_zones: Optional filter for climate zones. See [here](https://huggingface.co/datasets/BIFOLD-BigEarthNetv2-0/BigEarthNet.txt/sql-console/3xLT8_u) for possible climate_zones values or retrieve them by yourself using some kind of database tool on the Parquet file.
            transform: Optional transform applied to the rendered RGB PIL image.
            splits: Optional filter for dataset splits ('train', 'validation', 'test', 'bench').
            point_token: Optional tuple of [start_token, end_token] to wrap <point> tags in text.
            ref_token: Optional tuple of [start_token, end_token] to wrap <ref> tags in text.
            info_fn: Optional callback function for logging information during initialization.
        """
        super().__init__()

        if isinstance(bands, str):
            assert bands in _predefined_bandcombinations, f"{bands} not in predefined options: {_predefined_bandcombinations.keys()}"
            bands = _predefined_bandcombinations[bands]
        elif isinstance(bands, Iterable):
            bands = list(bands)
        elif bands is None:
            bands = _predefined_bandcombinations["RGB"]
        else:
            raise NotImplementedError(f"{bands} is not supported")

        self.image_reader = BENImageReader(lmdb_file, metadata_file, bands, img_size, upsample_mode, info_fn=info_fn)

        self.text_data = pd.read_parquet(metadata_file)

        # check the format of the text file
        assert self._expected_columns.issubset(set(self.text_data.columns)), f"The text data at {metadata_file} does not contain the expected columns"
        info_fn(f"Loaded text data with {len(self.text_data)} entries")
        if types is not None:
            self.text_data = self.text_data[self.text_data["type"].isin(types)]
        if categories is not None:
            self.text_data = self.text_data[self.text_data["category"].isin(categories)]
        if countries is not None:
            self.text_data = self.text_data[self.text_data["country"].isin(countries)]
        if seasons is not None:
            self.text_data = self.text_data[self.text_data["season"].isin(seasons)]
        if climate_zones is not None:
            self.text_data = self.text_data[self.text_data["climate_zone"].isin(climate_zones)]
        self.text_data = self.text_data.reset_index(drop=True)
        info_fn(f"After filtering, text data contains {len(self.text_data)} entries")

        if splits is not None:
            self.text_data = self.text_data[self.text_data["split"].isin(splits)].reset_index(drop=True)
            info_fn(f"Split {splits} text data contains {len(self.text_data)} entries")

        self.transform = transform
        self.point_token = ["", ""] if point_token is None else point_token
        assert len(self.point_token) == 2, "Point tokens must have length 2."
        self.ref_token = ["", ""] if ref_token is None else ref_token
        assert len(self.ref_token) == 2, "Reference tokens must have length 2."
        self.bands = list(bands)
        if self.bands != _rgb_band_order:
            raise ValueError(
                "BENTxTDataset currently emits VLM-ready shared samples and therefore "
                "requires RGB Sentinel-2 bands ('B04', 'B03', 'B02')."
            )

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.text_data)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing:
                - 'image': RGB PIL image for the shared VLM collator path.
                - 'input_text': The instruction or question for the VLM.
                - 'target_texts': List containing the expected text output(s).
                - 'lat': Latitude of the patch center.
                - 'lon': Longitude of the patch center.
        """
        sample = self.text_data.iloc[idx]
        img_id = sample.patch_id
        img_data = self.image_reader[img_id]
        image = _sentinel2_rgb_tensor_to_pil(img_data)
        if self.transform is not None:
            image = self.transform(image)
        if not isinstance(image, Image.Image):
            raise TypeError(
                "BENTxTDataset expects transforms to return a PIL image for the shared VLM path"
            )

        text_in = sample.input.replace("<ref>", self.ref_token[0]).replace("</ref>", self.ref_token[1])
        text_in = text_in.replace("<point>", self.point_token[0]).replace("</point>", self.point_token[1])

        if sample.type in {'binary', 'mcq', 'captioning', 'bounding box'}:
            output = sample.output
        else:
            raise NotImplementedError(f"{sample.type} is not supported")

        return {
            "image": image,
            "input_text": text_in,
            "target_texts": [str(output)],
            "lat": float(sample.latitude),
            "lon": float(sample.longitude),
        }


class BENTxTDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for BigEarthNet.txt.

    This DataModule provides a structured interface for loading BigEarthNet.txt
    for the shared VLM training/evaluation path. It automatically handles train,
    validation, test, and benchmark dataset splits and emits the repo sample schema.

    The module manages:
    - Automatic dataset setup for different training stages
    - Sentinel-2 RGB rendering into PIL images for the shared collator/model path
    - DataLoader creation with appropriate batch sizes and worker processes
    - GPU pinning when CUDA is available

    Attributes:
        train_ds (BENTxTDataset): Training dataset instance.
        val_ds (BENTxTDataset): Validation dataset instance.
        test_ds (BENTxTDataset): Test dataset instance.
        bench_ds (BENTxTDataset): Benchmark dataset instance.
    """
    train_ds = None
    val_ds = None
    test_ds = None
    bench_ds = None

    def __init__(
            self,
            image_lmdb_file: str | Path,
            metadata_file: str | Path,
            bands: Iterable[str] | str | None = "RGB",
            img_size: int = 120,
            upsample_mode: str = "nearest",
            types: Iterable[str] | None = None,
            categories: Iterable[str] | None = None,
            countries: Iterable[str] | None = None,
            seasons: Iterable[str] | None = None,
            climate_zones: Iterable[str] | None = None,
            num_workers_dataloader: int | None = 4,
            batch_size: int | None = 16,
            image_transforms_train: Callable | None = None,
            image_transforms_eval: Callable | None = None,
            point_token: Iterable[str] = None,
            ref_token: Iterable[str] = None,
            info_fn: Callable | None = lambda x: x,
    ):
        """
        Initialize the BigEarthNet.txt DataModule.

        Args:
            lmdb_file: Path to the LMDB file containing the BigEarthNet-v2.0 image data.
            metadata_file: Path to the BigEarthNet.txt Parquet file.
            bands: Band names to load. The current shared VLM path requires RGB
                Sentinel-2 bands ('B04', 'B03', 'B02'); defaults to 'RGB'.
            img_size: Target image size for interpolation (default: 120).
            upsample_mode: Interpolation mode for resizing ('nearest', 'bilinear', 'bicubic', etc.).
                Default: 'nearest'.
            types: Optional filter for annotation types (e.g., 'binary', 'mcq', 'captioning', 'bounding box').
            categories: Optional filter for annotation categories. See [here](https://huggingface.co/datasets/BIFOLD-BigEarthNetv2-0/BigEarthNet.txt/sql-console/KzrmYgF) for possible type-category combinations or retrieve them by yourself using some kind of database tool on the Parquet file.
            countries: Optional filter for acquisition countries (e.g., 'Austria', 'Belgium', 'Finland', 'Ireland', 'Kosovo', 'Lithuania', 'Luxembourg', 'Portugal', 'Serbia', 'Switzerland').
            seasons: Optional filter for seasons (e.g., 'Spring', 'Summer', 'Fall', 'Winter').
            climate_zones: Optional filter for climate zones. See [here](https://huggingface.co/datasets/BIFOLD-BigEarthNetv2-0/BigEarthNet.txt/sql-console/SUU1DwA) for possible climate_zones values or retrieve them by yourself using some kind of database tool on the Parquet file.
            num_workers_dataloader: Number of worker processes for DataLoaders (default: 4).
                Set to 0 to disable multiprocessing.
            batch_size: Batch size for DataLoaders (default: 16).
            image_transforms_train: Optional transform applied to rendered RGB PIL images for training.
            image_transforms_eval: Optional transform applied to rendered RGB PIL images for evaluation.
            point_token: Optional tuple of [start_token, end_token] to wrap <point> tags in text.
            ref_token: Optional tuple of [start_token, end_token] to wrap <ref> tags in text.
            info_fn: Optional callback function for logging during initialization.
        """
        super().__init__()
        self.num_workers_dataloader = num_workers_dataloader
        self.batch_size = batch_size
        self.pin_memory = torch.cuda.is_available()
        self._collator: Callable | None = None

        if isinstance(bands, str):
            assert bands in _predefined_bandcombinations, f"{bands} not in predefined options: {_predefined_bandcombinations.keys()}"
            self.bands = _predefined_bandcombinations[bands]
        elif isinstance(bands, Iterable):
            self.bands = list(bands)
        elif bands is None:
            self.bands = _predefined_bandcombinations["RGB"]
        else:
            raise NotImplementedError(f"{bands} is not supported")

        self.ds_kwargs = {
            "lmdb_file": image_lmdb_file,
            "metadata_file": metadata_file,
            "bands": self.bands,
            "img_size": img_size,
            "upsample_mode": upsample_mode,
            "types": types,
            "categories": categories,
            "countries": countries,
            "seasons": seasons,
            "climate_zones": climate_zones,
            "point_token": point_token,
            "ref_token": ref_token,
            "info_fn": info_fn,
        }

        # The shared Qwen/Unsloth path expects actual images and handles its own
        # resizing/tokenization. Optional transforms therefore operate on PIL
        # images and default to None.
        self.train_transforms = image_transforms_train
        self.eval_transforms = image_transforms_eval

    def set_collator(self, collator: Callable) -> None:
        self._collator = collator

    def setup(self, stage: str | None = None) -> None:
        """
        Create train/val/test/bench datasets based on the specified stage.

        This method is called by PyTorch Lightning during trainer initialization.

        Args:
            stage: The training stage - one of 'fit', 'test', 'bench', or None. If None,
                all datasets are created. Default: None.
                - 'fit': Creates train and validation datasets
                - 'test': Creates test dataset (includes both 'test' and 'bench' splits)
                - 'bench': Creates benchmark dataset
        """
        if stage == "fit" or stage is None:
            self.train_ds = BENTxTDataset(
                **self.ds_kwargs,
                splits=['train'],
                transform=self.train_transforms
            )
            self.val_ds = BENTxTDataset(
                **self.ds_kwargs,
                splits=['validation'],
                transform=self.eval_transforms
            )
        if stage == "test" or stage is None:
            self.test_ds = BENTxTDataset(
                **self.ds_kwargs,
                splits=['test', 'bench'],
                transform=self.eval_transforms
            )
        if stage == "bench" or stage is None:
            self.bench_ds = BENTxTDataset(
                **self.ds_kwargs,
                splits=['bench'],
                transform=self.eval_transforms
            )



    def _create_dataloader(self, dataset, *, shuffle: bool) -> DataLoader:
        collate_fn = self._collator if self._collator is not None else collate_normalized
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers_dataloader,
            shuffle=shuffle,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
        )

    def train_dataloader(self):
        """Create and return the training DataLoader with shuffling."""
        return self._create_dataloader(self.train_ds, shuffle=True)

    def val_dataloader(self):
        """Create and return the validation DataLoader without shuffling."""
        return self._create_dataloader(self.val_ds, shuffle=False)

    def test_dataloader(self):
        """Create and return the test DataLoader (includes both 'test' and 'bench' splits)."""
        return self._create_dataloader(self.test_ds, shuffle=False)

    def bench_dataloader(self):
        """Create and return the benchmark DataLoader."""
        return self._create_dataloader(self.bench_ds, shuffle=False)
