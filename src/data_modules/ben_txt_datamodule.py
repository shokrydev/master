"""BigEarthNet.txt datamodule, based on the dataset publication implementation.

Upstream source:
https://huggingface.co/datasets/BIFOLD-BigEarthNetv2-0/BigEarthNet.txt/blob/main/ben_txt_datamodule.py
"""

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Literal

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
_valid_bandnames = set(_s1_bandnames + _s2_bandnames)
_predefined_bandcombinations = {
    "RGB": ["B04", "B03", "B02"],
    "S2-10m20m": ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"],
    "S1S2-10m20m": ["VV", "VH", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"],
    "all":  _s1_bandnames + _s2_bandnames,
}
_rgb_band_order = ["B04", "B03", "B02"]
RGBRenderMode = Literal["copernicus", "quantile"]
_copernicus_rgb_scale = 3558.0
_default_rgb_quantile = 0.90

"""
Band statistics for BigEarthNet v2 (including S1 stats from v1) after
interpolating images to 120x120 with nearest-neighbor interpolation.
The statistics were calculated on the official train split.
"""

means = {
    "B01": 361.0767822265625,
    "B02": 438.3720703125,
    "B03": 614.0556640625,
    "B04": 588.4096069335938,
    "B05": 942.8433227539062,
    "B06": 1769.931640625,
    "B07": 2049.551513671875,
    "B08": 2193.2919921875,
    "B09": 2241.455322265625,
    "B11": 1568.226806640625,
    "B12": 997.7324829101562,
    "B8A": 2235.556640625,
    "VH": -19.352558135986328,
    "VV": -12.643863677978516,
}
stds = {
    "B01": 575.0687255859375,
    "B02": 607.02685546875,
    "B03": 603.2968139648438,
    "B04": 684.56884765625,
    "B05": 738.4326782226562,
    "B06": 1100.4560546875,
    "B07": 1275.805419921875,
    "B08": 1369.3717041015625,
    "B09": 1316.393310546875,
    "B11": 1070.1612548828125,
    "B12": 813.5276489257812,
    "B8A": 1356.5440673828125,
    "VH": 5.590505599975586,
    "VV": 5.133493900299072,
}


def _resolve_bands(bands: Iterable[str] | str | None, *, default: str = "all") -> list[str]:
    if isinstance(bands, str):
        if bands not in _predefined_bandcombinations:
            raise ValueError(
                f"{bands} not in predefined options: {_predefined_bandcombinations.keys()}"
            )
        resolved = list(_predefined_bandcombinations[bands])
    elif isinstance(bands, Iterable):
        resolved = list(bands)
    elif bands is None:
        resolved = list(_predefined_bandcombinations[default])
    else:
        raise TypeError(f"Unsupported bands value: {bands!r}")

    invalid = [band for band in resolved if band not in _valid_bandnames]
    if invalid:
        raise ValueError(f"Unknown BigEarthNet band names: {invalid}")
    if not resolved:
        raise ValueError("At least one BigEarthNet band must be selected")
    return resolved


def _union_bands(*band_groups: Iterable[str]) -> list[str]:
    bands = []
    for band_group in band_groups:
        for band in band_group:
            if band not in bands:
                bands.append(band)
    return bands


def _select_bands(
    image_tensor: torch.Tensor,
    source_bands: list[str],
    target_bands: list[str],
) -> torch.Tensor:
    missing = [band for band in target_bands if band not in source_bands]
    if missing:
        raise ValueError(f"Cannot select missing BigEarthNet bands: {missing}")
    indices = [source_bands.index(band) for band in target_bands]
    return image_tensor[indices]


def default_non_rgb_transform(mean: list[float], std: list[float]) -> Callable:
    mean_values = tuple(float(value) for value in mean)
    std_values = tuple(float(value) for value in std)

    def _normalize(image_tensor: torch.Tensor) -> torch.Tensor:
        mean_tensor = image_tensor.new_tensor(mean_values).view(-1, 1, 1)
        std_tensor = image_tensor.new_tensor(std_values).view(-1, 1, 1)
        return (image_tensor - mean_tensor) / std_tensor

    return _normalize


def collate_normalized(batch):
    images = []
    non_rgb_images = []
    input_texts = []
    target_texts = []
    latitudes = []
    longitudes = []
    non_rgb_bands = []

    for item in batch:
        images.append(item["image"])
        if "non_rgb_imagery" in item:
            non_rgb_images.append(item["non_rgb_imagery"])
            non_rgb_bands.append(item.get("non_rgb_bands"))
        input_texts.append(item["input_text"])
        target_texts.append(item["target_texts"])
        latitudes.append(item["lat"])
        longitudes.append(item["lon"])

    collated = {
        "image": images,
        "input_text": input_texts,
        "target_texts": target_texts,
        "lat": torch.tensor(latitudes, dtype=torch.float64),
        "lon": torch.tensor(longitudes, dtype=torch.float64),
    }

    if non_rgb_images:
        collated["non_rgb_imagery"] = torch.stack(non_rgb_images, dim=0)
        if all(bands == non_rgb_bands[0] for bands in non_rgb_bands):
            collated["non_rgb_bands"] = non_rgb_bands[0]
        else:
            collated["non_rgb_bands"] = non_rgb_bands

    return collated


def _sentinel2_rgb_tensor_to_pil(
    image_tensor: torch.Tensor,
    *,
    rgb_render_mode: RGBRenderMode = "copernicus",
    rgb_quantile: float = _default_rgb_quantile,
) -> Image.Image:
    """Render raw Sentinel-2 RGB bands into a deterministic PIL image.

    BigEarthNet reflectance values are not stored as byte RGB images. For the
    shared Qwen/Unsloth path we emit an actual RGB image rather than
    classifier-style normalized tensors.
    """
    if image_tensor.ndim != 3 or image_tensor.shape[0] != 3:
        raise ValueError(
            "Expected Sentinel-2 RGB tensor with shape (3, H, W) for VLM image rendering"
        )
    if rgb_render_mode == "copernicus":
        digital = (image_tensor / _copernicus_rgb_scale) * 255.0
    elif rgb_render_mode == "quantile":
        scale = image_tensor.quantile(rgb_quantile)
        if not bool(torch.isfinite(scale).item()) or float(scale.item()) <= 0.0:
            scale = image_tensor.new_tensor(_copernicus_rgb_scale)
        digital = (image_tensor / scale) * 255.0
    else:
        raise ValueError(f"Unsupported RGB render mode: {rgb_render_mode}")

    image_array = digital.clamp(0.0, 255.0).to(torch.uint8)
    image_array = image_array.detach().cpu().numpy()
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
        self.bands = list(bands)
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
    together with Sentinel-1/Sentinel-2 imagery from BigEarthNet-v2.0
    (converted to LMDB format) and emits the shared repo sample schema for the
    VLM path plus normalized non-RGB imagery for future S1/S2 towers.
    It supports various filtering options to create custom dataset splits based on
    textual annotation metadata, such as type or category, or image metadata
    like country, season, and climate zone.
    """
    _expected_columns = {'s1_name', 'output', 'longitude', 'country', 'climate_zone', 'type', 'input', 'split', 'latitude', 'ID', 'patch_id', 'category', 'season'}

    def __init__(
            self,
            lmdb_file: str | Path,
            metadata_file: str | Path,
            bands: Iterable[str] | str | None = None,
            img_size: int = 120,
            upsample_mode: str = "nearest",
            rgb_render_mode: RGBRenderMode = "copernicus",
            rgb_quantile: float = _default_rgb_quantile,
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
            bands: Sentinel-1/Sentinel-2 band names to normalize and expose. Can be a
                predefined combination key ('RGB', 'S2-10m20m', 'S1S2-10m20m',
                'all') or an iterable of band names. Defaults to 'all'.
                RGB bands are always loaded separately for the VLM image path.
            img_size: Target image size for interpolation (default: 120).
            upsample_mode: Interpolation mode for resizing ('nearest', 'bilinear', 'bicubic', etc.).
                Default: 'nearest'.
            rgb_render_mode: How to convert raw RGB reflectance bands into
                uint8 RGB for the VLM path. 'copernicus' applies the official
                3558 reflectance scale; 'quantile' applies a per-sample
                quantile stretch.
            rgb_quantile: Quantile used when rgb_render_mode='quantile'.
            types: Optional filter for annotation types (e.g., 'binary', 'mcq', 'captioning', 'bounding box').
            categories: Optional filter for annotation categories. See [here](https://huggingface.co/datasets/BIFOLD-BigEarthNetv2-0/BigEarthNet.txt/sql-console/8okbuKf) for possible type-category combinations or retrieve them by yourself using some kind of database tool on the Parquet file.
            countries: Optional filter for acquisition countries (e.g., 'Austria', 'Belgium', 'Finland', 'Ireland', 'Kosovo', 'Lithuania', 'Luxembourg', 'Portugal', 'Serbia', 'Switzerland').
            seasons: Optional filter for seasons (e.g., 'Spring', 'Summer', 'Fall', 'Winter').
            climate_zones: Optional filter for climate zones. See [here](https://huggingface.co/datasets/BIFOLD-BigEarthNetv2-0/BigEarthNet.txt/sql-console/3xLT8_u) for possible climate_zones values or retrieve them by yourself using some kind of database tool on the Parquet file.
            transform: Optional transform applied to the non-RGB imagery tensor.
            splits: Optional filter for dataset splits ('train', 'validation', 'test', 'bench').
            point_token: Optional tuple of [start_token, end_token] to wrap <point> tags in text.
            ref_token: Optional tuple of [start_token, end_token] to wrap <ref> tags in text.
            info_fn: Optional callback function for logging information during initialization.
        """
        super().__init__()

        self.bands = _resolve_bands(bands)
        if rgb_render_mode not in ("copernicus", "quantile"):
            raise ValueError(f"Unsupported RGB render mode: {rgb_render_mode}")
        if not 0.0 < rgb_quantile <= 1.0:
            raise ValueError("rgb_quantile must be in the interval (0, 1]")
        self.rgb_render_mode = rgb_render_mode
        self.rgb_quantile = rgb_quantile

        missing_stats = [band for band in self.bands if band not in means or band not in stds]
        if missing_stats:
            raise ValueError(f"Missing BigEarthNet normalization stats for bands: {missing_stats}")

        self.reader_bands = _union_bands(_rgb_band_order, self.bands)
        self.image_reader = BENImageReader(
            lmdb_file,
            metadata_file,
            self.reader_bands,
            img_size,
            upsample_mode,
            info_fn=info_fn,
        )

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
                - 'non_rgb_imagery': Normalized S1/S2 tensor.
                - 'non_rgb_bands': Band order of the non-RGB imagery tensor.
                - 'input_text': The instruction or question for the VLM.
                - 'target_texts': List containing the expected text output(s).
                - 'lat': Latitude of the patch center.
                - 'lon': Longitude of the patch center.
        """
        sample = self.text_data.iloc[idx]
        img_id = sample.patch_id
        img_data = self.image_reader[img_id]
        rgb_data = _select_bands(img_data, self.reader_bands, _rgb_band_order)
        non_rgb_imagery = _select_bands(img_data, self.reader_bands, self.bands)

        image = _sentinel2_rgb_tensor_to_pil(
            rgb_data,
            rgb_render_mode=self.rgb_render_mode,
            rgb_quantile=self.rgb_quantile,
        )
        if self.transform is not None:
            non_rgb_imagery = self.transform(non_rgb_imagery)
        if not isinstance(image, Image.Image):
            raise TypeError(
                "BENTxTDataset expects the shared VLM image path to produce a PIL image"
            )

        text_in = sample.input.replace("<ref>", self.ref_token[0]).replace("</ref>", self.ref_token[1])
        text_in = text_in.replace("<point>", self.point_token[0]).replace("</point>", self.point_token[1])

        if sample.type in {'binary', 'mcq', 'captioning', 'bounding box'}:
            output = sample.output
        else:
            raise NotImplementedError(f"{sample.type} is not supported")

        return {
            "image": image,
            "non_rgb_imagery": non_rgb_imagery,
            "non_rgb_bands": list(self.bands),
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
    - normalized non-RGB imagery tensors for the optional S1/S2 encoder path
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
            bands: Iterable[str] | str | None = None,
            img_size: int = 120,
            upsample_mode: str = "nearest",
            rgb_render_mode: RGBRenderMode = "copernicus",
            rgb_quantile: float = _default_rgb_quantile,
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
            bands: Sentinel-1/Sentinel-2 band names to normalize and expose. Can be a
                predefined combination key ('RGB', 'S2-10m20m', 'S1S2-10m20m',
                'all') or an iterable of band names. Defaults to 'all'.
                RGB bands are always loaded separately for the VLM image path.
            img_size: Target image size for interpolation (default: 120).
            upsample_mode: Interpolation mode for resizing ('nearest', 'bilinear', 'bicubic', etc.).
                Default: 'nearest'.
            rgb_render_mode: How to convert raw RGB reflectance bands into
                uint8 RGB for the VLM path. 'copernicus' applies the official
                3558 reflectance scale; 'quantile' applies a per-sample
                quantile stretch.
            rgb_quantile: Quantile used when rgb_render_mode='quantile'.
            types: Optional filter for annotation types (e.g., 'binary', 'mcq', 'captioning', 'bounding box').
            categories: Optional filter for annotation categories. See [here](https://huggingface.co/datasets/BIFOLD-BigEarthNetv2-0/BigEarthNet.txt/sql-console/KzrmYgF) for possible type-category combinations or retrieve them by yourself using some kind of database tool on the Parquet file.
            countries: Optional filter for acquisition countries (e.g., 'Austria', 'Belgium', 'Finland', 'Ireland', 'Kosovo', 'Lithuania', 'Luxembourg', 'Portugal', 'Serbia', 'Switzerland').
            seasons: Optional filter for seasons (e.g., 'Spring', 'Summer', 'Fall', 'Winter').
            climate_zones: Optional filter for climate zones. See [here](https://huggingface.co/datasets/BIFOLD-BigEarthNetv2-0/BigEarthNet.txt/sql-console/SUU1DwA) for possible climate_zones values or retrieve them by yourself using some kind of database tool on the Parquet file.
            num_workers_dataloader: Number of worker processes for DataLoaders (default: 4).
                Set to 0 to disable multiprocessing.
            batch_size: Batch size for DataLoaders (default: 16).
            image_transforms_train: Optional transform applied to normalized non-RGB imagery tensors for training.
            image_transforms_eval: Optional transform applied to normalized non-RGB imagery tensors for evaluation.
            point_token: Optional tuple of [start_token, end_token] to wrap <point> tags in text.
            ref_token: Optional tuple of [start_token, end_token] to wrap <ref> tags in text.
            info_fn: Optional callback function for logging during initialization.
        """
        super().__init__()
        self.num_workers_dataloader = num_workers_dataloader
        self.batch_size = batch_size
        self.pin_memory = torch.cuda.is_available()
        self._collator: Callable | None = None

        self.bands = _resolve_bands(bands)
        if rgb_render_mode not in ("copernicus", "quantile"):
            raise ValueError(f"Unsupported RGB render mode: {rgb_render_mode}")
        if not 0.0 < rgb_quantile <= 1.0:
            raise ValueError("rgb_quantile must be in the interval (0, 1]")

        missing_stats = [band for band in self.bands if band not in means or band not in stds]
        if missing_stats:
            raise ValueError(f"Missing BigEarthNet normalization stats for bands: {missing_stats}")

        self.ds_kwargs = {
            "lmdb_file": image_lmdb_file,
            "metadata_file": metadata_file,
            "bands": self.bands,
            "img_size": img_size,
            "upsample_mode": upsample_mode,
            "rgb_render_mode": rgb_render_mode,
            "rgb_quantile": rgb_quantile,
            "types": types,
            "categories": categories,
            "countries": countries,
            "seasons": seasons,
            "climate_zones": climate_zones,
            "point_token": point_token,
            "ref_token": ref_token,
            "info_fn": info_fn,
        }

        self.mean = [means[band] for band in self.bands]
        self.std = [stds[band] for band in self.bands]

        # The shared Qwen/Unsloth path gets unnormalized RGB PIL images. These
        # transforms are for the parallel non-RGB imagery tensor path only.
        default_transform = default_non_rgb_transform(self.mean, self.std)
        self.train_transforms = (
            image_transforms_train if image_transforms_train is not None else default_transform
        )
        self.eval_transforms = (
            image_transforms_eval if image_transforms_eval is not None else default_transform
        )

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
