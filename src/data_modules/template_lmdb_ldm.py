# Source: https://lightning.ai/docs/pytorch/stable/data/datamodule.html

from typing import List, Literal, Optional

import lightning as L
import torch
import lmdb
import pandas as pd
from safetensors.numpy import load as load_np_safetensor

from torchvision.transforms import v2

class TemplateLMDB(torch.utils.data.Dataset):
    def __init__(self, lmdb_path: str, metadata_parquet_path: str, bandorder: List[str], split=None, transform=None):
        """
        Dataset for the BigEarthNet dataset using an lmdb file.

        :param lmdb_path: path to the lmdb file
        :param metadata_parquet_path: path to the metadata parquet file
        :param bandorder: names of the bands to use, e.g. ["B04", "B03", "B02"] for RGB
        :param split: split of the dataset to use, one of 'train', 'validation', 'test', None (uses all data)
        :param transform: a torchvision transform to apply to the images after loading
        """
        self.lmdb_path = lmdb_path
        self.metadata_parquet_path = metadata_parquet_path
        self.bandorder = bandorder
        self.metadata = pd.read_parquet(metadata_parquet_path)
        if split is not None:
            self.metadata = self.metadata[self.metadata['split'] == split]
        self.keys = self.metadata['patch_id'].tolist()
        self.transform = transform

        # sort keys to ensure reproducibility; in playce
        self.keys.sort()

        # init later when workers are branched to avoid locking conflicts 
        self.env = None

    def __len__(self):
        return len(self.keys)

    def _open_lmdb(self):
        if self.env is None:
            self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=True, meminit=False)

    #TODO rethink this method
    def _stack(self, patch):
        # interpolate each channel to 120x120
        patch = {
            k: torch.nn.functional.interpolate(
                torch.from_numpy(v).float().unsqueeze(0),
                size=(120, 120),
                mode='bilinear',
                align_corners=False
            ).squeeze()
            for k, v in patch.items()
        }
        # stack channels
        patch = torch.stack([patch[k] for k in self.bandorder], dim=0)
        return patch

    def __getitem__(self, idx):
        """
        Get an item from the dataset.

        :param idx: index of the item to get
        :return: (patch, label) tuple where patch is a tensor of shape (C, H, W) and label is a tensor of shape (N,)
        """
        self._open_lmdb()
        patch_id = self.keys[idx]

        # load item from lmdb
        with self.env.begin(write=False) as txn:
            patch = load_np_safetensor(txn.get(patch_id.encode()))

        # data preprocessing
        patch = self._stack(patch)
        if self.transform:
            patch = self.transform(patch)

        # get label from metadata
        label = self.metadata[self.metadata['patch_id'] == patch_id]['labels']
        assert len(label) == 1
        label = label.values[0]
        label = _stringlist_to_tensor(label)
        return patch, label

#TODO check if usefull
class TemplateDataloader(torch.utils.data.DataLoader):
    pass

class TemplateDataModule(L.LightningDataModule):
    def __init__(
            self,
            batch_size: int,
            num_workers: int,
            bandorder: List,
            lmdb_path: str = None,
            metadata_parquet_path: str = None,
            #TODO check mean and dev
            train_transform = None, # use None as default to avoid mutable complex defaults that are shared across multiple instances and ensure compatibility with LightningCLI/YAML overrides
            val_test_transform = None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.bandorder = bandorder

        
         # Assign defaults inside __init__ to avoid shared state
        self.train_transform = train_transform or v2.Compose([
            v2.ToImage(), # wraps data in tv_tensors.Image object
            v2.ToDtype(torch.float32, scale=True), #  scale=True for [0,1] scaling
            v2.Normalize(mean=[0.1307], std=[0.3081])
        ])
        
        self.val_test_transform = val_test_transform or v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True), #  scale=True for [0,1] scaling
            v2.Normalize(mean=[0.1307], std=[0.3081])
        ])


    def prepare_data(self):
        # good according to docs
        # download_data()
        # tokenize()
        # etc()

        # bad according to docs
        # self.split = data_split
        # self.some_state = some_other_state()
        
        # set in LightningDataModule.__init__ self.prepare_data_per_node = True
        pass

    def setup(self, stage: str):
        if stage == 'fit' or stage is None:
            # First create dataset without transforms to compute statistics so that we don't have to laod the dataset twice
            self.train_dataset = self.dataset(
                split='train',
                transform=None  #TODO: Check: No transforms for statistics computation
            )
            
            # Validation dataset without augmentations
            self.val_dataset = self.dataset(
                split='validation',
                transform=self.val_test_transform
            )
            
        if stage == 'test' or stage is None:
            self.test_dataset = self.dataset(
                split='test',
                transform=self.val_test_transform
            )

        if stage == 'predict' or stage is None:
            raise Exception("LightningDataModule setup is not implemented for predict case")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.mnist_train, batch_size=32)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.mnist_val, batch_size=32)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.mnist_test, batch_size=32)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.mnist_predict, batch_size=32)