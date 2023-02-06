import logging
import os
from typing import Any, Dict, Set

import cv2
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from mammography.src import utils
from mammography.src.sampler import BreastSampler

logger = logging.getLogger(__name__)


class MinMaxScale(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _min = x.amin(dim=(-2, -1), keepdim=True)
        _max = x.amax(dim=(-2, -1), keepdim=True)
        x -= _min
        x /= _max - _min
        return x


class PercentileScale(torch.nn.Module):
    def __init__(self, min: float, max: float) -> None:
        super().__init__()
        self.register_buffer("percentiles", torch.tensor([min, max]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is (N, C, H, W) with range [0.0, 1.0]
        q = x.flatten(start_dim=2).quantile(q=self.get_buffer("percentiles"), dim=-1, keepdim=True)
        q.unsqueeze_(-1)
        _min, _max = q[0], q[1]
        x -= _min
        x /= _max - _min
        # x.clamp_(0.0, 1.0)
        return x


class CropCenterRight(torch.nn.Module):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.size = size

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return utils.crop_right_center(img=img, size=self.size)


class PNGDataset(Dataset):
    def __init__(
        self, df: pd.DataFrame, augmentation: torch.nn.Sequential, keys: Set[str], filepath_format: str
    ) -> None:
        super().__init__()
        self.df = df
        self.augmentation = augmentation
        self.keys = keys
        self.filepath_format = filepath_format

    def __len__(self) -> int:
        return len(self.df)

    @staticmethod
    def _read(filepath: str) -> torch.Tensor:
        arr = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if arr is None:
            raise RuntimeError(f"No data found at {filepath}")
        t = torch.from_numpy(arr)
        t.unsqueeze_(dim=0)
        return t

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.df.iloc[index].to_dict()
        row.update(
            {
                view: self.augmentation(self._read(self.filepath_format.format(image_id=np.random.choice(row[view]))))
                for view in ["CC", "MLO"]
            }
        )
        return row


class DataModule(LightningDataModule):
    def __init__(
        self,
        image_dir: str,
        augmentation: DictConfig,
        batch_size: int,
        prefetch_factor: int,
    ) -> None:
        super().__init__()
        self.save_hyperparameters("augmentation", "batch_size")
        self.image_dir = image_dir
        self.augmentation: torch.nn.Sequential = instantiate(self.hparams_initial["augmentation"])
        self.num_workers = torch.multiprocessing.cpu_count()
        logger.info(f"Detected {self.num_workers} cores.")
        self.prefetch_factor = prefetch_factor

    @staticmethod
    def compute_class_weights(y: pd.Series) -> pd.Series:
        return len(y) / (y.nunique() * y.value_counts())

    def _format_filepath(self, row: pd.Series):
        return self.filepath_format.format(**row)

    def _use_artifact(self) -> None:
        if not (self.trainer.logger.experiment.offline or self.trainer.logger.experiment.disabled):
            logger.info("Saving input artifact reference...")
            artifact = wandb.Artifact(name="input", type="dataset")
            artifact.add_reference(uri=f"file://{os.path.join(self.image_dir, 'train.pickle')}", name="train.pickle")
            artifact.add_reference(uri=f"file://{os.path.join(self.image_dir, 'val.pickle')}", name="val.pickle")
            # artifact.add_reference(
            #     uri=f"file://{self.image_dir}", name="image_dir", max_objects=len(self.df), checksum=False
            # )
            self.trainer.logger.use_artifact(artifact, artifact_type="dataset").save()
        else:
            logger.warning(f"Unable to use artifact when in {self.trainer.logger.experiment.mode} mode")

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_meta = pd.read_pickle(os.path.join(self.image_dir, "train.pickle"))
            self.val_meta = pd.read_pickle(os.path.join(self.image_dir, "val.pickle"))
            # class_weights = 1.0 / self.df["cancer"].value_counts(normalize=True)
            class_weights = {0: 1, 1: 3}
            self.train_meta["sample_weight"] = self.train_meta["cancer"].map(class_weights.get)
            self._use_artifact()
        elif stage == "predict":
            self.val_meta = pd.read_pickle(os.path.join(self.image_dir, "val.pickle"))

    def train_dataloader(self) -> DataLoader:
        pipe = PNGDataset(
            self.train_meta,
            augmentation=self.augmentation.train(),
            keys={"cancer"},
            filepath_format=os.path.join(self.image_dir, "{image_id}_0.png"),
        )
        return DataLoader(
            dataset=pipe,
            batch_size=self.hparams_initial["batch_size"],
            pin_memory=True,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            sampler=WeightedRandomSampler(
                weights=self.train_meta["sample_weight"],
                num_samples=len(self.train_meta),
                replacement=True,
            ),
            drop_last=False,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        pipe = PNGDataset(
            self.val_meta,
            augmentation=self.augmentation.eval(),
            keys={"cancer", "patient_id", "laterality"},
            filepath_format=os.path.join(self.image_dir, "{image_id}_0.png"),
        )
        return DataLoader(
            dataset=pipe,
            batch_size=self.hparams_initial["batch_size"] * 4,
            pin_memory=True,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            drop_last=False,
            persistent_workers=self.num_workers > 0,
        )

    def predict_dataloader(self) -> DataLoader:
        return self.val_dataloader()
