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
from torchvision.transforms import functional_tensor

from mammography.src import utils

# from torchdata.dataloader2 import DataLoader2, PrototypeMultiProcessingReadingService
# from torchdata.datapipes.map import MapDataPipe


logger = logging.getLogger(__name__)


class CropCenterRight(torch.nn.Module):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.size = size

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return utils.crop_right_center(img=img, size=self.size)


class DataframeDataPipe(Dataset):
    def __init__(self, df: pd.DataFrame, augmentation: torch.nn.Sequential, keys: Set) -> None:
        super().__init__()
        self.df = df
        self.augmentation = augmentation
        self.keys = keys

    def __len__(self) -> int:
        return len(self.df)

    @staticmethod
    def _read(filepath: str) -> npt.NDArray[np.uint8]:
        arr = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        return arr

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.df.iloc[index]
        logger.debug(f"Loading image {row['image_id']}")
        d = row.to_dict()
        arr = self._read(filepath=row["filepath"])
        pixels = torch.from_numpy(arr)
        pixels.unsqueeze_(dim=0)
        d["pixels"] = self.augmentation(pixels)
        d = {k: v for k, v in d.items() if k in self.keys}
        return d


class DataModule(LightningDataModule):
    def __init__(
        self,
        metadata_filepath: str,
        image_dir: str,
        augmentation: DictConfig,
        batch_size: int,
        prefetch_factor: int,
    ) -> None:
        super().__init__()
        self.save_hyperparameters("augmentation", "batch_size")
        self.metadata_filepath = metadata_filepath
        self.image_dir = image_dir
        self.augmentation: torch.nn.Sequential = instantiate(self.hparams.augmentation)
        self.num_workers = torch.multiprocessing.cpu_count()
        self.prefetch_factor = prefetch_factor
        self.filepath_format = os.path.join(self.image_dir, "{image_id}_0.png")

    @staticmethod
    def compute_class_weights(y: pd.Series) -> pd.Series:
        return len(y) / (y.nunique() * y.value_counts())

    def _format_filepath(self, row: pd.Series):
        return self.filepath_format.format(**row)

    def _use_artifact(self) -> None:
        if not (self.trainer.logger.experiment.offline or self.trainer.logger.experiment.disabled):
            artifact = wandb.Artifact(name="input", type="dataset")
            artifact.add_reference(uri=f"file://{self.metadata_filepath}", name="metadata.csv")
            artifact.add_reference(uri=f"file://{self.image_dir}", name="image_dir", checksum=False, max_objects=1e9)
            self.trainer.logger.use_artifact(artifact, artifact_type="dataset").save()
        else:
            logger.warning(f"Unable to use artifact when in {self.trainer.logger.experiment.mode} mode")

    def setup(self, stage: str) -> None:
        self.df = pd.read_csv(self.metadata_filepath)
        self.df["filepath"] = self.df.apply(self._format_filepath, axis=1)
        if stage == "fit":
            # self.df.query("patient_id != 27770", inplace=True)
            # self.df.query("image_id != 1942326353", inplace=True)
            class_weights = 1.0 / self.df["cancer"].value_counts()
            self.df["sample_weight"] = self.df["cancer"].map(class_weights.get)
            self.cancer_base_rate = self.df["cancer"].mean()

        self._use_artifact()

    def state_dict(self) -> Dict[str, Any]:
        return {"cancer_base_rate": self.cancer_base_rate}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.cancer_base_rate = state_dict["cancer_base_rate"]

    def train_dataloader(self) -> DataLoader:
        pipe = DataframeDataPipe(
            self.df, augmentation=self.augmentation.train(), keys={"pixels", "cancer"}
        )  # .to_iter_datapipe()
        return DataLoader(
            dataset=pipe,
            batch_size=self.hparams.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            sampler=WeightedRandomSampler(
                weights=self.df["sample_weight"],
                num_samples=len(self.df),
                replacement=True,
            ),
            drop_last=True,  # True to avoid retriggering CuDNN benchmarking
        )
        # pipe = pipe.map(dicom2tensor, input_col="filepath", output_col="pixels")
        # pipe = pipe.in_memory_cache()
        # pipe = pipe.shuffle(buffer_size=1)
        # self.augmentation.train()
        # pipe = pipe.map(self.augmentation, input_col="pixels", output_col="pixels")
        # pipe = pipe.map(partial(select_dict_keys, keys=["pixels", "cancer"]))
        # pipe = pipe.batch(8)
        # pipe = pipe.collate()
        # pipe = pipe.prefetch(buffer_size=1)
        # return DataLoader2(datapipe=pipe, reading_service=PrototypeMultiProcessingReadingService(num_workers=0))

    def val_dataloader(self) -> DataLoader:
        pipe = DataframeDataPipe(
            self.df,
            augmentation=self.augmentation.eval(),
            keys={"pixels", "cancer", "image_id", "patient_id", "laterality"},
        )  # .to_iter_datapipe()
        return DataLoader(
            dataset=pipe,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            drop_last=False,
        )

    def predict_dataloader(self) -> DataLoader:
        return self.val_dataloader()
