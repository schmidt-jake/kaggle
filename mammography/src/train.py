import logging
import os
from functools import partial
from inspect import signature
from typing import Any, Callable, Dict, List

import cv2
import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from lightning_lite.utilities.seed import seed_everything
from omegaconf import DictConfig
from pydicom import dcmread
from torch.utils.data import DataLoader, Dataset

# from torchdata.dataloader2 import DataLoader2, PrototypeMultiProcessingReadingService
# from torchdata.datapipes.map import MapDataPipe
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score
from torchvision.models.feature_extraction import create_feature_extractor

logger = logging.getLogger(__name__)


class FeatureExtractor(torch.nn.Module):
    def __init__(self, backbone: torch.nn.Module, layer_name: str) -> None:
        super().__init__()
        self.layer_name = layer_name
        backbone.features[0] = replace_layer(backbone.features[0], in_channels=1)
        self.feature_extractor = create_feature_extractor(model=backbone, return_nodes=[self.layer_name])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(x)[self.layer_name]


class DataframeDataPipe(Dataset):
    def __init__(self, df: pd.DataFrame, augmentation: torch.nn.Sequential) -> None:
        super().__init__()
        self.df = df
        self.augmentation = augmentation

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.df.iloc[index]
        d = row.to_dict()
        d["pixels"] = self.augmentation(dicom2tensor(row["filepath"]))
        d = {k: v for k, v in d.items() if k in ["pixels", "cancer"]}
        return d


def replace_layer(layer_to_replace: torch.nn.Module, **new_layer_kwargs) -> torch.nn.Module:
    class_signature = signature(layer_to_replace.__class__).parameters
    layer_params = {k: getattr(layer_to_replace, k) for k in class_signature.keys() if hasattr(layer_to_replace, k)}
    layer_params.update(new_layer_kwargs)
    return type(layer_to_replace)(**layer_params)


def dicom2tensor(filepath: str) -> torch.Tensor:
    img = dcmread(filepath)
    arr = img.pixel_array
    arr = arr.astype(np.int16)
    if img.PhotometricInterpretation == "MONOCHROME1":
        # https://dicom.nema.org/medical/Dicom/2017c/output/chtml/part03/sect_C.7.6.3.html#sect_C.7.6.3.1.2
        cv2.bitwise_not(arr, dst=arr)
    return torch.from_numpy(arr).unsqueeze(dim=0)


def select_dict_keys(input_dict: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
    output_dict: Dict[str, Any] = {}
    for k, v in input_dict.items():
        if k in keys:
            output_dict[k] = v
    return output_dict


class DataModule(pl.LightningDataModule):
    def __init__(self, root_dir: str, augmentation: torch.nn.Sequential) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.augmentation = augmentation

    def setup(self, stage: str) -> None:
        if stage == "fit":
            stage = "train"
        self.df = pd.read_csv(os.path.join(self.root_dir, f"{stage}.csv"))
        self.df["filepath"] = (
            self.root_dir
            + "/"
            + f"{stage}_images"
            + "/"
            + self.df["patient_id"].astype(str)
            + "/"
            + self.df["image_id"].astype(str)
            + ".dcm"
        )

    def train_dataloader(self) -> DataLoader:
        pipe = DataframeDataPipe(self.df, augmentation=self.augmentation.train())  # .to_iter_datapipe()
        return DataLoader(
            dataset=pipe,
            batch_size=8,
            shuffle=True,
            pin_memory=True,
            num_workers=2,
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
        pipe = DataframeDataPipe(self.df, augmentation=self.augmentation.eval())  # .to_iter_datapipe()
        return DataLoader(
            dataset=pipe,
            batch_size=8,
            shuffle=False,
            pin_memory=True,
            num_workers=2,
        )

    def test_dataloader(self) -> DataLoader:
        return self.val_dataloader()


class Model(pl.LightningModule):
    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        classifier: torch.nn.Sequential,
        loss: torch.nn.modules.loss._Loss,
        optimizer_config: Callable[..., torch.optim.Optimizer],
    ) -> None:
        super().__init__()
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.train_metrics = MetricCollection({"f1": BinaryF1Score(), "accuracy": BinaryAccuracy()}, prefix="train_")
        self.val_metrics = MetricCollection({"f1": BinaryF1Score(), "accuracy": BinaryAccuracy()}, prefix="val_")
        self.loss = loss
        self.optimizer_config = optimizer_config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.precision < 32:
            x = x.half()
        else:
            x = x.float()
        return self.classifier(self.feature_extractor(x)).squeeze(dim=1)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        prediction = self(batch["pixels"])
        loss = self.loss(input=prediction, target=batch["cancer"].float())
        self.train_metrics(preds=prediction, target=batch["cancer"])
        self.log_dict(self.train_metrics, on_step=True, on_epoch=False, sync_dist=True)
        return {"loss": loss}

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        prediction = self(batch["pixels"])
        loss = self.loss(input=prediction, target=batch["cancer"].float())
        self.val_metrics(preds=prediction, target=batch["cancer"])
        self.log_dict(self.val_metrics, on_step=True, on_epoch=True, sync_dist=True)
        return {"loss": loss}

    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self(batch["pixels"])

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return self.optimizer_config(params=self.parameters())


@hydra.main(config_path="../config", config_name="train", version_base=None)
def train(cfg: DictConfig) -> None:
    seed_everything(seed=42, workers=True)
    trainer: pl.Trainer = instantiate(cfg.trainer)
    datamodule: pl.LightningDataModule = instantiate(cfg.datamodule)
    model: pl.LightningModule = instantiate(cfg.model)
    trainer.fit(model=model, datamodule=datamodule)
    # trainer.test()


if __name__ == "__main__":
    logging.basicConfig(level=logging.NOTSET)
    train()
