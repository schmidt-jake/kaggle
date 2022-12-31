import logging
import os
from inspect import signature
from typing import Any, Callable, Dict, List

import cv2
import hydra
import numpy as np
import numpy.typing as npt
import pandas as pd
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from lightning_lite.utilities.seed import seed_everything
from omegaconf import DictConfig
from pydicom import dcmread
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset

# from torchdata.dataloader2 import DataLoader2, PrototypeMultiProcessingReadingService
# from torchdata.datapipes.map import MapDataPipe
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import BinaryAccuracy
from torchvision.models.feature_extraction import create_feature_extractor

logger = logging.getLogger(__name__)


class ProbabilisticBinaryF1Score(Metric):
    is_differentiable = True
    higher_is_better = True
    full_state_update = False

    def __init__(self) -> None:
        # https://torchmetrics.readthedocs.io/en/stable/pages/implement.html
        # https://www.kaggle.com/code/sohier/probabilistic-f-score/notebook
        super().__init__()
        self.add_state("y_true_count", default=torch.tensor(0, dtype=torch.int64), dist_reduce_fx="sum")
        self.add_state("ctp", default=torch.tensor(0.0, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("cfp", default=torch.tensor(0.0, dtype=torch.float32), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        self.y_true_count += target.numel()  # type: ignore[operator]
        self.ctp += preds[target].sum()
        self.cfp += preds[target.logical_not()].sum()

    def compute(self) -> torch.Tensor:
        c_precision = self.ctp / (self.ctp + self.cfp)  # type: ignore[operator]
        c_recall = self.ctp / self.y_true_count  # type: ignore[operator]
        result = 2 * (c_precision * c_recall) / (c_precision + c_recall)
        return result


class FeatureExtractor(torch.nn.Module):
    def __init__(self, backbone: torch.nn.Module, layer_name: str) -> None:
        super().__init__()
        self.layer_name = layer_name
        backbone.features[0] = replace_layer(backbone.features[0], in_channels=1)  # type: ignore[assignment,index,operator]
        self.feature_extractor = create_feature_extractor(model=backbone, return_nodes=[self.layer_name])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda:
            x = x.half()
        else:
            x = x.float()
        x /= 65534  # 2 ** 16 - 1
        return self.feature_extractor(x)[self.layer_name]


class DataframeDataPipe(Dataset):
    def __init__(self, df: pd.DataFrame, augmentation: torch.nn.Sequential) -> None:
        super().__init__()
        self.df = df
        self.augmentation = augmentation

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        cv2.setNumThreads(0)
        row = self.df.iloc[index]
        logger.debug(f"Loading image {row['image_id']}")
        d = row.to_dict()
        arr = dicom2numpy(row["filepath"])
        arr = crop(arr)
        pixels = torch.from_numpy(arr.astype(np.int32)).unsqueeze(dim=0)
        d["pixels"] = self.augmentation(pixels)
        d = {k: v for k, v in d.items() if k in ["pixels", "cancer"]}
        return d


def replace_layer(layer_to_replace: torch.nn.Module, **new_layer_kwargs) -> torch.nn.Module:
    class_signature = signature(layer_to_replace.__class__).parameters
    layer_params = {k: getattr(layer_to_replace, k) for k in class_signature.keys() if hasattr(layer_to_replace, k)}
    layer_params.update(new_layer_kwargs)
    return type(layer_to_replace)(**layer_params)


def crop(img: npt.NDArray[np.uint16]) -> npt.NDArray[np.uint16]:
    img_uint8 = cv2.convertScaleAbs(img)
    contours = cv2.findContours(img_uint8, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)[0]
    max_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_contour)
    return img[y : y + h, x : x + w]


def dicom2numpy(filepath: str) -> npt.NDArray[np.uint16]:
    with dcmread(filepath) as img:
        arr = img.pixel_array
        if img.PhotometricInterpretation == "MONOCHROME1":
            # https://dicom.nema.org/medical/Dicom/2017c/output/chtml/part03/sect_C.7.6.3.html#sect_C.7.6.3.1.2
            cv2.bitwise_not(arr, dst=arr)
    return arr


def select_dict_keys(input_dict: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
    output_dict: Dict[str, Any] = {}
    for k, v in input_dict.items():
        if k in keys:
            output_dict[k] = v
    return output_dict


class DataModule(pl.LightningDataModule):
    def __init__(self, root_dir: str, augmentation: torch.nn.Sequential, batch_size: int) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.augmentation = augmentation
        self.batch_size = batch_size

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
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=torch.multiprocessing.cpu_count(),
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
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=torch.multiprocessing.cpu_count(),
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
        self.train_metrics = MetricCollection(
            {"pf1": ProbabilisticBinaryF1Score(), "accuracy": BinaryAccuracy()}, postfix="/train"
        )
        self.val_metrics = MetricCollection(
            {"pf1": ProbabilisticBinaryF1Score(), "accuracy": BinaryAccuracy()}, postfix="/val"
        )
        self.loss = loss
        self.optimizer_config = optimizer_config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.precision == 16:
            x = x.half()
        else:
            x = x.float()
        return self.classifier(self.feature_extractor(x)).squeeze(dim=1)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        prediction = self(batch["pixels"])
        # loss = self.loss(input=prediction, target=batch["cancer"].float())
        metrics = self.train_metrics(preds=prediction, target=batch["cancer"])
        self.log_dict(self.train_metrics, on_step=True, on_epoch=False, sync_dist=True)  # type: ignore[arg-type]
        return {"loss": -metrics["pf1/train"]}

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        prediction = self(batch["pixels"])
        # loss = self.loss(input=prediction, target=batch["cancer"].float())
        metrics = self.val_metrics(preds=prediction, target=batch["cancer"])
        self.log_dict(self.val_metrics, on_step=True, on_epoch=True, sync_dist=True)  # type: ignore[arg-type]
        return {"loss": -metrics["pf1/val"]}

    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:  # type: ignore[override]
        return self(batch["pixels"])

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return self.optimizer_config(params=self.parameters())


@hydra.main(config_path="../config", config_name="train", version_base=None)
def train(cfg: DictConfig) -> None:
    seed_everything(seed=42, workers=True)
    trainer: pl.Trainer = instantiate(cfg.trainer)
    datamodule: pl.LightningDataModule = instantiate(cfg.datamodule)
    model: pl.LightningModule = instantiate(cfg.model)
    for logger in trainer.loggers:
        if isinstance(logger, WandbLogger):
            logger.watch(model, log="all", log_freq=1)
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    logging.basicConfig(level=logging.NOTSET)
    train()
