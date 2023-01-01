import logging
import os
from inspect import signature

# from tempfile import TemporaryDirectory
from typing import Any, Callable, Dict, List

import cv2
import dicomsdl
import hydra
import numpy as np
import numpy.typing as npt
import pandas as pd
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from lightning_lite.utilities.seed import seed_everything
from omegaconf import DictConfig
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset

# from torchdata.dataloader2 import DataLoader2, PrototypeMultiProcessingReadingService
# from torchdata.datapipes.map import MapDataPipe
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, BinaryROC
from torchvision.models.feature_extraction import create_feature_extractor

logger = logging.getLogger(__name__)


class ProbabilisticBinaryF1Score(Metric):
    is_differentiable = True
    higher_is_better = True
    full_state_update = False

    def __init__(self, **kwargs) -> None:
        # https://torchmetrics.readthedocs.io/en/stable/pages/implement.html
        # https://www.kaggle.com/code/sohier/probabilistic-f-score/notebook
        super().__init__(**kwargs)
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
        # x /= 65534.0  # 2 ** 16 - 1
        x /= 255.0  # 2 ** 8 - 1
        return self.feature_extractor(x)[self.layer_name]


class DataframeDataPipe(Dataset):
    def __init__(self, df: pd.DataFrame, augmentation: torch.nn.Sequential) -> None:
        super().__init__()
        self.df = df
        self.augmentation = augmentation
        # self._cache_dir = TemporaryDirectory()

    def __len__(self) -> int:
        return len(self.df)

    def _cached_read(self, filepath: str, image_id: str) -> npt.NDArray[np.uint16]:
        save_path = os.path.join(self._cache_dir.name, image_id, "pixels.npy")
        if os.path.exists(save_path):
            arr = np.load(save_path)
        else:
            arr = self._read(filepath)
            np.save(save_path, arr)
        return arr

    @staticmethod
    def _read(filepath: str) -> npt.NDArray[np.uint16]:
        arr = dicom2numpy(filepath)
        arr = crop(arr)
        return arr

    def __getitem__(self, index: int) -> Dict[str, Any]:
        cv2.setNumThreads(0)
        row = self.df.iloc[index]
        logger.debug(f"Loading image {row['image_id']}")
        d = row.to_dict()
        # arr = self._cached_read(filepath=row["filepath"], image_id=row["image_id"])
        arr = self._read(filepath=row["filepath"])
        pixels = torch.from_numpy(arr).unsqueeze(dim=0)
        d["pixels"] = self.augmentation(pixels)
        d = {k: v for k, v in d.items() if k in ["pixels", "cancer"]}
        return d


def replace_layer(layer_to_replace: torch.nn.Module, **new_layer_kwargs) -> torch.nn.Module:
    class_signature = signature(layer_to_replace.__class__).parameters
    layer_params = {k: getattr(layer_to_replace, k) for k in class_signature.keys() if hasattr(layer_to_replace, k)}
    layer_params.update(new_layer_kwargs)
    return type(layer_to_replace)(**layer_params)


def crop(img: npt.NDArray[np.uint16]) -> npt.NDArray[np.uint8]:
    thresh, mask = cv2.threshold(img, thresh=0, maxval=1, type=cv2.THRESH_OTSU)
    logger.debug(f"thresh={thresh}")
    contours = cv2.findContours(mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)[0]
    max_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_contour)
    cropped = img[y : y + h, x : x + w]
    # https://stackoverflow.com/a/72666415/14841071
    # img_uint8 = cv2.convertScaleAbs(cropped, alpha=1.0 / 256.0, beta=-0.49999)
    return cropped


def dicom2numpy(filepath: str) -> npt.NDArray[np.uint8]:
    dcm = dicomsdl.open(filepath)
    arr = dcm.pixelData(storedvalue=True)
    # arr = cv2.convertScaleAbs(arr, alpha=1.0 / 256.0, beta=-0.49999)
    arr = arr.astype(np.float32)
    arr /= 65_535.0
    arr *= 255.0
    arr = arr.astype(np.uint8)
    # https://escapetech.eu/manuals/qmedical/commands/index_Values_of_Interest__.html
    if dcm.getPixelDataInfo()["PhotometricInterpretation"] == "MONOCHROME1":
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
    def __init__(
        self, root_dir: str, augmentation: torch.nn.Sequential, batch_size: int, prefetch_batches: int
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.augmentation = augmentation
        self.batch_size = batch_size
        self.num_workers = torch.multiprocessing.cpu_count()
        self.prefetch = max(prefetch_batches * self.batch_size // self.num_workers, 2)

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
        # self._train_cache = pipe._cache_dir
        return DataLoader(
            dataset=pipe,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch,
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
        # self._val_cache = pipe._cache_dir
        return DataLoader(
            dataset=pipe,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch,
        )

    # def teardown(self, stage: str) -> None:
    #     self._train_cache.cleanup()
    #     self._val_cache.cleanup()
    #     super().teardown(stage)

    def test_dataloader(self) -> DataLoader:
        return self.val_dataloader()


class Model(pl.LightningModule):
    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        classifier: torch.nn.Sequential,
        optimizer_config: Callable[..., torch.optim.Optimizer],
    ) -> None:
        super().__init__()
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.train_metrics = MetricCollection({"pf1": ProbabilisticBinaryF1Score()}, postfix="/train")
        self.val_metrics = MetricCollection(
            {
                "pf1": ProbabilisticBinaryF1Score(),
                "accuracy": BinaryAccuracy(validate_args=False),
                # "roc": BinaryROC(validate_args=False),
                "auroc": BinaryAUROC(validate_args=False),
            },
            postfix="/val",
        )
        self.optimizer_config = optimizer_config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.precision == 16:
            x = x.half()
        else:
            x = x.float()
        return self.classifier(self.feature_extractor(x)).squeeze(dim=1)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        prediction = self(batch["pixels"])
        metrics = self.train_metrics(preds=prediction, target=batch["cancer"])
        self.log_dict(self.train_metrics, on_step=True, on_epoch=False, sync_dist=True)  # type: ignore[arg-type]
        return {"loss": -metrics["pf1/train"]}

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        prediction = self(batch["pixels"])
        self.val_metrics(preds=prediction, target=batch["cancer"])
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True, sync_dist=True)  # type: ignore[arg-type]

    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:  # type: ignore[override]
        return self(batch["pixels"])

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return self.optimizer_config(params=self.parameters())


@hydra.main(config_path="../config", config_name="train", version_base=None)
def train(cfg: DictConfig) -> None:
    seed_everything(seed=42, workers=True)
    trainer: pl.Trainer = instantiate(cfg.trainer)
    datamodule: DataModule = instantiate(cfg.datamodule)
    model: pl.LightningModule = instantiate(cfg.model)
    for logger in trainer.loggers:
        logger.log_hyperparams(cfg)  # type: ignore[arg-type]
        # if isinstance(logger, WandbLogger):
        #     logger.watch(model, log="all", log_freq=cfg.trainer.log_every_n_steps)
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    logging.basicConfig(level=logging.NOTSET)
    train()
