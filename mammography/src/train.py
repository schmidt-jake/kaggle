import logging
from inspect import signature
from typing import Any, Callable, Dict, List

import cv2
import dicomsdl
import hydra
import numpy as np
import numpy.typing as npt
import pandas as pd
import pytorch_lightning as pl
import torch
import wandb

# from functorch.compile import memory_efficient_fusion
from hydra.utils import instantiate
from lightning_lite.utilities.seed import seed_everything
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

# from torchdata.dataloader2 import DataLoader2, PrototypeMultiProcessingReadingService
# from torchdata.datapipes.map import MapDataPipe
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC
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
        is_y_true = target == 1
        self.y_true_count += is_y_true.sum()  # type: ignore[operator]
        self.ctp += preds[is_y_true].sum()
        self.cfp += preds[is_y_true.logical_not()].sum()

    def compute(self) -> torch.Tensor:
        c_precision: torch.Tensor = self.ctp / (self.ctp + self.cfp)  # type: ignore[operator]
        c_recall: torch.Tensor = self.ctp / self.y_true_count  # type: ignore[operator]
        result = 2 * (c_precision * c_recall) / (c_precision + c_recall)
        result.nan_to_num_()
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

    def __len__(self) -> int:
        return len(self.df)

    @staticmethod
    def _read(filepath: str) -> npt.NDArray[np.uint8]:
        if filepath.endswith(".dcm"):
            arr = dicom2numpy(filepath)
            arr = crop(arr)
            return arr
        elif filepath.endswith(".png"):
            arr = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            return arr
        else:
            raise ValueError(f"Got unknown file suffix in filepath: {filepath}")

    def __getitem__(self, index: int) -> Dict[str, Any]:
        cv2.setNumThreads(0)
        row = self.df.iloc[index]
        logger.debug(f"Loading image {row['image_id']}")
        d = row.to_dict()
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


def crop(img: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
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
    # arr = arr.astype(np.float32)
    # arr /= 65_535.0
    # arr *= 255.0
    # arr = arr.astype(np.uint8)
    arr = cv2.normalize(src=arr, dst=arr, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
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
        self,
        metadata_filepath: str,
        image_dir: str,
        augmentation: torch.nn.Sequential,
        batch_size: int,
        prefetch_batches: int,
    ) -> None:
        super().__init__()
        self.metadata_filepath = metadata_filepath
        self.image_dir = image_dir
        self.augmentation = augmentation
        self.batch_size = batch_size
        self.num_workers = torch.multiprocessing.cpu_count()
        self.prefetch = max(prefetch_batches * self.batch_size // max(self.num_workers, 1), 2)

    @staticmethod
    def compute_class_weights(y: pd.Series) -> pd.Series:
        return len(y) / (y.nunique() * y.value_counts())

    def setup(self, stage: str) -> None:
        self.df = pd.read_csv(self.metadata_filepath)
        if stage == "fit":
            self.df["filepath"] = self.image_dir + "/" + self.df["image_id"].astype(str) + ".png"
            self.class_weights = self.compute_class_weights(self.df["cancer"])
            self.df["sample_weight"] = self.df["cancer"].map(self.class_weights.get)
        elif stage == "predict":
            self.df["filepath"] = (
                self.image_dir
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
            pin_memory=True,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch,
            sampler=WeightedRandomSampler(
                weights=self.df["sample_weight"],
                num_samples=len(self.df),
                replacement=False,
            ),
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
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch,
        )

    def predict_dataloader(self) -> DataLoader:
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
        self.train_metrics = MetricCollection(
            {
                "pf1": ProbabilisticBinaryF1Score(),
                "accuracy": BinaryAccuracy(validate_args=False),
            },
            prefix="metrics/",
            postfix="/train",
        )
        self.val_metrics = MetricCollection(
            {
                "pf1": ProbabilisticBinaryF1Score(),
                "accuracy": BinaryAccuracy(validate_args=False),
                # "roc": BinaryROC(validate_args=False),
                "auroc": BinaryAUROC(validate_args=False),
            },
            prefix="metrics/",
            postfix="/val",
        )
        self.optimizer_config = optimizer_config

    @staticmethod
    def get_bias(y: pd.Series) -> float:
        """
        Gets the value of the input to the sigmoid function such that
        it outputs the probability of the postive class.

        Parameters
        ----------
        y : pd.Series
            The vector of binary labels for the training data

        Returns
        -------
        float
            The bias value.
        """
        p = y.mean()
        return np.log(p / (1 - p))

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.loss = torch.nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(self.trainer.datamodule.class_weights[1]),  # type: ignore[attr-defined]
            )
            torch.nn.init.constant_(
                tensor=self.classifier[-1].bias,
                val=self.get_bias(self.trainer.datamodule.df["cancer"]),  # type: ignore[attr-defined]
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.precision == 16:
            x = x.half()
        else:
            x = x.float()
        # features: torch.Tensor = checkpoint_sequential(
        #     self.feature_extractor, segments=5, input=x, preserve_rng_state=False
        # )
        features = self.feature_extractor(x)
        predictions: torch.Tensor = self.classifier(features)
        # predictions: torch.Tensor = checkpoint_sequential(
        #     self.classifier, segments=2, input=features, preserve_rng_state=False
        # )
        return predictions.squeeze(dim=1)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        prediction: torch.Tensor = self(batch["pixels"])
        wandb.log({"predictions/train": wandb.Histogram(prediction.detach().cpu())}, step=self.global_step)
        loss: torch.Tensor = self.loss(input=prediction, target=batch["cancer"].float())
        self.train_metrics(preds=prediction.sigmoid(), target=batch["cancer"])
        self.log_dict(self.train_metrics, on_step=True, on_epoch=False)  # type: ignore[arg-type]
        return {"loss": loss}

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        prediction: torch.Tensor = self(batch["pixels"])
        wandb.log({"predictions/val": wandb.Histogram(prediction.detach().cpu())}, step=self.global_step)
        loss: torch.Tensor = self.loss(input=prediction, target=batch["cancer"].float())
        self.val_metrics(preds=prediction.sigmoid(), target=batch["cancer"])
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True)  # type: ignore[arg-type]
        return {"loss": loss}

    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:  # type: ignore[override]
        return self(batch["pixels"])

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return self.optimizer_config(params=self.parameters())

    def optimizer_zero_grad(
        self, epoch: int, batch_idx: int, optimizer: torch.optim.Optimizer, optimizer_idx: int
    ) -> None:
        optimizer.zero_grad(set_to_none=True)


@hydra.main(config_path="../config", config_name="train", version_base=None)
def train(cfg: DictConfig) -> None:
    seed_everything(seed=42, workers=True)
    trainer: pl.Trainer = instantiate(cfg.trainer)
    datamodule: DataModule = instantiate(cfg.datamodule)
    model: pl.LightningModule = instantiate(cfg.model)
    wandb.watch(model, log="all", log_freq=cfg.trainer.log_every_n_steps, log_graph=True)
    trainer.fit(model=model, datamodule=datamodule)

    [logger.log_hyperparams(cfg) for logger in trainer.loggers]  # type: ignore[arg-type]

    if hasattr(trainer, "profiler"):
        profile_art = wandb.Artifact("trace", type="profile")
        profile_art.add_dir(trainer.profiler.dirpath)
        profile_art.save()


if __name__ == "__main__":
    logging.basicConfig(level=logging.NOTSET)
    train()
