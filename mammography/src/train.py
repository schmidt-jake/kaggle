import logging
from inspect import signature
from typing import TYPE_CHECKING, Any, Callable, Dict, List

import cv2
import hydra
import numpy as np
import numpy.typing as npt
import pandas as pd
import pytorch_lightning as pl
import torch
import wandb
from hydra.utils import instantiate
from lightning_lite.utilities.seed import seed_everything
from omegaconf import DictConfig, OmegaConf
from pydicom.pixel_data_handlers.util import apply_windowing
from pytorch_lightning.profilers import PyTorchProfiler
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

# from torchdata.dataloader2 import DataLoader2, PrototypeMultiProcessingReadingService
# from torchdata.datapipes.map import MapDataPipe
from torchmetrics import MeanMetric, Metric, MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryCalibrationError,
)
from torchvision.transforms import functional_tensor
from wandb.data_types import Classes

from mammography.src import utils

if TYPE_CHECKING:
    from pytorch_lightning.loggers import WandbLogger

logger = logging.getLogger(__name__)


class ProbabilisticBinaryF1Score(Metric):
    is_differentiable = False
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
        result.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        return result


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
            arr, dcm = utils.dicom2numpy(filepath, should_apply_window_fn=False)
            windows = utils.get_unique_windows(dcm)
            if not windows.empty:
                windowed_arr = apply_windowing(arr=arr.copy(), ds=dcm, index=np.random.choice(windows.index))
                if windowed_arr.ptp() > 10:
                    arr = windowed_arr
                else:
                    logger.warning(f"Got bad window for {filepath}")
            # dcm.BitsStored = utils.get_suspected_bit_depth(arr)
            arr = utils.maybe_invert(arr=arr, dcm=dcm)
            arr = utils.maybe_flip_left(arr=arr)
            # arr = utils.scale_to_01(arr=arr, dcm=dcm)
        elif filepath.endswith(".png"):
            arr = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        else:
            raise ValueError(f"Got unknown file suffix in filepath: {filepath}")
        return arr

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.df.iloc[index]
        logger.debug(f"Loading image {row['image_id']}")
        d = row.to_dict()
        arr = self._read(filepath=row["filepath"])
        pixels = torch.from_numpy(arr.astype(np.float32))
        _min, _max = pixels.min(), pixels.max()
        pixels -= _min
        pixels /= _max - _min
        pixels.unsqueeze_(dim=0)
        pixels = utils.crop_right_center(pixels, size=2048)
        pixels = functional_tensor.resize(pixels, size=512)
        # d["pixels"] = self.augmentation(pixels)
        d["pixels"] = pixels
        d = {k: v for k, v in d.items() if k in ["pixels", "cancer"]}
        return d


def replace_layer(layer_to_replace: torch.nn.Module, **new_layer_kwargs) -> torch.nn.Module:
    class_signature = signature(layer_to_replace.__class__).parameters
    layer_params = {k: getattr(layer_to_replace, k) for k in class_signature.keys() if hasattr(layer_to_replace, k)}
    layer_params.update(new_layer_kwargs)
    if "bias" in layer_params.keys() and isinstance(layer_params["bias"], torch.Tensor):
        layer_params["bias"] = True
    return type(layer_to_replace)(**layer_params)


def replace_submodules(module: torch.nn.Module, **replacement_kwargs: Dict[str, Any]) -> torch.nn.Module:
    for target, replacement in replacement_kwargs.items():
        parent_name, _, child_name = target.rpartition(".")
        parent = module.get_submodule(parent_name)
        child = parent.get_submodule(child_name)
        new_child = replace_layer(child, **replacement)
        setattr(parent, child_name, new_child)
    return module


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
        self.df.query("patient_id != 27770", inplace=True)
        self.df.query("image_id != 1942326353", inplace=True)
        if stage == "fit":
            # self.df["filepath"] = self.image_dir + "/" + self.df["image_id"].astype(str) + ".png"
            self.df["filepath"] = (
                self.image_dir
                + "/"
                + self.df["patient_id"].astype(str)
                + "/"
                + self.df["image_id"].astype(str)
                + ".dcm"
            )
            class_weights = 1.0 / self.df["cancer"].value_counts()
            self.df["sample_weight"] = self.df["cancer"].map(class_weights.get)
        elif stage == "predict":
            self.df["filepath"] = (
                self.image_dir
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
        pipe = DataframeDataPipe(self.df, augmentation=self.augmentation.eval())  # .to_iter_datapipe()
        return DataLoader(
            dataset=pipe,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch,
            drop_last=False,
        )

    def predict_dataloader(self) -> DataLoader:
        return self.val_dataloader()


class Model(pl.LightningModule):
    def __init__(
        self,
        feature_extractor: torch.nn.Module,
        classifier: torch.nn.Sequential,
        optimizer_config: Callable[..., torch.optim.Optimizer],
    ) -> None:
        super().__init__()
        self.logger: "WandbLogger"
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.train_metrics = MetricCollection(
            {
                "pf1": ProbabilisticBinaryF1Score(),
                "accuracy": BinaryAccuracy(validate_args=False),
                "loss": MeanMetric(nan_strategy="error"),
                # "predictions": CatMetric(nan_strategy="error"),
            },
            prefix="metrics/",
            postfix="/train",
        )
        self.val_metrics = MetricCollection(
            {
                "pf1": ProbabilisticBinaryF1Score(),
                "accuracy": BinaryAccuracy(validate_args=False),
                "loss": MeanMetric(nan_strategy="error"),
                # "roc": BinaryROC(validate_args=False),
                "auroc": BinaryAUROC(validate_args=False),
                "calibration_error": BinaryCalibrationError(validate_args=False),
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
        self.example_input_array = torch.rand(
            size=(self.trainer.datamodule.batch_size, 1, 512, 512), dtype=torch.float32
        )
        if stage == "fit":
            self.loss = torch.nn.BCEWithLogitsLoss(
                # pos_weight=torch.tensor(self.trainer.datamodule.class_weights[1]),  # type: ignore[attr-defined]
            )
            # for m in self.modules():
            #     if isinstance(m, torch.nn.Conv2d):
            #         torch.nn.init.kaiming_normal_(m.weight)
            #     elif isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            #         torch.nn.init.constant_(m.weight, 1)
            #         torch.nn.init.constant_(m.bias, 0)
            #     elif isinstance(m, torch.nn.Linear):
            #         torch.nn.init.constant_(m.bias, 0)
            # torch.nn.init.constant_(
            #     tensor=self.classifier[-1].bias, val=self.get_bias(self.trainer.datamodule.df["cancer"])
            # )
            # torch.nn.init.zeros_(self.classifier[-1].weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # features: torch.Tensor = checkpoint_sequential(
        #     self.feature_extractor, segments=5, input=x, preserve_rng_state=False
        # )
        features: torch.Tensor = self.feature_extractor(x)
        predictions: torch.Tensor = self.classifier(features)
        # predictions: torch.Tensor = checkpoint_sequential(
        #     self.classifier, segments=2, input=features, preserve_rng_state=False
        # )
        return predictions.squeeze(dim=1)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        logit: torch.Tensor = self(batch["pixels"])
        preds = logit.sigmoid()
        # self.logger.experiment.log(
        #     {
        #         "pixels/train": wandb.Histogram(batch["pixels"].detach().cpu()),
        #         "predictions/train": wandb.Histogram(preds.detach().cpu()),
        #         "labels/train": batch["cancer"].detach().float().mean().cpu(),
        #     },
        #     step=self.global_step,
        # )
        if self.global_step == 0 and self.global_rank == 0:
            pixels: npt.NDArray[np.float32] = batch["pixels"].detach().cpu().numpy()
            cancer = batch["cancer"].detach().cpu().numpy()
            self.logger.log_image(
                key="input_batch",
                images=[pixels[i, 0, :, :] for i in range(pixels.shape[0])],
                step=self.global_step,
                mode=["L"] * pixels.shape[0],
                classes=[Classes([{"id": c, "name": "cancer" if c == 1 else "no-cancer"}]) for c in cancer],
            )

        loss: torch.Tensor = self.loss(input=logit, target=batch["cancer"].float())
        self.train_metrics(preds=preds, target=batch["cancer"], value=loss)
        self.log_dict(self.train_metrics, on_step=True, on_epoch=False)  # type: ignore[arg-type]
        return {"loss": loss}

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        logit: torch.Tensor = self(batch["pixels"])
        preds = logit.sigmoid()
        # self.logger.experiment.log({"predictions/val": wandb.Histogram(prediction.detach().cpu())}, step=self.global_step)
        loss: torch.Tensor = self.loss(input=logit, target=batch["cancer"].float())
        self.val_metrics(preds=preds, target=batch["cancer"], value=loss)
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True)  # type: ignore[arg-type]

    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:  # type: ignore[override]
        logit: torch.Tensor = self(batch["pixels"])
        return logit.sigmoid()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        config = self.optimizer_config
        config["optimizer"] = config["optimizer"](params=self.parameters())
        try:
            config["lr_scheduler"]["scheduler"] = config["lr_scheduler"]["scheduler"](optimizer=config["optimizer"])
        except ValueError:
            config["lr_scheduler"]["scheduler"] = config["lr_scheduler"]["scheduler"](
                optimizer=config["optimizer"], total_steps=self.trainer.estimated_stepping_batches
            )
        del self.optimizer_config
        return OmegaConf.to_container(config)

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
    trainer.logger.log_hyperparams(cfg)
    trainer.logger.watch(model, log="all", log_freq=cfg.trainer.log_every_n_steps, log_graph=True)
    trainer.fit(model=model, datamodule=datamodule)
    if isinstance(trainer.profiler, PyTorchProfiler):
        profile_art = wandb.Artifact("trace", type="profile")
        profile_art.add_dir(trainer.profiler.dirpath)
        trainer.logger.experiment.log_artifact(profile_art)


if __name__ == "__main__":
    logging.basicConfig(level=logging.NOTSET)
    train()
