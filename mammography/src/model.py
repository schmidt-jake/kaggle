import logging
from inspect import signature
from typing import TYPE_CHECKING, Any, Dict, List, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torchmetrics import MeanMetric, MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryCalibrationError,
)
from wandb.data_types import Classes

from mammography.src.metrics import ProbabilisticBinaryF1Score

if TYPE_CHECKING:
    from pytorch_lightning.loggers import WandbLogger

logger = logging.getLogger(__name__)


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
        if isinstance(new_child, torch.nn.modules.conv._ConvNd):
            torch.nn.init.kaiming_normal_(new_child.weight, nonlinearity="relu")
        setattr(parent, child_name, new_child)
    return module


class Model(pl.LightningModule):
    def __init__(
        self,
        feature_extractor: DictConfig,
        optimizer_config: DictConfig,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.logger: "WandbLogger"
        self.feature_extractor: torch.nn.Module = instantiate(self.hparams.feature_extractor)
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

    def _init_metrics(self) -> None:
        for name, metric in self.train_metrics.items(keep_base=True):
            self.logger.experiment.define_metric(name=name, summary="max" if metric.higher_is_better else "min")
        for name, metric in self.val_metrics.items(keep_base=True):
            self.logger.experiment.define_metric(name=name, summary="max" if metric.higher_is_better else "min")

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
        self.example_input_array = torch.randint(
            low=0, high=255, size=(self.trainer.datamodule.hparams.batch_size, 1, 512, 512), dtype=torch.uint8
        )
        if stage == "fit":
            self.loss = torch.nn.BCEWithLogitsLoss(
                # pos_weight=torch.tensor(self.trainer.datamodule.class_weights[1]),  # type: ignore[attr-defined]
            )
            self._init_metrics()
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
        x = x.float()
        x /= 255.0
        predictions: torch.Tensor = self.feature_extractor(x)
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
                caption=["cancer" if c == 1 else "no-cancer" for c in cancer],
            )

        loss: torch.Tensor = self.loss(input=logit, target=batch["cancer"].float())
        self.train_metrics(preds=preds, target=batch["cancer"], value=loss)
        self.log_dict(self.train_metrics, on_step=True, on_epoch=False)  # type: ignore[arg-type]
        return {"loss": loss}

    def validation_step(self, batch: Dict[str, Union[torch.Tensor, List[str]]], batch_idx: int) -> None:
        logit: torch.Tensor = self(batch["pixels"])
        preds = logit.sigmoid()
        # self.logger.experiment.log({"predictions/val": wandb.Histogram(prediction.detach().cpu())}, step=self.global_step)
        loss: torch.Tensor = self.loss(input=logit, target=batch["cancer"].float())
        self.val_metrics(preds=preds, target=batch["cancer"], value=loss)
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True)  # type: ignore[arg-type]

    def predict_step(
        self, batch: Dict[str, Union[torch.Tensor, List[str]]], batch_idx: int
    ) -> Dict[str, Union[torch.Tensor, List[str]]]:
        logit: torch.Tensor = self(batch["pixels"])
        return {
            "image_id": batch["image_id"],
            "cancer": logit.sigmoid(),
            "patient_id": batch["patient_id"],
            "laterality": batch["laterality"],
        }

    def configure_optimizers(self) -> Dict[str, Any]:
        config = instantiate(self.hparams.optimizer_config)
        config["optimizer"]: torch.optim.Optimizer = config["optimizer"](params=self.parameters())
        try:
            config["lr_scheduler"]["scheduler"]: torch.optim.lr_scheduler._LRScheduler = config["lr_scheduler"][
                "scheduler"
            ](optimizer=config["optimizer"])
        except ValueError:
            config["lr_scheduler"]["scheduler"]: torch.optim.lr_scheduler._LRScheduler = config["lr_scheduler"][
                "scheduler"
            ](optimizer=config["optimizer"], total_steps=self.trainer.estimated_stepping_batches)
        return config

    def optimizer_zero_grad(
        self, epoch: int, batch_idx: int, optimizer: torch.optim.Optimizer, optimizer_idx: int
    ) -> None:
        optimizer.zero_grad(set_to_none=True)
