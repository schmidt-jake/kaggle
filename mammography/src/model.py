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
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC
from torchvision.transforms import ConvertImageDtype

from mammography.src.loss import SigmoidFocalLoss
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
        self.feature_extractor: torch.nn.Module = instantiate(self.hparams_initial["feature_extractor"])
        self.train_metrics = MetricCollection(
            {
                "pf1": ProbabilisticBinaryF1Score(),
                "accuracy": BinaryAccuracy(validate_args=False),
                "loss": MeanMetric(nan_strategy="error"),
                "auroc": BinaryAUROC(validate_args=False),
            },
            prefix="metrics/",
            postfix="/train",
        )
        self.val_metrics = MetricCollection(
            {
                "pf1": ProbabilisticBinaryF1Score(),
                "accuracy": BinaryAccuracy(validate_args=False),
                "loss": MeanMetric(nan_strategy="error"),
                "auroc": BinaryAUROC(validate_args=False),
            },
            prefix="metrics/",
            postfix="/val",
        )
        self.transform = torch.jit.script(torch.nn.Sequential(ConvertImageDtype(dtype=torch.float)))

    def _init_metrics(self) -> None:
        for attr in ["train_metrics", "val_metrics"]:
            for name, metric in getattr(self, attr).items(keep_base=True):
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
        x = torch.randint(
            low=0,
            high=255,
            size=(self.trainer.datamodule.hparams["batch_size"], 1, 512, 512),
            dtype=torch.uint8,
        )
        self.example_input_array = {"cc": x, "mlo": x}
        if stage == "fit":
            self.loss = torch.nn.BCEWithLogitsLoss()
            # self.loss = torch.jit.script(SigmoidFocalLoss())
            self._init_metrics()
            self.hparams["cancer_base_rate"] = self.trainer.datamodule.meta["train"]["cancer"].mean()

    def forward(self, cc: torch.Tensor, mlo: torch.Tensor) -> torch.Tensor:
        p1: torch.Tensor = self.feature_extractor(self.transform(cc).expand(-1, 3, -1, -1))
        p2: torch.Tensor = self.feature_extractor(self.transform(mlo).expand(-1, 3, -1, -1))
        predictions = torch.cat([p1, p2], dim=1)
        return predictions.max(dim=1).values

    def log_images(self, batch: Dict[str, torch.Tensor]) -> None:
        cancer = batch["cancer"].detach().cpu().numpy()
        pixels: npt.NDArray[np.uint8] = batch["CC"].detach().cpu().numpy()
        caption = ["cancer" if c == 1 else "no-cancer" for c in cancer]
        self.logger.log_image(
            key="input_CC",
            images=[pixels[i, 0, :, :] for i in range(pixels.shape[0])],
            step=self.global_step,
            mode=["L"] * pixels.shape[0],
            caption=caption,
        )
        pixels: npt.NDArray[np.uint8] = batch["MLO"].detach().cpu().numpy()
        self.logger.log_image(
            key="input_MLO",
            images=[pixels[i, 0, :, :] for i in range(pixels.shape[0])],
            step=self.global_step,
            mode=["L"] * pixels.shape[0],
            caption=caption,
        )

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        logit: torch.Tensor = self(cc=batch["CC"], mlo=batch["MLO"])
        if self.global_step == 0 and self.global_rank == 0:
            self.log_images(batch)
        # loss, preds = self.loss(input=logit, target=batch["cancer"].float())
        loss: torch.Tensor = self.loss(input=logit, target=batch["cancer"].float())
        self.train_metrics(preds=logit.sigmoid(), target=batch["cancer"], value=loss)
        self.log_dict(self.train_metrics, on_step=True, on_epoch=False)  # type: ignore[arg-type]
        return {"loss": loss}

    def validation_step(self, batch: Dict[str, Union[torch.Tensor, List[str]]], batch_idx: int) -> None:
        logit: torch.Tensor = self(cc=batch["CC"], mlo=batch["MLO"])
        # loss, preds = self.loss(input=logit, target=batch["cancer"].float())
        loss = self.loss(input=logit, target=batch["cancer"].float())
        self.val_metrics(preds=logit.sigmoid(), target=batch["cancer"], value=loss)
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True)  # type: ignore[arg-type]

    def predict_step(
        self, batch: Dict[str, Union[torch.Tensor, List[str]]], batch_idx: int
    ) -> Dict[str, Union[torch.Tensor, List[str]]]:
        logit: torch.Tensor = self(cc=batch["CC"], mlo=batch["MLO"])
        return {"cancer": logit.sigmoid(), "prediction_id": batch["prediction_id"]}

    def configure_optimizers(self) -> Dict[str, Any]:
        weight_decay = self.hparams_initial["optimizer_config"]["optimizer"].pop("weight_decay")
        config = instantiate(self.hparams_initial["optimizer_config"])
        config["optimizer"]: torch.optim.Optimizer = config["optimizer"](params=self._get_params(weight_decay))
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

    def _get_params(self, weight_decay: float):
        params = dict(self.named_parameters())
        decay_params = set()
        no_decay_params = set()
        for param_name, param in params.items():
            parent_name, _, child_name = param_name.rpartition(".")
            if child_name.endswith("weight") and isinstance(
                self.get_submodule(parent_name), (torch.nn.Linear, torch.nn.modules.conv._ConvNd)
            ):
                decay_params.add(param_name)
            else:
                no_decay_params.add(param_name)

        assert decay_params.isdisjoint(no_decay_params)
        assert decay_params.union(no_decay_params) == params.keys()

        logger.info(f"Applying weight decay to {len(decay_params)/len(params):.2%} of all parameters")

        return [
            {"params": [params[p] for p in decay_params], "weight_decay": weight_decay},
            {"params": [params[p] for p in no_decay_params], "weight_decay": 0.0},
        ]
