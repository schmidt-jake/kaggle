import logging
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

from mammography.src.metrics import ProbabilisticBinaryF1Score

if TYPE_CHECKING:
    from pytorch_lightning.loggers import WandbLogger

logger = logging.getLogger(__name__)


class Model(pl.LightningModule):
    def __init__(self, net: DictConfig, optimizer_config: DictConfig, loss: DictConfig) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.logger: "WandbLogger"
        self.net: torch.nn.Module = instantiate(self.hparams_initial["net"])
        for k in {"cancer", "density", "sum"}:
            setattr(self, f"loss_{k}_train", MeanMetric(nan_strategy="error"))
            setattr(self, f"loss_{k}_val", MeanMetric(nan_strategy="error"))
        for k in {"cancer", "density"}:
            setattr(
                self,
                f"metrics_{k}_train",
                MetricCollection(
                    {
                        "pf1": ProbabilisticBinaryF1Score(),
                        "accuracy": BinaryAccuracy(validate_args=False),
                        "auroc": BinaryAUROC(validate_args=False),
                    },
                    prefix=k + "/",
                    postfix="/train",
                ),
            )
            setattr(
                self,
                f"metrics_{k}_val",
                MetricCollection(
                    {
                        "pf1": ProbabilisticBinaryF1Score(),
                        "accuracy": BinaryAccuracy(validate_args=False),
                        "auroc": BinaryAUROC(validate_args=False),
                    },
                    prefix=k + "/",
                    postfix="/val",
                ),
            )

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
            size=(self.trainer.datamodule.hparams["train_batch_size"], 1, 512, 512),
            dtype=torch.uint8,
        )
        self.example_input_array = {
            "cc": x,
            "mlo": x,
            "age": torch.randint(
                low=30, high=90, size=(self.trainer.datamodule.hparams["train_batch_size"],), dtype=torch.uint8
            ),
        }
        if stage == "fit":
            self.loss = instantiate(self.hparams_initial["loss"])
            # self._init_metrics()
            self.cancer_base_rate = self.trainer.datamodule.meta["train"]["cancer"].mean()
            self.age_mean = self.trainer.datamodule.meta["train"]["age"].mean()
            self.age_std = self.trainer.datamodule.meta["train"]["age"].std()
            with torch.no_grad():
                self(**self.example_input_array)

    def forward(self, age: torch.Tensor, **imgs: torch.Tensor) -> torch.Tensor:
        age = age.float()
        age -= self.age_mean
        age /= self.age_std
        return self.net(age=age.unsqueeze(dim=1), **imgs)

    def get_extra_state(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in {"cancer_base_rate", "age_mean", "age_std"}}

    def set_extra_state(self, state: Dict[str, Any]) -> None:
        for k, v in state.items():
            setattr(self, k, v)

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
        logit: torch.Tensor = self(cc=batch["CC"], mlo=batch["MLO"], age=batch["age"])
        if self.global_step == 0 and self.global_rank == 0:
            self.log_images(batch)
        loss: torch.Tensor = self.loss(
            input=logit, target={"cancer": batch["cancer"].float(), "density": batch["density"].float()}
        )
        for k, v in loss.items():
            m = getattr(self, f"loss_{k}_train")
            m(value=v)
            self.log(name=f"loss/{k}/train", value=m, on_step=True, on_epoch=False)  # type: ignore[arg-type]
            if hasattr(self, f"metrics_{k}_train"):
                m = getattr(self, f"metrics_{k}_train")
                m(preds=logit["cancer"].sigmoid(), target=batch["cancer"])
                self.log_dict(m, on_step=True, on_epoch=False)
        return {"loss": loss["sum"]}

    def validation_step(self, batch: Dict[str, Union[torch.Tensor, List[str]]], batch_idx: int) -> None:
        logit: torch.Tensor = self(cc=batch["CC"], mlo=batch["MLO"], age=batch["age"])
        loss: torch.Tensor = self.loss(
            input=logit, target={"cancer": batch["cancer"].float(), "density": batch["density"].float()}
        )
        for k, v in loss.items():
            m = getattr(self, f"loss_{k}_val")
            m(value=v)
            self.log(name=f"loss/{k}/val", value=m, on_step=False, on_epoch=True)  # type: ignore[arg-type]
            if hasattr(self, f"metrics_{k}_val"):
                m = getattr(self, f"metrics_{k}_val")
                m(preds=logit["cancer"].sigmoid(), target=batch["cancer"])
                self.log_dict(m, on_step=False, on_epoch=True)

    def predict_step(
        self, batch: Dict[str, Union[torch.Tensor, List[str]]], batch_idx: int
    ) -> Dict[str, Union[torch.Tensor, List[str]]]:
        logit: torch.Tensor = self(cc=batch["CC"], mlo=batch["MLO"], age=batch["age"])
        return {"cancer": logit["cancer"].sigmoid(), "prediction_id": batch["prediction_id"]}

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
