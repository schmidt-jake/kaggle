import logging
from logging import getLogger
from typing import TYPE_CHECKING, Any, Dict, Optional

import hydra
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.profilers import PyTorchProfiler
from pytorch_lightning.tuner.batch_size_scaling import scale_batch_size

if TYPE_CHECKING:
    from pytorch_lightning import LightningDataModule, LightningModule, Trainer


logger = getLogger(__name__)


class BatchSizeFinder(Callback):
    def __init__(self, training: Optional[Dict[str, Any]] = None, inference: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        self.training = training
        self.inference = inference

    def on_fit_start(self, trainer: "Trainer", pl_module: "LightningModule") -> None:
        if self.training:
            logger.info("Finding training batch size...")
            scale_batch_size(trainer=trainer, model=pl_module, **self.training)

    def on_validation_start(self, trainer: "Trainer", pl_module: "LightningModule") -> None:
        if trainer.sanity_checking or trainer.state.fn != "validate":
            return

        if self.inference and trainer.current_epoch == 0:
            logger.info("Finding validation batch size...")
            scale_batch_size(trainer=trainer, model=pl_module, **self.inference)

    def on_predict_start(self, trainer: "Trainer", pl_module: "LightningModule") -> None:
        if self.inference and trainer.current_epoch == 0:
            logger.info("Finding predict batch size...")
            scale_batch_size(trainer=trainer, model=pl_module, **self.inference)


@hydra.main(config_path="../config", config_name="train", version_base=None)
def train(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    instantiate(cfg.seed_fn)
    trainer: "Trainer" = instantiate(cfg.trainer)
    datamodule: "LightningDataModule" = instantiate(cfg.datamodule, _recursive_=False)
    model: "LightningModule" = instantiate(cfg.model, _recursive_=False)
    # trainer.logger.watch(
    #     model.feature_extractor,
    #     log="all",
    #     # log_freq=trainer.log_every_n_steps,
    #     log_freq=25,
    #     log_graph=True,
    # )
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=getattr(cfg, "ckpt_path", None))
    if isinstance(trainer.profiler, PyTorchProfiler):
        profile_art = wandb.Artifact("trace", type="profile")
        profile_art.add_dir(trainer.profiler.dirpath)
        trainer.logger.experiment.log_artifact(profile_art)


if __name__ == "__main__":
    train()
