import logging
from typing import TYPE_CHECKING

import hydra
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning.profilers import PyTorchProfiler

if TYPE_CHECKING:
    from pytorch_lightning import LightningDataModule, LightningModule, Trainer


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
