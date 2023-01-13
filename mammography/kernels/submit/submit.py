import logging

import hydra
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


@hydra.main(config_path="../config", config_name="train", version_base=None)
def submit(cfg: DictConfig) -> None:
    model: pl.LightningModule = instantiate(cfg.model)
    datamodule: pl.LightningDataModule = instantiate(cfg.datamodule)
    trainer: pl.Trainer = instantiate(cfg.trainer)
    trainer.predict(model=model, datamodule=datamodule, return_predictions=False, ckpt_path=cfg.ckpt_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.NOTSET)
    submit()
