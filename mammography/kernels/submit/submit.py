import logging

import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


@hydra.main(config_path="../config", config_name="train", version_base=None)
def submit(cfg: DictConfig) -> None:
    model: pl.LightningModule = instantiate(cfg.model)
    datamodule: pl.LightningDataModule = instantiate(cfg.datamodule)
    trainer: pl.Trainer = instantiate(cfg.trainer)
    predictions = trainer.predict(model=model, datamodule=datamodule, return_predictions=True, ckpt_path=cfg.ckpt_path)
    print(pd.DataFrame(predictions))
    # test_df = pd.read_csv("../input/rsna-breast-cancer-detection/test.csv")
    # test_df["prediction_id"] = test_df["patient_id"].astype(str) + "_" + test_df["laterality"]
    # test_df.drop_duplicates("prediction_id", inplace=True)
    # test_df["cancer"] = np.random.random(len(test_df))
    # test_df[["prediction_id", "cancer"]].to_csv("submission.csv", index=False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.NOTSET)
    main()
