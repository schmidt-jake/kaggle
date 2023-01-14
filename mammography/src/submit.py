import logging
from collections import defaultdict
from typing import Any, Dict, Optional, Sequence

import hydra
import pandas as pd
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning.callbacks import BasePredictionWriter

logger = logging.getLogger(__name__)


class SubmissionWriter(BasePredictionWriter):
    def __init__(self, write_interval: str, output_filepath: str) -> None:
        super().__init__(write_interval)
        self.output_filepath = output_filepath

    def write_on_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        predictions: Sequence[Any],
        batch_indices: Optional[Sequence[Any]],
    ) -> None:
        if len(predictions) > 1:
            logger.warning(f"Expected predictions len 1 but got {len(predictions)}")
        preds = defaultdict(list)
        for batch in predictions[0]:
            batch: Dict[str, torch.Tensor]
            for key, tensor in batch.items():
                preds[key].extend(tensor.cpu().numpy())
        predictions = pd.DataFrame(preds)
        test_df: pd.DataFrame = trainer.datamodule.df.merge(
            pd.DataFrame(predictions), on="image_id", how="outer", validate="1:1"
        )
        test_df["prediction_id"] = test_df["patient_id"].astype(str) + "_" + test_df["laterality"]
        test_df["cancer"].fillna(trainer.datamodule.cancer_base_rate, inplace=True)
        test_df = test_df.groupby("prediction_id", as_index=False)["cancer"].mean()
        test_df.to_csv(self.output_filepath, index=False)


@hydra.main(config_path="../config", config_name="submit", version_base=None)
def submit(cfg: DictConfig) -> None:
    model: pl.LightningModule = instantiate(cfg.model)
    datamodule: pl.LightningDataModule = instantiate(cfg.datamodule)
    trainer: pl.Trainer = instantiate(cfg.trainer)
    trainer.predict(model=model, datamodule=datamodule, return_predictions=False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    submit()
