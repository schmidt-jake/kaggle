import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Union

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
            batch: Dict[str, Union[torch.Tensor, List[str]]]
            for key, val in batch.items():
                if isinstance(val, torch.Tensor):
                    val = val.cpu().numpy()
                preds[key].extend(val)
        predictions = pd.DataFrame(preds)
        predictions["prediction_id"] = predictions["patient_id"].astype(str) + "_" + predictions["laterality"]
        predictions["cancer"].fillna(trainer.datamodule.cancer_base_rate, inplace=True)
        predictions = predictions.groupby("prediction_id", as_index=False)["cancer"].mean()
        predictions.to_csv(self.output_filepath, index=False)


@hydra.main(config_path="../config", config_name="submit", version_base=None)
def submit(cfg: DictConfig) -> None:
    instantiate(cfg.seed_fn)
    model: pl.LightningModule = instantiate(cfg.model)
    datamodule: pl.LightningDataModule = instantiate(cfg.datamodule)
    trainer: pl.Trainer = instantiate(cfg.trainer)
    trainer.predict(model=model, datamodule=datamodule, return_predictions=False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    submit()
