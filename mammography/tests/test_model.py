from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from pytest import MonkeyPatch
from torchmetrics import MetricCollection

from mammography.src.train import ProbabilisticBinaryF1Score, train


def data_patch(index: int) -> Dict[str, Any]:
    return {"pixels": torch.randint(size=(1, 512, 512), low=0, high=255, dtype=torch.uint8), "cancer": 0}


def test_model_train(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr("torch.multiprocessing.cpu_count", lambda: 0)
    monkeypatch.setattr(
        "pandas.read_csv",
        lambda filepath: pd.DataFrame([{"image_id": 0, "cancer": 0}] * 2),
    )
    monkeypatch.setattr("mammography.src.train.DataframeDataPipe.__getitem__", staticmethod(data_patch))
    with initialize(version_base=None, config_path="../config"):
        cfg = compose(
            config_name="train",
            overrides=[
                "+trainer.limit_train_batches=1",
                "+trainer.limit_val_batches=1",
                "trainer.max_epochs=1",
                f"trainer.default_root_dir={tmp_path}",
                "datamodule.image_dir=''",
                "datamodule.metadata_filepath=''",
                "datamodule.batch_size=2",
                "datamodule.prefetch_batches=0",
                "+trainer.detect_anomaly=true",
                "trainer.benchmark=false",
                "+trainer.logger.mode=disabled",
                f"trainer.accelerator={'gpu' if torch.cuda.is_available() else 'cpu'}",
                f"trainer.precision={16 if torch.cuda.is_available() else 32}",
            ],
        )
        train(cfg)


def test_datamodule(monkeypatch: MonkeyPatch) -> None:
    # monkeypatch.setattr("mammography.src.train.dicom2numpy", data_patch)
    monkeypatch.setattr("torch.multiprocessing.cpu_count", lambda: 0)
    with initialize(version_base=None, config_path="../config"):
        cfg = compose(
            config_name="train",
            overrides=[
                # "datamodule.image_dir=mammography/data/uint8_crops/png",
                "datamodule.image_dir=mammography/data/raw/train_images",
                "datamodule.metadata_filepath=mammography/data/raw/train.csv",
                "datamodule.prefetch_batches=0",
            ],
        )
        datamodule: pl.LightningDataModule = instantiate(cfg.datamodule)
        datamodule.setup(stage="fit")
        dataloader = datamodule.train_dataloader()
        for batch in dataloader:
            assert "pixels" in batch.keys()
            assert "cancer" in batch.keys()
            assert batch["pixels"].shape == (cfg.datamodule.batch_size, 1, 512, 512)
            assert batch["cancer"].float().mean().item() > 0.1
            # break


def test_pf1_metric() -> None:
    metric = ProbabilisticBinaryF1Score()
    for _ in range(5):
        metric.update(
            preds=torch.testing.make_tensor(
                shape=(8, 1),
                device="cpu",
                dtype=torch.float32,
                low=0.0,
                high=1.0,
            ),
            target=torch.testing.make_tensor(
                shape=(8, 1),
                device="cpu",
                dtype=torch.int64,
                low=0,
                high=1,
            ),
        )
    result = metric.compute()
    assert not result.isnan()
