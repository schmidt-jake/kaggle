from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytorch_lightning as pl
import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from pytest import MonkeyPatch

from mammography.src.train import ProbabilisticBinaryF1Score, train


def data_patch(filepath: str) -> npt.NDArray[np.uint16]:
    return np.random.randint(size=(1, 512, 512), low=0, high=2**16 - 1, dtype=np.uint16)


def test_model_train(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    # monkeypatch.setattr("mammography.src.train.dicom2numpy", data_patch)
    with initialize(version_base=None, config_path="../config"):
        cfg = compose(
            config_name="train",
            overrides=[
                "+trainer.limit_train_batches=1",
                "+trainer.limit_val_batches=1",
                "trainer.max_epochs=1",
                "~trainer.precision",
                f"trainer.default_root_dir={tmp_path}",
                "datamodule.root_dir=mammography/data/raw",
                "trainer.logger=null",
            ],
        )
        train(cfg)


def test_datamodule(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr("mammography.src.train.dicom2tensor", data_patch)
    with initialize(version_base=None, config_path="../config"):
        cfg = compose(
            config_name="train",
            overrides=["datamodule.root_dir=mammography/data/raw"],
        )
        datamodule: pl.LightningDataModule = instantiate(cfg.datamodule)
        datamodule.setup(stage="fit")
        dataloader = datamodule.train_dataloader()
        for batch in dataloader:
            assert "pixels" in batch.keys()
            assert "cancer" in batch.keys()
            assert batch["pixels"].shape == (8, 1, 512, 512)
            break


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
