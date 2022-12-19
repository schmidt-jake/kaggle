from pathlib import Path

import pytorch_lightning as pl
import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from pytest import MonkeyPatch

from mammography.kernels.submit.submit import submit
from mammography.src.train import train


def data_patch(filepath: str) -> torch.Tensor:
    return torch.testing.make_tensor(shape=(1, 512, 512), dtype=torch.int16, low=0, device=torch.device("cpu"))


def test_model_train(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr("mammography.src.train.dicom2tensor", data_patch)
    with initialize(version_base=None, config_path="../config"):
        cfg = compose(
            config_name="train",
            overrides=[
                "+trainer.limit_train_batches=1",
                "+trainer.limit_val_batches=1",
                "trainer.max_epochs=1",
                "~trainer.precision",
                # f"trainer.default_root_dir={tmp_path}",
                "trainer.default_root_dir=mammography",
                "datamodule.root_dir=mammography/data",
            ],
        )
        train(cfg)
        submit(cfg)


def test_datamodule(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr("mammography.src.train.dicom2tensor", data_patch)
    with initialize(version_base=None, config_path="../config"):
        cfg = compose(
            config_name="train",
            overrides=["datamodule.root_dir=mammography/data"],
        )
        datamodule: pl.LightningDataModule = instantiate(cfg.datamodule)
        datamodule.setup(stage="fit")
        dataloader = datamodule.train_dataloader()
        for batch in dataloader:
            assert "pixels" in batch.keys()
            assert "cancer" in batch.keys()
            assert batch["pixels"].shape == (8, 1, 512, 512)
            break
