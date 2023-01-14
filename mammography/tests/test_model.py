from pathlib import Path
from typing import Any, Dict

import pandas as pd
import pytorch_lightning as pl
import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from pytest import MonkeyPatch
from tqdm import tqdm

from mammography.src.metrics import ProbabilisticBinaryF1Score
from mammography.src.submit import submit
from mammography.src.train import train


def data_patch(index: int) -> Dict[str, Any]:
    # return {"pixels": torch.randint(size=(1, 512, 512), low=0, high=255, dtype=torch.uint8), "cancer": 0}
    return {"pixels": torch.rand(size=(1, 512, 512), dtype=torch.float32), "cancer": 0}


def test_model_train(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr("torch.multiprocessing.cpu_count", lambda: 0)
    # monkeypatch.setattr(
    #     "pandas.read_csv",
    #     lambda filepath: pd.DataFrame([{"image_id": 0, "cancer": 0, "patient_id": 0}] * 2),
    # )
    # monkeypatch.setattr("mammography.src.train.DataframeDataPipe.__getitem__", staticmethod(data_patch))
    with initialize(version_base=None, config_path="../config"):
        train_cfg = compose(
            config_name="train",
            overrides=[
                "+trainer.limit_train_batches=1",
                "+trainer.limit_val_batches=1",
                "trainer.max_epochs=1",
                f"trainer.default_root_dir={tmp_path}",
                "datamodule.image_dir=mammography/data/raw/train_images",
                "datamodule.metadata_filepath=mammography/data/raw/train.csv",
                "datamodule.batch_size=2",
                "datamodule.prefetch_batches=0",
                "+trainer.detect_anomaly=true",
                "trainer.benchmark=false",
                "+trainer.logger.mode=offline",
                "+trainer.logger.id=pytest",
                f"trainer.accelerator={'gpu' if torch.cuda.is_available() else 'cpu'}",
                f"trainer.precision={16 if torch.cuda.is_available() else 32}",
            ],
        )
        train(train_cfg)

        ckpt_path = tmp_path / "checkpoints" / "last.ckpt"
        submit_cfg = compose(
            config_name="submit",
            overrides=[
                f"+trainer.default_root_dir={train_cfg.trainer.default_root_dir}",
                "datamodule.image_dir=mammography/data/raw/test_images",
                "datamodule.metadata_filepath=mammography/data/raw/test.csv",
                f"+trainer.logger.id={train_cfg.trainer.logger.id}",
                f"datamodule.checkpoint_path={ckpt_path}",
                f"model.checkpoint_path={ckpt_path}",
            ],
        )
        submit(submit_cfg)


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
        for batch in tqdm(dataloader):
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


def test_predict(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr("torch.multiprocessing.cpu_count", lambda: 0)
    # monkeypatch.setattr(
    #     "pandas.read_csv",
    #     lambda filepath: pd.DataFrame([{"image_id": 0, "cancer": 0, "patient_id": 0}] * 2),
    # )
    # monkeypatch.setattr("mammography.src.train.DataframeDataPipe.__getitem__", staticmethod(data_patch))
    with initialize(version_base=None, config_path="../config"):
        cfg = compose(
            config_name="train",
            overrides=[
                "+trainer.limit_train_batches=1",
                "+trainer.limit_val_batches=1",
                "trainer.max_epochs=1",
                f"trainer.default_root_dir={tmp_path}",
                "datamodule.image_dir=mammography/data/raw/train_images",
                "datamodule.metadata_filepath=mammography/data/raw/train.csv",
                "datamodule.batch_size=2",
                "datamodule.prefetch_batches=0",
                "+trainer.detect_anomaly=true",
                "trainer.benchmark=false",
                "+trainer.logger.mode=disabled",
                f"trainer.accelerator={'gpu' if torch.cuda.is_available() else 'cpu'}",
                f"trainer.precision={16 if torch.cuda.is_available() else 32}",
                "+dev=submit",
                "ckpt_path=null",
            ],
        )
        submit(cfg)
