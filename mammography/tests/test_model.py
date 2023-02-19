import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from pytest import MonkeyPatch
from tqdm import tqdm

from mammography.src.data.metadata import get_breast_metadata
from mammography.src.submit import submit
from mammography.src.train import train


def data_patch(index: int) -> Dict[str, Any]:
    return torch.rand(size=(1, 512, 512), dtype=torch.uint8)


def test_model_train(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr("torch.multiprocessing.cpu_count", lambda: 0)
    monkeypatch.setattr(
        "cv2.imread",
        lambda filepath, params=None: np.random.randint(
            low=0,
            high=255,
            size=np.random.randint(low=128, high=2046, size=2),
            dtype=np.uint8,
        ),
    )
    monkeypatch.setattr(
        "mammography.src.data.dicom.process_dicom",
        lambda filepath, raw=False: np.random.randint(
            low=0,
            high=255,
            size=np.random.randint(low=128, high=2046, size=2),
            dtype=np.uint8,
        ),
    )
    with initialize(version_base=None, config_path="../config"):
        train_cfg = compose(
            config_name="train",
            overrides=[
                "+trainer.limit_train_batches=1",
                "+trainer.limit_val_batches=1",
                "trainer.max_epochs=2",
                "datamodule.image_dir=mammography/data/resized2",
                "datamodule.train_batch_size=2",
                "datamodule.inference_batch_size=2",
                "datamodule.prefetch_factor=2",
                "+trainer.detect_anomaly=true",
                "trainer.benchmark=false",
                "+trainer.logger.mode=offline",
                "+trainer.logger.id=pytest",
                f"trainer.logger.save_dir={tmp_path}",
                f"trainer.accelerator={'gpu' if torch.cuda.is_available() else 'cpu'}",
                f"trainer.precision={16 if torch.cuda.is_available() else 32}",
            ],
        )

        train(train_cfg)

        ckpt_path = tmp_path / train_cfg.trainer.logger.project / train_cfg.trainer.logger.id / "checkpoints"
        ckpt_path /= os.listdir(ckpt_path)[0]
        meta = pd.read_csv("mammography/data/raw/test.csv")
        test_meta_path = tmp_path / "meta.json"
        get_breast_metadata(meta).to_json(test_meta_path)
        submit_cfg = compose(
            config_name="submit",
            overrides=[
                "datamodule.image_dir=mammography/data/raw/test_images",
                f"datamodule.metadata_paths.predict={test_meta_path}",
                f"ckpt_path='{ckpt_path}'",
                "+trainer.limit_predict_batches=1",
                "datamodule.prefetch_factor=2",
                f"trainer.accelerator={'gpu' if torch.cuda.is_available() else 'cpu'}",
                f"trainer.precision={16 if torch.cuda.is_available() else 32}",
            ],
        )
        submit(submit_cfg)


def test_datamodule(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr("torch.multiprocessing.cpu_count", lambda: 0)
    monkeypatch.setattr(
        "cv2.imread", lambda filepath, params: np.random.randint(low=0, high=255, size=(512, 512), dtype=np.uint8)
    )
    with initialize(version_base=None, config_path="../config"):
        cfg = compose(
            config_name="train",
            overrides=[
                "datamodule.image_dir=mammography/data/raw/train_images",
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
