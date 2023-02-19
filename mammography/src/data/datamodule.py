import logging
import os
from functools import partial
from operator import itemgetter
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
import torch
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, WeightedRandomSampler

from mammography.src.data import utils
from mammography.src.data.dataset import DataframeDataPipe
from mammography.src.data.transforms import CropCenterRight, ResizeLookup

logger = logging.getLogger(__name__)


class DataModule(LightningDataModule):
    def __init__(
        self,
        metadata_paths: Dict[str, str],
        image_dir: str,
        augmentation: DictConfig,
        train_batch_size: int,
        inference_batch_size: int,
        prefetch_factor: int,
        resizer: DictConfig,
        cropper: DictConfig,
    ) -> None:
        super().__init__()
        self.save_hyperparameters("augmentation", "train_batch_size", "inference_batch_size", "resizer", "cropper")
        self.image_dir = image_dir
        self.metadata_paths = metadata_paths
        self.augmentation: torch.nn.Sequential = instantiate(self.hparams_initial["augmentation"])
        self.num_workers = torch.multiprocessing.cpu_count()
        logger.info(f"Detected {self.num_workers} cores.")
        self.prefetch_factor = prefetch_factor
        self.resizer: ResizeLookup = instantiate(self.hparams_initial["resizer"])
        self.cropper: CropCenterRight = instantiate(self.hparams_initial["cropper"])

    @staticmethod
    def compute_class_weights(y: pd.Series) -> pd.Series:
        return len(y) / (y.nunique() * y.value_counts())

    def _use_artifact(self) -> None:
        if not (self.trainer.logger.experiment.offline or self.trainer.logger.experiment.disabled):
            logger.info("Saving input artifact reference...")
            artifact = wandb.Artifact(name="input", type="dataset")
            for k, v in self.metadata_paths.items():
                artifact.add_reference(uri=f"file://{v}", name=k)
            # artifact.add_reference(
            #     uri=f"file://{self.image_dir}", name="image_dir", max_objects=len(self.df), checksum=False
            # )
            self.trainer.logger.use_artifact(artifact, artifact_type="dataset").save()
        else:
            logger.warning(f"Unable to use artifact when in {self.trainer.logger.experiment.mode} mode")

    def setup(self, stage: str) -> None:
        self.meta = {k: pd.read_json(v) for k, v in self.metadata_paths.items()}
        if stage == "fit":
            # class_weights = 1.0 / self.df["cancer"].value_counts(normalize=True)
            class_weights = {0: 1.0, 1: 3.0}
            self.meta["train"]["sample_weight"] = self.meta["train"]["cancer"].map(class_weights.get)
            self._use_artifact()

    def train_dataloader(self) -> DataLoader:
        pipe = DataframeDataPipe(df=self.meta["train"], fns=self.train_val_fns())
        dataloader = DataLoader(
            dataset=pipe,
            batch_size=self.hparams["train_batch_size"],
            pin_memory=True,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            sampler=WeightedRandomSampler(
                weights=self.meta["train"]["sample_weight"],
                num_samples=len(self.meta["train"]),
                replacement=True,
            ),
            drop_last=False,
            persistent_workers=self.num_workers > 0,
            collate_fn=utils.collate_fn,
        )
        return dataloader

    def val_dataloader(self) -> DataLoader:
        pipe = DataframeDataPipe(df=self.meta["val"], fns=self.train_val_fns())
        dataloader = DataLoader(
            dataset=pipe,
            batch_size=self.hparams["inference_batch_size"],
            pin_memory=True,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            drop_last=False,
            persistent_workers=self.num_workers > 0,
            shuffle=False,
            collate_fn=utils.collate_fn,
        )
        return dataloader

    def predict_dataloader(self) -> DataLoader:
        from mammography.src.data.dicom import process_dicom

        fns = []
        for view in ["CC", "MLO"]:
            fns.extend(
                [
                    (np.random.choice, view, view),
                    (
                        partial(
                            utils.get_filepath,
                            template=os.path.join(self.image_dir, f"{{patient_id}}/{{{view}}}.dcm"),
                        ),
                        None,
                        view,
                    ),
                    (process_dicom, view, view),
                    # (itemgetter(0), view, view),
                    (self.augmentation, view, view),
                    (partial(self.resizer.resize, key=view), None, view),
                    (self.cropper, view, view),
                ]
            )
        fns.append((partial(utils.select_keys, keys={"cancer", "CC", "MLO", "prediction_id", "age"}), None, None))
        pipe = DataframeDataPipe(df=self.meta["predict"], fns=fns)
        dataloader = DataLoader(
            dataset=pipe,
            batch_size=self.hparams["inference_batch_size"],
            pin_memory=True,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            drop_last=False,
            persistent_workers=self.num_workers > 0,
            shuffle=False,
            collate_fn=utils.collate_fn,
        )
        return dataloader

    def train_val_fns(self) -> List[Callable]:
        fns = []

        for view in ["CC", "MLO"]:
            fns.extend(
                [
                    (np.random.choice, view, view),
                    (partial(utils.get_filepath, template=os.path.join(self.image_dir, f"{{{view}}}.png")), None, view),
                    (utils.read_png, view, view),
                    (self.augmentation, view, view),
                    (partial(self.resizer.resize, key=view), None, view),
                    (self.cropper, view, view),
                ]
            )
        fns.extend(
            [
                # map_fn({"A": 0, "B": 0, "C": 1, "D": 1}.get, input_key="density", output_key="density"),
                # partial(utils.select_keys, keys={"cancer", "CC", "MLO", "age", "density"}),
                (partial(utils.select_keys, keys={"cancer", "CC", "MLO"}), None, None)
            ]
        )
        return fns
