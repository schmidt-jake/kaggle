import logging
from typing import Any, Dict, Set

import cv2
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig
from pydicom.pixel_data_handlers.util import apply_windowing
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.transforms import functional_tensor

from mammography.src import utils

# from torchdata.dataloader2 import DataLoader2, PrototypeMultiProcessingReadingService
# from torchdata.datapipes.map import MapDataPipe


logger = logging.getLogger(__name__)


class DataframeDataPipe(Dataset):
    def __init__(self, df: pd.DataFrame, augmentation: torch.nn.Sequential, keys: Set) -> None:
        super().__init__()
        self.df = df
        self.augmentation = augmentation
        self.keys = keys

    def __len__(self) -> int:
        return len(self.df)

    @staticmethod
    def _read(filepath: str) -> npt.NDArray[np.uint8]:
        if filepath.endswith(".dcm"):
            arr, dcm = utils.dicom2numpy(filepath, should_apply_window_fn=False)
            windows = utils.get_unique_windows(dcm)
            if not windows.empty:
                window_index = np.random.choice(windows.index)
                window = windows.loc[window_index]
                pixel_bit_depth = utils.get_suspected_bit_depth(pixel_value=int(arr.max()))
                window_bit_depth = utils.get_suspected_bit_depth(
                    pixel_value=int(window["center"] + window["width"] // 2)
                )
                if pixel_bit_depth != window_bit_depth:
                    if window_bit_depth != dcm.BitsStored:
                        arr = utils.to_bit_depth(arr, src_depth=pixel_bit_depth, dest_depth=window_bit_depth)
                    else:
                        logger.warning(
                            f"Couldn't fix {filepath}"
                            f"\ncenter={window['center']}, width={window['width']}"
                            f"pixel_max={arr.max()}, bit_depth={pixel_bit_depth}"
                        )
                windowed_arr = apply_windowing(arr=arr.copy(), ds=dcm, index=np.random.choice(windows.index))
                if windowed_arr.ptp() > 10:
                    arr = windowed_arr
                else:
                    logger.warning(f"Got bad window for {filepath}, using raw pixels instead.")
            else:
                logger.info(f"Now window found for {filepath}, using raw pixels...")
            # dcm.BitsStored = utils.get_suspected_bit_depth(arr)
            arr = utils.maybe_invert(arr=arr, dcm=dcm)
            arr = utils.maybe_flip_left(arr=arr)
            # arr = utils.scale_to_01(arr=arr, dcm=dcm)
        elif filepath.endswith(".png"):
            arr = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        else:
            raise ValueError(f"Got unknown file suffix in filepath: {filepath}")
        return arr

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.df.iloc[index]
        logger.debug(f"Loading image {row['image_id']}")
        d = row.to_dict()
        arr = self._read(filepath=row["filepath"])
        pixels = torch.from_numpy(arr.astype(np.float32))
        _min, _max = pixels.min(), pixels.max()
        pixels -= _min
        pixels /= _max - _min
        pixels.unsqueeze_(dim=0)
        pixels = utils.crop_right_center(pixels, size=2048)
        pixels = functional_tensor.resize(pixels, size=512)
        # d["pixels"] = self.augmentation(pixels)
        d["pixels"] = pixels
        d = {k: v for k, v in d.items() if k in self.keys}
        return d


class DataModule(LightningDataModule):
    def __init__(
        self,
        metadata_filepath: str,
        image_dir: str,
        augmentation: DictConfig,
        batch_size: int,
        prefetch_batches: int,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["metadata_filepath", "image_dir"])
        self.metadata_filepath = metadata_filepath
        self.image_dir = image_dir
        self.augmentation: torch.nn.Sequential = instantiate(self.hparams.augmentation)
        self.num_workers = torch.multiprocessing.cpu_count()
        self.prefetch = max(self.hparams.prefetch_batches * self.hparams.batch_size // max(self.num_workers, 1), 2)

    @staticmethod
    def compute_class_weights(y: pd.Series) -> pd.Series:
        return len(y) / (y.nunique() * y.value_counts())

    def _use_artifact(self) -> None:
        if self.trainer.logger.experiment.mode == "online":
            artifact = wandb.Artifact(name="input", type="dataset")
            artifact.add_reference(uri=f"file://{self.metadata_filepath}", name="metadata.csv")
            artifact.add_reference(uri=f"file://{self.image_dir}", name="image_dir", checksum=False, max_objects=1e9)
            self.trainer.logger.use_artifact(artifact, artifact_type="dataset").save()
        else:
            logger.warning(f"Unable to use artifact when in {self.trainer.logger.experiment.mode} mode")

    def setup(self, stage: str) -> None:
        self.df = pd.read_csv(self.metadata_filepath)
        self.df["filepath"] = (
            self.image_dir + "/" + self.df["patient_id"].astype(str) + "/" + self.df["image_id"].astype(str) + ".dcm"
        )
        if stage == "fit":
            self.df.query("patient_id != 27770", inplace=True)
            self.df.query("image_id != 1942326353", inplace=True)
            class_weights = 1.0 / self.df["cancer"].value_counts()
            self.df["sample_weight"] = self.df["cancer"].map(class_weights.get)
            self.cancer_base_rate = self.df["cancer"].mean()

        self._use_artifact()

    def state_dict(self) -> Dict[str, Any]:
        return {"cancer_base_rate": self.cancer_base_rate}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.cancer_base_rate = state_dict["cancer_base_rate"]

    def train_dataloader(self) -> DataLoader:
        pipe = DataframeDataPipe(
            self.df, augmentation=self.augmentation.train(), keys={"pixels", "cancer"}
        )  # .to_iter_datapipe()
        return DataLoader(
            dataset=pipe,
            batch_size=self.hparams.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch,
            sampler=WeightedRandomSampler(
                weights=self.df["sample_weight"],
                num_samples=len(self.df),
                replacement=True,
            ),
            drop_last=True,  # True to avoid retriggering CuDNN benchmarking
        )
        # pipe = pipe.map(dicom2tensor, input_col="filepath", output_col="pixels")
        # pipe = pipe.in_memory_cache()
        # pipe = pipe.shuffle(buffer_size=1)
        # self.augmentation.train()
        # pipe = pipe.map(self.augmentation, input_col="pixels", output_col="pixels")
        # pipe = pipe.map(partial(select_dict_keys, keys=["pixels", "cancer"]))
        # pipe = pipe.batch(8)
        # pipe = pipe.collate()
        # pipe = pipe.prefetch(buffer_size=1)
        # return DataLoader2(datapipe=pipe, reading_service=PrototypeMultiProcessingReadingService(num_workers=0))

    def val_dataloader(self) -> DataLoader:
        pipe = DataframeDataPipe(
            self.df, augmentation=self.augmentation.eval(), keys={"pixels", "cancer", "image_id"}
        )  # .to_iter_datapipe()
        return DataLoader(
            dataset=pipe,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch,
            drop_last=False,
        )

    def predict_dataloader(self) -> DataLoader:
        return self.val_dataloader()
