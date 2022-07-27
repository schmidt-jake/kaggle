from math import log
from multiprocessing import cpu_count

import hydra
from omegaconf import DictConfig
import pandas as pd
import torch
import torch.backends.cudnn
from torch.utils.data import DataLoader
from torchmetrics import AUROC
from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy
from torchmetrics.classification import CalibrationError
from torchvision import models

from mayo_clinic_strip_ai.dataset import NEG_CLS
from mayo_clinic_strip_ai.dataset import POS_CLS
from mayo_clinic_strip_ai.dataset import TifDataset
from mayo_clinic_strip_ai.metadata import load_metadata
from mayo_clinic_strip_ai.model import Classifier
from mayo_clinic_strip_ai.model import FeatureExtractor
from mayo_clinic_strip_ai.model import Loss
from mayo_clinic_strip_ai.model import Model
from mayo_clinic_strip_ai.model import Normalizer


def get_pos_weight(meta: pd.DataFrame) -> float:
    cls_weights = len(meta) / (meta["label"].nunique() * meta["label"].value_counts())
    return cls_weights[POS_CLS] / cls_weights[NEG_CLS]


def get_bias(meta: pd.DataFrame) -> float:
    p = meta["label"].eq(POS_CLS).mean()
    return log(p / (1 - p))


@hydra.main(config_path=".", config_name="train_config", version_base=None)
def train(cfg: DictConfig) -> None:
    torch.backends.cudnn.benchmark = True

    train_meta = pd.merge(
        left=pd.read_csv(cfg.roi_path, dtype={"image_id": "string", "roi_num": "uint8"}),
        right=load_metadata(cfg.metadata_path),
        how="left",
        validate="m:1",
        on="image_id",
    )
    train_meta.query(
        f"h >= {cfg.hyperparameters.data.crop_size} and w >= {cfg.hyperparameters.data.crop_size}",
        inplace=True,
    )

    model = Model(
        normalizer=Normalizer(),
        feature_extractor=FeatureExtractor(backbone=getattr(models, cfg.hyperparameters.model.backbone)()),
        classifier=Classifier(initial_logit_bias=get_bias(train_meta), in_features=2208),  # FIXME: auto-set in_features
    )
    optimizer: torch.optim.Optimizer = getattr(torch.optim, cfg.hyperparameters.optimizer.cls)(
        model.parameters(),
        lr=cfg.hyperparameters.optimizer.lr,
        weight_decay=cfg.hyperparameters.optimizer.weight_decay,
    )
    loss_fn = Loss(pos_weight=get_pos_weight(train_meta))
    metrics = MetricCollection(
        {
            "raw_accuracy": Accuracy(),
            "calibration_error": CalibrationError(),
            "weighted_accuracy": Accuracy(threshold=train_meta["label"].eq(POS_CLS).mean()),
            "auroc": AUROC(),
        }
    )

    # move things to the right device
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    print("device:", device)
    model.to(device=device, memory_format=torch.channels_last, non_blocking=True)
    loss_fn.to(device=device, non_blocking=True)
    metrics.to(device=device, non_blocking=True)

    train_dataset = TifDataset(
        metadata=train_meta,
        training=True,
        data_dir=cfg.tif_dir,
    )
    num_workers = cpu_count()
    prefetch_batches = 4
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.hyperparameters.data.batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        pin_memory_device=str(device) if torch.cuda.is_available() else "",
        prefetch_factor=max(2, prefetch_batches * cfg.hyperparameters.data.batch_size // num_workers),
        num_workers=num_workers,
        drop_last=torch.backends.cudnn.benchmark,
    )

    grad_scaler = torch.cuda.amp.GradScaler()

    print("num workers:", num_workers)
    print("prefetch factor:", train_dataloader.prefetch_factor)
    for epoch in range(cfg.hyperparameters.data.epochs):
        print("Starting epoch", epoch)
        for img, label_id in train_dataloader:
            img = img.to(device=device, memory_format=torch.channels_last, non_blocking=True)
            label_id = label_id.to(device=device, non_blocking=True)
            with torch.autocast(device_type=img.device.type):
                optimizer.zero_grad(set_to_none=True)
                logit: torch.Tensor = model(img)
                loss: torch.Tensor = loss_fn(logit=logit, label=label_id)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()
                metrics.update(preds=logit.sigmoid(), target=label_id)
                m = {k: v.item() for k, v in metrics.compute().items()}
                m["loss"] = loss.item()
                metrics.reset()
            print(m)


if __name__ == "__main__":
    train()
