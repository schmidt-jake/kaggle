from logging import getLogger
from math import log
from multiprocessing import cpu_count
import os
import random

from functorch.compile import memory_efficient_fusion
import hydra
from hydra.utils import call
from hydra.utils import instantiate
import numpy as np
from omegaconf import DictConfig
import pandas as pd
import torch
import torch.backends.cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from mayo_clinic_strip_ai.data import NEG_CLS
from mayo_clinic_strip_ai.data import POS_CLS
from mayo_clinic_strip_ai.data import ROIDataset
from mayo_clinic_strip_ai.data import StratifiedBatchSampler
from mayo_clinic_strip_ai.metrics import Metrics
from mayo_clinic_strip_ai.model import Classifier
from mayo_clinic_strip_ai.model import FeatureExtractor
from mayo_clinic_strip_ai.model import Loss
from mayo_clinic_strip_ai.model import Model
from mayo_clinic_strip_ai.model import Normalizer

logger = getLogger(__name__)


def get_pos_weight(y: pd.Series) -> float:
    """
    Computes the weight of the positive class, such that the negative class weight is 1.0
    and the weights are balanced according to their frequency.

    Parameters
    ----------
    y : pd.Series
        The vector of binary labels for the training data

    Returns
    -------
    float
        The positive class weight
    """
    cls_weights = len(y) / (y.nunique() * y.value_counts())
    return cls_weights[POS_CLS] / cls_weights[NEG_CLS]


def get_bias(y: pd.Series) -> float:
    """
    Gets the value of the input to the sigmoid function such that
    it outputs the probability of the postive class.

    Parameters
    ----------
    y : pd.Series
        The vector of binary labels for the training data

    Returns
    -------
    float
        The bias value.
    """
    p = y.eq(POS_CLS).mean()
    return log(p / (1 - p))


@hydra.main(config_path="config", config_name="train", version_base=None)
def train(cfg: DictConfig) -> None:
    """
    The entrypoint into the training job.

    Parameters
    ----------
    cfg : DictConfig
        The config object containing e.g. hyperparameter settings.
    """
    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)
    # Let the pytorch backend seelct the fastest convolutional kernel
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"Using device: {device}")

    # Load the training metadata, including image labels and ROI coordinates
    train_meta = pd.merge(
        left=pd.read_csv(cfg.roi_path),
        right=pd.read_csv(cfg.metadata_path),
        how="inner",
        validate="m:1",
        on="image_id",
    ).astype(
        {
            "image_id": "string",
            "center_id": "category",
            "patient_id": "category",
            "image_num": "uint8",
            "label": "category",
            "roi_num": "uint8",
        }
    )

    # Filter out ROIs that are too small
    train_meta.query(
        f"h >= {cfg.hparams.data.crop_size} and w >= {cfg.hparams.data.crop_size}",
        inplace=True,
    )

    # filter ROIs with extreme aspect ratios
    train_meta.query("h / w > 1 / 12 and h / w < 12", inplace=True)

    # filter blurry ROIs
    train_meta.query("blur > 1.0", inplace=True)

    # Create the model
    backbone = instantiate(cfg.hparams.model.backbone)
    model = Model(
        normalizer=Normalizer(),
        feature_extractor=FeatureExtractor(backbone=backbone),
        classifier=Classifier(
            initial_logit_bias=get_bias(train_meta["label"]),
            in_features=backbone.classifier.in_features,
        ),
    )
    model: Model = memory_efficient_fusion(model)  # type: ignore[no-redef]

    # https://hydra.cc/docs/advanced/instantiate_objects/overview/
    optimizer: torch.optim.Optimizer = instantiate(cfg.hparams.optimizer, params=model.parameters())

    # Create the loss function
    loss_fn = Loss(pos_weight=get_pos_weight(train_meta["label"]))

    # Create metrics
    writer = SummaryWriter()
    # TODO log hyperparameters to tensorboard run

    # move things to the right device

    model = model.to(device=device, memory_format=torch.channels_last, non_blocking=True)  # type: ignore[call-overload]
    loss_fn = loss_fn.to(device=device, non_blocking=True)
    train_metrics = Metrics(acc_thresh=train_meta["label"].eq(POS_CLS).mean()).to(device=device, non_blocking=True)

    # Create dataset and dataloader
    train_dataset = ROIDataset(
        metadata=train_meta,
        training=True,
        tif_dir=cfg.tif_dir,
        outline_dir=os.path.dirname(cfg.roi_path),
        crop_size=cfg.hparams.data.crop_size,
        final_size=cfg.hparams.data.final_size,
        min_intersect_pct=cfg.hparams.data.min_intersect_pct,
    )
    num_workers = cpu_count()  # use all CPUs
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_sampler=StratifiedBatchSampler(  # type: ignore[arg-type]
            levels=train_meta[cfg.hparams.data.stratification_levels],
            batch_size=cfg.hparams.data.batch_size,
            seed=cfg.random_seed,
        ),
        pin_memory=torch.cuda.is_available(),
        pin_memory_device=str(device) if torch.cuda.is_available() else "",
        prefetch_factor=max(2, cfg.prefetch_batches * cfg.hparams.data.batch_size // num_workers),
        num_workers=num_workers,
        persistent_workers=True,
    )

    # Gradient scaler is used for automatic mixed precision training
    # Docs: https://pytorch.org/docs/stable/notes/amp_examples.html
    grad_scaler = torch.cuda.amp.GradScaler()

    lr_scheduler: torch.optim.lr_scheduler._LRScheduler = instantiate(
        cfg.hparams.lr_scheduler,
        optimizer=optimizer,
        verbose=True,
    )

    logger.info(f"num workers: {num_workers}")
    logger.info(f"prefetch samples per worker: {train_dataloader.prefetch_factor}")

    # begin the training loop
    for epoch in range(cfg.hparams.data.epochs):
        logger.info(f"Starting epoch {epoch}...")
        model.train()
        for global_step, (img, label_id) in enumerate(train_dataloader):
            img = img.to(device=device, memory_format=torch.channels_last, non_blocking=True)
            label_id = label_id.to(device=device, non_blocking=True)
            with torch.autocast(device_type=img.device.type):
                logit: torch.Tensor = model(img)
                loss: torch.Tensor = loss_fn(logit=logit, label=label_id)

            writer.add_histogram(
                tag="logit",
                values=logit.detach(),
                global_step=global_step,
                bins="auto",
            )

            train_metrics.update(logit=logit, target=label_id)
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)

            call(
                cfg.hparams.backward,
                parameters=model.parameters(),
                error_if_nonfinite=False,
            )
            grad_scaler.step(optimizer)
            grad_scaler.update()
            optimizer.zero_grad(set_to_none=True)
            m = train_metrics.compute()
            train_metrics.reset()
            m["loss"] = loss.item()

            writer.add_scalars(main_tag="train", tag_scalar_dict=m, global_step=global_step)

        lr_scheduler.step()

    logger.info("Saving model...")
    torch.jit.script(model).save("model.torchscript")


if __name__ == "__main__":
    train()
