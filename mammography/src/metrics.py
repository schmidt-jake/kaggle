from typing import Dict, Set, Union

import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.trainer.states import RunningStage
from torchmetrics import MeanMetric, Metric, MetricCollection
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC


class ProbabilisticBinaryF1Score(Metric):
    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    def __init__(self, **kwargs) -> None:
        # https://torchmetrics.readthedocs.io/en/stable/pages/implement.html
        # https://www.kaggle.com/code/sohier/probabilistic-f-score/notebook
        super().__init__(**kwargs)
        self.add_state("y_true_count", default=torch.tensor(0, dtype=torch.int64), dist_reduce_fx="sum")
        self.add_state("ctp", default=torch.tensor(0.0, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("cfp", default=torch.tensor(0.0, dtype=torch.float32), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        if not torch.all((0 <= preds) * (preds <= 1)):
            # preds is logits, convert with sigmoid
            preds = preds.sigmoid()
        is_y_true = target == 1
        self.y_true_count += is_y_true.sum()  # type: ignore[operator]
        self.ctp += preds[is_y_true].sum()
        self.cfp += preds[is_y_true.logical_not()].sum()

    def compute(self) -> torch.Tensor:
        c_precision: torch.Tensor = self.ctp / (self.ctp + self.cfp)  # type: ignore[operator]
        c_recall: torch.Tensor = self.ctp / self.y_true_count  # type: ignore[operator]
        result = 2 * (c_precision * c_recall) / (c_precision + c_recall)
        result.nan_to_num_(nan=0.0)
        return result


def set_metrics(pl_module: LightningModule) -> None:
    # A hack; required because pytorch-lightning requires that all metrics must be
    # direct attributes of the LightningModule, not part of a data structure besides `MetricContainer``
    for stage in {"train", "val"}:
        setattr(pl_module, f"loss_sum_{stage}", MeanMetric(nan_strategy="error"))
        for loss in {"cancer", "density"}:
            setattr(pl_module, f"loss_{loss}_{stage}", MeanMetric(nan_strategy="error"))
            acc_metrics = {"accuracy": BinaryAccuracy(validate_args=False), "auroc": BinaryAUROC(validate_args=False)}
            if loss == "cancer":
                acc_metrics["pf1"] = ProbabilisticBinaryF1Score()
            setattr(
                pl_module,
                f"metrics_{loss}_{stage}",
                MetricCollection(acc_metrics, prefix=loss + "/", postfix="/" + stage),
            )


def _get_stage(pl_module: LightningModule) -> str:
    if pl_module.trainer.state.stage == RunningStage.TRAINING:
        return "train"
    elif pl_module.trainer.state.stage == RunningStage.VALIDATING:
        return "val"


def _log(pl_module: LightningModule, attr: str, **metric_args) -> None:
    metric: Union[Metric, MetricCollection] = getattr(pl_module, attr)
    metric(**metric_args)
    if isinstance(metric, Metric):
        pl_module.log(name=attr.replace("_", "/"), value=metric)
    elif isinstance(metric, MetricCollection):
        pl_module.log_dict(metric)


def log_metrics(
    pl_module: LightningModule,
    losses: Dict[str, torch.Tensor],
    preds: Dict[str, torch.Tensor],
    target: Dict[str, torch.Tensor],
) -> None:
    stage = _get_stage(pl_module)
    for k, v in losses.items():
        _log(pl_module, f"loss_{k}_{stage}", value=v)
    for k, v in preds.items():
        _log(pl_module, f"metrics_{k}_{stage}", preds=preds[k], target=target[k])
