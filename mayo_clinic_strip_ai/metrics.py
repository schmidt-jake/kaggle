from typing import Dict

import torch
from torchmetrics import MaxMetric
from torchmetrics import MetricCollection
from torchmetrics import MinMetric
from torchmetrics.classification import Accuracy


class EvalMetrics(torch.nn.Module):
    def __init__(self, acc_thresh: float) -> None:
        super().__init__()
        self.metrics = MetricCollection(
            {
                "raw_accuracy": Accuracy(),
                # "calibration_error": CalibrationError(),
                "weighted_accuracy": Accuracy(threshold=acc_thresh),
                # "auroc": AUROC(),
            }
        )
        self.min_pred = MinMetric()
        self.max_pred = MaxMetric()

    @torch.inference_mode()
    @torch.cuda.amp.autocast()
    def update(self, logit: torch.Tensor, target: torch.Tensor) -> None:
        preds = logit.detach().sigmoid()
        self.metrics.update(preds=preds, target=target.detach())
        self.min_pred.update(value=preds)
        self.max_pred.update(value=preds)

    @torch.inference_mode()
    @torch.cuda.amp.autocast()
    def compute(self) -> Dict[str, float]:
        metrics = self.metrics.compute()
        metrics.update(min_pred=self.min_pred.compute(), max_pred=self.max_pred.compute())
        return {k: v.item() for k, v in metrics.items()}

    @torch.inference_mode()
    @torch.cuda.amp.autocast()
    def reset(self) -> None:
        self.metrics.reset()
        self.min_pred.reset()
        self.max_pred.reset()


class TrainMetrics(EvalMetrics):
    def __init__(self, acc_thresh: float) -> None:
        super().__init__(acc_thresh)
        self.max_grad_norm = MaxMetric()

    def update_max_grad_norm(self, model: torch.nn.Module) -> None:
        for p in model.parameters():
            if p.grad is not None and p.requires_grad:
                self.max_grad_norm.update(p.grad.detach().data.norm(2))

    def compute(self) -> Dict[str, float]:
        metrics = super().compute()
        metrics.update(max_grad_norm=self.max_grad_norm.compute().item())
        return metrics

    def reset(self) -> None:
        super().reset()
        self.max_grad_norm.reset()
