from typing import Dict, List

import torch
from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy


class Metrics(torch.nn.Module):
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

    @torch.inference_mode()
    @torch.cuda.amp.autocast()
    def update(self, logit: torch.Tensor, target: torch.Tensor) -> None:
        preds = logit.detach().sigmoid()
        self.metrics.update(preds=preds.squeeze(dim=1), target=target.detach())

    @torch.inference_mode()
    @torch.cuda.amp.autocast()
    def compute(self) -> Dict[str, float]:
        metrics: Dict[str, torch.Tensor] = self.metrics.compute()
        return {k: v.item() for k, v in metrics.items()}

    @torch.inference_mode()
    @torch.cuda.amp.autocast()
    def reset(self) -> None:
        self.metrics.reset()


def get_grad_norms(model: torch.nn.Module) -> List[torch.Tensor]:
    return [p.grad.detach().data.norm(2) for p in model.parameters() if p.grad is not None and p.requires_grad]
