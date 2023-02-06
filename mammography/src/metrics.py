import torch
from torchmetrics import Metric


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
