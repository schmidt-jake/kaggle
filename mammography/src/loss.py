from typing import Tuple

import torch


class SigmoidFocalLoss(torch.nn.modules.loss._Loss):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.bce_loss = torch.nn.BCEWithLogitsLoss(**kwargs)
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bce_loss: torch.Tensor = self.bce_loss(input=input, target=target)
        p = input.sigmoid()
        p_t = p * target + (1 - p) * (1 - target)
        loss = bce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
            loss = alpha_t * loss

        # loss = getattr(loss, self.reduction)()
        loss = loss.mean()
        return loss, p
