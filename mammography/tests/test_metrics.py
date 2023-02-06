import torch

from mammography.src.metrics import ProbabilisticBinaryF1Score


def test_pf1_metric() -> None:
    metric = ProbabilisticBinaryF1Score()
    for _ in range(5):
        metric.update(
            preds=torch.testing.make_tensor(8, 1, device="cpu", dtype=torch.float32, low=0.0, high=1.0),
            target=torch.testing.make_tensor(8, 1, device="cpu", dtype=torch.int64, low=0, high=2),
        )
    result = metric.compute().item()
    assert result >= 0.0 and result <= 1.0

    metric.reset()
    y_true = torch.testing.make_tensor(128, dtype=torch.int64, device="cpu", low=0, high=2)
    metric(preds=y_true, target=y_true)
    val = metric.compute().item()
    assert val == 1.0

    metric.reset()
    y_true = torch.testing.make_tensor(128, dtype=torch.int64, device="cpu", low=0, high=1)
    metric(preds=y_true, target=y_true)
    val = metric.compute().item()
    assert val == 0.0, f"{val} != 0.0"
