import torch

from splicepath.seams.splice import clipped_mse


def test_clipped_mse_zero_when_equal() -> None:
    x = torch.rand(4, 5)
    assert torch.isclose(clipped_mse(x, x), torch.tensor(0.0), atol=1e-7)


def test_clipped_mse_clamps_target() -> None:
    pred = torch.zeros(3)
    target = torch.tensor([2.0, 2.0, 2.0])
    assert torch.isclose(clipped_mse(pred, target), torch.tensor(1.0))
