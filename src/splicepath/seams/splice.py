from torch import Tensor
from torch.nn import functional as F


def clipped_mse(prediction: Tensor, target: Tensor) -> Tensor:
    bounded = target.clamp(0.0, 1.0)
    return F.mse_loss(prediction, bounded)
