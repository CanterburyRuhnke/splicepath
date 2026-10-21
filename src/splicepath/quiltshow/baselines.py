import numpy as np
import torch
from torch import Tensor, nn

from splicepath.gauge.regression import mean_pcc


class LinearPSIHead(nn.Module):
    def __init__(self, patch_dim: int, num_events: int) -> None:
        super().__init__()
        self.linear = nn.Linear(patch_dim, num_events)

    def forward(self, patches: Tensor) -> Tensor:
        pooled = patches.mean(dim=1)
        return torch.sigmoid(self.linear(pooled))


def mean_predictor_pcc(train_psi: np.ndarray, test_psi: np.ndarray) -> float:
    means = np.asarray(train_psi, dtype=np.float64).mean(axis=0, keepdims=True)
    prediction = np.repeat(means, test_psi.shape[0], axis=0)
    return mean_pcc(prediction, test_psi)
