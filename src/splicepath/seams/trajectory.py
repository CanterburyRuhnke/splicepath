import torch
from torch import Tensor

from splicepath.sashing.gnn import knn_indices


def trajectory_smoothness(pseudotime: Tensor, coords: Tensor, k: int = 8) -> Tensor:
    neighbors = knn_indices(coords, k)
    gathered = torch.gather(
        pseudotime.unsqueeze(1).expand(-1, pseudotime.shape[1], -1), 2, neighbors
    )
    diff = pseudotime.unsqueeze(-1) - gathered
    return (diff**2).mean()
