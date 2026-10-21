import torch
from torch import Tensor, nn


def knn_indices(coords: Tensor, k: int) -> Tensor:
    num_patches = coords.shape[1]
    k_eff = min(k, num_patches - 1)
    distances = torch.cdist(coords, coords)
    eye = torch.eye(num_patches, device=coords.device, dtype=torch.bool)
    distances = distances.masked_fill(eye.unsqueeze(0), float("inf"))
    neighbors = torch.topk(distances, k_eff, dim=-1, largest=False).indices
    return neighbors


class SpatialGNN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int = 128, k: int = 8) -> None:
        super().__init__()
        self.k = k
        self.self_transform = nn.Linear(dim, hidden_dim)
        self.neighbor_transform = nn.Linear(dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, state: Tensor, coords: Tensor) -> Tensor:
        neighbors = knn_indices(coords, self.k)
        batch, num_patches, _ = state.shape
        index = neighbors.unsqueeze(-1).expand(-1, -1, -1, state.shape[-1])
        gathered = torch.gather(
            state.unsqueeze(1).expand(-1, num_patches, -1, -1), 2, index
        )
        messages = self.neighbor_transform(gathered).mean(dim=2)
        updated = self.self_transform(state) + messages
        return self.norm(torch.nn.functional.gelu(updated))
