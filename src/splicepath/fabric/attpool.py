import torch
from torch import Tensor, nn


class AttentionPool(nn.Module):
    def __init__(self, dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.value = nn.Linear(dim, hidden_dim)
        self.gate = nn.Linear(dim, hidden_dim)
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, patches: Tensor) -> tuple[Tensor, Tensor]:
        projected = torch.tanh(self.value(patches))
        gated = torch.sigmoid(self.gate(patches))
        logits = self.score(projected * gated).squeeze(-1)
        weights = torch.softmax(logits, dim=-1)
        pooled = torch.einsum("bp,bpd->bd", weights, patches)
        return pooled, weights
