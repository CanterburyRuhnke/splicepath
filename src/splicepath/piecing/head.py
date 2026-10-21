import torch
from torch import Tensor, nn


class PSIHead(nn.Module):
    def __init__(self, event_dim: int, genetic_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(event_dim + genetic_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, events: Tensor, genetic: Tensor) -> Tensor:
        num_events = events.shape[1]
        expanded = genetic.unsqueeze(1).expand(-1, num_events, -1)
        joined = torch.cat([events, expanded], dim=-1)
        logits = self.mlp(joined).squeeze(-1)
        return torch.sigmoid(logits)
