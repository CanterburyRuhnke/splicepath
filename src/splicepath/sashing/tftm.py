from dataclasses import dataclass

from torch import Tensor, nn

from splicepath.sashing.gnn import SpatialGNN


@dataclass(frozen=True)
class TrajectoryOutput:
    pseudotime: Tensor
    state: Tensor


class TFTM(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 128, k: int = 8) -> None:
        super().__init__()
        self.gnn = SpatialGNN(state_dim, hidden_dim, k)
        self.time_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: Tensor, coords: Tensor) -> TrajectoryOutput:
        refined = self.gnn(state, coords)
        pseudotime = self.time_head(refined).squeeze(-1)
        return TrajectoryOutput(pseudotime=pseudotime, state=refined)
