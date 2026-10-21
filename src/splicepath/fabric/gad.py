from dataclasses import dataclass

import torch
from torch import Tensor, nn

from splicepath.fabric.attpool import AttentionPool


@dataclass(frozen=True)
class GeneticOutput:
    mutation: Tensor
    cnv: Tensor
    hrd: Tensor
    weights: Tensor

    def vector(self) -> Tensor:
        return torch.cat([self.mutation, self.cnv, self.hrd.unsqueeze(-1)], dim=-1)


class GeneticArchitectureDecoder(nn.Module):
    def __init__(
        self,
        dim: int,
        num_genes: int = 50,
        num_cnv_arms: int = 22,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.num_genes = num_genes
        self.num_cnv_arms = num_cnv_arms
        self.pool = AttentionPool(dim, hidden_dim)
        self.mutation_head = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, num_genes)
        )
        self.cnv_head = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, num_cnv_arms)
        )
        self.hrd_head = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, patches: Tensor) -> GeneticOutput:
        pooled, weights = self.pool(patches)
        mutation = torch.sigmoid(self.mutation_head(pooled))
        cnv = torch.sigmoid(self.cnv_head(pooled))
        hrd = self.hrd_head(pooled).squeeze(-1)
        return GeneticOutput(mutation=mutation, cnv=cnv, hrd=hrd, weights=weights)
