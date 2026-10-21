from dataclasses import dataclass

import torch
from torch import Tensor, nn

from splicepath.templates.kmer import kmer_dim


@dataclass(frozen=True)
class TopologyFeatures:
    event_type: Tensor
    log_lengths: Tensor
    kmer: Tensor

    def to(self, device: torch.device) -> "TopologyFeatures":
        return TopologyFeatures(
            event_type=self.event_type.to(device),
            log_lengths=self.log_lengths.to(device),
            kmer=self.kmer.to(device),
        )


class TopologyEncoder(nn.Module):
    def __init__(
        self,
        d_topo: int = 128,
        kmer_k: int = 3,
        num_event_types: int = 5,
        type_dim: int = 16,
        hidden_dim: int = 256,
        use_topology: bool = True,
    ) -> None:
        super().__init__()
        self.d_topo = d_topo
        self.use_topology = use_topology
        self.type_embedding = nn.Embedding(num_event_types, type_dim)
        in_dim = type_dim + 3 + kmer_dim(kmer_k)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_topo),
        )

    def forward(self, features: TopologyFeatures) -> Tensor:
        type_vec = self.type_embedding(features.event_type)
        if self.use_topology:
            stacked = torch.cat([type_vec, features.log_lengths, features.kmer], dim=-1)
        else:
            zeros = torch.zeros_like(features.log_lengths)
            zero_kmer = torch.zeros_like(features.kmer)
            stacked = torch.cat([type_vec, zeros, zero_kmer], dim=-1)
        encoded: Tensor = self.mlp(stacked)
        return encoded
