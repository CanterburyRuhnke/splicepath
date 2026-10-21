from typing import Protocol, runtime_checkable

import torch
from torch import Tensor, nn

_EMBED_DIMS = {
    "conch": 512,
    "virchow2": 1280,
    "uni": 1024,
    "resnet50": 2048,
}


@runtime_checkable
class Backbone(Protocol):
    embed_dim: int

    def embed(self, pixels: Tensor) -> Tensor: ...


class SurrogateBackbone(nn.Module):
    weight: Tensor
    bias: Tensor

    def __init__(
        self,
        embed_dim: int = 512,
        pooled_grid: int = 8,
        seed: int = 1234,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.pooled_grid = pooled_grid
        self.pool = nn.AdaptiveAvgPool2d((pooled_grid, pooled_grid))
        generator = torch.Generator().manual_seed(seed)
        in_dim = 3 * pooled_grid * pooled_grid
        weight = torch.randn(embed_dim, in_dim, generator=generator)
        weight /= float(in_dim) ** 0.5
        bias = torch.zeros(embed_dim)
        self.register_buffer("weight", weight)
        self.register_buffer("bias", bias)
        self.norm = nn.LayerNorm(embed_dim)
        for parameter in self.norm.parameters():
            parameter.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def embed(self, pixels: Tensor) -> Tensor:
        pooled = self.pool(pixels).flatten(1)
        projected = torch.nn.functional.linear(pooled, self.weight, self.bias)
        activated = torch.nn.functional.gelu(projected)
        return self.norm(activated)

    def forward(self, pixels: Tensor) -> Tensor:
        return self.embed(pixels)


def build_backbone(name: str, seed: int = 1234) -> SurrogateBackbone:
    key = name.lower()
    if key not in _EMBED_DIMS:
        known = ", ".join(sorted(_EMBED_DIMS))
        raise KeyError(f"unknown backbone '{name}'; known: {known}")
    return SurrogateBackbone(embed_dim=_EMBED_DIMS[key], seed=seed)
