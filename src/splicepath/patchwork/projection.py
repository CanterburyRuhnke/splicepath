import torch
from torch import Tensor, nn


class MultiScaleProjector(nn.Module):
    def __init__(
        self,
        embed_dim: int = 512,
        proj_dim: int = 256,
        num_scales: int = 3,
    ) -> None:
        super().__init__()
        self.num_scales = num_scales
        self.proj_dim = proj_dim
        self.output_dim = proj_dim * num_scales
        self.projections = nn.ModuleList(
            nn.Linear(embed_dim, proj_dim) for _ in range(num_scales)
        )

    def forward(self, multiscale: Tensor) -> Tensor:
        if multiscale.shape[-2] != self.num_scales:
            raise ValueError("scale dimension mismatch")
        parts = []
        for scale in range(self.num_scales):
            projected = self.projections[scale](multiscale[..., scale, :])
            parts.append(projected)
        return torch.cat(parts, dim=-1)
