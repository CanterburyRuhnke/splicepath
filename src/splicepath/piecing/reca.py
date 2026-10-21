import torch
from torch import Tensor, nn


class RECA(nn.Module):
    def __init__(
        self,
        patch_dim: int,
        d_topo: int = 128,
        num_event_types: int = 5,
        num_heads: int = 4,
        head_dim: int = 64,
    ) -> None:
        super().__init__()
        self.num_event_types = num_event_types
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner = num_heads * head_dim
        self.scale = head_dim**-0.5
        self.query = nn.ModuleList(
            nn.Linear(d_topo, self.inner) for _ in range(num_event_types)
        )
        self.key = nn.ModuleList(
            nn.Linear(patch_dim, self.inner) for _ in range(num_event_types)
        )
        self.value = nn.ModuleList(
            nn.Linear(patch_dim, self.inner) for _ in range(num_event_types)
        )
        self.out = nn.Linear(self.inner, self.inner)

    def forward(
        self, topo_query: Tensor, patches: Tensor, event_type: Tensor
    ) -> Tensor:
        batch = patches.shape[0]
        num_events = topo_query.shape[0]
        out = patches.new_zeros(batch, num_events, self.inner)
        for kind in range(self.num_event_types):
            mask = event_type == kind
            if not bool(mask.any()):
                continue
            idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
            q = self.query[kind](topo_query[idx])
            q = q.view(idx.shape[0], self.num_heads, self.head_dim)
            k = self.key[kind](patches).view(batch, -1, self.num_heads, self.head_dim)
            v = self.value[kind](patches).view(batch, -1, self.num_heads, self.head_dim)
            logits = torch.einsum("ehd,bphd->behp", q, k) * self.scale
            alpha = torch.softmax(logits, dim=-1)
            pooled = torch.einsum("behp,bphd->behd", alpha, v)
            pooled = pooled.reshape(batch, idx.shape[0], self.inner)
            out[:, idx] = self.out(pooled)
        return out


class StandardMHA(nn.Module):
    def __init__(
        self,
        patch_dim: int,
        num_events: int,
        num_heads: int = 4,
        head_dim: int = 64,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner = num_heads * head_dim
        self.scale = head_dim**-0.5
        self.query = nn.Embedding(num_events, self.inner)
        self.key = nn.Linear(patch_dim, self.inner)
        self.value = nn.Linear(patch_dim, self.inner)
        self.out = nn.Linear(self.inner, self.inner)

    def forward(
        self, topo_query: Tensor, patches: Tensor, event_type: Tensor
    ) -> Tensor:
        batch = patches.shape[0]
        num_events = topo_query.shape[0]
        q = self.query.weight.view(num_events, self.num_heads, self.head_dim)
        k = self.key(patches).view(batch, -1, self.num_heads, self.head_dim)
        v = self.value(patches).view(batch, -1, self.num_heads, self.head_dim)
        logits = torch.einsum("ehd,bphd->behp", q, k) * self.scale
        alpha = torch.softmax(logits, dim=-1)
        pooled = torch.einsum("behp,bphd->behd", alpha, v)
        pooled = pooled.reshape(batch, num_events, self.inner)
        return self.out(pooled)


class MILPool(nn.Module):
    def __init__(self, patch_dim: int, inner: int) -> None:
        super().__init__()
        self.inner = inner
        self.proj = nn.Linear(patch_dim, inner)

    def forward(
        self, topo_query: Tensor, patches: Tensor, event_type: Tensor
    ) -> Tensor:
        num_events = topo_query.shape[0]
        pooled = self.proj(patches.mean(dim=1))
        return pooled.unsqueeze(1).expand(-1, num_events, -1)
