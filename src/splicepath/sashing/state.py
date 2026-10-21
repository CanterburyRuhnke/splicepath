import torch
from torch import Tensor


def build_state_vectors(patches: Tensor, psi: Tensor, genetic: Tensor) -> Tensor:
    num_patches = patches.shape[1]
    psi_field = psi.unsqueeze(1).expand(-1, num_patches, -1)
    genetic_field = genetic.unsqueeze(1).expand(-1, num_patches, -1)
    return torch.cat([patches, psi_field, genetic_field], dim=-1)
