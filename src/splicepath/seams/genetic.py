from torch import Tensor
from torch.nn import functional as F

from splicepath.fabric.gad import GeneticOutput


def genetic_bce(
    output: GeneticOutput,
    mutation: Tensor,
    cnv: Tensor,
    hrd: Tensor,
) -> Tensor:
    mutation_loss = F.binary_cross_entropy(output.mutation, mutation)
    cnv_loss = F.binary_cross_entropy(output.cnv, cnv)
    hrd_loss = F.mse_loss(output.hrd, hrd)
    return mutation_loss + cnv_loss + hrd_loss
