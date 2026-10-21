from dataclasses import dataclass

from torch import Tensor

from splicepath.patchwork.dataset import Batch
from splicepath.piecing.model import ModelOutput
from splicepath.seams.genetic import genetic_bce
from splicepath.seams.splice import clipped_mse
from splicepath.seams.trajectory import trajectory_smoothness


@dataclass(frozen=True)
class LossTerms:
    total: Tensor
    splice: Tensor
    genetic: Tensor
    trajectory: Tensor


class CompositeLoss:
    def __init__(
        self, lambda_g: float = 0.5, lambda_t: float = 0.1, k: int = 8
    ) -> None:
        self.lambda_g = lambda_g
        self.lambda_t = lambda_t
        self.k = k

    def __call__(self, output: ModelOutput, batch: Batch) -> LossTerms:
        splice = clipped_mse(output.psi, batch["psi"])
        zero = output.psi.new_zeros(())
        genetic = zero
        if output.genetic is not None:
            genetic = genetic_bce(
                output.genetic, batch["mutation"], batch["cnv"], batch["hrd"]
            )
        trajectory = zero
        if output.trajectory is not None:
            trajectory = trajectory_smoothness(
                output.trajectory.pseudotime, batch["coords"], self.k
            )
        total = splice + self.lambda_g * genetic + self.lambda_t * trajectory
        return LossTerms(
            total=total, splice=splice, genetic=genetic, trajectory=trajectory
        )
