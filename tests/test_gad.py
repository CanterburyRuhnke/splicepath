import torch

from splicepath.fabric.attpool import AttentionPool
from splicepath.fabric.gad import GeneticArchitectureDecoder


def test_attpool_weights_sum_to_one() -> None:
    pool = AttentionPool(dim=40, hidden_dim=32)
    patches = torch.randn(3, 15, 40)
    pooled, weights = pool(patches)
    assert pooled.shape == (3, 40)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(3), atol=1e-5)


def test_gad_outputs_bounded_and_vector_dim() -> None:
    gad = GeneticArchitectureDecoder(dim=40, num_genes=7, num_cnv_arms=5)
    patches = torch.randn(2, 9, 40)
    out = gad(patches)
    assert out.mutation.shape == (2, 7)
    assert out.cnv.shape == (2, 5)
    assert out.hrd.shape == (2,)
    assert (out.mutation >= 0).all() and (out.mutation <= 1).all()
    assert out.vector().shape == (2, 7 + 5 + 1)
