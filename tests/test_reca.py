import torch

from splicepath.piecing.reca import RECA, MILPool, StandardMHA


def test_reca_output_shape() -> None:
    reca = RECA(patch_dim=96, d_topo=64, num_event_types=5, num_heads=4, head_dim=16)
    topo = torch.randn(10, 64)
    patches = torch.randn(3, 20, 96)
    event_type = torch.randint(0, 5, (10,))
    out = reca(topo, patches, event_type)
    assert out.shape == (3, 10, 64)


def test_reca_attention_rows_sum_to_one() -> None:
    torch.manual_seed(1)
    reca = RECA(patch_dim=32, d_topo=16, num_event_types=1, num_heads=2, head_dim=8)
    topo = torch.randn(4, 16)
    patches = torch.randn(2, 7, 32)
    event_type = torch.zeros(4, dtype=torch.long)
    q = reca.query[0](topo).view(4, 2, 8)
    k = reca.key[0](patches).view(2, 7, 2, 8)
    v = reca.value[0](patches).view(2, 7, 2, 8)
    logits = torch.einsum("ehd,bphd->behp", q, k) * reca.scale
    alpha = torch.softmax(logits, dim=-1)
    assert torch.allclose(alpha.sum(dim=-1), torch.ones(2, 4, 2), atol=1e-5)
    pooled = torch.einsum("behp,bphd->behd", alpha, v).reshape(2, 4, 16)
    manual = reca.out(pooled)
    assert torch.allclose(manual, reca(topo, patches, event_type), atol=1e-5)


def test_standard_mha_and_mil_shapes() -> None:
    topo = torch.randn(6, 32)
    patches = torch.randn(2, 12, 48)
    event_type = torch.randint(0, 5, (6,))
    mha = StandardMHA(patch_dim=48, num_events=6, num_heads=2, head_dim=16)
    assert mha(topo, patches, event_type).shape == (2, 6, 32)
    mil = MILPool(patch_dim=48, inner=32)
    out = mil(topo, patches, event_type)
    assert out.shape == (2, 6, 32)
    assert torch.allclose(out[:, 0], out[:, 1])
