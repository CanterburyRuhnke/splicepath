import torch

from splicepath.sashing.gnn import SpatialGNN, knn_indices
from splicepath.sashing.state import build_state_vectors
from splicepath.seams.trajectory import trajectory_smoothness


def test_knn_excludes_self_and_returns_k() -> None:
    coords = torch.rand(2, 16, 2)
    neighbors = knn_indices(coords, k=5)
    assert neighbors.shape == (2, 16, 5)
    for b in range(2):
        for i in range(16):
            assert i not in neighbors[b, i].tolist()


def test_state_vector_dim() -> None:
    patches = torch.randn(2, 10, 24)
    psi = torch.rand(2, 6)
    genetic = torch.rand(2, 4)
    state = build_state_vectors(patches, psi, genetic)
    assert state.shape == (2, 10, 24 + 6 + 4)


def test_gnn_shape_and_smoothness_nonnegative() -> None:
    gnn = SpatialGNN(dim=12, hidden_dim=16, k=4)
    state = torch.randn(2, 9, 12)
    coords = torch.rand(2, 9, 2)
    refined = gnn(state, coords)
    assert refined.shape == (2, 9, 16)
    pseudotime = torch.randn(2, 9)
    assert trajectory_smoothness(pseudotime, coords, k=4).item() >= 0.0


def test_constant_pseudotime_has_zero_smoothness() -> None:
    coords = torch.rand(1, 12, 2)
    pseudotime = torch.full((1, 12), 0.5)
    assert torch.isclose(
        trajectory_smoothness(pseudotime, coords, k=4), torch.tensor(0.0), atol=1e-6
    )
