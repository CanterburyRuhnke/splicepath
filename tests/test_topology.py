import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from splicepath.templates.kmer import kmer_dim
from splicepath.templates.topology import TopologyEncoder, TopologyFeatures


def _features(num_events: int, kmer_k: int) -> TopologyFeatures:
    return TopologyFeatures(
        event_type=torch.randint(0, 5, (num_events,)),
        log_lengths=torch.randn(num_events, 3),
        kmer=torch.rand(num_events, kmer_dim(kmer_k)),
    )


@settings(max_examples=25, deadline=None)
@given(
    num_events=st.integers(min_value=1, max_value=12),
    d_topo=st.sampled_from([32, 64, 128]),
)
def test_topology_shape_and_finite(num_events: int, d_topo: int) -> None:
    encoder = TopologyEncoder(d_topo=d_topo, kmer_k=3, num_event_types=5)
    out = encoder(_features(num_events, 3))
    assert out.shape == (num_events, d_topo)
    assert torch.isfinite(out).all()


def test_disabling_topology_changes_output() -> None:
    features = _features(8, 3)
    on = TopologyEncoder(d_topo=64, use_topology=True)
    off = TopologyEncoder(d_topo=64, use_topology=False)
    off.load_state_dict(on.state_dict())
    assert not torch.allclose(on(features), off(features))
