import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from splicepath.notions.config import RunConfig
from splicepath.patchwork.cohort import CohortBundle
from splicepath.patchwork.dataset import Batch
from splicepath.piecing.model import SplicePath


@settings(max_examples=10, deadline=None)
@given(batch_size=st.integers(min_value=1, max_value=3))
def test_psi_within_unit_interval(
    cfg: RunConfig, bundle: CohortBundle, batch_size: int
) -> None:
    model = SplicePath(cfg)
    model.eval()
    samples = [bundle.train[i % len(bundle.train)] for i in range(batch_size)]
    embeddings = torch.stack([s.embeddings for s in samples])
    coords = torch.stack([s.coords for s in samples])
    with torch.no_grad():
        out = model(embeddings, coords, bundle.events)
    assert out.psi.shape == (batch_size, cfg.data.num_events)
    assert (out.psi >= 0).all() and (out.psi <= 1).all()


def test_ablation_switches_remove_modules(cfg: RunConfig) -> None:
    from dataclasses import replace

    stripped = replace(
        cfg,
        model=replace(
            cfg.model,
            reca=replace(cfg.model.reca, use_reca=False),
            gad=replace(cfg.model.gad, use_gad=False),
            tftm=replace(cfg.model.tftm, use_tftm=False),
        ),
    )
    model = SplicePath(stripped)
    assert model.gad is None
    assert model.tftm is None


def test_trajectory_present_in_full_model(
    cfg: RunConfig, bundle: CohortBundle, batch: Batch
) -> None:
    model = SplicePath(cfg)
    out = model(batch["embeddings"], batch["coords"], bundle.events)
    assert out.trajectory is not None
    assert out.genetic is not None
    assert out.trajectory.pseudotime.shape == (
        batch["embeddings"].shape[0],
        batch["embeddings"].shape[1],
    )
