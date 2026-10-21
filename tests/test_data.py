import numpy as np
import torch

from splicepath.notions.config import RunConfig
from splicepath.patchwork.backbone import build_backbone
from splicepath.patchwork.cohort import CohortBundle
from splicepath.patchwork.projection import MultiScaleProjector
from splicepath.patchwork.spliceseq import load_psi_matrix


def test_backbone_dims_and_frozen() -> None:
    backbone = build_backbone("conch")
    assert backbone.embed_dim == 512
    pixels = torch.randn(4, 3, 64, 64)
    out = backbone.embed(pixels)
    assert out.shape == (4, 512)
    assert all(not p.requires_grad for p in backbone.norm.parameters())


def test_backbone_variants() -> None:
    assert build_backbone("virchow2").embed_dim == 1280
    assert build_backbone("uni").embed_dim == 1024


def test_projection_concatenates_scales() -> None:
    projector = MultiScaleProjector(embed_dim=512, proj_dim=256, num_scales=3)
    feats = torch.randn(2, 10, 3, 512)
    out = projector(feats)
    assert out.shape == (2, 10, 768)
    assert projector.output_dim == 768


def test_spliceseq_loader_filters_reads(tmp_path) -> None:
    psi = tmp_path / "psi.tsv"
    reads = tmp_path / "reads.tsv"
    psi.write_text("event\ts1\ts2\nE1\t0.5\t0.6\nE2\t0.1\t0.2\n")
    reads.write_text("event\ts1\ts2\nE1\t20\t30\nE2\t2\t3\n")
    ids, matrix = load_psi_matrix(psi, reads, min_reads=8)
    assert ids == ["E1"]
    assert matrix.shape == (1, 2)
    assert np.all((matrix >= 0) & (matrix <= 1))


def test_cohort_shapes(cfg: RunConfig, bundle: CohortBundle) -> None:
    sample = bundle.train[0]
    assert sample.embeddings.shape[1] == 3
    assert sample.psi.shape == (cfg.data.num_events,)
    assert (
        bundle.genetic_dim == cfg.model.gad.num_genes + cfg.model.gad.num_cnv_arms + 1
    )
