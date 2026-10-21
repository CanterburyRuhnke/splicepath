from dataclasses import dataclass

import torch
from torch import Generator

from splicepath.notions.config import RunConfig
from splicepath.patchwork.dataset import SlideDataset, SlideSample
from splicepath.templates.kmer import kmer_frequency
from splicepath.templates.topology import TopologyFeatures


@dataclass(frozen=True)
class CohortBundle:
    train: SlideDataset
    val: SlideDataset
    test: SlideDataset
    events: TopologyFeatures
    genetic_dim: int


def _random_sequence(length: int, generator: Generator) -> str:
    alphabet = "ACGT"
    indices = torch.randint(0, 4, (length,), generator=generator)
    return "".join(alphabet[int(i)] for i in indices)


def _build_events(cfg: RunConfig, generator: Generator) -> TopologyFeatures:
    k = cfg.data.num_events
    topo = cfg.model.topology
    event_type = torch.randint(0, topo.num_event_types, (k,), generator=generator)
    lengths = torch.empty(k, 3)
    lengths[:, 0] = torch.randint(40, 400, (k,), generator=generator).float()
    lengths[:, 1] = torch.randint(40, 400, (k,), generator=generator).float()
    lengths[:, 2] = torch.randint(200, 8000, (k,), generator=generator).float()
    log_lengths = torch.log(lengths)
    kmer = torch.stack(
        [
            kmer_frequency(_random_sequence(120, generator), topo.kmer_k)
            for _ in range(k)
        ]
    )
    return TopologyFeatures(event_type=event_type, log_lengths=log_lengths, kmer=kmer)


class SyntheticCohort:
    def __init__(self, cfg: RunConfig) -> None:
        self.cfg = cfg
        self.generator = torch.Generator().manual_seed(cfg.seed)

    def build(self) -> CohortBundle:
        cfg = self.cfg
        gen = self.generator
        data = cfg.data
        latent = data.latent_dim
        embed_dim = cfg.model.backbone.embed_dim
        scales = len(cfg.model.backbone.patch_sizes)
        num_types = len(data.cancer_types)
        num_genes = cfg.model.gad.num_genes
        num_cnv = cfg.model.gad.num_cnv_arms
        genetic_dim = num_genes + num_cnv + 1
        events = _build_events(cfg, gen)

        type_mean = torch.randn(num_types, latent, generator=gen)
        grad_dir = torch.randn(num_types, 2, generator=gen)
        w_mut = torch.randn(latent, num_genes, generator=gen) / latent**0.5
        w_cnv = torch.randn(latent, num_cnv, generator=gen) / latent**0.5
        w_hrd = torch.randn(latent, generator=gen) / latent**0.5
        w_morph = torch.randn(data.num_events, latent, generator=gen) / latent**0.5
        w_gen = (
            torch.randn(data.num_events, genetic_dim, generator=gen) / genetic_dim**0.5
        )
        psi_bias = torch.randn(data.num_events, generator=gen) * 0.3
        scale_proj = torch.randn(scales, latent, embed_dim, generator=gen) / latent**0.5
        scale_bias = torch.randn(scales, embed_dim, generator=gen) * 0.1

        grid = data.grid_size
        coords_grid = torch.stack(
            torch.meshgrid(
                torch.arange(grid).float(),
                torch.arange(grid).float(),
                indexing="ij",
            ),
            dim=-1,
        ).reshape(-1, 2)
        coords_norm = coords_grid / max(grid - 1, 1)

        def make_slide() -> SlideSample:
            cancer = int(torch.randint(0, num_types, (1,), generator=gen).item())
            z = type_mean[cancer] + data.noise * torch.randn(latent, generator=gen)
            direction = grad_dir[cancer]
            projection = coords_norm @ direction
            projection = (projection - projection.mean()) / (projection.std() + 1e-6)
            patches = z.unsqueeze(0) + 0.6 * projection.unsqueeze(
                1
            ) * direction.norm() * torch.ones(latent)
            patches = patches + data.noise * torch.randn(
                grid * grid, latent, generator=gen
            )
            pooled = patches.mean(dim=0)
            mutation = (torch.sigmoid(pooled @ w_mut) > 0.5).float()
            cnv = (torch.sigmoid(pooled @ w_cnv) > 0.5).float()
            hrd = torch.sigmoid(pooled @ w_hrd)
            genetic = torch.cat([mutation, cnv, hrd.reshape(1)])
            psi = torch.sigmoid(w_morph @ pooled + w_gen @ genetic + psi_bias)
            emb = torch.einsum("pl,sld->psd", patches, scale_proj) + scale_bias
            emb = torch.nn.functional.gelu(emb)
            if data.randomize_input:
                emb = torch.randn_like(emb)
            if data.zero_patch:
                emb = torch.zeros_like(emb)
            coords = coords_norm.clone()
            pseudotime = projection.clone()
            if data.zero_spatial:
                coords = torch.zeros_like(coords)
            return SlideSample(
                embeddings=emb,
                coords=coords,
                psi=psi,
                mutation=mutation,
                cnv=cnv,
                hrd=hrd,
                pseudotime=pseudotime,
                cancer=cancer,
            )

        train = [make_slide() for _ in range(data.train_slides)]
        val = [make_slide() for _ in range(data.val_slides)]
        test = [make_slide() for _ in range(data.test_slides)]
        return CohortBundle(
            train=SlideDataset(train, events),
            val=SlideDataset(val, events),
            test=SlideDataset(test, events),
            events=events,
            genetic_dim=genetic_dim,
        )
