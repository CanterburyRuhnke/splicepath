from dataclasses import dataclass

import torch
from torch import Tensor, nn

from splicepath.fabric.gad import GeneticArchitectureDecoder, GeneticOutput
from splicepath.notions.config import RunConfig
from splicepath.patchwork.projection import MultiScaleProjector
from splicepath.piecing.head import PSIHead
from splicepath.piecing.reca import RECA, MILPool, StandardMHA
from splicepath.sashing.state import build_state_vectors
from splicepath.sashing.tftm import TFTM, TrajectoryOutput
from splicepath.templates.topology import TopologyEncoder, TopologyFeatures


@dataclass(frozen=True)
class ModelOutput:
    psi: Tensor
    genetic: GeneticOutput | None
    trajectory: TrajectoryOutput | None


class SplicePath(nn.Module):
    def __init__(self, cfg: RunConfig) -> None:
        super().__init__()
        model = cfg.model
        backbone = model.backbone
        num_scales = len(backbone.patch_sizes)
        self.condition_genetic = not cfg.data.zero_genetic
        self.genetic_dim = model.gad.num_genes + model.gad.num_cnv_arms + 1
        self.projector = MultiScaleProjector(
            backbone.embed_dim, backbone.proj_dim, num_scales
        )
        patch_dim = self.projector.output_dim
        self.topology = TopologyEncoder(
            d_topo=model.topology.d_topo,
            kmer_k=model.topology.kmer_k,
            num_event_types=model.topology.num_event_types,
            use_topology=model.topology.use_topology,
        )
        inner = model.reca.num_heads * model.reca.head_dim
        self.event_dim = inner
        self.aggregator: nn.Module
        if not model.reca.use_reca:
            self.aggregator = MILPool(patch_dim, inner)
        elif model.reca.standard_mha:
            self.aggregator = StandardMHA(
                patch_dim,
                cfg.data.num_events,
                model.reca.num_heads,
                model.reca.head_dim,
            )
        else:
            self.aggregator = RECA(
                patch_dim,
                model.topology.d_topo,
                model.topology.num_event_types,
                model.reca.num_heads,
                model.reca.head_dim,
            )
        self.use_gad = model.gad.use_gad
        self.gad: GeneticArchitectureDecoder | None
        if self.use_gad:
            self.gad = GeneticArchitectureDecoder(
                patch_dim,
                model.gad.num_genes,
                model.gad.num_cnv_arms,
                model.gad.hidden_dim,
            )
        else:
            self.gad = None
        self.psi_head = PSIHead(inner, self.genetic_dim)
        self.use_tftm = model.tftm.use_tftm
        self.tftm: TFTM | None
        if self.use_tftm:
            state_dim = patch_dim + cfg.data.num_events + self.genetic_dim
            self.tftm = TFTM(state_dim, model.tftm.hidden_dim, model.tftm.k_spatial)
        else:
            self.tftm = None

    def forward(
        self, embeddings: Tensor, coords: Tensor, events: TopologyFeatures
    ) -> ModelOutput:
        patches = self.projector(embeddings)
        topo_query = self.topology(events)
        z = self.aggregator(topo_query, patches, events.event_type)
        genetic_output: GeneticOutput | None = None
        if self.gad is not None:
            genetic_output = self.gad(patches)
            genetic = genetic_output.vector()
        else:
            genetic = patches.new_zeros(patches.shape[0], self.genetic_dim)
        conditioning = genetic if self.condition_genetic else torch.zeros_like(genetic)
        psi = self.psi_head(z, conditioning)
        trajectory: TrajectoryOutput | None = None
        if self.tftm is not None:
            state = build_state_vectors(patches, psi.detach(), genetic.detach())
            trajectory = self.tftm(state, coords)
        return ModelOutput(psi=psi, genetic=genetic_output, trajectory=trajectory)
