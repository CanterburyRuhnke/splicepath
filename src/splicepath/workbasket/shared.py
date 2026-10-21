from dataclasses import dataclass, field, replace

import torch

from splicepath.basting.checkpoint import load_checkpoint
from splicepath.notions.config import RunConfig, load_experiment
from splicepath.patchwork.cohort import CohortBundle, SyntheticCohort
from splicepath.piecing.model import SplicePath


@dataclass
class CliOptions:
    experiment: str = "main"
    configs_dir: str = "configs"
    seed: int = field(default=0)
    device: str = "cpu"
    output_dir: str = "outputs"
    max_steps: int | None = None
    checkpoint: str = ""


def build_config(options: CliOptions) -> RunConfig:
    cfg = load_experiment(options.experiment, options.configs_dir)
    return replace(
        cfg,
        seed=options.seed,
        device=options.device,
        output_dir=options.output_dir,
    )


def build_bundle(cfg: RunConfig) -> CohortBundle:
    return SyntheticCohort(cfg).build()


def build_model(cfg: RunConfig, checkpoint: str = "") -> SplicePath:
    model = SplicePath(cfg).to(torch.device(cfg.device))
    if checkpoint:
        state = load_checkpoint(checkpoint, map_location=cfg.device)
        model.load_state_dict(state["model"])
    return model
