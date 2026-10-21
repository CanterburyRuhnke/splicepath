from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

from splicepath.notions.io import deep_merge, read_yaml


@dataclass(frozen=True)
class BackboneConfig:
    name: str = "conch"
    embed_dim: int = 512
    patch_sizes: tuple[int, ...] = (256, 512, 1024)
    proj_dim: int = 256
    patch_dropout: float = 0.1


@dataclass(frozen=True)
class TopologyConfig:
    d_topo: int = 128
    kmer_k: int = 3
    num_event_types: int = 5
    use_topology: bool = True


@dataclass(frozen=True)
class RecaConfig:
    num_heads: int = 4
    head_dim: int = 64
    use_reca: bool = True
    standard_mha: bool = False


@dataclass(frozen=True)
class GadConfig:
    num_genes: int = 50
    num_cnv_arms: int = 22
    hidden_dim: int = 256
    use_gad: bool = True


@dataclass(frozen=True)
class TftmConfig:
    k_spatial: int = 8
    hidden_dim: int = 128
    use_tftm: bool = True


@dataclass(frozen=True)
class ModelConfig:
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    topology: TopologyConfig = field(default_factory=TopologyConfig)
    reca: RecaConfig = field(default_factory=RecaConfig)
    gad: GadConfig = field(default_factory=GadConfig)
    tftm: TftmConfig = field(default_factory=TftmConfig)


@dataclass(frozen=True)
class DataConfig:
    cancer_types: tuple[str, ...] = ("BRCA", "UVM", "LAML", "LUAD", "COAD")
    num_events: int = 200
    num_patches: int = 64
    grid_size: int = 8
    train_slides: int = 24
    val_slides: int = 6
    test_slides: int = 12
    randomize_input: bool = False
    mean_predictor: bool = False
    zero_patch: bool = False
    zero_genetic: bool = False
    zero_spatial: bool = False
    latent_dim: int = 32
    noise: float = 0.35


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 50
    batch_size: int = 16
    grad_accum: int = 4
    world_size: int = 4
    lr: float = 5e-4
    weight_decay: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.999)
    warmup_steps: int = 500
    grad_clip: float = 1.0
    lambda_g: float = 0.5
    lambda_t: float = 0.1
    amp: bool = True
    amp_dtype: str = "bfloat16"
    ema_decay: float = 0.999


@dataclass(frozen=True)
class RunConfig:
    name: str = "main"
    seed: int = 0
    device: str = "cpu"
    output_dir: str = "outputs"
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def _coerce(annotation: Any, value: Any) -> Any:
    origin = getattr(annotation, "__origin__", None)
    if annotation is tuple or origin is tuple:
        return tuple(value)
    return value


def _build(cls: Any, payload: dict[str, Any]) -> Any:
    kwargs: dict[str, Any] = {}
    for spec in fields(cls):
        if spec.name not in payload:
            continue
        raw = payload[spec.name]
        if is_dataclass(spec.type) and isinstance(raw, dict):
            kwargs[spec.name] = _build(spec.type, raw)
        else:
            kwargs[spec.name] = _coerce(spec.type, raw)
    return cls(**kwargs)


def config_from_dict(payload: dict[str, Any]) -> RunConfig:
    result: RunConfig = _build(RunConfig, payload)
    return result


def load_experiment(
    name: str,
    configs_dir: str | Path = "configs",
    overrides: dict[str, Any] | None = None,
) -> RunConfig:
    root = Path(configs_dir)
    payload = read_yaml(root / "experiment" / f"{name}.yaml")
    inherit = payload.pop("inherit", None)
    if inherit is not None:
        base = read_yaml(root / "experiment" / f"{inherit}.yaml")
        payload = deep_merge(base, payload)
    if overrides:
        payload = deep_merge(payload, overrides)
    return config_from_dict(payload)
