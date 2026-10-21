from pathlib import Path

import torch
from simple_parsing import parse
from torch import Tensor, nn

from splicepath.notions.logs import get_logger
from splicepath.patchwork.cohort import SyntheticCohort
from splicepath.piecing.model import SplicePath
from splicepath.templates.topology import TopologyFeatures
from splicepath.workbasket.shared import CliOptions, build_config, build_model


class ExportWrapper(nn.Module):
    event_type: Tensor
    log_lengths: Tensor
    kmer: Tensor

    def __init__(self, model: SplicePath, events: TopologyFeatures) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("event_type", events.event_type)
        self.register_buffer("log_lengths", events.log_lengths)
        self.register_buffer("kmer", events.kmer)

    def forward(self, embeddings: Tensor, coords: Tensor) -> Tensor:
        features = TopologyFeatures(
            event_type=self.event_type,
            log_lengths=self.log_lengths,
            kmer=self.kmer,
        )
        return self.model(embeddings, coords, features).psi


def main() -> None:
    options = parse(CliOptions)
    cfg = build_config(options)
    logger = get_logger("splicepath.workbasket.cast")
    bundle = SyntheticCohort(cfg).build()
    model = build_model(cfg, options.checkpoint)
    wrapper = ExportWrapper(model, bundle.events).eval()
    sample = bundle.test[0]
    embeddings = sample.embeddings.unsqueeze(0)
    coords = sample.coords.unsqueeze(0)
    target = Path(cfg.output_dir) / "splicepath.onnx"
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (embeddings, coords),
        str(target),
        input_names=["embeddings", "coords"],
        output_names=["psi"],
        dynamic_axes={
            "embeddings": {0: "batch", 1: "patches"},
            "coords": {0: "batch", 1: "patches"},
            "psi": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )
    logger.info("exported ONNX graph to %s", target)


if __name__ == "__main__":
    main()
