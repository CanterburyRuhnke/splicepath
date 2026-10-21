import torch
from simple_parsing import parse

from splicepath.notions.logs import get_logger
from splicepath.patchwork.dataset import collate
from splicepath.workbasket.shared import (
    CliOptions,
    build_bundle,
    build_config,
    build_model,
)


def main() -> None:
    options = parse(CliOptions)
    cfg = build_config(options)
    logger = get_logger("splicepath.workbasket.read")
    bundle = build_bundle(cfg)
    model = build_model(cfg, options.checkpoint)
    model.eval()
    device = torch.device(cfg.device)
    batch = collate([bundle.test[0]])
    with torch.no_grad():
        output = model(
            batch["embeddings"].to(device),
            batch["coords"].to(device),
            bundle.events.to(device),
        )
    psi = output.psi.squeeze(0)
    logger.info("predicted PSI for %d splice events", psi.shape[0])
    logger.info("first 8 PSI values: %s", [round(v, 3) for v in psi[:8].tolist()])
    if output.genetic is not None:
        burden = float(output.genetic.mutation.squeeze(0).mean())
        logger.info("mean mutation probability %.3f", burden)


if __name__ == "__main__":
    main()
