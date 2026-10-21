from pathlib import Path

from simple_parsing import parse

from splicepath.basting.trainer import Trainer
from splicepath.notions.logs import get_logger
from splicepath.workbasket.shared import CliOptions, build_bundle, build_config


def main() -> None:
    options = parse(CliOptions)
    cfg = build_config(options)
    logger = get_logger("splicepath.workbasket.fit")
    bundle = build_bundle(cfg)
    trainer = Trainer(cfg, bundle)
    history = trainer.fit(max_steps=options.max_steps)
    target = Path(cfg.output_dir) / f"{cfg.name}.ckpt"
    trainer.save(target)
    logger.info("final loss %.4f over %d steps", history[-1], len(history))
    logger.info("checkpoint written to %s", target)


if __name__ == "__main__":
    main()
