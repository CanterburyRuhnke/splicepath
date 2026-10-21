import torch
from simple_parsing import parse

from splicepath.notions.logs import get_logger
from splicepath.quiltshow.evaluate import collect_predictions, run_evaluation
from splicepath.workbasket.shared import (
    CliOptions,
    build_bundle,
    build_config,
    build_model,
)


def main() -> None:
    options = parse(CliOptions)
    cfg = build_config(options)
    logger = get_logger("splicepath.workbasket.assess")
    bundle = build_bundle(cfg)
    model = build_model(cfg, options.checkpoint)
    predictions = collect_predictions(
        model, bundle.test, bundle.events, torch.device(cfg.device)
    )
    report = run_evaluation(predictions, ci_seed=cfg.seed)
    logger.info(
        "PSI mean PCC %.3f [%.3f, %.3f] | spearman %.3f | rmse %.3f",
        report.mean_pcc,
        report.pcc_ci_low,
        report.pcc_ci_high,
        report.spearman,
        report.rmse,
    )
    logger.info(
        "genetic AUROC mut %.3f cnv %.3f hrd %.3f",
        report.mutation_auroc,
        report.cnv_auroc,
        report.hrd_auroc,
    )
    logger.info(
        "trajectory concordance %.3f | Moran's I %.3f",
        report.trajectory_concordance,
        report.morans_i,
    )


if __name__ == "__main__":
    main()
