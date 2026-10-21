from splicepath.quiltshow.evaluate import EvaluationReport


def crossdb_degradation(
    in_distribution: EvaluationReport, external: EvaluationReport
) -> float:
    return in_distribution.mean_pcc - external.mean_pcc
