from dataclasses import dataclass

import numpy as np
import torch

from splicepath.gauge.classification import auroc, auroc_multi
from splicepath.gauge.intervals import bootstrap_ci
from splicepath.gauge.regression import (
    mae,
    mean_pcc,
    median_pcc,
    per_event_pcc,
    r2_score,
    rmse,
    spearman,
)
from splicepath.gauge.spatial import morans_i, trajectory_concordance
from splicepath.patchwork.dataset import SlideDataset, collate
from splicepath.piecing.model import SplicePath
from splicepath.templates.topology import TopologyFeatures


@dataclass(frozen=True)
class Predictions:
    psi_pred: np.ndarray
    psi_true: np.ndarray
    mutation_pred: np.ndarray
    mutation_true: np.ndarray
    cnv_pred: np.ndarray
    cnv_true: np.ndarray
    hrd_pred: np.ndarray
    hrd_true: np.ndarray
    pseudotime_pred: np.ndarray
    pseudotime_true: np.ndarray
    coords: np.ndarray


@dataclass(frozen=True)
class EvaluationReport:
    mean_pcc: float
    median_pcc: float
    spearman: float
    rmse: float
    r2: float
    mae: float
    pcc_ci_low: float
    pcc_ci_high: float
    mutation_auroc: float
    cnv_auroc: float
    hrd_auroc: float
    trajectory_concordance: float
    morans_i: float


@torch.no_grad()
def collect_predictions(
    model: SplicePath,
    dataset: SlideDataset,
    events: TopologyFeatures,
    device: torch.device,
) -> Predictions:
    model.eval()
    bound_events = events.to(device)
    psi_pred, psi_true = [], []
    mut_pred, mut_true = [], []
    cnv_pred, cnv_true = [], []
    hrd_pred, hrd_true = [], []
    time_pred, time_true, coords = [], [], []
    samples = [dataset[i] for i in range(len(dataset))]
    batch = collate(samples)
    embeddings = batch["embeddings"].to(device)
    coords_in = batch["coords"].to(device)
    output = model(embeddings, coords_in, bound_events)
    psi_pred.append(output.psi.cpu().numpy())
    psi_true.append(batch["psi"].numpy())
    if output.genetic is not None:
        mut_pred.append(output.genetic.mutation.cpu().numpy())
        cnv_pred.append(output.genetic.cnv.cpu().numpy())
        hrd_pred.append(output.genetic.hrd.cpu().numpy())
    mut_true.append(batch["mutation"].numpy())
    cnv_true.append(batch["cnv"].numpy())
    hrd_true.append(batch["hrd"].numpy())
    if output.trajectory is not None:
        time_pred.append(output.trajectory.pseudotime.cpu().numpy())
    time_true.append(batch["pseudotime"].numpy())
    coords.append(batch["coords"].numpy())
    empty = np.zeros((0, 0), dtype=np.float64)
    return Predictions(
        psi_pred=np.concatenate(psi_pred),
        psi_true=np.concatenate(psi_true),
        mutation_pred=np.concatenate(mut_pred) if mut_pred else empty,
        mutation_true=np.concatenate(mut_true),
        cnv_pred=np.concatenate(cnv_pred) if cnv_pred else empty,
        cnv_true=np.concatenate(cnv_true),
        hrd_pred=np.concatenate(hrd_pred) if hrd_pred else empty,
        hrd_true=np.concatenate(hrd_true),
        pseudotime_pred=np.concatenate(time_pred) if time_pred else empty,
        pseudotime_true=np.concatenate(time_true),
        coords=np.concatenate(coords),
    )


def run_evaluation(predictions: Predictions, ci_seed: int = 0) -> EvaluationReport:
    per_event = per_event_pcc(predictions.psi_pred, predictions.psi_true)
    finite = per_event[np.isfinite(per_event)]
    ci_low, ci_high = bootstrap_ci(finite, seed=ci_seed)
    mutation_auroc = (
        float(
            np.nanmean(
                auroc_multi(predictions.mutation_pred, predictions.mutation_true)
            )
        )
        if predictions.mutation_pred.size
        else float("nan")
    )
    cnv_auroc = (
        float(np.nanmean(auroc_multi(predictions.cnv_pred, predictions.cnv_true)))
        if predictions.cnv_pred.size
        else float("nan")
    )
    if predictions.hrd_pred.size:
        threshold = float(np.median(predictions.hrd_true))
        hrd_labels = (predictions.hrd_true > threshold).astype(np.float64)
        hrd_auroc = auroc(predictions.hrd_pred, hrd_labels)
    else:
        hrd_auroc = float("nan")
    if predictions.pseudotime_pred.size:
        concordance = trajectory_concordance(
            predictions.pseudotime_pred.ravel(), predictions.pseudotime_true.ravel()
        )
        moran_values = [
            morans_i(predictions.pseudotime_pred[i], predictions.coords[i])
            for i in range(predictions.pseudotime_pred.shape[0])
        ]
        moran = float(np.nanmean(moran_values))
    else:
        concordance = float("nan")
        moran = float("nan")
    return EvaluationReport(
        mean_pcc=mean_pcc(predictions.psi_pred, predictions.psi_true),
        median_pcc=median_pcc(predictions.psi_pred, predictions.psi_true),
        spearman=spearman(predictions.psi_pred, predictions.psi_true),
        rmse=rmse(predictions.psi_pred, predictions.psi_true),
        r2=r2_score(predictions.psi_pred, predictions.psi_true),
        mae=mae(predictions.psi_pred, predictions.psi_true),
        pcc_ci_low=ci_low,
        pcc_ci_high=ci_high,
        mutation_auroc=mutation_auroc,
        cnv_auroc=cnv_auroc,
        hrd_auroc=hrd_auroc,
        trajectory_concordance=concordance,
        morans_i=moran,
    )
