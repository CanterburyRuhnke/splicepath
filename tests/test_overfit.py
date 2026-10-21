import torch
from torch.optim import Adam

from splicepath.notions.config import RunConfig
from splicepath.patchwork.cohort import CohortBundle
from splicepath.patchwork.dataset import Batch
from splicepath.piecing.model import SplicePath
from splicepath.seams.splice import clipped_mse


def test_overfit_single_batch_drives_splice_loss_down(
    cfg: RunConfig, bundle: CohortBundle, batch: Batch
) -> None:
    torch.manual_seed(0)
    model = SplicePath(cfg)
    optimizer = Adam(model.parameters(), lr=1e-3)
    events = bundle.events
    first = None
    last = None
    for step in range(150):
        out = model(batch["embeddings"], batch["coords"], events)
        loss = clipped_mse(out.psi, batch["psi"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == 0:
            first = float(loss.detach())
        last = float(loss.detach())
    assert first is not None and last is not None
    assert last < 0.4 * first
