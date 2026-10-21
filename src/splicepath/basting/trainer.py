from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from splicepath.basting.amp import autocast_context
from splicepath.basting.checkpoint import save_atomic
from splicepath.basting.ema import EMA
from splicepath.basting.schedule import cosine_warmup_factor
from splicepath.notions.config import RunConfig
from splicepath.notions.logs import get_logger
from splicepath.notions.seed import set_seed
from splicepath.patchwork.cohort import CohortBundle
from splicepath.patchwork.dataset import Batch, collate
from splicepath.piecing.model import SplicePath
from splicepath.seams.composite import CompositeLoss
from splicepath.templates.topology import TopologyFeatures


def _to_device(batch: Batch, device: torch.device) -> Batch:
    return Batch(
        embeddings=batch["embeddings"].to(device),
        coords=batch["coords"].to(device),
        psi=batch["psi"].to(device),
        mutation=batch["mutation"].to(device),
        cnv=batch["cnv"].to(device),
        hrd=batch["hrd"].to(device),
        pseudotime=batch["pseudotime"].to(device),
        cancer=batch["cancer"].to(device),
    )


class Trainer:
    def __init__(self, cfg: RunConfig, bundle: CohortBundle) -> None:
        set_seed(cfg.seed)
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.model = SplicePath(cfg).to(self.device)
        self.events: TopologyFeatures = bundle.events.to(self.device)
        self.loss = CompositeLoss(
            cfg.train.lambda_g, cfg.train.lambda_t, cfg.model.tftm.k_spatial
        )
        self.loader: DataLoader[object] = DataLoader(
            bundle.train,
            batch_size=cfg.train.batch_size,
            shuffle=True,
            collate_fn=collate,
            drop_last=False,
        )
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=cfg.train.lr,
            betas=cfg.train.betas,
            weight_decay=cfg.train.weight_decay,
        )
        steps_per_epoch = max(1, len(self.loader) // cfg.train.grad_accum)
        self.total_steps = max(1, steps_per_epoch * cfg.train.epochs)
        warmup = cfg.train.warmup_steps
        total = self.total_steps
        self.scheduler = LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: cosine_warmup_factor(step, warmup, total),
        )
        self.ema = EMA(self.model, cfg.train.ema_decay)
        self.logger = get_logger("splicepath.basting")
        self.global_step = 0

    def fit(self, max_steps: int | None = None) -> list[float]:
        cfg = self.cfg.train
        history: list[float] = []
        self.model.train()
        accum = 0
        self.optimizer.zero_grad()
        for _ in range(cfg.epochs):
            for raw in self.loader:
                batch = _to_device(raw, self.device)
                with autocast_context(self.device, cfg.amp, cfg.amp_dtype):
                    output = self.model(
                        batch["embeddings"], batch["coords"], self.events
                    )
                    terms = self.loss(output, batch)
                (terms.total / cfg.grad_accum).backward()
                accum += 1
                if accum < cfg.grad_accum:
                    continue
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                self.optimizer.step()
                self.scheduler.step()
                self.ema.update(self.model)
                self.optimizer.zero_grad()
                accum = 0
                self.global_step += 1
                history.append(float(terms.total.detach().cpu()))
                self.logger.info("step %d loss %.4f", self.global_step, history[-1])
                if max_steps is not None and self.global_step >= max_steps:
                    return history
        return history

    def save(self, path: str | Path) -> None:
        save_atomic(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "ema": self.ema.state_dict(),
                "seed": self.cfg.seed,
                "step": self.global_step,
                "experiment": self.cfg.name,
            },
            path,
        )
