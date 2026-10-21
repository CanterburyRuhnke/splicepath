from splicepath.basting.trainer import Trainer
from splicepath.notions.config import RunConfig
from splicepath.patchwork.cohort import CohortBundle


def test_training_smoke_decreases(cfg: RunConfig, bundle: CohortBundle) -> None:
    trainer = Trainer(cfg, bundle)
    history = trainer.fit(max_steps=8)
    assert len(history) == 8
    assert history[-1] < history[0]


def test_checkpoint_roundtrip(tmp_path, cfg: RunConfig, bundle: CohortBundle) -> None:
    trainer = Trainer(cfg, bundle)
    trainer.fit(max_steps=2)
    target = tmp_path / "ckpt.pt"
    trainer.save(target)
    from splicepath.basting.checkpoint import load_checkpoint

    state = load_checkpoint(target)
    assert state["seed"] == cfg.seed
    assert "model" in state and "ema" in state
