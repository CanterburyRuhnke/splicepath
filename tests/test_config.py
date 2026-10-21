from splicepath.notions.config import load_experiment


def test_main_inherits_base() -> None:
    cfg = load_experiment("main")
    assert cfg.name == "main"
    assert cfg.model.topology.d_topo == 128
    assert cfg.train.lambda_g == 0.5
    assert cfg.model.backbone.patch_sizes == (256, 512, 1024)


def test_smoke_overrides_apply() -> None:
    cfg = load_experiment("_smoke")
    assert cfg.data.num_events == 16
    assert cfg.train.epochs == 8
    assert cfg.train.amp is False


def test_tuple_coercion() -> None:
    cfg = load_experiment("main")
    assert isinstance(cfg.train.betas, tuple)
    assert isinstance(cfg.data.cancer_types, tuple)
