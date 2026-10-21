# Repository Plan

## Tree

    splicepath/
    |-- README.md
    |-- LICENSE
    |-- pyproject.toml
    |-- requirements.txt
    |-- environment.yml
    |-- Dockerfile
    |-- .gitignore
    |-- .pre-commit-config.yaml
    |-- configs/experiment/
    |   |-- base.yaml                       # full authoritative defaults
    |   |-- main.yaml                       # paper's primary result
    |   |-- ablation_tier0_*.yaml           # random input, mean predictor
    |   |-- ablation_tier1_*.yaml           # wo_reca / wo_gad / wo_tftm / wo_topology
    |   |-- ablation_tier2_*.yaml           # wo_reca_gad / wo_reca_tftm / reca_to_mha
    |   |-- ablation_tier3_*.yaml           # d_topo and lambda_g sweeps
    |   |-- supplementary_*.yaml            # multiscale, backbone, cross-db, lr, heads
    |   `-- _smoke.yaml                     # pytest only
    |-- src/splicepath/
    |   |-- patchwork/   # tiling, frozen backbone, projection, cohort, SpliceSeq, dataset
    |   |-- templates/   # splice-event topology encoding, event registry, k-mer
    |   |-- piecing/     # RECA cross-attention, PSI head, full SplicePath model
    |   |-- fabric/      # GAD: attention pooling + mutation/CNV/HRD heads
    |   |-- sashing/     # TFTM: state vectors, k-NN spatial GNN, pseudotime
    |   |-- seams/       # losses: splice, genetic, trajectory, composite
    |   |-- gauge/       # metrics + statistics battery
    |   |-- basting/     # trainer, schedule, AMP, DDP, EMA, atomic checkpoint
    |   |-- quiltshow/   # evaluation, retrospective, cross-db, baselines
    |   |-- notions/     # config dataclasses, seed, logging, io, registry
    |   `-- workbasket/  # cli: fit / assess / read / cast
    |-- scripts/
    |   |-- lay_train.sh
    |   |-- lay_eval.sh
    |   `-- cut_tesserae.sh
    |-- tests/
    `-- docs/

## Module responsibilities
- **patchwork** - cuts slides into multi-scale patches; owns the frozen-backbone
  protocol, Otsu tissue masking and Macenko stain normalisation, the synthetic
  cohort, SpliceSeq PSI loading, and the dataset/collate boundary.
- **templates** - builds the per-event topology query (event type, log exon/
  intron lengths, k-mer context) into a d_topo vector.
- **piecing** - RECA event cross-attention with per-event-type projections, the
  sigmoid PSI head conditioned on the genetic vector, the full SplicePath
  assembly, plus standard-MHA and mean-instance fallbacks for ablations.
- **fabric** - attention pooling and mutation/CNV/HRD heads; emits the genetic
  architecture conditioning vector.
- **sashing** - concatenates state vectors, runs a hand-rolled k-NN spatial GNN,
  predicts pseudotime, and supplies the trajectory smoothness term.
- **seams** - the three loss terms and their lambda-weighted composite.
- **gauge** - regression/classification metrics and the statistics battery
  (bootstrap CI, Wilcoxon, Holm-Bonferroni, BH-FDR, Cohen's d, McNemar, Wilson
  CI, Moran's I, trajectory concordance).
- **basting** - AdamW + cosine-warmup training loop, AMP, DDP plumbing, EMA,
  atomic checkpoints, seed capture.
- **quiltshow** - evaluation for tasks A/B/C, graduated retrospective
  concordance, cross-database degradation, baseline PSI-head adapters.
- **notions** - typed config dataclasses (loaded from YAML with inheritance),
  seeding, logging, YAML io, a small component registry.
- **workbasket** - four CLI entry points parsed with simple-parsing: fit
  (train), assess (eval), read (infer), cast (export ONNX).

## Pinned dependencies
runtime: torch>=2.1, numpy>=1.26, scipy>=1.11, pyyaml>=6.0, simple-parsing>=0.1
dev: pytest>=8, hypothesis>=6, ruff>=0.5, black>=24, isort>=5, mypy>=1.10,
onnx>=1.16

## Test coverage
- property (hypothesis): PSIin[0,1]; attention weights row-stochastic; topology
  encoding shape/finiteness.
- numerical-agreement: RECA forward vs explicit attention; PCC vs SciPy; Moran's
  I on a fixed lattice; Holm-Bonferroni vs hand-computed ordering; bootstrap CI
  brackets the estimate.
- overfit-single-batch: SplicePath drives splice loss toward zero on one batch.
- integration smoke: optimizer steps on `_smoke.yaml`, total loss decreases;
  checkpoint round-trip.
- unit: event registry, k-mer counts, projection dims, attention pooling, state
  vectors, Otsu/tiling/Macenko, SpliceSeq read filtering, config inheritance.
