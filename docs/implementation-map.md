# Implementation Map

Every numbered equation, algorithmic component, reported table, ablation, and
metric mapped to its implementation site.

| paper location | item | file | module symbol | notes |
|---|---|---|---|---|
| Methods Section 5.1, Eq. (1) | composite loss L_total | `src/splicepath/seams/composite.py` | `CompositeLoss` | lambda_g=0.5, lambda_t=0.1 |
| Methods Section 5.1, Eq. (1) | splice loss L_splice | `src/splicepath/seams/splice.py` | `clipped_mse` | MSE on PSI clipped [0,1] |
| Methods Section 5.1, Eq. (1) | genetic loss L_genetic | `src/splicepath/seams/genetic.py` | `genetic_bce` | BCE mutation + CNV |
| Methods Section 5.5, Section 5.1 | trajectory loss L_traj | `src/splicepath/seams/trajectory.py` | `trajectory_smoothness` | Sigma|tau_i-tau_j|^2/|E| |
| Methods Section 5.2 | multi-scale backbone embed | `src/splicepath/patchwork/backbone.py` | `Backbone`, `SurrogateBackbone`, `build_backbone` | CONCH/Virchow2/UNI/ResNet (D1) |
| Methods Section 5.2 | 20x tiling + Otsu + Macenko | `src/splicepath/patchwork/tiling.py` | `multiscale_tiles`, `tissue_mask`, `macenko_stain_normalize` | 256/512/1024 |
| Methods Section 5.2 | scale projection d_proj=256 | `src/splicepath/patchwork/projection.py` | `MultiScaleProjector` | 3 independent linears + concat |
| Methods Section 5.3, Eq. (2) | splice topology encoding | `src/splicepath/templates/topology.py` | `TopologyEncoder` | type+lengths+kmer, d_topo=128 |
| Methods Section 5.3 | event registry (5 types) | `src/splicepath/templates/events.py` | `EventType`, `SpliceEvent` | ES/A5/A3/RI/ME |
| Methods Section 5.3 | k-mer (k=3) context | `src/splicepath/templates/kmer.py` | `kmer_frequency` | 64-dim |
| Methods Section 5.3, Eq. (3) | RECA cross-attention | `src/splicepath/piecing/reca.py` | `RECA` | 4 heads, d_h=64, topo query |
| Methods Section 5.3, Eq. (4) | PSI prediction head | `src/splicepath/piecing/head.py` | `PSIHead` | sigmoid, conditioned on g |
| Tier-2 ablation (Table 2) | standard MHA control | `src/splicepath/piecing/reca.py` | `StandardMHA` | capacity-matched |
| Methods Section 5.4, Eq. (5)-(7) | GAD heads | `src/splicepath/fabric/gad.py` | `GeneticArchitectureDecoder` | mutation/CNV/HRD |
| Methods Section 5.4, Eq. (5) | attention pooling | `src/splicepath/fabric/attpool.py` | `AttentionPool` | slide-level vector |
| Methods Section 5.5, Eq. (8) | spatial state vector s_i | `src/splicepath/sashing/state.py` | `build_state_vectors` | [x; psi_hat; g] |
| Methods Section 5.5, Eq. (9) | k-NN spatial GNN | `src/splicepath/sashing/gnn.py` | `SpatialGNN` | hand-rolled, k=8 (D3) |
| Methods Section 5.5 | pseudotime + TFTM | `src/splicepath/sashing/tftm.py` | `TFTM` | MLP_time over s'_i |
| Section 2.1, Table 1 | full model assembly | `src/splicepath/piecing/model.py` | `SplicePath` | RECA+GAD+TFTM |
| Methods Section 5.6 | trainer / optim / sched | `src/splicepath/basting/trainer.py` | `Trainer` | AdamW, cosine+warmup |
| R4 | atomic ckpt + EMA + seed | `src/splicepath/basting/checkpoint.py`, `ema.py` | `save_atomic`, `EMA` | tmp+os.replace |
| Methods Section 5.6 | DDP / AMP plumbing | `src/splicepath/basting/distributed.py`, `amp.py` | `setup_ddp`, `autocast_context` | optional |
| Section 2.1, Table 1 | PCC/Spearman/RMSE/R^2/MAE | `src/splicepath/gauge/regression.py` | `pcc`, `spearman`, `rmse`, `r2`, `mae` | per-event |
| Table 3 | AUROC (mut/CNV/HRD) | `src/splicepath/gauge/classification.py` | `auroc` | genetic tasks |
| Section 2.1, Methods Section 5.9 | bootstrap 95% CI | `src/splicepath/gauge/intervals.py` | `bootstrap_ci` | 1000 resamples |
| Section 2.1, Methods Section 5.9 | Cohen's d, Wilcoxon | `src/splicepath/gauge/significance.py` | `cohens_d`, `wilcoxon_signed_rank` | |
| Methods Section 5.9 | Holm-Bonferroni, BH-FDR | `src/splicepath/gauge/significance.py` | `holm_bonferroni`, `benjamini_hochberg` | |
| Section 2.5, Table 4 | McNemar + Wilson CI | `src/splicepath/gauge/significance.py` | `mcnemar`, `wilson_ci` | concordance |
| Section 2.6 | Moran's I | `src/splicepath/gauge/spatial.py` | `morans_i` | spatial autocorrelation |
| Section 2.6, Table 4 | trajectory concordance | `src/splicepath/gauge/spatial.py` | `trajectory_concordance` | Spearman pseudotime |
| Section 2.5, Table S11 | retrospective validation | `src/splicepath/quiltshow/retrospective.py` | `concordance_level`, `retrospective_report` | 5-level graduated |
| Section 2.4, Table S1 | cross-DB generalization | `src/splicepath/quiltshow/crossdb.py` | `crossdb_degradation` | CPTAC / Visium |
| Section 2.1-2.3 | evaluation loops | `src/splicepath/quiltshow/evaluate.py` | `run_evaluation` | tasks A/B/C |
| Methods Section 5.7 | baseline PSI-head adapter | `src/splicepath/quiltshow/baselines.py` | `LinearPSIHead` | adapted baselines |
| Data Section 6 | synthetic cohort (D2) | `src/splicepath/patchwork/cohort.py` | `SyntheticCohort` | self-consistent signal |
| Data Section 6 | SpliceSeq PSI loader | `src/splicepath/patchwork/spliceseq.py` | `load_psi_matrix` | filter >=8 reads |
| Data Section 6 | dataset / collate | `src/splicepath/patchwork/dataset.py` | `SlideDataset`, `collate` | |
| R4 | seed / logging / io | `src/splicepath/notions/{seed,logs,io}.py` | `set_seed`, `get_logger`, `read_yaml` | |
| R4 | typed config dataclasses | `src/splicepath/notions/config.py` | `RunConfig`, `ModelConfig`, ... | simple-parsing |
| CLI | train/eval/infer/export | `src/splicepath/workbasket/{fit,assess,read,cast}.py` | `main` | onnx export |
| Table 2 Tier 0-3 | ablation switches | `configs/experiment/ablation_*.yaml` | - | wo_* flags |
| Table S2 | interaction / multiscale | `configs/experiment/supplementary_*.yaml` | - | |
| Table S3 | d_topo / lambda_g / lr / heads sweep | `configs/experiment/ablation_*.yaml` | - | |
| Table S7 | backbone study | `configs/experiment/supplementary_backbone_*.yaml` | - | CONCH/Virchow2/UNI/ResNet |
