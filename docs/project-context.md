# Project Context

    project_name       : splicepath                                  [HIGH]
    domain             : computational pathology - splice-PSI         [HIGH]
                         prediction from H&E whole slide images
    framework          : PyTorch 2.x (frozen CONCH ViT-B/16 backbone) [HIGH]
    venue              : Nature Communications                        [MED]
    primary_datasets   : 5 live sources (see Section 6); 1 dropped           [HIGH]
    compute_target     : 4x A100 80GB, ~72h train, 8.3s/WSI infer     [HIGH]
    hparams_reference  : Methods Section 5.6 + Supplementary Table S3        [HIGH]
    supp_path          : embedded (Supplementary Information, pp. 30-36)
    extra_signals      : Eqs 1-9; ablations Table 2 + S2/S3; tasks
                         A/B/C (Tables 1/3 + Section 2.6); 5 cancer types

    NEEDS_USER_DECISION: 0

## 1. project_name
`splicepath` - the framework's own name throughout the manuscript (Abstract
L11, "a new computational pathology framework called SplicePath"). Package is
the lowercase form. [HIGH]

## 2. supp_path
Supplementary Information is bound into the same PDF (pp. 30-36): Tables
S2-S12. No separate sibling files were found by globbing. [HIGH]

## 3. domain
Computational pathology, specifically prediction of alternative-splicing
percent-spliced-in (PSI) values, somatic genetic architecture, and tumour
microenvironment fate trajectories from H&E whole slide images. Derived from
Abstract and Introduction Section 1. [HIGH]

## 4. framework
PyTorch 2.x. The Reproducibility statement (p. 21) names PyTorch 2.1; the
Methods Section 5.2 backbone is a frozen vision-language model (CONCH, ViT-B/16,
d=512) with alternates Virchow2 (d=1280, ViT-H/14) and UNI (d=1024, ViT-L/16),
and a ResNet-50 control (Tables S6, S7). Only RECA/GAD/TFTM are trained; all
backbones are frozen. [HIGH]

## 5. venue
Nature Communications. Inferred from structured abstract, numbered
superscript citations, and the Reporting summary / Ethics declarations /
Reproducibility statement sections characteristic of the Nature Portfolio
template. [MED]

## 6. primary_datasets
Verified against the manuscript Data availability (p. 20) and each canonical
source URL on 2026-06-17.

| name | version | license | access | status |
|---|---|---|---|---|
| TCGA Pan-Cancer (WSI + molecular) | 2013-2024 | TCGA / NIH open | https://portal.gdc.cancer.gov/ | live |
| TCGA SpliceSeq (PSI) | v2.0 (2016) | open academic | https://bioinformatics.mdanderson.org/TCGASpliceSeq/ | live |
| CPTAC proteogenomics | PDC | open academic | https://proteomic.datacommons.cancer.gov/pdc/ | live |
| HEST-1k (10x Visium ST + H&E) | 2024 | CC-BY-NC-SA 4.0 | https://github.com/mahmoodlab/HEST | live (1,276 paired samples) |
| 10x Visium platform | - | vendor docs | https://www.10xgenomics.com/platforms/visium | live |

DROPPED: OncoSplicing v3.0 (oncosplicing.com) - reference-only database cited
in Data availability; unreachable on two fetch attempts (timeout). Removed from
the release per the dead-link rule; it is not a training/evaluation source.

## 7. compute_target
4x NVIDIA A100 80GB (Methods Section 5.6, Reproducibility Section 2). batch_size 16 tiles,
grad_accum 4, world_size 4 -> effective batch 256. 50 epochs, ~72h wall-clock,
inference ~8.3s/WSI after patch extraction. [HIGH]

## 8. hparams_reference
Methods Section 5.6 and Supplementary Table S3.
- optimizer AdamW (beta1=0.9, beta2=0.999), weight_decay 1e-4 (ref [27])
- cosine LR schedule, base lr 5e-4, warmup_steps 500, epochs 50
- d_topo 128 (optimal in Tier-3 sweep over {32,64,128,256,512})
- lambda_g 0.5, lambda_t 0.1 (Tier-3 sensitivity); k-mer k=3 -> 64 features
- attention heads 4, d_h 64, 5 splice event types (ES/A5/A3/RI/ME)
- d_proj 256 (multi-scale projection), patch sizes 256/512/1024 at 20x
- |G|=50 mutation genes, |C|=22 chromosome arms, k_spatial=8
- augmentation: random patch dropout p=0.1, Macenko stain norm, spatial
  jitter +/-32 px
- AMP enabled (Table S6); mixed precision during training

## 9. extra_signals
- Numbered equations 1-9 (composite loss, topology encoding, RECA cross-
  attention, PSI head, GAD heads, TFTM state/GNN). Implement each exactly.
- Ablations: Table 2 tiers 0-3 (controls, necessity, interaction, sensitivity)
  + Supplementary Table S2 (interaction matrix, input-feature, multi-scale) +
  Table S3 (hyperparameter sweeps).
- Three downstream tasks: Task A splice PSI (Table 1, S1), Task B genetic
  architecture (Table 3), Task C TME fate trajectory (Section 2.6, Moran's I).
- Cross-database generalization (CPTAC, 10x Visium/HEST-1k) and retrospective
  validation of 100 splice events with a graduated 5-level concordance scale
  (Table 4, S11).
- Backbone study across CONCH/Virchow2/UNI/ResNet-50 (Tables S6, S7).
- Five primary cancer types: BRCA, UVM, LAML, LUAD, COAD (of 33 pan-cancer).
- A code-availability statement exists in the manuscript; it is NOT reproduced
  into the README per release policy.
