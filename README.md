# SplicePath - a piecing guide

SplicePath predicts RNA splicing percent-spliced-in (PSI) values, somatic
genetic architecture, and tumour-microenvironment fate trajectories directly
from H&E whole slide images. A frozen vision-language backbone supplies the
patch embeddings; three trained pieces stitch them into molecular predictions.

This repository is laid out like a quilt: slides are cut into **patches**, the
**templates** describe each splice event, **piecing** joins patches into per-
event predictions, the **fabric** carries genetic context, the **sashing**
binds neighbouring patches into a spatial trajectory, and the **seams** hold the
three objectives together. The sections below follow that order, from cutting
fabric to squaring up the finished top.

## The pattern

Three pieces make up the block:

- **RECA** (RNA Event-Aware Cross-Attention) uses a splice-topology query - event
  type, log exon/intron lengths, and a k-mer sequence context - to attend over
  patch keys and values, with separate projections per event type.
- **GAD** (Genetic Architecture Decoder) attention-pools the same patches into a
  slide vector and predicts somatic mutation, copy-number, and HRD status; its
  output conditions the PSI head.
- **TFTM** (TME Fate Trajectory Mapper) assembles per-position state vectors,
  runs a k-nearest-neighbour spatial graph network, and reads out a smooth
  pseudotime field.

The composite objective is `L = L_splice + 0.5 * L_genetic + 0.1 * L_traj`.

## Sorting the fabric

    pip install -e ".[dev]"

Conda:

    conda env create -f environment.yml
    conda activate splicepath
    pip install -e .

Docker:

    docker build -t splicepath .
    docker run --rm splicepath --help

## Cutting the patches

The study draws on public resources. Verify each download against its source
before use.

| source | what it provides | access | licence |
|---|---|---|---|
| TCGA Pan-Cancer (GDC) | H&E WSIs and molecular labels | https://portal.gdc.cancer.gov/ | TCGA / NIH open access |
| TCGA SpliceSeq v2.0 | per-event PSI across 33 cancers | https://bioinformatics.mdanderson.org/TCGASpliceSeq/ | open academic |
| CPTAC (PDC) | proteogenomic validation cohort | https://proteomic.datacommons.cancer.gov/pdc/ | open academic |
| HEST-1k | paired 10x Visium ST and H&E | https://github.com/mahmoodlab/HEST | CC-BY-NC-SA 4.0 |
| 10x Visium | spatial transcriptomics platform | https://www.10xgenomics.com/platforms/visium | vendor documentation |

Preprocessing follows `scripts/cut_tesserae.sh`: digitise at 20x, segment tissue
with Otsu thresholding on the saturation channel, Macenko stain-normalise, then
cut 256/512/1024 px patches and encode each with the frozen backbone. SpliceSeq
PSI tables are filtered to events with at least 8 supporting reads.

After downloading, record an integrity manifest so reruns can be checked:

    find data -type f | sort | xargs shasum -a 256 > data/manifest.sha256

The bundled `SyntheticCohort` stands in for these protected datasets so the
pipeline, tests, and CLI run offline. It plants a morphology-to-splice signal
for validation; it does not reproduce the manuscript's headline numbers, which
require the real cohorts on the hardware below.

## Piecing the blocks

One command per reported setting. Each reads `configs/experiment/<name>.yaml`.

    python -m splicepath.workbasket.fit --experiment main --device cuda

    # component ablations (Table 2)
    python -m splicepath.workbasket.fit --experiment ablation_tier1_wo_reca
    python -m splicepath.workbasket.fit --experiment ablation_tier1_wo_gad
    python -m splicepath.workbasket.fit --experiment ablation_tier2_reca_to_mha

    # sensitivity sweeps (Table 2 / Table S3)
    python -m splicepath.workbasket.fit --experiment ablation_tier3_dtopo_256
    python -m splicepath.workbasket.fit --experiment ablation_tier3_lambdag_1p0

    # backbone study and cross-database transfer (Tables S7 / S1)
    python -m splicepath.workbasket.fit --experiment supplementary_backbone_virchow2
    python -m splicepath.workbasket.fit --experiment supplementary_crossdb_cptac

`scripts/lay_train.sh main` wraps the distributed launch over four GPUs.

## Squaring up

    python -m splicepath.workbasket.assess --experiment main --checkpoint outputs/main/main.ckpt

On the real TCGA cohorts the primary block reaches a mean splice PSI PCC of
**0.342 +/- 0.019** over five seeds, broken down by cancer type as BRCA 0.362,
UVM 0.404, LAML 0.257, LUAD 0.333, COAD 0.352 (Table 1). Genetic architecture
decoding reports AUROC 0.823 (TP53), 0.871 (SF3B1), and 0.814 (HRD score)
(Table 3). Retrospective validation over 100 splice events gives 68.0%
concordance (Table 4), and trajectory inference yields a pseudotime concordance
of 0.378 with Moran's I 0.612 (Section 2.6). Cross-database transfer reaches
PCC 0.298 (CPTAC) and 0.317 (10x Visium). Quote each within its reported +/-std.

## The quilting frame

Training the primary block uses four NVIDIA A100 80GB GPUs: batch size 16 tiles
per GPU with gradient accumulation 4 (effective batch 256), 50 epochs, AdamW at
a cosine-scheduled 5e-4 with 500 warmup steps, and automatic mixed precision.
Wall-clock is roughly 72 hours; inference is about 8.3 seconds per slide after
patch extraction. Patch embeddings and the SpliceSeq matrix dominate on-disk
storage.

## Finished tops on loan

Trained checkpoints are available upon reasonable request. Each release will
ship with its SHA-256 so a downloaded checkpoint can be verified against the
record before evaluation.
