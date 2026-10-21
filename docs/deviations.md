# Deviations

Each entry links the manuscript location it concerns and the justification.
The deviation is also annotated at its call site in code.

## D1 - Frozen backbone surrogate
Methods Section 5.2 specifies frozen CONCH (ViT-B/16, d=512) with Virchow2 (d=1280)
and UNI (d=1024) alternates. Their published weights are access-gated and no
raw whole slide images are bundled with this release. We therefore provide a
deterministic frozen surrogate exposing the identical `Backbone` protocol and
output dimensionality per backbone; the real encoders drop in unchanged. The
trained components (RECA, GAD, TFTM) and every downstream computation are
unaffected because they consume only the embedding interface.
Site: `src/splicepath/patchwork/backbone.py`.

## D2 - Synthetic self-consistent cohort
Data availability lists TCGA / CPTAC / HEST-1k as the sources; none are
redistributable here and require substantial download. To keep the pipeline
runnable and the tests meaningful, `SyntheticCohort` generates multi-scale
patch embeddings, a SpliceSeq-style PSI matrix, mutation/CNV/HRD labels, and
spatial coordinates with a planted morphology->splice signal, so overfit and
smoke tests exercise real gradient flow. Substituting the real loaders (same
dataset/collate interface) reproduces the manuscript protocol.
Site: `src/splicepath/patchwork/cohort.py`.

## D3 - Hand-rolled spatial graph operations
Methods Section 5.5, Eq. (9) defines a k-nearest-spatial-neighbour GNN (k=8). Rather
than depend on a heavyweight graph library, the message passing and k-NN
construction are implemented directly in PyTorch. This keeps the dependency
surface minimal and matches the policy of implementing a paper detail manually
when a library cannot express it natively.
Site: `src/splicepath/sashing/gnn.py`.
