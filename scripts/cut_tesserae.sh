#!/usr/bin/env bash
set -euo pipefail

echo "Tile extraction expects whole slide images digitised at 20x magnification."
echo "Each slide is segmented with Otsu thresholding on the saturation channel,"
echo "Macenko stain-normalised, then cut into 256/512/1024 px patches before the"
echo "frozen backbone produces multi-scale embeddings."
echo
echo "Provide WSIs under data/slides and SpliceSeq PSI tables under data/spliceseq,"
echo "then point the data loader at those paths. See README for sources."
