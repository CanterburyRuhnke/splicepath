#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT="${1:-main}"
CHECKPOINT="${2:-outputs/${EXPERIMENT}/${EXPERIMENT}.ckpt}"

python -m splicepath.workbasket.assess \
    --experiment "${EXPERIMENT}" \
    --device "${DEVICE:-cpu}" \
    --checkpoint "${CHECKPOINT}"
