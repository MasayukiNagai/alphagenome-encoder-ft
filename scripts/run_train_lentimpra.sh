#!/bin/bash
# Two-stage lentiMPRA fine-tune. Needs a GPU.
#
#   bash scripts/run_train_lentimpra.sh [CELLTYPE] [extra flags...]
#
# Any flag after the cell type is forwarded to train_lentimpra.py and overrides the config.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ROOT="$HERE"
while [[ "$ROOT" != / && ! -f "$ROOT/pyproject.toml" ]]; do ROOT="$(dirname "$ROOT")"; done
[[ -f "$ROOT/pyproject.toml" ]] || { echo "no pyproject.toml above $HERE" >&2; exit 1; }

PY="${PYTHON:-$ROOT/.venv/bin/python}"
[[ -x "$PY" ]] || { echo "no interpreter at $PY" >&2; exit 1; }

CELLTYPE="${1:-K562}"
[[ $# -gt 0 ]] && shift

CONFIG="$ROOT/configs/lentimpra_${CELLTYPE}.json"
[[ -f "$CONFIG" ]] || { echo "no config: $CONFIG" >&2; exit 1; }

INPUT_TSV="${LENTI_INPUT_TSV:-/grid/koo/home/shared/data/lentimpra/agarwal_2025/${CELLTYPE}.tsv}"
[[ -f "$INPUT_TSV" ]] || { echo "dataset not found: $INPUT_TSV" >&2; exit 1; }

WEIGHTS="${ALPHAGENOME_WEIGHTS:?set ALPHAGENOME_WEIGHTS to the pretrained backbone}"
[[ -f "$WEIGHTS" ]] || { echo "pretrained weights not found: $WEIGHTS" >&2; exit 1; }

timestamp=$(date +"%m%d_%H%M")

exec "$PY" "$ROOT/scripts/train_lentimpra.py" \
  --config "$CONFIG" \
  --input_tsv "$INPUT_TSV" \
  --pretrained_weights "$WEIGHTS" \
  --wandb_name "mpra_${CELLTYPE}_${timestamp}" \
  --checkpoint_dir "${CHECKPOINT_DIR:-$ROOT/results/mpra_${CELLTYPE}_${timestamp}}" \
  "$@"
