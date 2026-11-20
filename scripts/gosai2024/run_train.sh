#!/bin/bash
# Two-stage Gosai 2024 episomal MPRA fine-tune. Needs a GPU.
#
#   INPUT_TSV=data/gosai_K562_inserts.tsv ALPHAGENOME_WEIGHTS=/path/to/model.safetensors \
#   CHECKPOINT_DIR=results/gosai2024_K562_locon_ema \
#     bash scripts/gosai2024/run_train.sh gosai2024_locon_ema [flags for 1_train.py...]
#
# CONFIG is a path or the name of a file in configs/ (with or without .json). INPUT_TSV is the
# table 0_prepare.py writes. The run goes to CHECKPOINT_DIR (default:
# results/<config name>). Flags after CONFIG override the config for this run.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ROOT="$HERE"
while [[ "$ROOT" != / && ! -f "$ROOT/pyproject.toml" ]]; do ROOT="$(dirname "$ROOT")"; done
[[ -f "$ROOT/pyproject.toml" ]] || { echo "no pyproject.toml above $HERE" >&2; exit 1; }

PY="${PYTHON:-$ROOT/.venv/bin/python}"
[[ -x "$PY" ]] || { echo "no interpreter at $PY" >&2; exit 1; }

[[ $# -ge 1 ]] || { echo "usage: $0 CONFIG [flags...]" >&2; exit 2; }
CONFIG="$1"; shift
[[ -f "$CONFIG" ]] || CONFIG="$ROOT/configs/${CONFIG%.json}.json"
[[ -f "$CONFIG" ]] || { echo "no config: $CONFIG" >&2; exit 1; }
NAME="$(basename "$CONFIG" .json)"

INPUT_TSV="${INPUT_TSV:?set INPUT_TSV to the TSV 0_prepare.py writes}"
[[ -f "$INPUT_TSV" ]] || { echo "dataset not found: $INPUT_TSV" >&2; exit 1; }
WEIGHTS="${ALPHAGENOME_WEIGHTS:?set ALPHAGENOME_WEIGHTS to the pretrained backbone}"
[[ -f "$WEIGHTS" ]] || { echo "pretrained weights not found: $WEIGHTS" >&2; exit 1; }

exec "$PY" "$HERE/1_train.py" \
  --config "$CONFIG" \
  --input_tsv "$INPUT_TSV" \
  --pretrained_weights "$WEIGHTS" \
  --checkpoint_dir "${CHECKPOINT_DIR:-$ROOT/results/$NAME}" \
  "$@"
