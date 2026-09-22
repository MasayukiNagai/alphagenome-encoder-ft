#!/bin/bash
# Score both stage checkpoints of a lentiMPRA run.
#
#   bash scripts/run_evaluate_lentimpra.sh [RUN_DIR] [extra flags...]
#
# The construct and the input TSV come from the checkpoint and the run's run.json, so no
# dataset arguments are needed here.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ROOT="$HERE"
while [[ "$ROOT" != / && ! -f "$ROOT/pyproject.toml" ]]; do ROOT="$(dirname "$ROOT")"; done
[[ -f "$ROOT/pyproject.toml" ]] || { echo "no pyproject.toml above $HERE" >&2; exit 1; }

PY="${PYTHON:-$ROOT/.venv/bin/python}"
[[ -x "$PY" ]] || { echo "no interpreter at $PY" >&2; exit 1; }

RUN_DIR="${1:-${CHECKPOINT_DIR:-}}"
[[ -n "$RUN_DIR" ]] || { echo "usage: $0 RUN_DIR [flags...]" >&2; exit 1; }
[[ $# -gt 0 ]] && shift
[[ -d "$RUN_DIR" ]] || { echo "no run directory: $RUN_DIR" >&2; exit 1; }

found=0
for stage in stage1 stage2; do
  ckpt="$RUN_DIR/$stage/best.pt"
  if [[ ! -f "$ckpt" ]]; then
    echo "skipping $stage: no checkpoint at $ckpt" >&2
    continue
  fi
  found=1
  echo "=== evaluating $stage"
  "$PY" "$ROOT/scripts/evaluate_lentimpra.py" \
    --checkpoint_path "$ckpt" \
    --output_dir "$RUN_DIR/$stage/evaluation" \
    "$@"
done
[[ "$found" -eq 1 ]] || { echo "no checkpoints found under $RUN_DIR" >&2; exit 1; }
