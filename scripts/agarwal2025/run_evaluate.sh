#!/bin/bash
# Score the stage checkpoints of a lentiMPRA run on validation and test. Needs a GPU.
#
#   bash scripts/agarwal2025/run_evaluate.sh RUN_DIR [flags for 2_evaluate.py...]
#
# The construct, input TSV and held-out chromosomes come from the checkpoint and run.json.
# Results go to RUN_DIR/<stage>/evaluation/{val,test}_metrics.json. A stage directory that is
# a symlink (a stage 1 shared between runs, as reproduce_locon_ema.sh makes) is skipped: it is
# scored with the run it came from.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ROOT="$HERE"
while [[ "$ROOT" != / && ! -f "$ROOT/pyproject.toml" ]]; do ROOT="$(dirname "$ROOT")"; done
[[ -f "$ROOT/pyproject.toml" ]] || { echo "no pyproject.toml above $HERE" >&2; exit 1; }

PY="${PYTHON:-$ROOT/.venv/bin/python}"
[[ -x "$PY" ]] || { echo "no interpreter at $PY" >&2; exit 1; }

[[ $# -ge 1 ]] || { echo "usage: $0 RUN_DIR [flags...]" >&2; exit 2; }
RUN_DIR="$1"; shift
[[ -d "$RUN_DIR" ]] || { echo "no run directory: $RUN_DIR" >&2; exit 1; }

found=0
for stage in stage1 stage2; do
  ckpt="$RUN_DIR/$stage/best.pt"
  if [[ ! -f "$ckpt" ]]; then
    echo "skipping $stage: no checkpoint at $ckpt" >&2
    continue
  fi
  if [[ -L "$RUN_DIR/$stage" ]]; then
    echo "skipping $stage: linked from $(readlink -f "$RUN_DIR/$stage")" >&2
    continue
  fi
  found=1
  for split in val test; do
    echo "=== $stage $split"
    "$PY" "$HERE/2_evaluate.py" \
      --checkpoint_path "$ckpt" \
      --output_dir "$RUN_DIR/$stage/evaluation" \
      --split "$split" \
      "$@"
  done
done
[[ "$found" -eq 1 ]] || { echo "no checkpoints found under $RUN_DIR" >&2; exit 1; }
