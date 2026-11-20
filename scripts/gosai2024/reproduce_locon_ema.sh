#!/bin/bash
# One seed of the stage-2 comparison in docs/LoCon_EMA.md on Gosai 2024 K562. Needs a GPU.
#
#   INPUT_TSV=data/gosai_K562_inserts.tsv ALPHAGENOME_WEIGHTS=/path/to/model.safetensors \
#     bash scripts/gosai2024/reproduce_locon_ema.sh 42
#
# Trains stage 1 once, then runs stage 2 four times from that same checkpoint, one per config
# (configs/gosai2024_K562_{fullft,fullft_ema,locon,locon_ema}.json; their stage 1 sections
# are identical), and scores every stage on validation and test, strand-averaged:
#
#   OUT_DIR/stage1-s<seed>/            stage 1, scored in stage1/evaluation/
#   OUT_DIR/<setting>-s<seed>/         stage 2 of one setting; stage1 -> the shared stage 1
#
# OUT_DIR defaults to results/gosai2024_K562_locon_ema_reproduce. A finished step is skipped
# and an interrupted one restarts, so rerunning the same seed resumes it. The doc used seeds
# 42, 43 and 44; summarize with
#
#   python scripts/summarize_runs.py OUT_DIR
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ROOT="$HERE"
while [[ "$ROOT" != / && ! -f "$ROOT/pyproject.toml" ]]; do ROOT="$(dirname "$ROOT")"; done
[[ -f "$ROOT/pyproject.toml" ]] || { echo "no pyproject.toml above $HERE" >&2; exit 1; }

PY="${PYTHON:-$ROOT/.venv/bin/python}"
[[ -x "$PY" ]] || { echo "no interpreter at $PY" >&2; exit 1; }

[[ $# -ge 1 ]] || { echo "usage: $0 SEED [flags for 1_train.py...]" >&2; exit 2; }
SEED="$1"; shift

INPUT_TSV="${INPUT_TSV:?set INPUT_TSV to the TSV 0_prepare.py writes}"
[[ -f "$INPUT_TSV" ]] || { echo "dataset not found: $INPUT_TSV" >&2; exit 1; }
WEIGHTS="${ALPHAGENOME_WEIGHTS:?set ALPHAGENOME_WEIGHTS to the pretrained backbone}"
[[ -f "$WEIGHTS" ]] || { echo "pretrained weights not found: $WEIGHTS" >&2; exit 1; }

PREFIX="gosai2024_K562"
OUT_DIR="${OUT_DIR:-$ROOT/results/${PREFIX}_locon_ema_reproduce}"
STAGE1_RUN="$OUT_DIR/stage1-s$SEED"

train() {  # train <config name> <run dir> [flags...]
  local config="$ROOT/configs/$1.json" run_dir="$2"; shift 2
  "$PY" "$HERE/1_train.py" --config "$config" --input_tsv "$INPUT_TSV" --pretrained_weights "$WEIGHTS" \
    --checkpoint_dir "$run_dir" --wandb_name "$(basename "$run_dir")" --seed "$SEED" "$@"
}

evaluate() {  # evaluate <stage dir>
  for split in val test; do
    [[ -f "$1/evaluation/${split}_metrics.json" ]] && continue
    "$PY" "$HERE/2_evaluate.py" --checkpoint_path "$1/best.pt" --output_dir "$1/evaluation" --split "$split"
  done
}

# history.json is written when training ends, so it marks a finished run; best.pt does not.
if [[ ! -f "$STAGE1_RUN/history.json" ]]; then
  echo "=== stage 1, seed $SEED"
  train "${PREFIX}_fullft" "$STAGE1_RUN" --stage1_only "$@"
fi
evaluate "$STAGE1_RUN/stage1"

for setting in fullft fullft_ema locon locon_ema; do
  run_dir="$OUT_DIR/$setting-s$SEED"
  if [[ ! -f "$run_dir/history.json" ]]; then
    echo "=== stage 2 $setting, seed $SEED"
    mkdir -p "$run_dir"
    ln -sfn "$STAGE1_RUN/stage1" "$run_dir/stage1"
    train "${PREFIX}_$setting" "$run_dir" --resume_from_stage2 "$@"
  fi
  evaluate "$run_dir/stage2"
done
