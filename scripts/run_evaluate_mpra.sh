#!/bin/bash

set -e

REPO_ROOT="/grid/koo/home/nagai/projects/ag_mpra_torch/alphagenome-encoder-ft"
cd "$REPO_ROOT"

PYTHON="$REPO_ROOT/.venv/bin/python"
script="$REPO_ROOT/scripts/evaluate_mpra.py"
celltype=${1:-"K562"}

input_tsv="/grid/koo/home/shared/data/lentimpra/agarwal_2025/${celltype}.tsv"
pretrained_weights="/grid/koo/home/shared/models/alphagenome_pytorch/model_all_folds.safetensors"
run_dir="$REPO_ROOT/results/mpra_${celltype}"


for stage in stage1 stage2; do
  checkpoint_path="$run_dir/$stage/best.pt"
  if [[ ! -f "$checkpoint_path" ]]; then
    echo "Checkpoint not found: $checkpoint_path" >&2
    exit 1
  fi

  output_dir="$run_dir/$stage/evaluation"
  mkdir -p "$output_dir"

  cmd="$PYTHON $script \
    --checkpoint_path $checkpoint_path \
    --output_dir $output_dir "

  echo "Running command for $stage:"
  echo "$cmd"
  eval "$cmd"
done
