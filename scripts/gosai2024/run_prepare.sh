#!/bin/bash
# Build the Gosai 2024 training TSV for one cell type. CPU only, a few minutes.
#
#   bash scripts/gosai2024/run_prepare.sh SOURCE OUTPUT [flags for 0_prepare.py...]
#   bash scripts/gosai2024/run_prepare.sh DATA-Table_S2__MPRA_dataset.txt data/gosai_K562_inserts.tsv
#
# SOURCE is Supplementary Table S2 of Gosai et al. 2024. K562 unless --cell_type says otherwise.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ROOT="$HERE"
while [[ "$ROOT" != / && ! -f "$ROOT/pyproject.toml" ]]; do ROOT="$(dirname "$ROOT")"; done
[[ -f "$ROOT/pyproject.toml" ]] || { echo "no pyproject.toml above $HERE" >&2; exit 1; }

PY="${PYTHON:-$ROOT/.venv/bin/python}"
[[ -x "$PY" ]] || { echo "no interpreter at $PY" >&2; exit 1; }

[[ $# -ge 2 ]] || { echo "usage: $0 SOURCE OUTPUT [flags...]" >&2; exit 2; }
SOURCE="$1"; OUTPUT="$2"; shift 2

exec "$PY" "$HERE/0_prepare.py" --source "$SOURCE" --output "$OUTPUT" "$@"
