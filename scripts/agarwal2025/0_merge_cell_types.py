#!/usr/bin/env python
"""Merge per-cell-type lentiMPRA tables into one table with a target column per cell type.

    python scripts/agarwal2025/0_merge_cell_types.py \\
        --cell_types K562 HepG2 WTC11 \\
        --input_tsvs data/K562_with_chrom.tsv data/HepG2_with_chrom.tsv data/WTC11_with_chrom.tsv \\
        --output data/merged_with_chrom.tsv

Each input is a ``<celltype>_with_chrom.tsv`` from ``0_add_coordinates.py``. The cell types'
libraries are mostly different elements, so rows are joined on the sequence itself (``seq``,
the 230 bp element with its adapters), not on ``seq_id``, which the libraries name
differently. Only ``rev == 0`` rows are read, as in training.

Writes ``seq_id, seq, <cell type>..., rev, chrom``: one row per distinct sequence, in order of
first appearance (cell types in the order given), with the cell type's ``mean_value`` in its
column and a blank where the element was not measured in that cell type. ``seq_id`` lists the
distinct ids the sequence carries across tables, separated by ``;``. ``chrom`` is the first
table's; a sequence whose tables name different primary chromosomes aborts the run, since its
split would depend on the table. ``start``/``end``/``strand`` are dropped because the tables'
coordinates are on different assemblies.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import primary_chrom  # noqa: E402

from alphagenome_encoder_ft import read_tsv_rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cell_types", nargs="+", required=True, help="column names, one per table")
    parser.add_argument("--input_tsvs", nargs="+", type=Path, required=True,
                        help="<celltype>_with_chrom.tsv tables, in the order of --cell_types")
    parser.add_argument("--output", type=Path, required=True, help="merged table to write")
    return parser


def merge(cell_types: list[str], input_tsvs: list[Path]) -> dict[str, dict]:
    """``seq`` -> ``{"seq_ids": [...], "chrom": str, "values": {cell type: mean_value}}``."""

    merged: dict[str, dict] = {}
    for cell_type, path in zip(cell_types, input_tsvs, strict=True):
        rows = read_tsv_rows(path, keep=lambda row: int(row["rev"]) == 0)
        if rows and "chrom" not in rows[0]:
            raise SystemExit(f"{path}: no chrom column; run 0_add_coordinates.py first")
        for row in rows:
            entry = merged.setdefault(row["seq"], {"seq_ids": [], "chrom": row["chrom"], "values": {}})
            if cell_type in entry["values"]:
                raise SystemExit(f"{path}: sequence of {row['seq_id']!r} appears twice")
            if primary_chrom(row["chrom"]) != primary_chrom(entry["chrom"]):
                raise SystemExit(
                    f"{row['seq_id']!r} is on {row['chrom']} in {path} but on {entry['chrom']} in an "
                    "earlier table; its split would depend on the table"
                )
            if row["seq_id"] not in entry["seq_ids"]:
                entry["seq_ids"].append(row["seq_id"])
            entry["values"][cell_type] = row["mean_value"]
        print(f"{cell_type}: {len(rows):,} rows from {path}")
    return merged


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    if len(args.cell_types) != len(args.input_tsvs):
        parser.error("--cell_types and --input_tsvs must have the same length")
    if len(set(args.cell_types)) != len(args.cell_types):
        parser.error("--cell_types must be distinct")

    merged = merge(args.cell_types, args.input_tsvs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["seq_id", "seq", *args.cell_types, "rev", "chrom"])
        for seq, entry in merged.items():
            values = [entry["values"].get(cell_type, "") for cell_type in args.cell_types]
            writer.writerow([";".join(entry["seq_ids"]), seq, *values, 0, entry["chrom"]])

    labelled = {cell_type: sum(cell_type in e["values"] for e in merged.values()) for cell_type in args.cell_types}
    by_count: dict[int, int] = {}
    for entry in merged.values():
        by_count[len(entry["values"])] = by_count.get(len(entry["values"]), 0) + 1
    summary = {
        "inputs": dict(zip(args.cell_types, map(str, args.input_tsvs))),
        "rows": len(merged),
        "labelled_rows_per_cell_type": labelled,
        "rows_by_number_of_cell_types": dict(sorted(by_count.items())),
    }
    meta_path = args.output.with_name(args.output.stem + "_metadata.json")
    meta_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.output} and {meta_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
