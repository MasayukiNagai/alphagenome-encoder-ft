#!/usr/bin/env python
"""Convert the Gosai 2024 episomal MPRA table into the TSV EpisomalMPRAGosai2024Dataset reads.

    python scripts/gosai2024/0_prepare.py --source DATA-Table_S2__MPRA_dataset.txt \\
        --cell_type K562 --output data/gosai_K562_inserts.tsv

The source is Supplementary Table S2 of Gosai et al. 2024 (*Nature*), the MPRA dataset with
columns ``IDs, chr, sequence, <cell>_log2FC, <cell>_lfcSE``. It ships inserts with no split column and one target column per cell type.
This picks the cell type, drops unmeasured rows and Malinois's outlier rows, and writes
`seq_id, seq, mean_value, rev, chrom, insert_len, max_lfc_se`.

`seq` is the insert verbatim. The reporter around it is NOT written here: it is
`EpisomalMPRAGosai2024Library.construct()`, applied when the dataset is loaded, so it travels
with the checkpoint and the same rule builds the model input at inference time. See
`library.py` for the cassette and where its 384 bp window stops.

Splits
------
No split is written. `chrom` is copied from the source `chr` column, and the train/evaluate
drivers select validation and test rows by chromosome (`--val_chroms`, `--test_chroms`).

Malinois rows
-------------
The rows match the ones Malinois was trained, selected and scored on (Gosai et al. 2024,
Methods; boda2 `MPRA_DataModule` with the released checkpoint's settings):

- Outliers are dropped from every split: a row goes if any of the three cell types'
  log2FC is >= mean + 6 SD + 3 or <= mean - 6 SD, with mean and SD taken per cell type
  over the whole table. On Table S2 that is 19 rows.
- No standard-error filter is applied here. `max_lfc_se`, the largest lfcSE across the
  three cell types, is written instead, and the dataset reader uses it to restrict only
  the test split to Malinois's reported test set (lfcSE <= 1 in every cell type).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from library import EpisomalMPRAGosai2024Library as LIBRARY  # noqa: E402

CELL_TYPES = ("K562", "HepG2", "SKNSH")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare Gosai 2024 episomal MPRA for encoder-only fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--source", type=str, required=True, help="Gosai et al. 2024 Table S2")
    parser.add_argument("--cell_type", type=str, default="K562", choices=CELL_TYPES)
    parser.add_argument("--output", type=str, required=True, help="TSV to write")
    parser.add_argument("--outlier_sd", type=float, default=6.0,
                        help="drop rows beyond this many SD from the per-cell-type mean in "
                             "any cell type (Malinois: 6)")
    parser.add_argument("--outlier_up_shift", type=float, default=3.0,
                        help="added to the upper outlier bound (Malinois: 3)")
    parser.add_argument("--limit", type=int, default=None,
                        help="stop after N source rows (smoke test)")
    return parser


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_source(source: Path, limit) -> list[dict[str, str]]:
    """All source rows (or the first ``limit``), after checking the needed columns exist."""
    needed = ["IDs", "chr", "sequence"]
    needed += [f"{cell}_log2FC" for cell in CELL_TYPES] + [f"{cell}_lfcSE" for cell in CELL_TYPES]
    with open(source, newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for column in needed:
            if column not in (reader.fieldnames or []):
                raise SystemExit(f"{source} has no column {column!r}")
        rows = []
        for n, row in enumerate(reader):
            if limit is not None and n >= limit:
                break
            rows.append(row)
    return rows


def outlier_mask(rows: list[dict[str, str]], n_sd: float, up_shift: float) -> tuple[np.ndarray, dict]:
    """True for rows boda2's ``MPRA_DataModule`` drops as extreme values.

    Bounds are per cell type over all rows (pandas ``std``, ddof=1); a row is an outlier if
    any cell type reaches ``mean + n_sd*sd + up_shift`` or ``mean - n_sd*sd``.
    """
    values = np.array(
        [[float(row[f"{cell}_log2FC"]) for cell in CELL_TYPES] for row in rows], dtype=np.float64
    )
    mean = values.mean(axis=0)
    sd = values.std(axis=0, ddof=1)
    upper = mean + n_sd * sd + up_shift
    lower = mean - n_sd * sd
    mask = (values >= upper).any(axis=1) | (values <= lower).any(axis=1)
    bounds = {cell: {"lower": float(lo), "upper": float(hi)} for cell, lo, hi in zip(CELL_TYPES, lower, upper)}
    return mask, bounds


def main() -> int:
    args = build_arg_parser().parse_args()
    source = Path(args.source)
    if not source.is_file():
        raise SystemExit(f"source table not found: {source}")

    target_col = f"{args.cell_type}_log2FC"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    rows = read_source(source, args.limit)
    for row in rows:
        for cell in CELL_TYPES:
            if row[f"{cell}_log2FC"] in ("", "NA", "NaN") or row[f"{cell}_lfcSE"] in ("", "NA", "NaN"):
                raise SystemExit(f"unmeasured {cell} value for {row['IDs']!r}")
    outliers, outlier_bounds = outlier_mask(rows, args.outlier_sd, args.outlier_up_shift)

    print(f"writing {output}", flush=True)
    chrom_counts: Counter[str] = Counter()
    length_counts: Counter[int] = Counter()
    dropped_by_chrom: Counter[str] = Counter()
    written = 0
    with open(output, "w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        # chrom selects the split at load time. insert_len is annotation that lets
        # evaluation stratify by insert length. max_lfc_se selects Malinois's test rows.
        writer.writerow(["seq_id", "seq", "mean_value", "rev", "chrom", "insert_len", "max_lfc_se"])
        for row, is_outlier in zip(rows, outliers):
            ids, chrom = row["IDs"], row["chr"]
            if is_outlier:
                dropped_by_chrom[chrom] += 1
                continue
            value = float(row[target_col])
            insert = row["sequence"].strip().upper()
            max_lfc_se = max(float(row[f"{cell}_lfcSE"]) for cell in CELL_TYPES)
            if not insert or set(insert) - set("ACGT"):
                raise SystemExit(f"non-ACGT insert for {ids!r}")
            if len(insert) > LIBRARY.INSERT_LENGTH:
                raise SystemExit(
                    f"insert {ids!r} is {len(insert)} bp, longer than the "
                    f"{LIBRARY.INSERT_LENGTH} bp the construct leaves for it; it would be "
                    f"trimmed into the vector flanks"
                )
            # rev is always 0: the source has no reverse-complement duplicate rows, and
            # RC is applied as an on-the-fly augmentation by the dataset.
            writer.writerow([ids, insert, f"{value:.10g}", 0, chrom, len(insert), f"{max_lfc_se:.10g}"])
            chrom_counts[chrom] += 1
            length_counts[len(insert)] += 1
            written += 1

    if written == 0:
        raise SystemExit("no rows survived the filters")

    construct = LIBRARY.construct()
    metadata = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": str(source),
        "source_sha256": sha256(source),
        "source_mtime_utc": datetime.fromtimestamp(
            source.stat().st_mtime, timezone.utc
        ).isoformat(timespec="seconds"),
        "output": str(output),
        "cell_type": args.cell_type,
        "target_column": target_col,
        "rows_written": written,
        "outliers": {
            "rule": "any cell type log2FC >= mean + outlier_sd*SD + outlier_up_shift or <= mean - outlier_sd*SD",
            "bounds": outlier_bounds,
            "dropped": int(outliers.sum()),
            "dropped_by_chrom": dict(sorted(dropped_by_chrom.items())),
        },
        "args": vars(args),
        "construct": {
            "applied_at": "load time, by EpisomalMPRAGosai2024Library.construct()",
            "note": "seq holds the insert only; this is the reporter the drivers wrap it in",
            "default": construct.to_dict(),
            "flank_source": "boda2 MPRA_UPSTREAM / MPRA_DOWNSTREAM (300 bp each)",
            "regulatory_landmarks_in_downstream": {
                "polylinker": "28-78", "HindIII": "79-84",
                "minP": "85-127", "TATA": 98, "utr_kozak": "128-154", "gfp_cds": "155-299",
            },
        },
        "rows_by_chrom": dict(sorted(chrom_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "insert_lengths": {
            "min": min(length_counts), "max": max(length_counts),
            "n_full": length_counts.get(LIBRARY.INSERT_LENGTH, 0),
            "n_short": sum(c for l, c in length_counts.items() if l < LIBRARY.INSERT_LENGTH),
        },
    }
    meta_path = output.with_name(output.stem + "_metadata.json")
    with open(meta_path, "w") as handle:
        json.dump(metadata, handle, indent=2)

    n_short = sum(c for l, c in length_counts.items() if l < LIBRARY.INSERT_LENGTH)
    print(f"\nwrote {written:,} rows -> {output}")
    print(f"      outliers dropped  {int(outliers.sum())} {dict(sorted(dropped_by_chrom.items()))}")
    print(f"      metadata          -> {meta_path}")
    print(f"      construct         applied at load time: {len(construct.prefix)} + insert + "
          f"{len(construct.suffix)}, {construct.length} bp window from {construct.window_start}")
    print(f"      insert lengths    {min(length_counts)}-{max(length_counts)} bp "
          f"({length_counts.get(LIBRARY.INSERT_LENGTH, 0):,} at {LIBRARY.INSERT_LENGTH}, "
          f"{n_short:,} shorter)")
    print(f"      chromosomes       {len(chrom_counts)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
