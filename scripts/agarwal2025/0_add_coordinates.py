#!/usr/bin/env python
"""Append genomic coordinates to an Agarwal 2025 lentiMPRA table.

The human_legnet tables (``datasets/original/<celltype>.tsv`` of
https://github.com/autosome-ru/human_legnet; columns ``seq_id, seq, mean_value, fold, rev``)
carry no coordinates, so they cannot be split by chromosome. ENCODE publishes a BED for each
large-scale library (Agarwal et al. 2025, Data availability) whose name column (column 4)
equals the table's ``seq_id``. This script downloads that BED, verifies its md5, joins on
``seq_id`` and writes the table with ``chrom``, ``start``, ``end`` and ``strand`` appended
(BED convention: 0-based start, exclusive end). Row order and row count are unchanged.
Dinucleotide-shuffled controls have no locus and get ``.`` in all four columns; any other
``seq_id`` without coordinates aborts the run.

    python scripts/agarwal2025/0_add_coordinates.py --celltype K562 \\
        --input_tsv data/K562.tsv --output data/K562_with_chrom.tsv
"""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import re
import sys
import urllib.request
from pathlib import Path

ENCODE_URL = "https://www.encodeproject.org/files/{acc}/@@download/{acc}.bed.gz"

# One ENCODE BED per cell type. Columns 1-3 are chrom/start/end, column 4 equals ``seq_id``,
# column 6 is the strand. Coordinates are copied in each file's own assembly. The WTC11
# elements reference (ENCFF508RRB) is BED3 without names, so its quantification file, which
# carries the same names and coordinates, is used instead.
COORDINATES: dict[str, dict[str, str]] = {
    "K562": {"experiment": "ENCSR382BVV", "accession": "ENCFF975XXG", "assembly": "GRCh38",
             "md5": "7f01be3f9da6a0815ddc267a9e45aa47"},
    "HepG2": {"experiment": "ENCSR022GQD", "accession": "ENCFF970GJR", "assembly": "hg19",
              "md5": "27d8cc955e35fcae05821fd586f91f1b"},
    "WTC11": {"experiment": "ENCSR244FWB", "accession": "ENCFF971UBA", "assembly": "GRCh38",
              "md5": "fc4ccde78287261de209efee9a40498e"},
}

NEW_COLUMNS = ("chrom", "start", "end", "strand")
NO_LOCUS = (".",) * len(NEW_COLUMNS)
SYNTHETIC = re.compile(r"shuffled", re.IGNORECASE)  # seq_id pattern of elements with no locus
ORIENTATION = re.compile(r"(_Reversed:|_[FR])$")  # orientation suffix of seq_id


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--celltype", required=True, choices=sorted(COORDINATES))
    parser.add_argument("--input_tsv", type=Path, required=True, help="the human_legnet <celltype>.tsv")
    parser.add_argument("--output", type=Path, required=True, help="table with coordinates to write")
    parser.add_argument("--bed_dir", type=Path, default=None,
                        help="where the ENCODE BED is cached (default: next to --input_tsv)")
    parser.add_argument("--force", action="store_true", help="overwrite an existing output file")
    return parser


def md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(accession: str, dest: Path) -> None:
    url = ENCODE_URL.format(acc=accession)
    print(f"downloading {url} -> {dest}")
    with urllib.request.urlopen(url, timeout=120) as response, dest.open("wb") as out:
        for chunk in iter(lambda: response.read(1 << 20), b""):
            out.write(chunk)


def load_coords(bed: Path) -> dict[str, tuple[str, str, str, str]]:
    """Map BED column 4 (equal to the table's ``seq_id``) to (chrom, start, end, strand)."""

    coords: dict[str, tuple[str, str, str, str]] = {}
    with gzip.open(bed, "rt") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            name, record = fields[3], (fields[0], fields[1], fields[2], fields[5])
            if coords.get(name, record) != record:
                sys.exit(f"{bed.name}: {name!r} listed twice with different coordinates: "
                         f"{coords[name]} vs {record}")
            coords[name] = record
    return coords


def fetch_bed(celltype: str, bed_dir: Path) -> Path:
    spec = COORDINATES[celltype]
    bed = bed_dir / f"{spec['accession']}.bed.gz"
    if not bed.is_file():
        download(spec["accession"], bed)
    got = md5sum(bed)
    if got != spec["md5"]:
        sys.exit(f"{bed}: md5 {got} != expected {spec['md5']} (ENCODE {spec['accession']})")
    print(f"{bed.name}: md5 verified ({spec['assembly']}, ENCODE {spec['experiment']})")
    return bed


def main() -> None:
    args = build_arg_parser().parse_args()
    if not args.input_tsv.is_file():
        sys.exit(f"table not found: {args.input_tsv}")
    if args.output.exists() and not args.force:
        sys.exit(f"{args.output} exists; pass --force to overwrite")

    coords = load_coords(fetch_bed(args.celltype, args.bed_dir or args.input_tsv.parent))

    with args.input_tsv.open() as handle:
        header = handle.readline().rstrip("\n").split("\t")
        if "seq_id" not in header:
            sys.exit(f"{args.input_tsv}: no seq_id column in header {header}")
        clash = [column for column in NEW_COLUMNS if column in header]
        if clash:
            sys.exit(f"{args.input_tsv}: already has column(s) {clash}")
        idx = header.index("seq_id")
        lines = handle.read().splitlines()

    unmatched: collections.Counter[str] = collections.Counter()
    records: list[tuple[str, str, str, str]] = []
    synthetic = 0
    for line in lines:
        seq_id = line.split("\t")[idx]
        record = coords.get(seq_id)
        if record is None and SYNTHETIC.search(seq_id):
            record = NO_LOCUS
            synthetic += 1
        if record is None:
            unmatched[re.sub(r"\d+", "#", seq_id)] += 1
            record = NO_LOCUS
        records.append(record)
    if unmatched:
        sys.exit(f"{sum(unmatched.values())} rows have no coordinates; nothing written. "
                 f"Patterns: {unmatched.most_common(10)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        handle.write("\t".join(header + list(NEW_COLUMNS)) + "\n")
        for line, record in zip(lines, records):
            handle.write(line + "\t" + "\t".join(record) + "\n")

    elements = {ORIENTATION.sub("", line.split("\t")[idx]): record[0] for line, record in zip(lines, records)}
    per_chrom = collections.Counter(chrom for chrom in elements.values() if chrom != NO_LOCUS[0])
    print(f"wrote {args.output}: {len(lines)} rows, {len(elements)} unique elements, "
          f"{len(per_chrom)} contigs, {synthetic} rows with no locus")


if __name__ == "__main__":
    main()
