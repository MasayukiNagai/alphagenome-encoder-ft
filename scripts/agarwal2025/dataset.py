"""Reader for the lentiMPRA table with coordinates, split by chromosome.

Reads ``<celltype>_with_chrom.tsv`` written by ``0_add_coordinates.py``: the human_legnet
columns ``seq_id, seq, mean_value, fold, rev`` plus ``chrom, start, end, strand``. Rows on
``val_chroms`` form the validation split, rows on ``test_chroms`` the test split, and all
other rows the training split. The defaults are the holdout Malinois used on the Gosai 2024
episomal MPRA (validation chr19, 21, X; test chr7, 13), so both libraries share one split.

Chromosomes are compared by primary name: ``7``, ``chr7`` and ``chr7_KI270803v1_alt`` are the
same chromosome. Elements without a locus (``chrom == "."``) go to training.

Only ``rev == 0`` rows are read; reverse complements come from augmentation instead. Adapter
stripping, the insert and the construct are those of ``LentiMPRAAgarwal2025Dataset``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from pathlib import Path

from alphagenome_encoder_ft import (
    LentiMPRAAgarwal2025Dataset,
    MPRADataset,
    read_tsv_rows,
    strip_flanks,
)

MALINOIS_VAL_CHROMS = ("19", "21", "X")
MALINOIS_TEST_CHROMS = ("7", "13")

_PRIMARY = re.compile(r"^(?:chr)?([0-9]+|X|Y|M|MT)(?:_.*)?$", re.IGNORECASE)


def primary_chrom(chrom: str) -> str:
    """``7``, ``chr7`` and ``chr7_KI270803v1_alt`` -> ``7``; anything unrecognised is returned as-is."""

    match = _PRIMARY.match(chrom.strip())
    return match.group(1).upper() if match else chrom.strip()


def check_holdout(val_chroms: Iterable[str], test_chroms: Iterable[str]) -> tuple[frozenset[str], frozenset[str]]:
    """Normalise both lists and refuse an empty list or a chromosome listed in both."""

    val = frozenset(primary_chrom(c) for c in val_chroms)
    test = frozenset(primary_chrom(c) for c in test_chroms)
    if not val or not test:
        raise ValueError("val_chroms and test_chroms must each name at least one chromosome")
    overlap = val & test
    if overlap:
        raise ValueError(f"chromosome(s) in both val_chroms and test_chroms: {sorted(overlap)}")
    return val, test


def split_of(chrom: str, val: frozenset[str], test: frozenset[str]) -> str:
    primary = primary_chrom(chrom)
    if primary in test:
        return "test"
    if primary in val:
        return "val"
    return "train"


class LentiMPRAAgarwal2025ChromDataset(MPRADataset):
    """lentiMPRA reader over ``<celltype>_with_chrom.tsv`` with a chromosome holdout."""

    LIBRARY = LentiMPRAAgarwal2025Dataset.LIBRARY

    @classmethod
    def read_rows(
        cls,
        input_tsv: str | Path,
        split: str,
        val_chroms: Sequence[str] = MALINOIS_VAL_CHROMS,
        test_chroms: Sequence[str] = MALINOIS_TEST_CHROMS,
    ) -> list[dict[str, str]]:
        """``rev == 0`` rows of one split, in file order."""

        if split not in ("train", "val", "test"):
            raise ValueError(f"Unknown split: {split!r}")
        val, test = check_holdout(val_chroms, test_chroms)

        def keep(row: dict[str, str]) -> bool:
            if "chrom" not in row:
                raise ValueError(f"{input_tsv}: no chrom column; run 0_add_coordinates.py first")
            return int(row["rev"]) == 0 and split_of(row["chrom"], val, test) == split

        rows = read_tsv_rows(input_tsv, keep=keep)
        if split != "train":
            wanted = val if split == "val" else test
            missing = wanted - {primary_chrom(row["chrom"]) for row in rows}
            if missing:
                raise ValueError(f"{split} chromosome(s) with no rows in {input_tsv}: {sorted(missing)}")
        return rows

    def __init__(
        self,
        input_tsv: str | Path,
        split: str = "train",
        *,
        val_chroms: Sequence[str] = MALINOIS_VAL_CHROMS,
        test_chroms: Sequence[str] = MALINOIS_TEST_CHROMS,
        sequence_column: str = "seq",
        target_column: str = "mean_value",
        **kwargs,
    ) -> None:
        self.input_tsv = Path(input_tsv)
        self.split = split
        rows = self.read_rows(self.input_tsv, split, val_chroms, test_chroms)
        # rev == 1 rows carry the adapters reverse-complemented at the opposite ends, so
        # stripping runs after the rev filter, as in LentiMPRAAgarwal2025Dataset.
        inserts = strip_flanks(
            [row[sequence_column] for row in rows],
            self.LIBRARY.LEFT_ADAPTER,
            self.LIBRARY.RIGHT_ADAPTER,
            labels=[row.get("seq_id", str(index)) for index, row in enumerate(rows)],
        )
        super().__init__(inserts, [float(row[target_column]) for row in rows], **kwargs)
