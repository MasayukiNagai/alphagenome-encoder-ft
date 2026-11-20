"""Reader for the table 0_prepare.py writes, split by chromosome.

Same shape as the ``*Dataset`` readers in :mod:`alphagenome_encoder_ft.data`: one reader per
published table, parsing it and handing the inserts to ``MPRADataset``. Here the insert is
the assayed sequence as the source ships it, and ``EpisomalMPRAGosai2024Library.construct()``
wraps it in the reporter, so the reporter travels with the checkpoint instead of being frozen
into a file.

Rows on ``val_chroms`` form the validation split, rows on ``test_chroms`` the test split, and
all other rows the training split. The defaults are the Malinois holdout (Gosai et al. 2024,
Methods "Data preprocessing"; boda2 ``MPRA_DataModule`` defaults): validation = chr19, 21,
X; test = chr7, 13. Chromosomes are compared by primary name, so ``7`` and ``chr7`` match.

The test split is further restricted to Malinois's reported test set: rows whose
``max_lfc_se`` (largest lfcSE across K562, HepG2 and SK-N-SH) is at most 1 (Methods, "Test set
performance metrics"; 62,582 rows on chr7 and 13). Training and validation rows are not
filtered on standard error, as in Malinois's training loop.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from pathlib import Path

from alphagenome_encoder_ft import MPRADataset, read_tsv_rows
from library import EpisomalMPRAGosai2024Library

MALINOIS_VAL_CHROMS = ("19", "21", "X")
MALINOIS_TEST_CHROMS = ("7", "13")
MALINOIS_TEST_MAX_LFC_SE = 1.0

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


class EpisomalMPRAGosai2024Dataset(MPRADataset):
    """Reader for the Gosai 2024 episomal MPRA table prepared by ``0_prepare.py``.

    Columns are ``seq_id, seq, mean_value, rev, chrom, insert_len, max_lfc_se``. ``seq`` is
    the insert (73-200 bp); ``insert_len`` is annotation the evaluator reads through
    :meth:`read_rows`, never through training. ``max_lfc_se`` selects the test rows.

    ``rev`` is always 0 in this table — the source has no reverse-complement duplicate rows —
    but it is filtered on anyway so the schema matches the other assays' readers.
    """

    LIBRARY = EpisomalMPRAGosai2024Library

    @classmethod
    def read_rows(
        cls,
        input_tsv: str | Path,
        split: str,
        val_chroms: Sequence[str] = MALINOIS_VAL_CHROMS,
        test_chroms: Sequence[str] = MALINOIS_TEST_CHROMS,
    ) -> list[dict[str, str]]:
        """Rows for one split, in file order.

        The evaluator reads the annotation columns through this, so its rows and the
        dataset's items come off the same filter and cannot drift apart.
        """

        if split not in ("train", "val", "test"):
            raise ValueError(f"Unknown split: {split!r}")
        val, test = check_holdout(val_chroms, test_chroms)
        rows = read_tsv_rows(
            input_tsv,
            keep=lambda row: int(row["rev"]) == 0 and split_of(row["chrom"], val, test) == split,
        )
        if split == "test":
            if rows and "max_lfc_se" not in rows[0]:
                raise ValueError(f"{input_tsv} has no max_lfc_se column; re-run 0_prepare.py")
            rows = [row for row in rows if float(row["max_lfc_se"]) <= MALINOIS_TEST_MAX_LFC_SE]
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
        super().__init__(
            [row[sequence_column] for row in rows],
            [float(row[target_column]) for row in rows],
            **kwargs,
        )
