"""Reader for the lentiMPRA table, holding out chromosomes or the published folds.

Two splits are in use for this library and they are not interchangeable:

``chrom``
    Rows on ``val_chroms`` are validation, rows on ``test_chroms`` are test, everything
    else trains. The defaults are the holdout Malinois used on the Gosai 2024 episomal
    MPRA (validation chr19, 21, X; test chr7, 13), so both libraries share one split.
    Needs ``<celltype>_with_chrom.tsv`` from ``0_add_coordinates.py``.

``fold``
    The fold assignment Agarwal et al. 2025 ship in the table itself: train on folds
    2-9, validate on fold 1, test on fold 10. This is the split the published
    checkpoints were trained and scored on, so it is the one to use for a number
    comparable with theirs. Needs only the published table. A row whose fold is in no
    split is dropped, which is how a fold is held out of a run entirely.

Which one a run used is recorded in run.json as ``split_by`` and read back by the
evaluator, so a checkpoint is always scored on the rows it was trained against.
**A run.json with no ``split_by`` is a chromosome split**: it predates the fold option.

Chromosomes are compared by primary name: ``7``, ``chr7`` and ``chr7_KI270803v1_alt`` are
the same chromosome. Elements without a locus (``chrom == "."``) go to training.

Only ``rev == 0`` rows are read; reverse complements come from augmentation instead.
Adapter stripping, the insert and the construct are those of ``LentiMPRAAgarwal2025Dataset``.

The merged table ``0_merge_cell_types.py`` writes has one target column per cell type, blank
where the element was not measured in that cell type. ``target_columns`` names the columns to
read; a blank becomes a NaN label, which the default loss and the metrics skip.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from alphagenome_encoder_ft import (
    LentiMPRAAgarwal2025Dataset,
    MPRADataset,
    read_tsv_rows,
    strip_flanks,
)

MALINOIS_VAL_CHROMS = ("19", "21", "X")
MALINOIS_TEST_CHROMS = ("7", "13")

#: The publication's own split, from the package reader, so the two cannot drift apart.
DEFAULT_FOLD_SPLITS = LentiMPRAAgarwal2025Dataset.DEFAULT_FOLD_SPLITS

SPLIT_BY = ("chrom", "fold")
SPLITS = ("train", "val", "test")

#: Command-line flags that mean something under one holdout and nothing under the other.
SPLIT_FLAGS = {
    "chrom": ("val_chroms", "test_chroms"),
    "fold": ("train_folds", "val_folds", "test_folds"),
}


def foreign_split_flags(split_by: str, values: Mapping[str, object]) -> list[str]:
    """The other holdout's flags that were given a value, as ``--name`` strings.

    A caller passing ``--val_chroms`` to a fold run has misunderstood something; the
    scripts stop rather than accept the flag and ignore it.
    """

    other = next(flags for kind, flags in SPLIT_FLAGS.items() if kind != split_by)
    return [f"--{name}" for name in other if values.get(name) is not None]

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


def check_folds(
    train_folds: Iterable[int], val_folds: Iterable[int], test_folds: Iterable[int]
) -> tuple[frozenset[int], frozenset[int], frozenset[int]]:
    """A fold in two splits would leak, and an empty split would train or score on nothing."""

    splits = {name: frozenset(int(f) for f in folds)
              for name, folds in (("train", train_folds), ("val", val_folds), ("test", test_folds))}
    for name, folds in splits.items():
        if not folds:
            raise ValueError(f"{name}_folds is empty")
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = splits[a] & splits[b]
        if shared:
            raise ValueError(f"fold(s) {sorted(shared)} are in both {a} and {b}; that would leak")
    return splits["train"], splits["val"], splits["test"]


@dataclass(frozen=True)
class Splitter:
    """Assigns a table row to train, val, test or no split at all.

    ``kind`` is one of :data:`SPLIT_BY`. ``column`` is the table column it reads, and
    ``hint`` says what to do when that column is absent. ``metadata()`` returns the keys
    to record in run.json, and :meth:`from_run_metadata` reads them back.
    """

    kind: str
    column: str
    hint: str
    assign: Callable[[dict[str, str]], str | None]
    key: Callable[[dict[str, str]], object]
    wanted: Mapping[str, tuple]

    def split_of(self, row: dict[str, str]) -> str | None:
        """The split this row belongs to, or ``None`` if it belongs to none."""

        if self.column not in row:
            raise ValueError(f"no {self.column} column; {self.hint}")
        return self.assign(row)

    def metadata(self) -> dict[str, object]:
        if self.kind == "chrom":
            return {"val_chroms": list(self.wanted["val"]), "test_chroms": list(self.wanted["test"])}
        return {name + "_folds": list(self.wanted[name]) for name in SPLITS}

    @classmethod
    def from_run_metadata(cls, run: Mapping[str, object], **overrides) -> Splitter:
        """Rebuild the splitter a run used.

        A run.json with no ``split_by`` is a chromosome split: the fold option came
        later, so treating a missing key as anything else would silently score an old
        checkpoint on rows it never saw.
        """

        split_by = str(run.get("split_by") or "chrom")
        if split_by not in SPLIT_BY:
            raise ValueError(f"run.json records split_by={split_by!r}; expected one of {list(SPLIT_BY)}")
        if split_by == "chrom":
            val = overrides.get("val_chroms") or run.get("val_chroms")
            test = overrides.get("test_chroms") or run.get("test_chroms")
            if not val or not test:
                raise ValueError("no val_chroms/test_chroms in run.json; pass --val_chroms and --test_chroms")
            return chrom_splitter(val, test)
        folds = {name: overrides.get(f"{name}_folds") or run.get(f"{name}_folds") for name in SPLITS}
        missing = [f"{name}_folds" for name, value in folds.items() if not value]
        if missing:
            raise ValueError(f"no {'/'.join(missing)} in run.json; pass --{' --'.join(missing)}")
        return fold_splitter(**{f"{name}_folds": value for name, value in folds.items()})


def chrom_splitter(
    val_chroms: Sequence[str] = MALINOIS_VAL_CHROMS,
    test_chroms: Sequence[str] = MALINOIS_TEST_CHROMS,
) -> Splitter:
    """Hold out whole chromosomes. Every row lands in a split; an element with no locus trains."""

    val, test = check_holdout(val_chroms, test_chroms)

    def assign(row: dict[str, str]) -> str:
        primary = primary_chrom(row["chrom"])
        if primary in test:
            return "test"
        if primary in val:
            return "val"
        return "train"

    return Splitter(
        kind="chrom",
        column="chrom",
        hint="run 0_add_coordinates.py first, or use --split_by fold",
        assign=assign,
        key=lambda row: primary_chrom(row["chrom"]),
        wanted={"train": (), "val": tuple(sorted(val)), "test": tuple(sorted(test))},
    )


def fold_splitter(
    train_folds: Sequence[int] | None = None,
    val_folds: Sequence[int] | None = None,
    test_folds: Sequence[int] | None = None,
) -> Splitter:
    """Hold out folds. A row whose fold is in no split is dropped, not trained on."""

    train, val, test = check_folds(
        DEFAULT_FOLD_SPLITS["train"] if train_folds is None else train_folds,
        DEFAULT_FOLD_SPLITS["val"] if val_folds is None else val_folds,
        DEFAULT_FOLD_SPLITS["test"] if test_folds is None else test_folds,
    )

    def assign(row: dict[str, str]) -> str | None:
        fold = int(row["fold"])
        if fold in test:
            return "test"
        if fold in val:
            return "val"
        return "train" if fold in train else None

    return Splitter(
        kind="fold",
        column="fold",
        hint="this is not an Agarwal 2025 lentiMPRA table",
        assign=assign,
        key=lambda row: int(row["fold"]),
        wanted={"train": tuple(sorted(train)), "val": tuple(sorted(val)), "test": tuple(sorted(test))},
    )


def make_splitter(split_by: str, **kwargs) -> Splitter:
    """``chrom_splitter`` or ``fold_splitter``, by name, ignoring the other one's arguments."""

    if split_by not in SPLIT_BY:
        raise ValueError(f"Unknown split_by: {split_by!r}; expected one of {list(SPLIT_BY)}")
    if split_by == "chrom":
        return chrom_splitter(
            kwargs.get("val_chroms") or MALINOIS_VAL_CHROMS,
            kwargs.get("test_chroms") or MALINOIS_TEST_CHROMS,
        )
    return fold_splitter(kwargs.get("train_folds"), kwargs.get("val_folds"), kwargs.get("test_folds"))


class LentiMPRAAgarwal2025SplitDataset(MPRADataset):
    """lentiMPRA reader over one split of the table, chosen by a :class:`Splitter`."""

    LIBRARY = LentiMPRAAgarwal2025Dataset.LIBRARY

    @classmethod
    def read_rows(cls, input_tsv: str | Path, split: str, splitter: Splitter) -> list[dict[str, str]]:
        """``rev == 0`` rows of one split, in file order."""

        if split not in SPLITS:
            raise ValueError(f"Unknown split: {split!r}")

        def keep(row: dict[str, str]) -> bool:
            if splitter.column not in row:
                raise ValueError(f"{input_tsv}: no {splitter.column} column; {splitter.hint}")
            return int(row["rev"]) == 0 and splitter.assign(row) == split

        rows = read_tsv_rows(input_tsv, keep=keep)
        if split != "train":
            missing = set(splitter.wanted[split]) - {splitter.key(row) for row in rows}
            if missing:
                raise ValueError(
                    f"{split} {splitter.column}(s) with no rows in {input_tsv}: {sorted(missing, key=str)}"
                )
        return rows

    def __init__(
        self,
        input_tsv: str | Path,
        split: str = "train",
        *,
        splitter: Splitter,
        sequence_column: str = "seq",
        target_columns: Sequence[str] = ("mean_value",),
        target_standardization: Mapping[str, Mapping[str, float]] | None = None,
        **kwargs,
    ) -> None:
        self.input_tsv = Path(input_tsv)
        self.split = split
        self.splitter = splitter
        self.target_columns = tuple(target_columns)
        rows = self.read_rows(self.input_tsv, split, splitter)
        # rev == 1 rows carry the adapters reverse-complemented at the opposite ends, so
        # stripping runs after the rev filter, as in LentiMPRAAgarwal2025Dataset.
        inserts = strip_flanks(
            [row[sequence_column] for row in rows],
            self.LIBRARY.LEFT_ADAPTER,
            self.LIBRARY.RIGHT_ADAPTER,
            labels=[row.get("seq_id", str(index)) for index, row in enumerate(rows)],
        )
        targets = read_targets(rows, self.target_columns, target_standardization)
        super().__init__(inserts, targets, **kwargs)


def _target_value(text: str) -> float:
    """A blank or NA cell is a cell type the element was not measured in: a NaN label."""

    return float("nan") if text.strip() in ("", "NA", "NaN", "nan") else float(text)


def read_targets(
    rows: Sequence[dict[str, str]],
    target_columns: Sequence[str],
    standardization: Mapping[str, Mapping[str, float]] | None = None,
) -> list[float] | list[list[float]]:
    """One value per row for a single column (a single-output head), else one list per row.

    ``standardization`` maps a column to its ``mean`` and ``sd``; those columns are returned
    as ``(value - mean) / sd``.
    """

    if not target_columns:
        raise ValueError("target_columns must name at least one column")
    if rows and (missing := [column for column in target_columns if column not in rows[0]]):
        raise ValueError(f"no column(s) {missing} in the table")
    scale = {column: (0.0, 1.0) for column in target_columns}
    for column, stats in (standardization or {}).items():
        scale[column] = (float(stats["mean"]), float(stats["sd"]))
    values = [
        [(_target_value(row[column]) - scale[column][0]) / scale[column][1] for column in target_columns]
        for row in rows
    ]
    return [value[0] for value in values] if len(target_columns) == 1 else values


def target_standardization(
    input_tsv: str | Path,
    target_columns: Sequence[str],
    splitter: Splitter,
) -> dict[str, dict[str, float]]:
    """Mean and SD (ddof=0) of each target column over the labelled training rows.

    The statistics come from the training rows of *this* split, so switching to a
    different holdout restandardizes rather than reusing another split's numbers.
    """

    rows = LentiMPRAAgarwal2025SplitDataset.read_rows(input_tsv, "train", splitter)
    stats = {}
    for column in target_columns:
        values = np.array([_target_value(row[column]) for row in rows], dtype=np.float64)
        values = values[~np.isnan(values)]
        if values.size < 2 or values.std() == 0:
            raise ValueError(f"cannot standardize {column!r}: fewer than two distinct training labels")
        stats[column] = {"mean": float(values.mean()), "sd": float(values.std()), "n": int(values.size)}
    return stats
