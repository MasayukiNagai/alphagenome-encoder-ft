"""The agarwal2025 split selection: hold out chromosomes, or the published folds.

The scripts under ``scripts/agarwal2025/`` are not importable as a package (``1_train.py``
is not an identifier), so ``dataset.py`` is loaded by path. All of the split logic lives
there; the two scripts only parse flags and hand it a splitter.
"""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest

from alphagenome_encoder_ft import LentiMPRAAgarwal2025Library as AGARWAL

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "agarwal2025"


def _load_dataset_module():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("agarwal2025_dataset", SCRIPTS / "dataset.py")
    module = importlib.util.module_from_spec(spec)
    # Registered before execution: the module uses postponed annotations, and @dataclass
    # resolves them through sys.modules[cls.__module__].
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


dataset = _load_dataset_module()

# chrom / fold are chosen so the two holdouts disagree about several rows: the last row
# trains under the fold split and is held out under the chromosome split.
ROWS = [
    {"seq": "AAAA", "rev": 0, "fold": 2, "mean_value": 1.0, "chrom": "chr1"},
    {"seq": "CCCC", "rev": 0, "fold": 1, "mean_value": 2.0, "chrom": "chr19"},
    {"seq": "GGGG", "rev": 0, "fold": 10, "mean_value": 3.0, "chrom": "chr7"},
    {"seq": "TTTT", "rev": 1, "fold": 10, "mean_value": 4.0, "chrom": "chr7"},
    {"seq": "ACAC", "rev": 0, "fold": 3, "mean_value": 5.0, "chrom": "."},
    {"seq": "AGAG", "rev": 0, "fold": 11, "mean_value": 6.0, "chrom": "chr13"},
    {"seq": "ATAT", "rev": 0, "fold": 2, "mean_value": 10.0, "chrom": "chr7"},
]

VAL_CHROMS = ["19"]
TEST_CHROMS = ["7", "13"]
TRAIN_FOLDS, VAL_FOLDS, TEST_FOLDS = [2, 3], [1], [10]


def _write(path: Path, rows: list[dict], fieldnames: list[str]) -> Path:
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "seq": AGARWAL.LEFT_ADAPTER + row["seq"] + AGARWAL.RIGHT_ADAPTER})
    return path


@pytest.fixture
def table(tmp_path: Path) -> Path:
    return _write(tmp_path / "K562_with_chrom.tsv", ROWS, ["seq", "rev", "fold", "mean_value", "chrom"])


@pytest.fixture
def published_table(tmp_path: Path) -> Path:
    """The table as published: no coordinates, so no chrom column."""

    rows = [{key: value for key, value in row.items() if key != "chrom"} for row in ROWS]
    return _write(tmp_path / "K562.tsv", rows, ["seq", "rev", "fold", "mean_value"])


def _inserts(table: Path, split: str, splitter) -> list[str]:
    return dataset.LentiMPRAAgarwal2025SplitDataset(table, split=split, splitter=splitter).inserts


def _chrom():
    return dataset.chrom_splitter(VAL_CHROMS, TEST_CHROMS)


def _fold():
    return dataset.fold_splitter(TRAIN_FOLDS, VAL_FOLDS, TEST_FOLDS)


def test_chromosome_split_selects_rows_by_locus(table: Path):
    # An element with no locus (chrom ".") trains; the rev == 1 row is never read.
    assert _inserts(table, "train", _chrom()) == ["AAAA", "ACAC"]
    assert _inserts(table, "val", _chrom()) == ["CCCC"]
    assert _inserts(table, "test", _chrom()) == ["GGGG", "AGAG", "ATAT"]


def test_fold_split_selects_rows_by_fold(table: Path):
    assert _inserts(table, "train", _fold()) == ["AAAA", "ACAC", "ATAT"]
    assert _inserts(table, "val", _fold()) == ["CCCC"]
    assert _inserts(table, "test", _fold()) == ["GGGG"]


def test_the_two_holdouts_disagree(table: Path):
    """The point of recording split_by: the same row trains under one and is held out
    under the other, so scoring a checkpoint with the wrong one leaks."""

    assert "ATAT" in _inserts(table, "train", _fold())
    assert "ATAT" in _inserts(table, "test", _chrom())


def test_a_fold_in_no_split_is_dropped(table: Path):
    """Fold 11 is in none of the three lists, so it is held out of the run entirely."""

    everywhere = set().union(*(_inserts(table, split, _fold()) for split in ("train", "val", "test")))
    assert "AGAG" not in everywhere


def test_the_fold_split_needs_no_coordinates(published_table: Path):
    assert _inserts(published_table, "test", _fold()) == ["GGGG"]


def test_a_missing_chrom_column_names_the_fix(published_table: Path):
    with pytest.raises(ValueError, match="0_add_coordinates"):
        _inserts(published_table, "train", _chrom())


def test_a_held_out_chromosome_with_no_rows_is_an_error(table: Path):
    with pytest.raises(ValueError, match="no rows"):
        _inserts(table, "val", dataset.chrom_splitter(["21"], TEST_CHROMS))


def test_a_held_out_fold_with_no_rows_is_an_error(table: Path):
    with pytest.raises(ValueError, match="no rows"):
        _inserts(table, "test", dataset.fold_splitter(TRAIN_FOLDS, VAL_FOLDS, [9]))


def test_run_metadata_without_split_by_is_a_chromosome_split():
    """Every run.json written before the fold option lacks the key. Reading it as
    anything else would score an old checkpoint on rows it never saw."""

    splitter = dataset.Splitter.from_run_metadata({"val_chroms": ["19"], "test_chroms": ["7"]})
    assert splitter.kind == "chrom"


@pytest.mark.parametrize("make", [_chrom, _fold])
def test_metadata_round_trips_through_run_json(make):
    splitter = make()
    run = {"split_by": splitter.kind, **splitter.metadata()}
    assert dataset.Splitter.from_run_metadata(run).metadata() == splitter.metadata()


def test_run_metadata_rejects_an_unknown_split_by():
    with pytest.raises(ValueError, match="split_by"):
        dataset.Splitter.from_run_metadata({"split_by": "random"})


def test_run_metadata_reports_missing_folds():
    with pytest.raises(ValueError, match="test_folds"):
        dataset.Splitter.from_run_metadata({"split_by": "fold", "train_folds": [2], "val_folds": [1]})


def test_check_folds_refuses_a_leak_and_an_empty_split():
    with pytest.raises(ValueError, match="would leak"):
        dataset.check_folds([1, 2], [1], [10])
    with pytest.raises(ValueError, match="val_folds is empty"):
        dataset.check_folds([2], [], [10])


def test_the_other_holdouts_flags_are_caught():
    """Accepting --val_chroms on a fold run and ignoring it would train on a split the
    caller did not ask for."""

    assert dataset.foreign_split_flags("fold", {"val_chroms": ["19"], "train_folds": [2]}) == ["--val_chroms"]
    assert dataset.foreign_split_flags("chrom", {"test_folds": [10]}) == ["--test_folds"]
    assert dataset.foreign_split_flags("fold", {"train_folds": [2], "val_chroms": None}) == []


def test_standardization_follows_the_runs_own_training_rows(table: Path):
    """The two holdouts have different training rows, so they must not share statistics."""

    by_chrom = dataset.target_standardization(table, ["mean_value"], _chrom())["mean_value"]
    by_fold = dataset.target_standardization(table, ["mean_value"], _fold())["mean_value"]

    assert by_chrom["n"] == 2 and by_chrom["mean"] == pytest.approx(3.0)
    assert by_fold["n"] == 3 and by_fold["mean"] == pytest.approx(16.0 / 3)
