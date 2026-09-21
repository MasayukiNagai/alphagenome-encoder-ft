"""Dataset base (construct + augmentation) and the per-assay readers."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import torch
from alphagenome_pytorch.utils.sequence import onehot_to_sequence

from alphagenome_encoder_ft.constructs import (
    LENTIMPRA_LEFT_ADAPTER,
    LENTIMPRA_RIGHT_ADAPTER,
    Construct,
    lentimpra_construct,
)
from alphagenome_encoder_ft.data import (
    DeepSTARRDataset,
    LentiMPRADataset,
    MPRADataset,
    strip_flanks,
)


def _decode(item: torch.Tensor) -> str:
    return onehot_to_sequence(item.numpy())


def _write_tsv(path: Path, fieldnames: list[str], rows: list[dict]) -> Path:
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return path


# -------------------------
# Base
# -------------------------


def test_base_applies_the_construct():
    # A + CC + GT is 5 bases; padding to 6 adds the odd base on the suffix side.
    ds = MPRADataset(["cc"], [1.0], construct=Construct(prefix="A", suffix="GT", length=6))
    onehot, target = ds[0]
    assert _decode(onehot) == "ACCGTN"
    assert onehot.shape == (6, 4)
    assert target.item() == pytest.approx(1.0)


def test_base_passes_sequences_through_without_a_construct():
    ds = MPRADataset(["acgt"], [2.0])
    assert _decode(ds[0][0]) == "ACGT"


def test_base_supports_multi_output_targets():
    ds = MPRADataset(["AC", "GT"], [[1.0, 2.0], [3.0, 4.0]])
    assert ds[1][1].shape == (2,)
    torch.testing.assert_close(ds[1][1], torch.tensor([3.0, 4.0]))


def test_base_rejects_mismatched_targets():
    with pytest.raises(ValueError, match="targets has"):
        MPRADataset(["AC", "GT"], [1.0])


def test_random_shift_requires_a_construct_length():
    with pytest.raises(ValueError, match="random_shift requires"):
        MPRADataset(["AC"], [1.0], random_shift=True)
    with pytest.raises(ValueError, match="random_shift requires"):
        MPRADataset(["AC"], [1.0], construct=Construct(prefix="A"), random_shift=True)


def test_jitter_slides_the_window_and_keeps_the_length():
    construct = Construct(prefix="A" * 10, suffix="G" * 10, length=20)
    ds = MPRADataset(
        ["C" * 4] * 50,
        [0.0] * 50,
        construct=construct,
        random_shift=True,
        shift_prob=1.0,
        max_shift=5,
        seed=0,
    )
    sequences = {_decode(ds[i][0]) for i in range(len(ds))}
    assert all(len(seq) == 20 for seq in sequences)
    # With shift_prob=1 and max_shift=5 the window lands in several distinct places.
    assert len(sequences) > 1
    # Every window still contains part of the insert; none is pure flank.
    assert all("C" in seq for seq in sequences)


def test_no_jitter_is_deterministic():
    # assembled = AAACCCCGGG (10) windowed to 8 -> drop 1 left, 1 right.
    ds = MPRADataset(["CCCC"] * 5, [0.0] * 5, construct=Construct(prefix="AAA", suffix="GGG", length=8))
    assert {_decode(ds[i][0]) for i in range(len(ds))} == {"AACCCCGG"}


def test_reverse_complement_applies_to_the_whole_assembled_sequence():
    construct = Construct(prefix="AA", suffix="GG", length=8)
    ds = MPRADataset(
        ["CCCC"] * 20,
        [0.0] * 20,
        construct=construct,
        reverse_complement=True,
        rc_prob=1.0,
        seed=0,
    )
    # forward is AACCCCGG; its reverse complement is CCGGGGTT.
    assert {_decode(ds[i][0]) for i in range(len(ds))} == {"CCGGGGTT"}


def test_subset_frac_samples_rows_and_keeps_targets_aligned():
    inserts = ["AA", "AC", "AG", "AT", "CA", "CC", "CG", "CT", "GA", "GC"]
    ds = MPRADataset(inserts, [float(i) for i in range(len(inserts))], subset_frac=0.5, seed=1)

    assert len(ds) == 5
    for index in range(len(ds)):
        insert = _decode(ds[index][0])
        assert insert == ds.inserts[index]
        # each sampled row keeps the target it had before sampling
        assert ds[index][1].item() == pytest.approx(float(inserts.index(insert)))


def test_validation_of_numeric_arguments():
    with pytest.raises(ValueError, match="subset_frac"):
        MPRADataset(["AC"], [1.0], subset_frac=0.0)
    with pytest.raises(ValueError, match="rc_prob"):
        MPRADataset(["AC"], [1.0], rc_prob=1.5)
    with pytest.raises(ValueError, match="shift_prob"):
        MPRADataset(["AC"], [1.0], shift_prob=-0.1)
    with pytest.raises(ValueError, match="max_shift"):
        MPRADataset(["AC"], [1.0], max_shift=-1)


# -------------------------
# Readers
# -------------------------


def _published(element: str) -> str:
    """A seq column value: the element between the two cloning adapters."""

    return LENTIMPRA_LEFT_ADAPTER + element + LENTIMPRA_RIGHT_ADAPTER


@pytest.fixture
def lentimpra_tsv(tmp_path: Path) -> Path:
    return _write_tsv(
        tmp_path / "K562.tsv",
        ["seq", "rev", "fold", "mean_value"],
        [
            {"seq": _published("AC"), "rev": 0, "fold": 2, "mean_value": 1.0},
            {"seq": _published("GT"), "rev": 0, "fold": 1, "mean_value": 2.0},
            {"seq": _published("AA"), "rev": 1, "fold": 10, "mean_value": 3.0},
            {"seq": _published("CC"), "rev": 0, "fold": 10, "mean_value": 4.0},
        ],
    )


def test_lentimpra_reader_filters_split_and_reverse_rows(lentimpra_tsv: Path):
    assert len(LentiMPRADataset(lentimpra_tsv, split="train")) == 1
    assert len(LentiMPRADataset(lentimpra_tsv, split="val")) == 1
    # fold 10 has two rows but one is rev == 1.
    test_ds = LentiMPRADataset(lentimpra_tsv, split="test")
    assert len(test_ds) == 1
    assert test_ds.inserts == ["CC"]


def test_lentimpra_reader_honours_custom_folds(lentimpra_tsv: Path):
    ds = LentiMPRADataset(lentimpra_tsv, split="train", train_folds=[1, 2])
    assert sorted(ds.inserts) == ["AC", "GT"]


def test_lentimpra_reader_passes_the_construct_through(lentimpra_tsv: Path):
    ds = LentiMPRADataset(lentimpra_tsv, split="test", construct=Construct(suffix="GG", length=4))
    assert _decode(ds[0][0]) == "CCGG"


def test_lentimpra_reader_rejects_an_unknown_split(lentimpra_tsv: Path):
    with pytest.raises(ValueError, match="Unknown split"):
        LentiMPRADataset(lentimpra_tsv, split="holdout")


def test_reader_reports_a_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Dataset file not found"):
        LentiMPRADataset(tmp_path / "absent.tsv", split="train")


@pytest.fixture
def deepstarr_tsv(tmp_path: Path) -> Path:
    return _write_tsv(
        tmp_path / "deepstarr.tsv",
        ["sequence", "set", "Dev_log2_enrichment", "Hk_log2_enrichment"],
        [
            {"sequence": "AC", "set": "train", "Dev_log2_enrichment": 1.0, "Hk_log2_enrichment": 2.0},
            {"sequence": "GT", "set": "test", "Dev_log2_enrichment": 3.0, "Hk_log2_enrichment": 4.0},
        ],
    )


def test_deepstarr_reader_selects_the_split_and_two_targets(deepstarr_tsv: Path):
    ds = DeepSTARRDataset(deepstarr_tsv, split="test")
    assert len(ds) == 1
    onehot, target = ds[0]
    assert _decode(onehot) == "GT"
    torch.testing.assert_close(target, torch.tensor([3.0, 4.0]))


def test_deepstarr_reader_applies_the_construct(deepstarr_tsv: Path):
    ds = DeepSTARRDataset(deepstarr_tsv, split="train", construct=Construct(prefix="TT", suffix="AA", length=6))
    assert _decode(ds[0][0]) == "TTACAA"


def test_deepstarr_reader_rejects_empty_target_columns(deepstarr_tsv: Path):
    with pytest.raises(ValueError, match="target_columns"):
        DeepSTARRDataset(deepstarr_tsv, split="train", target_columns=())


# -------------------------
# Adapter stripping
# -------------------------


ELEMENT = "ACGT" * 50  # 200 bp, as in the published tables


@pytest.fixture
def adapter_tsv(tmp_path: Path) -> Path:
    """A miniature Agarwal-style table: seq is adapter + element + adapter, both strands."""

    forward = LENTIMPRA_LEFT_ADAPTER + ELEMENT + LENTIMPRA_RIGHT_ADAPTER
    reverse = _reverse_complement(forward)
    return _write_tsv(
        tmp_path / "adapters.tsv",
        ["seq_id", "seq", "rev", "fold", "mean_value"],
        [
            {"seq_id": "peak1", "seq": forward, "rev": 0, "fold": 10, "mean_value": 1.0},
            # the rev == 1 partner carries each adapter's reverse complement at the other end
            {"seq_id": "peak1_Reversed:", "seq": reverse, "rev": 1, "fold": 10, "mean_value": 1.0},
        ],
    )


def _reverse_complement(sequence: str) -> str:
    return sequence[::-1].translate(str.maketrans("ACGT", "TGCA"))


def test_the_insert_is_the_element(adapter_tsv: Path):
    ds = LentiMPRADataset(adapter_tsv, split="test")
    assert ds.inserts == [ELEMENT]
    assert len(ds.inserts[0]) == 200


def test_the_reverse_complement_rows_are_dropped_before_stripping(adapter_tsv: Path):
    """They carry each adapter's reverse complement at the opposite end."""

    assert len(LentiMPRADataset(adapter_tsv, split="test")) == 1


def test_the_construct_rebuilds_the_published_sequence(adapter_tsv: Path):
    ds = LentiMPRADataset(adapter_tsv, split="test", construct=lentimpra_construct())
    onehot = ds[0][0]

    assert onehot.shape == (281, 4)
    # the first 230 bp are the seq column as published
    assert _decode(onehot)[:230] == LENTIMPRA_LEFT_ADAPTER + ELEMENT + LENTIMPRA_RIGHT_ADAPTER


def test_a_file_without_adapters_fails_loudly(tmp_path: Path):
    path = _write_tsv(
        tmp_path / "bare.tsv",
        ["seq", "rev", "fold", "mean_value"],
        [{"seq": "AC", "rev": 0, "fold": 10, "mean_value": 1.0}],
    )
    with pytest.raises(ValueError, match="too short to carry"):
        LentiMPRADataset(path, split="test")


def test_a_row_with_wrong_flanks_is_named(tmp_path: Path):
    path = _write_tsv(
        tmp_path / "mixed.tsv",
        ["seq_id", "seq", "rev", "fold", "mean_value"],
        [
            {
                "seq_id": "oddball",
                "seq": "T" * 15 + ELEMENT + LENTIMPRA_RIGHT_ADAPTER,
                "rev": 0,
                "fold": 10,
                "mean_value": 1.0,
            }
        ],
    )
    with pytest.raises(ValueError, match="oddball"):
        LentiMPRADataset(path, split="test")


def test_strip_flanks_is_reusable_on_plain_sequences():
    assert strip_flanks(["AAtttGG"], "AA", "GG") == ["TTT"]
    with pytest.raises(ValueError, match="does not carry the expected flanks"):
        strip_flanks(["CCtttGG"], "AA", "GG")
