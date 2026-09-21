"""Construct assembly: window rule, jitter offset, one-hot equivalence, serialization."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from alphagenome_pytorch.utils.sequence import sequence_to_onehot

from alphagenome_encoder_ft.constructs import (
    LENTIMPRA_BARCODE,
    LENTIMPRA_LEFT_ADAPTER,
    LENTIMPRA_RIGHT_ADAPTER,
    Construct,
    deepstarr_construct,
    lentimpra_construct,
    lentimpra_promoter_barcode_construct,
)


def _onehot(sequence: str) -> torch.Tensor:
    return torch.from_numpy(sequence_to_onehot(sequence).astype(np.float32))


def test_no_length_concatenates_flanks():
    construct = Construct(prefix="AA", suffix="TTT")
    assert construct.assemble_sequence("cg") == "AACGTTT"
    assert construct.assemble_sequences(["cg", "gc"]) == ["AACGTTT", "AAGCTTT"]
    # __call__ is the same method.
    assert construct("cg") == construct.assemble_sequence("cg")


def test_no_length_rejects_offset():
    with pytest.raises(ValueError, match="offset requires"):
        Construct(prefix="A").assemble_sequence("CG", offset=1)


def test_trim_puts_the_odd_base_on_the_suffix_side():
    # assembled = A + CCCCCCCCCCC + G = 13 bases, windowed to 10: 1 off the left, 2 off the right.
    construct = Construct(prefix="A", suffix="G", length=10)
    assert construct.assemble_sequence("C" * 11) == "CCCCCCCCCC"

    # even trim: 12 -> 10 removes 1 from each end.
    construct = Construct(prefix="AA", suffix="GG", length=10)
    assert construct.assemble_sequence("C" * 8) == "ACCCCCCCCG"


def test_pad_puts_the_odd_base_on_the_suffix_side():
    # assembled = A + CC + G = 4 bases, windowed to 7: 1 N on the left, 2 on the right.
    construct = Construct(prefix="A", suffix="G", length=7)
    assert construct.assemble_sequence("cc") == "NACCGNN"

    # even pad: 4 -> 6 adds 1 N on each side.
    assert Construct(prefix="A", suffix="G", length=6).assemble_sequence("cc") == "NACCGN"


def test_offset_slides_the_window():
    construct = Construct(prefix="AAA", suffix="GGG", length=6)
    # assembled = AAACCCCGGG (10), centered window starts at 2.
    assert construct.assemble_sequence("CCCC") == "ACCCCG"
    assert construct.assemble_sequence("CCCC", offset=2) == "CCCGGG"
    assert construct.assemble_sequence("CCCC", offset=-2) == "AAACCC"


def test_offset_at_exact_fit_drops_one_side_and_pads_the_other():
    construct = Construct(prefix="A", suffix="G", length=4)
    assert construct.assemble_sequence("CC") == "ACCG"
    assert construct.assemble_sequence("CC", offset=1) == "CCGN"
    assert construct.assemble_sequence("CC", offset=-1) == "NACC"


@pytest.mark.parametrize("offset", [-2, 0, 1])
@pytest.mark.parametrize(
    "construct",
    [
        Construct(prefix="AG", suffix="TC", length=8),
        Construct(prefix="AG", suffix="TC", length=4),
        Construct(suffix="TCTC", length=6),
        Construct(prefix="AG"),
    ],
    ids=["pad", "trim", "suffix_only", "no_length"],
)
def test_assemble_onehot_matches_assemble_sequence(construct: Construct, offset: int):
    if construct.length is None and offset != 0:
        pytest.skip("offset needs a length")
    insert = "ACGT"
    expected = _onehot(construct.assemble_sequence(insert, offset=offset))
    got = construct.assemble_onehot(_onehot(insert), offset=offset)
    torch.testing.assert_close(got, expected)


def test_assemble_onehot_handles_batches():
    construct = Construct(prefix="A", suffix="G", length=6)
    batch = torch.stack([_onehot("ACGT"), _onehot("TTTT")], dim=0)
    got = construct.assemble_onehot(batch)
    assert got.shape == (2, 6, 4)
    torch.testing.assert_close(got[0], construct.assemble_onehot(_onehot("ACGT")))
    torch.testing.assert_close(got[1], construct.assemble_onehot(_onehot("TTTT")))


def test_assemble_onehot_is_differentiable_through_the_insert():
    construct = Construct(prefix="AAA", suffix="GGG", length=8)
    insert = _onehot("ACGT").unsqueeze(0).requires_grad_(True)
    construct.assemble_onehot(insert).sum().backward()
    assert insert.grad is not None
    assert insert.grad.shape == insert.shape
    assert torch.count_nonzero(insert.grad) > 0


def test_assemble_onehot_rejects_bad_shapes():
    construct = Construct(length=4)
    with pytest.raises(ValueError, match="rank 2 or 3"):
        construct.assemble_onehot(torch.zeros(4))
    with pytest.raises(ValueError, match="last dimension 4"):
        construct.assemble_onehot(torch.zeros(4, 5))


def test_flanks_are_normalized_and_length_validated():
    construct = Construct(prefix=" ac ", suffix="gt")
    assert (construct.prefix, construct.suffix) == ("AC", "GT")
    with pytest.raises(ValueError, match="length must be > 0"):
        Construct(length=0)


def test_to_dict_from_dict_roundtrip():
    construct = Construct(prefix="AC", suffix="GT", length=12)
    assert Construct.from_dict(construct.to_dict()) == construct
    assert Construct.from_dict(None) is None
    assert Construct.from_dict({}) == Construct()


def test_lentimpra_construct_wraps_a_bare_element_in_the_whole_reporter():
    construct = lentimpra_construct()
    assembled = construct.assemble_sequence("A" * 200)

    assert construct.length == 281
    # 15 + 200 + 15 + 36 + 15 = 281, so nothing is trimmed or padded.
    assert len(assembled) == 281
    assert "N" not in assembled
    assert assembled.startswith(LENTIMPRA_LEFT_ADAPTER)
    assert assembled.endswith(LENTIMPRA_BARCODE)


def test_promoter_barcode_construct_adds_no_adapters():
    construct = lentimpra_promoter_barcode_construct()
    seq = "A" * 230  # an Agarwal row: adapters already inline
    assembled = construct.assemble_sequence(seq)

    assert construct.prefix == ""
    assert construct.length == 281
    # 230 + 36 + 15 = 281.
    assert len(assembled) == 281
    assert "N" not in assembled
    assert assembled.startswith(seq)


def test_the_two_lentimpra_presets_agree_once_the_adapters_are_inline():
    """Wrapping a bare element must equal appending to the same element with adapters."""

    element = "ACGT" * 50  # 200 bp
    with_adapters = LENTIMPRA_LEFT_ADAPTER + element + LENTIMPRA_RIGHT_ADAPTER

    assert lentimpra_construct().assemble_sequence(element) == (
        lentimpra_promoter_barcode_construct().assemble_sequence(with_adapters)
    )


def test_deepstarr_construct_layout():
    construct = deepstarr_construct()
    assert construct.length == 256
    assert len(construct.assemble_sequence("A" * 249)) == 256
