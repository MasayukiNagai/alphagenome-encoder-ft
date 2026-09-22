"""Construct assembly: the fixed reporter context around a variable MPRA insert.

In an MPRA the insert is the only variable part. Everything around it (adapters, minimal
promoter, barcode, vector backbone) is fixed for a given library and therefore a property
of the model trained on it. ``Construct`` holds that fixed part so the same rule builds the
model input at training time and at inference time, and travels with the checkpoint.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from alphagenome_pytorch.utils.sequence import sequence_to_onehot_tensor


def _encode(sequence: str) -> Tensor:
    if not sequence:
        return torch.zeros(0, 4, dtype=torch.float32)
    return sequence_to_onehot_tensor(sequence, dtype=torch.float32)


def _window_start(assembled_length: int, length: int, offset: int) -> int:
    """Start of the ``length``-wide window over an assembled sequence.

    ``trunc`` toward zero puts the odd base on the suffix side in both directions: trimming
    13 -> 10 removes 1 from the left and 2 from the right; padding 10 -> 13 adds 1 on the
    left and 2 on the right. ``offset`` slides the window (positive = toward the suffix).
    """

    return int(math.trunc((assembled_length - length) / 2)) + offset


@dataclass(frozen=True)
class Construct:
    """Fixed flanks around an insert, plus an optional fixed output length.

    ``assemble_*`` returns ``prefix + insert + suffix``. When ``length`` is set the result is
    windowed to exactly ``length`` bases: longer sequences are trimmed from both ends, shorter
    ones padded with ``N`` (all-zero one-hot) on both ends, the odd base going to the suffix
    side. ``offset`` shifts that window and is the training-time jitter; inference leaves it
    at 0.
    """

    prefix: str = ""
    suffix: str = ""
    length: int | None = None

    _prefix_onehot: Tensor = field(init=False, repr=False, compare=False)
    _suffix_onehot: Tensor = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "prefix", str(self.prefix).strip().upper())
        object.__setattr__(self, "suffix", str(self.suffix).strip().upper())
        if self.length is not None:
            if int(self.length) <= 0:
                raise ValueError("length must be > 0 when set")
            object.__setattr__(self, "length", int(self.length))
        object.__setattr__(self, "_prefix_onehot", _encode(self.prefix))
        object.__setattr__(self, "_suffix_onehot", _encode(self.suffix))

    # -------------------------
    # Serialization
    # -------------------------

    def to_dict(self) -> dict[str, Any]:
        return {"prefix": self.prefix, "suffix": self.suffix, "length": self.length}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "Construct | None":
        if data is None:
            return None
        return cls(
            prefix=data.get("prefix", ""),
            suffix=data.get("suffix", ""),
            length=data.get("length"),
        )

    # -------------------------
    # String assembly
    # -------------------------

    def _check_offset(self, offset: int) -> None:
        if offset != 0 and self.length is None:
            raise ValueError("offset requires a Construct with length")

    def assemble_sequence(self, insert: str, *, offset: int = 0) -> str:
        self._check_offset(offset)
        assembled = f"{self.prefix}{insert.strip().upper()}{self.suffix}"
        if self.length is None:
            return assembled

        n = len(assembled)
        start = _window_start(n, self.length, offset)
        end = start + self.length
        left_pad = max(0, -start)
        right_pad = max(0, end - n)
        core = assembled[max(0, start) : min(n, end)]
        return f"{'N' * left_pad}{core}{'N' * right_pad}"

    def assemble_sequences(self, inserts: Iterable[str], *, offset: int = 0) -> list[str]:
        return [self.assemble_sequence(insert, offset=offset) for insert in inserts]

    __call__ = assemble_sequence

    # -------------------------
    # One-hot assembly
    # -------------------------

    def assemble_onehot(self, insert: Tensor, *, offset: int = 0) -> Tensor:
        """Attach one-hot flanks to insert one-hots of shape ``(L, 4)`` or ``(B, L, 4)``.

        The result depends differentiably on ``insert``, so gradients flow back to the
        insert positions for attribution.
        """

        self._check_offset(offset)
        if insert.ndim == 2:
            batch, squeeze = insert.unsqueeze(0), True
        elif insert.ndim == 3:
            batch, squeeze = insert, False
        else:
            raise ValueError(f"Expected rank 2 or 3 input, got rank {insert.ndim}")
        if batch.shape[-1] != 4:
            raise ValueError(f"Expected last dimension 4, got {tuple(batch.shape)}")

        batch_size = batch.shape[0]

        def _expand(flank: Tensor) -> Tensor:
            return flank.to(device=batch.device, dtype=batch.dtype).unsqueeze(0).expand(batch_size, -1, -1)

        pieces: list[Tensor] = []
        if self._prefix_onehot.shape[0]:
            pieces.append(_expand(self._prefix_onehot))
        pieces.append(batch)
        if self._suffix_onehot.shape[0]:
            pieces.append(_expand(self._suffix_onehot))
        assembled = torch.cat(pieces, dim=1)

        if self.length is not None:
            n = assembled.shape[1]
            start = _window_start(n, self.length, offset)
            end = start + self.length
            left_pad = max(0, -start)
            right_pad = max(0, end - n)
            core = assembled[:, max(0, start) : min(n, end)]
            pads = []
            if left_pad:
                pads.append(assembled.new_zeros(batch_size, left_pad, 4))
            pads.append(core)
            if right_pad:
                pads.append(assembled.new_zeros(batch_size, right_pad, 4))
            assembled = torch.cat(pads, dim=1)

        return assembled.squeeze(0) if squeeze else assembled


# -------------------------
# Published libraries
# -------------------------
#
# One class per assayed library: the fixed pieces of its reporter, and the Construct that
# assembles them around an insert. The pieces are specific to the library, not to the assay
# type, so each is keyed by its publication.


class LentiMPRAAgarwal2025Library:
    """Reporter design of the lentiMPRA library in Agarwal et al. 2025 (K562, HepG2, WTC11)."""

    LEFT_ADAPTER = "AGGACCGGATCAACT"
    RIGHT_ADAPTER = "CATTGCGTGAACCGA"
    PROMOTER = "TCCATTATATACCCTCTAGTGTCGGTTCACGCAATG"
    BARCODE = "AGAGACTGAGGCCAC"
    INSERT_LENGTH = 200
    INPUT_LENGTH = 281

    @classmethod
    def construct(cls) -> Construct:
        """``left adapter + insert + right adapter + minP + barcode``."""

        return Construct(
            prefix=cls.LEFT_ADAPTER,
            suffix=cls.RIGHT_ADAPTER + cls.PROMOTER + cls.BARCODE,
            length=cls.INPUT_LENGTH,
        )


class DeepSTARRDeAlmeida2022Library:
    """Reporter design of the Drosophila STARR-seq library in de Almeida et al. 2022."""

    ADAPTER_UP = "TCCCTACACGACGCTCTTCCGATCT"
    ADAPTER_DOWN = "AGATCGGAAGAGCACACGTCTGAACT"
    INPUT_LENGTH = 256

    @classmethod
    def construct(cls) -> Construct:
        """``adapter + insert + adapter``, windowed to 256 bp."""

        return Construct(prefix=cls.ADAPTER_UP, suffix=cls.ADAPTER_DOWN, length=cls.INPUT_LENGTH)
