"""Construct assembly utilities for MPRA insert sequences."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import torch

from alphagenome_pytorch.utils.sequence import sequence_to_onehot_tensor

LENTIMPRA_LEFT_ADAPTER = "AGGACCGGATCAACT"
LENTIMPRA_RIGHT_ADAPTER = "CATTGCGTGAACCGA"
LENTIMPRA_PROMOTER = "TCCATTATATACCCTCTAGTGTCGGTTCACGCAATG"
LENTIMPRA_BARCODE = "AGAGACTGAGGCCAC"


@dataclass(frozen=True)
class ConstructSpec:
    """Reusable construct assembly rules for MPRA insert sequences."""

    left_adapter: str | None = LENTIMPRA_LEFT_ADAPTER
    right_adapter: str | None = LENTIMPRA_RIGHT_ADAPTER
    promoter_seq: str | None = LENTIMPRA_PROMOTER
    barcode_seq: str | None = LENTIMPRA_BARCODE
    _left_adapter_onehot: torch.Tensor | None = field(init=False, repr=False, default=None)
    _right_adapter_onehot: torch.Tensor | None = field(init=False, repr=False, default=None)
    _promoter_onehot: torch.Tensor | None = field(init=False, repr=False, default=None)
    _barcode_onehot: torch.Tensor | None = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_left_adapter_onehot", self._encode_constant(self.left_adapter))
        object.__setattr__(self, "_right_adapter_onehot", self._encode_constant(self.right_adapter))
        object.__setattr__(self, "_promoter_onehot", self._encode_constant(self.promoter_seq))
        object.__setattr__(self, "_barcode_onehot", self._encode_constant(self.barcode_seq))

    @classmethod
    def lentimpra_default(cls) -> "ConstructSpec":
        return cls(
            left_adapter=LENTIMPRA_LEFT_ADAPTER,
            right_adapter=LENTIMPRA_RIGHT_ADAPTER,
            promoter_seq=LENTIMPRA_PROMOTER,
            barcode_seq=LENTIMPRA_BARCODE,
        )

    @staticmethod
    def _encode_constant(sequence: str | None) -> torch.Tensor | None:
        if sequence is None:
            return None
        return sequence_to_onehot_tensor(sequence, dtype=torch.float32)

    @staticmethod
    def validate_mode(mode: str) -> str:
        normalized = mode.lower()
        if normalized not in {"core", "flanked", "full"}:
            raise ValueError(f"Invalid mode {mode!r}. Must be one of: core, flanked, full")
        return normalized

    def _validate_required_components(self, mode: str) -> None:
        missing: list[str] = []
        if mode == "core":
            if self.left_adapter is None:
                missing.append("left_adapter")
            if self.right_adapter is None:
                missing.append("right_adapter")
            if self.promoter_seq is None:
                missing.append("promoter_seq")
            if self.barcode_seq is None:
                missing.append("barcode_seq")
        elif mode == "flanked":
            if self.promoter_seq is None:
                missing.append("promoter_seq")
            if self.barcode_seq is None:
                missing.append("barcode_seq")

        if missing:
            raise ValueError(
                f"Mode {mode!r} requires construct components that are missing: {', '.join(missing)}"
            )

    @staticmethod
    def _normalize_insert_sequence(insert_seq: str) -> str:
        return insert_seq.strip().upper()

    def assemble_sequence(self, insert_seq: str, mode: str = "core") -> str:
        normalized_mode = self.validate_mode(mode)
        self._validate_required_components(normalized_mode)
        normalized_insert = self._normalize_insert_sequence(insert_seq)
        parts: list[str] = []
        if normalized_mode == "core":
            parts.append(self.left_adapter)
            parts.append(normalized_insert)
            parts.append(self.right_adapter)
            parts.append(self.promoter_seq)
            parts.append(self.barcode_seq)
            return "".join(parts)
        if normalized_mode == "flanked":
            parts.append(normalized_insert)
            parts.append(self.promoter_seq)
            parts.append(self.barcode_seq)
            return "".join(parts)
        return normalized_insert

    def assemble_sequences(self, insert_seqs: Iterable[str], mode: str = "core") -> list[str]:
        normalized_mode = self.validate_mode(mode)
        self._validate_required_components(normalized_mode)
        return [self.assemble_sequence(insert_seq, mode=normalized_mode) for insert_seq in insert_seqs]

    @staticmethod
    def _normalize_onehot(onehot: torch.Tensor) -> tuple[torch.Tensor, bool]:
        if onehot.ndim == 2:
            if onehot.shape[-1] != 4:
                raise ValueError(f"Expected shape (L, 4), got {tuple(onehot.shape)}")
            return onehot.unsqueeze(0), True
        if onehot.ndim == 3:
            if onehot.shape[-1] != 4:
                raise ValueError(f"Expected shape (B, L, 4), got {tuple(onehot.shape)}")
            return onehot, False
        raise ValueError(f"Expected rank 2 or 3 onehot input, got rank {onehot.ndim}")

    @staticmethod
    def _expand_piece(
        piece: torch.Tensor | None,
        *,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor | None:
        if piece is None:
            return None
        return piece.to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)

    def assemble_onehot(self, onehot: torch.Tensor, mode: str = "core") -> torch.Tensor:
        normalized_mode = self.validate_mode(mode)
        self._validate_required_components(normalized_mode)
        batch, squeeze_output = self._normalize_onehot(onehot)
        pieces: list[torch.Tensor] = []
        batch_size = batch.shape[0]
        kwargs = {"batch_size": batch_size, "dtype": batch.dtype, "device": batch.device}

        if normalized_mode == "core":
            left_onehot = self._expand_piece(self._left_adapter_onehot, **kwargs)
            right_onehot = self._expand_piece(self._right_adapter_onehot, **kwargs)
            promoter_onehot = self._expand_piece(self._promoter_onehot, **kwargs)
            barcode_onehot = self._expand_piece(self._barcode_onehot, **kwargs)
            pieces.append(left_onehot)
            pieces.append(batch)
            pieces.append(right_onehot)
            pieces.append(promoter_onehot)
            pieces.append(barcode_onehot)
        elif normalized_mode == "flanked":
            promoter_onehot = self._expand_piece(self._promoter_onehot, **kwargs)
            barcode_onehot = self._expand_piece(self._barcode_onehot, **kwargs)
            pieces.append(batch)
            pieces.append(promoter_onehot)
            pieces.append(barcode_onehot)
        else:
            pieces.append(batch)

        assembled = torch.cat(pieces, dim=1)
        return assembled.squeeze(0) if squeeze_output else assembled
