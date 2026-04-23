"""Utility MPRA head implementations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
import torch.nn as nn

PoolingType = Literal["flatten", "center", "mean", "sum", "max"]
ENCODER_RESOLUTION_BP = 128
ENCODER_DIM = 1536


def _parse_hidden_sizes(hidden_sizes: int | Sequence[int]) -> list[int]:
    if isinstance(hidden_sizes, int):
        sizes = [hidden_sizes]
    else:
        sizes = list(hidden_sizes)
    if not sizes:
        raise ValueError("hidden_sizes must contain at least one layer")
    if any(size <= 0 for size in sizes):
        raise ValueError("hidden_sizes must be positive")
    return sizes


def _make_activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    raise ValueError("activation must be 'relu' or 'gelu'")


class MPRAHead(nn.Module):
    """Scalar regression head over encoder outputs.

    The head consumes AlphaGenome encoder features with shape ``(B, L, D)``,
    where ``B`` is batch size, ``L`` is the number of encoder positions, and
    ``D`` is the encoder channel dimension (1536).

    Pooling behavior depends on ``pooling_type``:

    - flatten:
      ``(B, L, D) -> (B, L * D) -> hidden MLP -> (B, 1) -> (B,)``
      All encoder positions are flattened into a single feature vector before prediction.
      ``center_bp`` is ignored.

    - center:
      ``(B, L, D) -> position-wise hidden MLP -> (B, L, 1) -> pick L // 2 -> (B,)``
      The hidden stack and output layer are applied independently at each
      encoder position, and only the exact center position is used. ``center_bp``
      is ignored.

    - mean / sum / max:
      ``(B, L, D) -> position-wise hidden MLP -> (B, L, 1) -> centered window -> reduce -> (B,)``
      The model first produces a scalar prediction per encoder position, then
      reduces over a centered window of positions. The window size is
      ``max(1, center_bp // 128)`` because encoder features are at 128 bp
      resolution.
    """

    def __init__(
        self,
        pooling_type: PoolingType = "flatten",
        center_bp: int = 256,
        hidden_sizes: int | Sequence[int] = 1024,
        dropout: float | None = 0.1,
        activation: Literal["relu", "gelu"] = "relu",
        num_outputs: int = 1,
    ) -> None:
        super().__init__()
        if pooling_type not in {"flatten", "center", "mean", "sum", "max"}:
            raise ValueError(f"Unknown pooling_type: {pooling_type}")
        if pooling_type in {"mean", "sum", "max"} and center_bp <= 0:
            raise ValueError("center_bp must be > 0")
        if dropout is not None and not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if num_outputs < 1:
            raise ValueError("num_outputs must be >= 1")

        self.pooling_type = pooling_type
        self.center_bp = center_bp
        self.hidden_sizes = _parse_hidden_sizes(hidden_sizes)
        self.dropout = dropout
        self.activation = activation
        self.num_outputs = int(num_outputs)
        self.norm = nn.LayerNorm(ENCODER_DIM)
        self.hidden_layers = nn.ModuleList()
        in_features: int | None = None
        for hidden_size in self.hidden_sizes:
            linear = nn.LazyLinear(hidden_size) if in_features is None else nn.Linear(in_features, hidden_size)
            self.hidden_layers.append(linear)
            in_features = hidden_size
        assert in_features is not None
        self.output_layer = nn.Linear(in_features, self.num_outputs)

    def _apply_hidden_layers(self, x: torch.Tensor) -> torch.Tensor:
        for linear in self.hidden_layers:
            x = linear(x)
            if self.dropout is not None:
                x = nn.functional.dropout(x, p=self.dropout, training=self.training)
            x = _make_activation(self.activation)(x)
        return x

    def _normalize_encoder_output(self, encoder_output: torch.Tensor) -> torch.Tensor:
        x = encoder_output
        if x.ndim == 3 and x.shape[-1] != ENCODER_DIM and x.shape[1] == ENCODER_DIM:
            x = x.transpose(1, 2)
        return self.norm(x)

    def _pool_predictions(self, preds: torch.Tensor) -> torch.Tensor:
        # preds: (B, L, K). For K==1, squeeze(-1) to preserve the legacy (B,) output.
        if preds.ndim != 3:
            raise ValueError(f"Expected per-position predictions rank 3, got {preds.ndim}")

        seq_len = preds.shape[1]
        if self.pooling_type == "center":
            center_idx = seq_len // 2
            pooled = preds[:, center_idx, :]
        else:
            window_positions = max(1, self.center_bp // ENCODER_RESOLUTION_BP)
            window_positions = min(window_positions, seq_len)
            start = max((seq_len - window_positions) // 2, 0)
            center_window = preds[:, start : start + window_positions, :]
            if self.pooling_type == "mean":
                pooled = center_window.mean(dim=1)
            elif self.pooling_type == "sum":
                pooled = center_window.sum(dim=1)
            elif self.pooling_type == "max":
                pooled = center_window.max(dim=1).values
            else:
                raise RuntimeError(f"Unhandled pooling type: {self.pooling_type}")
        return pooled.squeeze(-1) if self.num_outputs == 1 else pooled

    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        x = self._normalize_encoder_output(encoder_output)
        if self.pooling_type == "flatten":
            x = x.flatten(1)
            x = self._apply_hidden_layers(x)
            preds = self.output_layer(x)
            return preds.squeeze(-1) if self.num_outputs == 1 else preds

        x = self._apply_hidden_layers(x)
        preds = self.output_layer(x)
        return self._pool_predictions(preds)


class DeepSTARRHead(MPRAHead):
    """Dual-output regression head for Drosophila STARR-seq (dev + hk).

    Mirrors Al Murphy's JAX ``DeepSTARRHead`` in ``alphagenome_ft_mpra/mpra_heads.py``:
    same pooling modes, same configuration surface, same ``LayerNorm → MLP → Linear``
    layout. Architecture and state_dict keys are inherited from :class:`MPRAHead`,
    with ``num_outputs`` defaulted to 2 (``dev``, ``hk`` enhancer activity).
    """

    def __init__(
        self,
        pooling_type: PoolingType = "flatten",
        center_bp: int = 256,
        hidden_sizes: int | Sequence[int] = 2048,
        dropout: float | None = 0.5,
        activation: Literal["relu", "gelu"] = "relu",
        num_outputs: int = 2,
    ) -> None:
        super().__init__(
            pooling_type=pooling_type,
            center_bp=center_bp,
            hidden_sizes=hidden_sizes,
            dropout=dropout,
            activation=activation,
            num_outputs=num_outputs,
        )
