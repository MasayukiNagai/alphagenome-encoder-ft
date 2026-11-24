"""Utility MPRA head implementations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
import torch.nn as nn

PoolingType = Literal["flatten", "center", "mean", "sum", "max"]
NormType = Literal["layer", "batch", "group", "none"]
ENCODER_RESOLUTION_BP = 128
ENCODER_DIM = 1536
NORM_GROUPS = 8

# Norms that normalize over a channel dimension and therefore need (B, D, L) rather than
# the (B, L, D) the encoder emits.
_CHANNELS_FIRST_NORMS = {"batch", "group"}


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

def _make_norm(norm_type: str | None, channels: int, num_groups: int = NORM_GROUPS) -> nn.Module:
    """Normalization applied to the encoder output, over its ``channels`` dimension."""

    norm_type = "none" if norm_type is None else norm_type.lower()
    if norm_type == "layer":
        return nn.LayerNorm(channels)
    if norm_type == "batch":
        return nn.BatchNorm1d(channels)
    if norm_type == "group":
        groups = min(num_groups, channels)
        while groups > 1 and channels % groups != 0:
            groups -= 1
        return nn.GroupNorm(groups, channels)
    if norm_type == "none":
        return nn.Identity()
    raise ValueError(f"Unsupported norm_type: {norm_type}")


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

    ``proj_dim``, if set, adds a bias-free ``Linear(1536, proj_dim)`` at every position after
    the norm. ``first_layer_rank``, if set, makes the first hidden layer low-rank: a bias-free
    ``Linear(in, r)`` followed by ``Linear(r, H)``, with no activation or dropout between them.

    With ``num_outputs > 1`` the outputs share every layer and differ only in the rows of the
    final ``Linear(H, num_outputs)``. ``per_output_hidden_layers`` instead gives each output
    its own hidden layers and ``Linear(H, 1)``; the norm, ``proj_dim`` and the
    ``first_layer_rank`` factor stay shared. Without ``first_layer_rank`` or ``proj_dim`` the
    outputs then share no weights at all.

    ``norm_type`` selects the normalization applied to the encoder output before the MLP:

    - ``layer`` (default): ``LayerNorm(1536)`` over the channel dimension, per position.
      Independent of batch composition, which is why it is the default.
    - ``batch``: ``BatchNorm1d(1536)`` over batch and position per channel. Keeps running
      statistics, so predictions depend on batch composition during training.
    - ``group``: ``GroupNorm(8, 1536)``, between the two.
    - ``none``: no normalization (``Identity``), for feeding the encoder output straight in.
    """

    def __init__(
        self,
        pooling_type: PoolingType = "flatten",
        center_bp: int = 256,
        hidden_sizes: int | Sequence[int] = 1024,
        dropout: float | None = 0.1,
        activation: Literal["relu", "gelu"] = "relu",
        num_outputs: int = 1,
        norm_type: NormType = "layer",
        proj_dim: int | None = None,
        first_layer_rank: int | None = None,
        per_output_hidden_layers: bool = False,
    ) -> None:
        super().__init__()
        if pooling_type not in {"flatten", "center", "mean", "sum", "max"}:
            raise ValueError(f"Unknown pooling_type: {pooling_type}")
        if pooling_type in {"mean", "sum", "max"} and center_bp <= 0:
            raise ValueError("center_bp must be > 0")
        if proj_dim is not None and proj_dim <= 0:
            raise ValueError("proj_dim must be > 0")
        if first_layer_rank is not None and first_layer_rank <= 0:
            raise ValueError("first_layer_rank must be > 0")
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
        self.norm_type = "none" if norm_type is None else str(norm_type).lower()
        self.norm = _make_norm(self.norm_type, ENCODER_DIM)
        self.proj_dim = proj_dim
        self.first_layer_rank = first_layer_rank
        self.per_output_hidden_layers = bool(per_output_hidden_layers)
        # Registered only when set, so other heads keep their state_dict keys.
        if proj_dim is not None:
            self.proj = nn.Linear(ENCODER_DIM, proj_dim, bias=False)
        if first_layer_rank is not None:
            self.first_layer_factor = nn.LazyLinear(first_layer_rank, bias=False)
        if self.per_output_hidden_layers:
            self.branches = nn.ModuleList()
            for _ in range(self.num_outputs):
                hidden_layers, output_layer = self._build_hidden_stack(first_layer_rank, 1)
                self.branches.append(nn.ModuleDict({"hidden_layers": hidden_layers, "output_layer": output_layer}))
        else:
            self.hidden_layers, self.output_layer = self._build_hidden_stack(first_layer_rank, self.num_outputs)

    def _build_hidden_stack(self, in_features: int | None, num_outputs: int) -> tuple[nn.ModuleList, nn.Linear]:
        """Hidden layers of ``hidden_sizes`` and the output layer after them."""

        hidden_layers = nn.ModuleList()
        for hidden_size in self.hidden_sizes:
            linear = nn.LazyLinear(hidden_size) if in_features is None else nn.Linear(in_features, hidden_size)
            hidden_layers.append(linear)
            in_features = hidden_size
        assert in_features is not None
        return hidden_layers, nn.Linear(in_features, num_outputs)

    def _apply_hidden_layers(self, x: torch.Tensor, hidden_layers: nn.ModuleList) -> torch.Tensor:
        for linear in hidden_layers:
            x = linear(x)
            if self.dropout is not None:
                x = nn.functional.dropout(x, p=self.dropout, training=self.training)
            x = _make_activation(self.activation)(x)
        return x

    def _predict(self, x: torch.Tensor) -> torch.Tensor:
        """Features after the norm (and projection) -> ``(..., num_outputs)``."""

        if self.first_layer_rank is not None:
            x = self.first_layer_factor(x)
        if self.per_output_hidden_layers:
            return torch.cat(
                [
                    branch["output_layer"](self._apply_hidden_layers(x, branch["hidden_layers"]))
                    for branch in self.branches
                ],
                dim=-1,
            )
        return self.output_layer(self._apply_hidden_layers(x, self.hidden_layers))

    def _normalize_encoder_output(self, encoder_output: torch.Tensor) -> torch.Tensor:
        x = encoder_output
        if x.ndim == 3 and x.shape[-1] != ENCODER_DIM and x.shape[1] == ENCODER_DIM:
            x = x.transpose(1, 2)
        # LayerNorm and Identity act on the last dimension, which is already the channel
        # dimension. BatchNorm1d and GroupNorm act on dim 1, so they need (B, D, L).
        if x.ndim == 3 and self.norm_type in _CHANNELS_FIRST_NORMS:
            return self.norm(x.transpose(1, 2)).transpose(1, 2)
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
        if self.proj_dim is not None:
            x = self.proj(x)
        if self.pooling_type == "flatten":
            preds = self._predict(x.flatten(1))
            return preds.squeeze(-1) if self.num_outputs == 1 else preds

        return self._pool_predictions(self._predict(x))


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
        norm_type: NormType = "layer",
        proj_dim: int | None = None,
        first_layer_rank: int | None = None,
        per_output_hidden_layers: bool = False,
    ) -> None:
        super().__init__(
            pooling_type=pooling_type,
            center_bp=center_bp,
            hidden_sizes=hidden_sizes,
            dropout=dropout,
            activation=activation,
            num_outputs=num_outputs,
            norm_type=norm_type,
            proj_dim=proj_dim,
            first_layer_rank=first_layer_rank,
            per_output_hidden_layers=per_output_hidden_layers,
        )
