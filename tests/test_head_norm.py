"""Normalization applied to the encoder output before the head MLP (``norm_type``).

Pooling modes, output shapes and the head registry are covered in test_mpra_head.py.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from alphagenome_encoder_ft.config import HeadConfig, build_head
from alphagenome_encoder_ft.heads import ENCODER_DIM, DeepSTARRHead, MPRAHead, _make_norm

POOLING = ["flatten", "center", "mean", "sum", "max"]


@pytest.mark.parametrize(
    ("norm_type", "expected"),
    [
        ("layer", nn.LayerNorm),
        ("batch", nn.BatchNorm1d),
        ("group", nn.GroupNorm),
        ("none", nn.Identity),
    ],
)
def test_make_norm_builds_the_requested_module(norm_type: str, expected: type):
    assert isinstance(_make_norm(norm_type, ENCODER_DIM), expected)


def test_make_norm_sizes_every_norm_to_the_channel_dimension():
    assert _make_norm("layer", ENCODER_DIM).normalized_shape == (ENCODER_DIM,)
    assert _make_norm("batch", ENCODER_DIM).num_features == ENCODER_DIM
    group = _make_norm("group", ENCODER_DIM)
    assert (group.num_groups, group.num_channels) == (8, ENCODER_DIM)


def test_make_norm_falls_back_to_a_divisor_group_count():
    # 8 does not divide 12, so the group count is reduced until it does.
    group = _make_norm("group", 12, num_groups=8)
    assert group.num_groups == 6
    assert 12 % group.num_groups == 0


def test_make_norm_treats_none_as_identity_and_rejects_unknown():
    assert isinstance(_make_norm(None, ENCODER_DIM), nn.Identity)
    with pytest.raises(ValueError, match="Unsupported norm_type"):
        _make_norm("instance", ENCODER_DIM)


@pytest.mark.parametrize("norm_type", ["layer", "batch", "group", "none"])
@pytest.mark.parametrize("pooling_type", POOLING)
def test_every_norm_works_with_every_pooling_mode(norm_type: str, pooling_type: str):
    head = MPRAHead(pooling_type=pooling_type, hidden_sizes=16, norm_type=norm_type)
    preds = head(torch.randn(4, 3, ENCODER_DIM))
    assert preds.shape == (4,)
    assert torch.isfinite(preds).all()


def test_layer_is_the_default_so_existing_checkpoints_still_load():
    head = MPRAHead(hidden_sizes=16)
    assert head.norm_type == "layer"
    assert isinstance(head.norm, nn.LayerNorm)
    assert {"norm.weight", "norm.bias"} <= set(head.state_dict())


def test_none_leaves_the_encoder_output_untouched():
    head = MPRAHead(pooling_type="flatten", hidden_sizes=16, norm_type="none")
    encoder_output = torch.randn(4, 3, ENCODER_DIM)
    torch.testing.assert_close(head._normalize_encoder_output(encoder_output), encoder_output)
    assert not any(key.startswith("norm.") for key in head.state_dict())


def test_channels_first_norms_are_applied_over_channels_not_positions():
    # BatchNorm1d and GroupNorm act on dim 1, so the head hands them (B, D, L) and
    # transposes the result back; getting this wrong would normalize over positions.
    head = MPRAHead(pooling_type="flatten", hidden_sizes=16, norm_type="batch")
    head.eval()  # running stats, so this is a plain affine transform
    encoder_output = torch.randn(4, 3, ENCODER_DIM)

    got = head._normalize_encoder_output(encoder_output)

    assert got.shape == encoder_output.shape
    expected = head.norm(encoder_output.transpose(1, 2)).transpose(1, 2)
    torch.testing.assert_close(got, expected)


def test_group_norm_standardizes_each_group():
    head = MPRAHead(pooling_type="flatten", hidden_sizes=16, norm_type="group")
    encoder_output = torch.randn(2, 4, ENCODER_DIM)

    # affine weight=1 bias=0 at init, so each group comes out zero-mean.
    normalized = head._normalize_encoder_output(encoder_output).transpose(1, 2)
    grouped = normalized.reshape(2, 8, (ENCODER_DIM // 8) * 4)
    torch.testing.assert_close(grouped.mean(dim=-1), torch.zeros(2, 8), atol=1e-5, rtol=1e-4)


def test_transposed_encoder_output_is_still_handled():
    head = MPRAHead(pooling_type="flatten", hidden_sizes=16, norm_type="batch")
    head.eval()
    encoder_output = torch.randn(2, 3, ENCODER_DIM)

    torch.testing.assert_close(
        head._normalize_encoder_output(encoder_output.transpose(1, 2)),
        head._normalize_encoder_output(encoder_output),
    )


def test_norm_type_reaches_the_head_through_build_head():
    head = build_head("mpra", {"pooling_type": "flatten", "hidden_sizes": [8], "norm_type": "group"})
    assert isinstance(head.norm, nn.GroupNorm)


def test_head_config_defaults_to_layer_and_validates():
    assert HeadConfig().norm_type == "layer"
    with pytest.raises(ValueError, match="head.norm_type"):
        HeadConfig(norm_type="instance")


def test_deepstarr_head_passes_norm_type_through():
    assert isinstance(DeepSTARRHead(hidden_sizes=8, norm_type="batch").norm, nn.BatchNorm1d)
