from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from alphagenome_encoder_ft.config import build_head
from alphagenome_encoder_ft.heads import DeepSTARRHead, MPRAHead


@pytest.mark.parametrize("pooling_type", ["flatten", "center", "mean", "sum", "max"])
def test_mpra_head_output_shape(pooling_type: str):
    head = MPRAHead(pooling_type=pooling_type, hidden_sizes=16, center_bp=256)
    encoder_output = torch.randn(4, 3, 1536)
    preds = head(encoder_output)
    assert preds.shape == (4,)


def test_mpra_head_lazy_init_for_flatten():
    head = MPRAHead(pooling_type="flatten", hidden_sizes=[32, 16])
    encoder_output = torch.randn(2, 5, 1536)
    preds = head(encoder_output)
    assert preds.shape == (2,)


@pytest.mark.parametrize(
    ("pooling_type", "expected"),
    [
        ("center", 3.0),
        ("mean", 1.5),
        ("sum", 3.0),
        ("max", 3.0),
    ],
)
def test_mpra_head_non_flatten_pools_position_scores(pooling_type: str, expected: float):
    head = MPRAHead(pooling_type=pooling_type, hidden_sizes=[1], center_bp=256, dropout=None)
    encoder_output = torch.zeros(1, 2, 1536)
    encoder_output[0, 0, 0] = -1.0
    encoder_output[0, 1, 0] = 3.0

    _ = head(encoder_output)
    head.norm = nn.Identity()
    with torch.no_grad():
        head.hidden_layers[0].weight.zero_()
        head.hidden_layers[0].bias.zero_()
        head.hidden_layers[0].weight[0, 0] = 1.0
        head.output_layer.weight.fill_(1.0)
        head.output_layer.bias.zero_()

    preds = head(encoder_output)
    assert preds.shape == (1,)
    assert torch.allclose(preds, torch.tensor([expected]))


@pytest.mark.parametrize("pooling_type", ["flatten", "center", "mean", "sum", "max"])
def test_deepstarr_head_output_shape(pooling_type: str):
    head = DeepSTARRHead(pooling_type=pooling_type, hidden_sizes=16, center_bp=256)
    encoder_output = torch.randn(4, 3, 1536)
    preds = head(encoder_output)
    assert preds.shape == (4, 2)


def test_deepstarr_head_matches_in_tree_defaults():
    head = DeepSTARRHead()
    encoder_output = torch.randn(2, 2, 1536)
    preds = head(encoder_output)
    assert preds.shape == (2, 2)
    assert head.num_outputs == 2
    assert head.pooling_type == "flatten"
    assert head.hidden_sizes == [2048]


def test_mpra_head_num_outputs_multi():
    head = MPRAHead(pooling_type="flatten", hidden_sizes=16, num_outputs=3)
    preds = head(torch.randn(4, 3, 1536))
    assert preds.shape == (4, 3)


def test_build_head_registry_mpra_default():
    head = build_head("mpra", {"pooling_type": "flatten", "hidden_sizes": [16], "center_bp": 256})
    assert isinstance(head, MPRAHead)
    preds = head(torch.randn(2, 3, 1536))
    assert preds.shape == (2,)


def test_build_head_registry_deepstarr():
    head = build_head(
        "deepstarr",
        {"pooling_type": "flatten", "hidden_sizes": [16], "dropout": 0.5, "num_outputs": 2},
    )
    assert isinstance(head, DeepSTARRHead)
    preds = head(torch.randn(2, 3, 1536))
    assert preds.shape == (2, 2)


def test_build_head_registry_drops_unknown_keys():
    # head_type is stored on HeadConfig but is not a constructor arg — must be dropped.
    head = build_head(
        "mpra",
        {"pooling_type": "flatten", "hidden_sizes": [16], "head_type": "mpra", "bogus": 9},
    )
    assert isinstance(head, MPRAHead)


def test_build_head_unknown_head_type_raises():
    with pytest.raises(ValueError, match="Unknown head_type"):
        build_head("nope", {})
