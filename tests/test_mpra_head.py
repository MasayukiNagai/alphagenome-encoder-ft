from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from alphagenome_encoder_ft.heads import MPRAHead


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
