"""proj_dim / first_layer_rank: factorize the first hidden layer of the head."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from alphagenome_encoder_ft import cli
from alphagenome_encoder_ft.config import HeadConfig, TrainConfig, head_kwargs_of
from alphagenome_encoder_ft.constructs import Construct
from alphagenome_encoder_ft.heads import DeepSTARRHead, MPRAHead
from alphagenome_encoder_ft.model import AlphaGenomeEncoderModel
from alphagenome_encoder_ft.train import save_checkpoint
from conftest import DummyAlphaGenome, make_config

L, D = 3, 1536


def _n_params(head: MPRAHead) -> int:
    _ = head(torch.zeros(1, L, D))  # materialize lazy layers
    return sum(p.numel() for p in head.parameters())


def _n_mlp_weights(head: MPRAHead) -> int:
    """Parameters excluding the norm and the biases, i.e. the weight matrices alone."""

    _ = head(torch.zeros(1, L, D))
    return sum(p.numel() for name, p in head.named_parameters() if name.endswith("weight") and not name.startswith("norm"))


@pytest.mark.parametrize("kwargs", [{"proj_dim": 16}, {"first_layer_rank": 16}])
@pytest.mark.parametrize("pooling_type", ["flatten", "center", "mean"])
def test_factorized_head_output_shape(kwargs: dict, pooling_type: str):
    head = MPRAHead(pooling_type=pooling_type, hidden_sizes=8, **kwargs)
    assert head(torch.randn(4, L, D)).shape == (4,)


@pytest.mark.parametrize("kwargs", [{"proj_dim": 16}, {"first_layer_rank": 16}])
def test_deepstarr_head_takes_factorization(kwargs: dict):
    head = DeepSTARRHead(hidden_sizes=8, **kwargs)
    assert head(torch.randn(4, L, D)).shape == (4, 2)


def test_proj_dim_weight_count():
    # 1536 * C + L * C * H + H * 1
    head = MPRAHead(hidden_sizes=1024, proj_dim=64)
    assert _n_mlp_weights(head) == D * 64 + L * 64 * 1024 + 1024
    assert head.hidden_layers[0].in_features == L * 64


def test_proj_256_matches_hidden_256_flatten_head():
    # The sweep pairs these on purpose: 1536*256 + 3*256*1024 == 3*1536*256.
    assert _n_mlp_weights(MPRAHead(hidden_sizes=1024, proj_dim=256)) - 1024 == _n_mlp_weights(
        MPRAHead(hidden_sizes=256)
    ) - 256


def test_first_layer_rank_weight_count():
    # (L * 1536) * r + r * H + H * 1
    head = MPRAHead(hidden_sizes=1024, first_layer_rank=64)
    assert _n_mlp_weights(head) == L * D * 64 + 64 * 1024 + 1024
    assert head.first_layer_factor.bias is None
    assert head.hidden_layers[0].in_features == 64


def test_proj_is_shared_across_positions():
    head = MPRAHead(hidden_sizes=4, proj_dim=2, dropout=None)
    _ = head(torch.zeros(1, L, D))
    assert head.proj.weight.shape == (2, D)
    assert head.proj.bias is None


def test_unfactorized_head_keeps_its_state_dict_keys():
    head = MPRAHead(hidden_sizes=8)
    _ = head(torch.zeros(1, L, D))
    keys = set(head.state_dict())
    assert not any(key.startswith(("proj", "first_layer_factor")) for key in keys)


def test_proj_and_rank_compose():
    # 1536 * C + L * C * r + r * H + H * 1
    head = MPRAHead(hidden_sizes=1024, proj_dim=64, first_layer_rank=64)
    assert _n_mlp_weights(head) == D * 64 + L * 64 * 64 + 64 * 1024 + 1024
    assert head.first_layer_factor.in_features == L * 64
    assert head(torch.randn(2, L, D)).shape == (2,)
    assert HeadConfig(proj_dim=64, first_layer_rank=64).proj_dim == 64


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"proj_dim": 0}, "proj_dim must be > 0"),
        ({"first_layer_rank": -1}, "first_layer_rank must be > 0"),
    ],
)
def test_factorization_validation(kwargs: dict, match: str):
    with pytest.raises(ValueError, match=match):
        MPRAHead(hidden_sizes=8, **kwargs)
    with pytest.raises(ValueError, match=f"head.*{match}"):
        HeadConfig(**kwargs)


@pytest.mark.parametrize(("flag", "field"), [("--proj_dim", "proj_dim"), ("--first_layer_rank", "first_layer_rank")])
def test_cli_flags_set_factorization(tmp_path: Path, flag: str, field: str):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(make_config(tmp_path).to_dict()))
    parser = cli.add_train_arguments(argparse.ArgumentParser())
    config = cli.load_config(parser, parser.parse_args(["--config", str(config_path), flag, "32"]))
    assert getattr(config.head, field) == 32


def test_config_without_factorization_still_loads(tmp_path: Path):
    raw = make_config(tmp_path).to_dict()
    del raw["head"]["proj_dim"]
    del raw["head"]["first_layer_rank"]
    head = TrainConfig.from_dict(raw).head
    assert head.proj_dim is None and head.first_layer_rank is None


@pytest.mark.parametrize("kwargs", [{"proj_dim": 4}, {"first_layer_rank": 4}, {"proj_dim": 4, "first_layer_rank": 2}])
def test_checkpoint_roundtrip_keeps_factorization(tmp_path: Path, kwargs: dict):
    torch.manual_seed(0)
    model = AlphaGenomeEncoderModel(
        DummyAlphaGenome(),
        MPRAHead(hidden_sizes=8, **kwargs),
        construct=Construct(prefix="A", suffix="GT", length=6),
    )
    model.initialize_head(6, device="cpu")
    model.eval()
    before = model.predict_inserts(["cc"])
    recorded = head_kwargs_of(model.head)
    for key, value in kwargs.items():
        assert recorded[key] == value

    path = save_checkpoint(tmp_path / "factorized.pt", model, save_mode="minimal", config=make_config(tmp_path))
    restored = AlphaGenomeEncoderModel.from_checkpoint(path, device="cpu", backbone_factory=DummyAlphaGenome)

    for key, value in kwargs.items():
        assert getattr(restored.head, key) == value
    np.testing.assert_allclose(restored.predict_inserts(["cc"]).numpy(), before.numpy(), rtol=1e-5, atol=1e-5)
