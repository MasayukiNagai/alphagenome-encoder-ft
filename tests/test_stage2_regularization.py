"""Stage-2 options: train only the top encoder blocks, EMA of weights, L2-SP to the start."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from alphagenome_pytorch.convolutions import StandardizedConv1d
from torch.utils.data import DataLoader, TensorDataset

from alphagenome_encoder_ft import cli
from alphagenome_encoder_ft.config import Stage2Config, TrainConfig
from alphagenome_encoder_ft.heads import MPRAHead
from alphagenome_encoder_ft.model import AlphaGenomeEncoderModel
from alphagenome_encoder_ft.train import (
    ParameterEMA,
    create_stage1_optimizer,
    create_stage2_optimizer,
    make_l2sp_penalty,
    run_two_stage_training,
)
from conftest import make_config


class BlockEncoder(nn.Module):
    """AlphaGenome-like layout: an embedder and a list of down blocks, per position."""

    def __init__(self, n_blocks: int = 3):
        super().__init__()
        self.dna_embedder = nn.Linear(4, 8)
        self.down_blocks = nn.ModuleList(nn.Linear(8, 8) for _ in range(n_blocks))
        self.out = nn.Linear(8, 1536)

    def forward(self, x):
        x = self.dna_embedder(x)
        for block in self.down_blocks:
            x = x + torch.relu(block(x))
        return self.out(x)


class BlockBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = BlockEncoder()

    def forward(self, sequences, organism_idx, encoder_only=False):
        return {"encoder_output": self.encoder(sequences)}


def _model() -> AlphaGenomeEncoderModel:
    torch.manual_seed(0)
    model = AlphaGenomeEncoderModel(BlockBackbone(), MPRAHead(hidden_sizes=8, dropout=None))
    model.initialize_head(2, device="cpu")
    return model


def _trainable(module: nn.Module) -> bool:
    return all(p.requires_grad for p in module.parameters())


def _frozen(module: nn.Module) -> bool:
    return not any(p.requires_grad for p in module.parameters())


# --- top blocks --------------------------------------------------------------------------


def test_top_blocks_unfreezes_only_the_last_blocks():
    model = _model()
    model.set_encoder_trainable(True, top_blocks=2)
    enc = model.encoder
    assert _frozen(enc.dna_embedder) and _frozen(enc.down_blocks[0])
    assert _trainable(enc.down_blocks[1]) and _trainable(enc.down_blocks[2])


def test_top_blocks_none_trains_everything_and_false_freezes_everything():
    model = _model()
    model.set_encoder_trainable(True)
    assert _trainable(model.encoder)
    model.set_encoder_trainable(False, top_blocks=2)
    assert _frozen(model.encoder)


def test_top_blocks_out_of_range_or_without_down_blocks():
    with pytest.raises(ValueError, match=r"top_blocks must be in \[1, 3\]"):
        _model().set_encoder_trainable(True, top_blocks=4)
    flat = AlphaGenomeEncoderModel(BlockBackbone(), MPRAHead(hidden_sizes=8))
    flat.backbone.encoder = nn.Linear(4, 1536)
    with pytest.raises(AttributeError, match="down_blocks"):
        flat.set_encoder_trainable(True, top_blocks=1)


def test_stage2_optimizer_holds_only_unfrozen_encoder_params(tmp_path: Path):
    model = _model()
    model.set_encoder_trainable(True, top_blocks=1)
    config = make_config(tmp_path, stage2={"encoder_lr": 1e-4, "head_lr": 1e-4})
    optimizer = create_stage2_optimizer(config, model)
    encoder_group = next(g for g in optimizer.param_groups if g["name"] == "encoder")
    expected = {id(p) for p in model.encoder.down_blocks[-1].parameters()}
    assert {id(p) for p in encoder_group["params"]} == expected


# --- EMA ---------------------------------------------------------------------------------


def test_ema_tracks_with_warmup_and_swaps_in_and_out():
    layer = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        layer.weight.fill_(0.0)
    ema = ParameterEMA(layer, decay=0.9)
    with torch.no_grad():
        layer.weight.fill_(1.0)
    ema.update()  # warm-up decay min(0.9, 2/11)
    expected = 1 - 2 / 11
    assert torch.allclose(ema.shadow["weight"], torch.tensor([[expected]]))
    with ema.applied():
        assert torch.allclose(layer.weight, torch.tensor([[expected]]))
    assert torch.allclose(layer.weight, torch.tensor([[1.0]]))


def test_ema_covers_only_trainable_params():
    model = _model()
    model.set_encoder_trainable(True, top_blocks=1)
    names = set(ParameterEMA(model, 0.99).shadow)
    assert any(n.startswith("backbone.encoder.down_blocks.2") for n in names)
    assert not any(n.startswith("backbone.encoder.down_blocks.0") for n in names)


# --- L2-SP -------------------------------------------------------------------------------


def test_l2sp_is_zero_at_start_and_grows_with_change():
    model = _model()
    model.set_encoder_trainable(True)
    penalty = make_l2sp_penalty(model.encoder)
    assert float(penalty()) == 0.0
    with torch.no_grad():
        model.encoder.down_blocks[0].weight.add_(0.1)
    assert float(penalty()) == pytest.approx(0.1**2 * 64, rel=1e-5)


def test_l2sp_compares_standardized_kernels_and_skips_frozen():
    torch.manual_seed(0)
    encoder = nn.Module()
    encoder.conv = StandardizedConv1d(4, 6, kernel_size=3)
    encoder.frozen = nn.Linear(2, 2)
    encoder.frozen.weight.requires_grad_(False)
    encoder.frozen.bias.requires_grad_(False)
    penalty = make_l2sp_penalty(encoder)
    with torch.no_grad():
        encoder.conv.weight.mul_(3.0).add_(0.5)  # same standardized kernel
        encoder.frozen.weight.add_(1.0)  # frozen: not penalized
    assert float(penalty()) == pytest.approx(0.0, abs=1e-8)
    with torch.no_grad():
        encoder.conv.scale.add_(0.2)  # scale is compared as is
    assert float(penalty()) == pytest.approx(6 * 0.2**2, rel=1e-5)


def test_l2sp_gradient_pulls_back_toward_the_anchor():
    model = _model()
    model.set_encoder_trainable(True)
    penalty = make_l2sp_penalty(model.encoder)
    with torch.no_grad():
        model.encoder.down_blocks[1].bias.add_(1.0)
    penalty().backward()
    assert torch.all(model.encoder.down_blocks[1].bias.grad > 0)


# --- config, CLI and the two-stage run ---------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"encoder_top_blocks": 0}, "encoder_top_blocks must be >= 1"),
        ({"ema_decay": 1.0}, r"ema_decay must be in \(0, 1\)"),
        ({"l2sp": -1.0}, "l2sp must be >= 0"),
    ],
)
def test_stage2_config_validation(kwargs: dict, match: str):
    with pytest.raises(ValueError, match=match):
        Stage2Config(**kwargs)


def test_cli_flags_set_stage2_options(tmp_path: Path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(make_config(tmp_path, stage2={"encoder_lr": 1e-4}).to_dict()))
    parser = cli.add_train_arguments(argparse.ArgumentParser())
    args = parser.parse_args([
        "--config", str(config_path), "--stage2_encoder_top_blocks", "2",
        "--stage2_ema_decay", "0.999", "--stage2_l2sp", "0.01",
    ])
    stage2 = cli.load_config(parser, args).stage2
    assert (stage2.encoder_top_blocks, stage2.ema_decay, stage2.l2sp) == (2, 0.999, 0.01)


def test_config_without_the_new_fields_still_loads(tmp_path: Path):
    raw = make_config(tmp_path, stage2={"encoder_lr": 1e-4}).to_dict()
    for key in ("encoder_top_blocks", "ema_decay", "l2sp"):
        del raw["stage2"][key]
    stage2 = TrainConfig.from_dict(raw).stage2
    assert stage2.encoder_top_blocks is None and stage2.ema_decay is None and stage2.l2sp == 0.0


@pytest.mark.parametrize("options", [{"encoder_top_blocks": 1}, {"ema_decay": 0.9}, {"l2sp": 0.1}])
def test_two_stage_run_with_each_option(tmp_path: Path, options: dict):
    torch.manual_seed(0)
    model = _model()
    x = torch.randn(12, 2, 4)
    loader = DataLoader(TensorDataset(x, x.sum(dim=(1, 2))), batch_size=4)
    stage = {"num_epochs": 1, "early_stopping_patience": 5, "head_lr": 1e-3, "lr_scheduler": "constant"}
    config = make_config(tmp_path, stage1=stage, stage2={**stage, "encoder_lr": 1e-3, **options})
    frozen_before = model.encoder.dna_embedder.weight.detach().clone()
    result = run_two_stage_training(
        model, loader,
        stage1_optimizer=create_stage1_optimizer(config, model),
        stage2_optimizer_factory=lambda m: create_stage2_optimizer(config, m),
        config=config, device="cpu", val_loader=loader,
    )
    assert result["stage2"]["best_checkpoint_path"] is not None
    if "encoder_top_blocks" in options:
        assert torch.equal(model.encoder.dna_embedder.weight, frozen_before)
