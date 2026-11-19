"""LoCon adapters on the encoder's block convolutions."""

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
from alphagenome_encoder_ft.adapters import LoConConv1d, add_locon
from alphagenome_encoder_ft.config import Stage2Config
from alphagenome_encoder_ft.heads import MPRAHead
from alphagenome_encoder_ft.model import AlphaGenomeEncoderModel
from alphagenome_encoder_ft.train import (
    create_stage1_optimizer,
    create_stage2_optimizer,
    load_checkpoint,
    run_two_stage_training,
    save_checkpoint,
)
from conftest import make_config


class Block(nn.Module):
    def __init__(self, channels: int, standardized: bool):
        super().__init__()
        conv_cls = StandardizedConv1d if standardized else nn.Conv1d
        kwargs = {} if standardized else {"padding": "same"}
        self.conv = conv_cls(channels, channels, kernel_size=5, **kwargs)

    def forward(self, x):
        return x + self.conv(torch.relu(x))


class ConvEncoder(nn.Module):
    """(B, L, 4) -> (B, L, 1536) with conv blocks named ``conv``, like AlphaGenome's ConvBlocks."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(4, 8, kernel_size=3, padding="same")  # embedder-style, not wrapped
        self.blocks = nn.ModuleList([Block(8, standardized=True), Block(8, standardized=False)])
        self.out = nn.Linear(8, 1536)

    def forward(self, x):
        h = self.conv1(x.transpose(1, 2))
        for block in self.blocks:
            h = block(h)
        return self.out(h.transpose(1, 2))


class ConvBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ConvEncoder()

    def forward(self, sequences, organism_idx, encoder_only=False):
        return {"encoder_output": self.encoder(sequences)}


def _model() -> AlphaGenomeEncoderModel:
    torch.manual_seed(0)
    model = AlphaGenomeEncoderModel(ConvBackbone(), MPRAHead(hidden_sizes=8, dropout=None))
    model.initialize_head(6, device="cpu")
    return model


def _x(n: int = 4) -> torch.Tensor:
    return torch.nn.functional.one_hot(torch.randint(0, 4, (n, 6)), 4).float()


def test_wrapped_model_starts_identical_and_wraps_only_block_convs():
    model, x = _model(), _x()
    with torch.no_grad():
        before = model(x)
    wrapped = model.add_encoder_locon(2)
    assert wrapped == ["blocks.0.conv", "blocks.1.conv"]
    assert not isinstance(model.encoder.conv1, LoConConv1d)
    with torch.no_grad():
        assert torch.allclose(model(x), before)


def test_only_adapters_are_trainable_in_the_encoder():
    model = _model()
    model.add_encoder_locon(2)
    trainable = {n for n, p in model.encoder.named_parameters() if p.requires_grad}
    assert trainable == {f"blocks.{i}.conv.{w}" for i in (0, 1) for w in ("down", "up")}
    assert model.encoder.blocks[0].conv.down.shape == (2, 8, 5)
    assert model.encoder.blocks[0].conv.up.shape == (8, 2, 1)


def test_low_rank_path_adds_up_times_down_to_the_effective_kernel():
    torch.manual_seed(0)
    base = nn.Conv1d(3, 4, kernel_size=5, padding="same", bias=False)
    layer = LoConConv1d(base, rank=2)
    with torch.no_grad():
        layer.up.normal_()
    x = torch.randn(2, 3, 9)
    delta = torch.einsum("or,rik->oik", layer.up[..., 0], layer.down)
    expected = torch.nn.functional.conv1d(x, base.weight + delta, padding="same")
    assert torch.allclose(layer(x), expected, atol=1e-5)


def test_cannot_wrap_twice_or_without_convs():
    model = _model()
    model.add_encoder_locon(2)
    with pytest.raises(ValueError, match="already"):
        model.add_encoder_locon(2)
    with pytest.raises(ValueError, match="no 'conv'"):
        add_locon(nn.Linear(2, 2), 2)


def test_stage2_optimizer_holds_only_adapters(tmp_path: Path):
    model = _model()
    model.add_encoder_locon(2)
    optimizer = create_stage2_optimizer(make_config(tmp_path, stage2={"encoder_lr": 1e-3}), model)
    encoder_group = next(g for g in optimizer.param_groups if g["name"] == "encoder")
    assert len(encoder_group["params"]) == 4


def test_checkpoint_roundtrip_rebuilds_adapters(tmp_path: Path):
    model, x = _model(), _x()
    model.add_encoder_locon(2)
    with torch.no_grad():
        for module in model.encoder.modules():
            if isinstance(module, LoConConv1d):
                module.up.normal_()
    model.eval()
    with torch.no_grad():
        before = model(x)
    path = save_checkpoint(tmp_path / "locon.pt", model, save_mode="minimal", config=make_config(tmp_path))
    assert torch.load(path, weights_only=False)["encoder_locon_rank"] == 2
    restored = AlphaGenomeEncoderModel.from_checkpoint(path, device="cpu", backbone_factory=ConvBackbone)
    with torch.no_grad():
        assert torch.allclose(restored(x), before, atol=1e-6)


@pytest.mark.parametrize("save_mode", ["minimal", "full"])
def test_plain_checkpoint_loads_into_a_locon_model_as_a_no_op(tmp_path: Path, save_mode: str):
    # The stage-1 best.pt has no adapters; --evaluate_test reloads it after stage 2 added them.
    model, x = _model(), _x()
    model.eval()
    with torch.no_grad():
        plain = model(x)
    path = save_checkpoint(tmp_path / "plain.pt", model, save_mode=save_mode, config=make_config(tmp_path))
    model.add_encoder_locon(2)
    with torch.no_grad():
        for module in model.encoder.modules():
            if isinstance(module, LoConConv1d):
                module.up.normal_()
                module.base.weight.add_(1.0)
    load_checkpoint(path, model)
    with torch.no_grad():
        assert torch.allclose(model(x), plain, atol=1e-6)
    assert all(m.up.abs().sum() == 0 for m in model.encoder.modules() if isinstance(m, LoConConv1d))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"encoder_locon_rank": 0}, "encoder_locon_rank must be >= 1"),
        ({"encoder_unfreeze_first_conv": True}, "needs stage2.encoder_locon_rank"),
    ],
)
def test_config_validation(kwargs: dict, match: str):
    with pytest.raises(ValueError, match=match):
        Stage2Config(**kwargs)


def test_cli_flags(tmp_path: Path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(make_config(tmp_path, stage2={"encoder_lr": 1e-4}).to_dict()))
    parser = cli.add_train_arguments(argparse.ArgumentParser())
    args = parser.parse_args([
        "--config", str(config_path), "--stage2_encoder_locon_rank", "16", "--stage2_encoder_unfreeze_first_conv",
    ])
    stage2 = cli.load_config(parser, args).stage2
    assert stage2.encoder_locon_rank == 16 and stage2.encoder_unfreeze_first_conv


def test_two_stage_run_with_locon_keeps_base_weights(tmp_path: Path):
    model = _model()
    x = _x(12)
    loader = DataLoader(TensorDataset(x, x.sum(dim=(1, 2))), batch_size=4)
    stage = {"num_epochs": 1, "early_stopping_patience": 5, "head_lr": 1e-3, "lr_scheduler": "constant"}
    config = make_config(tmp_path, stage1=stage, stage2={**stage, "encoder_lr": 1e-2, "encoder_locon_rank": 2})
    base_before = model.encoder.blocks[0].conv.weight.detach().clone()
    result = run_two_stage_training(
        model, loader,
        stage1_optimizer=create_stage1_optimizer(config, model),
        stage2_optimizer_factory=lambda m: create_stage2_optimizer(config, m),
        config=config, device="cpu", val_loader=loader,
    )
    assert torch.equal(model.encoder.blocks[0].conv.base.weight, base_before)
    assert model.encoder.blocks[0].conv.up.abs().sum() > 0
    assert torch.load(result["stage2"]["best_checkpoint_path"], weights_only=False)["encoder_locon_rank"] == 2


def _trainable(model: AlphaGenomeEncoderModel) -> set[str]:
    return {n for n, p in model.encoder.named_parameters() if p.requires_grad}


ADAPTERS = {f"blocks.{i}.conv.{w}" for i in (0, 1) for w in ("down", "up")}


def test_unfreeze_first_conv_trains_the_unwrapped_conv_in_full():
    model = _model()
    model.add_encoder_locon(2, unfreeze_first_conv=True)
    assert _trainable(model) == ADAPTERS | {"conv1.weight", "conv1.bias"}


def test_two_stage_run_with_conv1_trains_it_and_checkpoint_restores_it(tmp_path: Path):
    model = _model()
    x = _x(12)
    loader = DataLoader(TensorDataset(x, x.sum(dim=(1, 2))), batch_size=4)
    stage = {"num_epochs": 1, "early_stopping_patience": 5, "head_lr": 1e-3, "lr_scheduler": "constant"}
    stage2 = {**stage, "encoder_lr": 1e-2, "encoder_locon_rank": 2, "encoder_unfreeze_first_conv": True}
    config = make_config(tmp_path, stage1=stage, stage2=stage2)
    kernel_before = model.encoder.blocks[0].conv.weight.detach().clone()
    scale_before = model.encoder.blocks[0].conv.scale.detach().clone()
    conv1_before = model.encoder.conv1.weight.detach().clone()
    result = run_two_stage_training(
        model, loader,
        stage1_optimizer=create_stage1_optimizer(config, model),
        stage2_optimizer_factory=lambda m: create_stage2_optimizer(config, m),
        config=config, device="cpu", val_loader=loader,
    )
    conv = model.encoder.blocks[0].conv
    assert torch.equal(conv.base.weight, kernel_before)
    assert torch.equal(conv.base.scale, scale_before)
    assert not torch.equal(model.encoder.conv1.weight, conv1_before)

    path = result["stage2"]["best_checkpoint_path"]
    load_checkpoint(path, model)
    model.eval()
    with torch.no_grad():
        expected = model(x)
    restored = AlphaGenomeEncoderModel.from_checkpoint(path, device="cpu", backbone_factory=ConvBackbone)
    with torch.no_grad():
        assert torch.allclose(restored(x), expected, atol=1e-6)
