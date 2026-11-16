"""Stage-2 EMA of the trainable weights (stage2.ema_decay)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from alphagenome_encoder_ft import cli
from alphagenome_encoder_ft.config import Stage2Config, TrainConfig
from alphagenome_encoder_ft.heads import MPRAHead
from alphagenome_encoder_ft.model import AlphaGenomeEncoderModel
from alphagenome_encoder_ft.train import (
    ParameterEMA,
    create_stage1_optimizer,
    create_stage2_optimizer,
    run_two_stage_training,
)
from conftest import make_config


class Encoder(nn.Module):
    """(B, L, 4) -> (B, L, 1536), per position."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(4, 8)
        self.out = nn.Linear(8, 1536)

    def forward(self, x):
        return self.out(torch.relu(self.embed(x)))


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()

    def forward(self, sequences, organism_idx, encoder_only=False):
        return {"encoder_output": self.encoder(sequences)}


def _model() -> AlphaGenomeEncoderModel:
    torch.manual_seed(0)
    model = AlphaGenomeEncoderModel(Backbone(), MPRAHead(hidden_sizes=8, dropout=None))
    model.initialize_head(2, device="cpu")
    return model


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
    model.set_encoder_trainable(True)
    model.encoder.embed.requires_grad_(False)
    names = set(ParameterEMA(model, 0.99).shadow)
    assert "backbone.encoder.out.weight" in names
    assert not any(n.startswith("backbone.encoder.embed") for n in names)


def test_config_validation():
    with pytest.raises(ValueError, match=r"ema_decay must be in \(0, 1\)"):
        Stage2Config(ema_decay=1.0)


def test_cli_flag(tmp_path: Path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(make_config(tmp_path, stage2={"encoder_lr": 1e-4}).to_dict()))
    parser = cli.add_train_arguments(argparse.ArgumentParser())
    args = parser.parse_args(["--config", str(config_path), "--stage2_ema_decay", "0.999"])
    assert cli.load_config(parser, args).stage2.ema_decay == 0.999


def test_config_without_the_field_still_loads(tmp_path: Path):
    raw = make_config(tmp_path, stage2={"encoder_lr": 1e-4}).to_dict()
    del raw["stage2"]["ema_decay"]
    assert TrainConfig.from_dict(raw).stage2.ema_decay is None


def test_two_stage_run_saves_the_averaged_weights(tmp_path: Path):
    torch.manual_seed(0)
    model = _model()
    x = torch.randn(12, 2, 4)
    loader = DataLoader(TensorDataset(x, x.sum(dim=(1, 2))), batch_size=4)
    stage = {"num_epochs": 1, "early_stopping_patience": 5, "head_lr": 1e-3, "lr_scheduler": "constant"}
    config = make_config(tmp_path, stage1=stage, stage2={**stage, "encoder_lr": 1e-3, "ema_decay": 0.9})
    result = run_two_stage_training(
        model, loader,
        stage1_optimizer=create_stage1_optimizer(config, model),
        stage2_optimizer_factory=lambda m: create_stage2_optimizer(config, m),
        config=config, device="cpu", val_loader=loader,
    )
    saved = torch.load(result["stage2"]["best_checkpoint_path"], weights_only=False)["encoder_state_dict"]
    # The live weights kept training after the averaged ones were saved.
    assert not torch.equal(saved["out.weight"], model.encoder.out.weight.detach())
