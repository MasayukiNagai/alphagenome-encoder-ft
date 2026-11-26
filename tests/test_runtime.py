"""runtime.amp_dtype / compile / cudnn_benchmark: config, flags, loss scaling and EMA skips."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest
import torch

import alphagenome_encoder_ft.train as train_module
from alphagenome_encoder_ft.cli import add_train_arguments, build_overrides
from alphagenome_encoder_ft.config import TrainConfig
from alphagenome_encoder_ft.train import ParameterEMA, make_grad_scaler, run_training_stage, train_epoch
from test_train import _make_config, _make_loader, _make_model


def test_runtime_defaults_keep_the_previous_behaviour():
    runtime = TrainConfig().runtime
    assert (runtime.use_amp, runtime.amp_dtype, runtime.compile, runtime.cudnn_benchmark) == (
        False, "bf16", False, False,
    )


def test_amp_dtype_is_validated():
    with pytest.raises(ValueError, match="runtime.amp_dtype"):
        TrainConfig.from_dict({"runtime": {"amp_dtype": "fp8"}})


def test_runtime_flags_reach_the_config():
    parser = add_train_arguments(argparse.ArgumentParser())
    args = parser.parse_args(["--use_amp", "--amp_dtype", "fp16", "--compile", "--cudnn_benchmark"])
    runtime = TrainConfig.from_dict(build_overrides(args)).runtime
    assert (runtime.use_amp, runtime.amp_dtype, runtime.compile, runtime.cudnn_benchmark) == (
        True, "fp16", True, True,
    )


def test_grad_scaler_only_for_fp16_autocast_on_cuda():
    assert make_grad_scaler("cpu", True, "fp16") is None
    assert make_grad_scaler("cuda", True, "bf16") is None
    assert make_grad_scaler("cuda", False, "fp16") is None
    assert isinstance(make_grad_scaler("cuda", True, "fp16"), torch.amp.GradScaler)


def test_a_scaled_step_trains_like_an_unscaled_one():
    torch.manual_seed(0)
    plain, scaled = _make_model(), _make_model()
    scaled.load_state_dict(plain.state_dict())
    for model, scaler in ((plain, None), (scaled, torch.amp.GradScaler("cpu", init_scale=1024.0))):
        optimizer = torch.optim.SGD(model.head.parameters(), lr=1e-2)
        train_epoch(model, _make_loader(), optimizer, "cpu", use_amp=False, scaler=scaler)
    for a, b in zip(plain.head.parameters(), scaled.head.parameters()):
        torch.testing.assert_close(a, b)


def test_an_overflowing_step_is_skipped_and_so_is_its_ema_update():
    model = _make_model()
    optimizer = torch.optim.SGD(model.head.parameters(), lr=1e-2)
    ema = ParameterEMA(model, 0.9)
    before = {n: p.detach().clone() for n, p in model.head.named_parameters()}

    def inf_loss(preds, targets):
        return (preds - targets).pow(2).mean() * float("inf")

    scaler = torch.amp.GradScaler("cpu")
    train_epoch(model, _make_loader(), optimizer, "cpu", use_amp=False, scaler=scaler, loss_fn=inf_loss, ema=ema)

    assert ema.num_updates == 0
    assert scaler.get_scale() < 2.0**16
    for n, p in model.head.named_parameters():
        torch.testing.assert_close(p, before[n])


def test_runtime_settings_apply_at_each_stage_start(tmp_path: Path, monkeypatch):
    model = _make_model()
    config = _make_config(tmp_path)
    config.runtime.compile = True
    config.runtime.cudnn_benchmark = True
    compiled = []
    monkeypatch.setattr(type(model.encoder), "compile", lambda self, *a, **k: compiled.append(self), raising=False)
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", False)
    keys = set(model.state_dict())

    optimizer = train_module.create_stage1_optimizer(config, model)
    run_training_stage(
        model, _make_loader(), optimizer=optimizer, config=config, stage_config=config.stage1,
        device="cpu", stage="stage1", train_encoder=False, val_loader=_make_loader(),
    )

    assert compiled == [model.encoder]
    assert torch.backends.cudnn.benchmark is True
    assert set(model.state_dict()) == keys
