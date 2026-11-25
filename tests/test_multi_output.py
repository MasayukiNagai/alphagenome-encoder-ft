"""Multi-output heads: split_at, the masked default loss and NaN-aware metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from alphagenome_encoder_ft import cli
from alphagenome_encoder_ft.config import HeadConfig, head_kwargs_of
from alphagenome_encoder_ft.constructs import Construct
from alphagenome_encoder_ft.heads import DeepSTARRHead, MPRAHead
from alphagenome_encoder_ft.metrics import pearsonr, regression_metrics
from alphagenome_encoder_ft.model import AlphaGenomeEncoderModel
from alphagenome_encoder_ft.train import _compute_metrics, _default_loss_fn, save_checkpoint
from conftest import DummyAlphaGenome, make_config

L, D = 3, 1536


@pytest.mark.parametrize("split_at", ["output", "hidden", "input"])
@pytest.mark.parametrize("pooling_type", ["flatten", "center", "mean"])
@pytest.mark.parametrize("first_layer_rank", [None, 4])
def test_split_head_shape(split_at: str, pooling_type: str, first_layer_rank: int | None):
    head = MPRAHead(
        pooling_type=pooling_type, hidden_sizes=8, num_outputs=3,
        first_layer_rank=first_layer_rank, split_at=split_at,
    )
    assert head(torch.randn(5, L, D)).shape == (5, 3)


@pytest.mark.parametrize("split_at", ["hidden", "input"])
def test_single_output_split_head_keeps_scalar_output(split_at: str):
    assert MPRAHead(hidden_sizes=8, split_at=split_at)(torch.randn(5, L, D)).shape == (5,)


@pytest.mark.parametrize("split_at", ["hidden", "input"])
def test_deepstarr_head_takes_split_at(split_at: str):
    head = DeepSTARRHead(hidden_sizes=8, split_at=split_at)
    assert head(torch.randn(2, L, D)).shape == (2, 2)


def test_split_at_validation():
    with pytest.raises(ValueError, match="split_at"):
        MPRAHead(hidden_sizes=8, split_at="norm")
    with pytest.raises(ValueError, match="head.split_at"):
        HeadConfig(split_at="norm")


def _grad_sum(module: torch.nn.Module) -> float:
    return sum(0.0 if p.grad is None else float(p.grad.abs().sum()) for p in module.parameters())


def test_split_at_hidden_shares_the_factor_and_nothing_after_it():
    torch.manual_seed(0)
    head = MPRAHead(hidden_sizes=8, num_outputs=3, first_layer_rank=4, split_at="hidden", dropout=None)
    head(torch.randn(4, L, D))[:, 1].sum().backward()
    assert head.first_layer_factor.weight.grad.abs().sum() > 0
    assert [_grad_sum(branch) > 0 for branch in head.branches] == [False, True, False]


def test_split_at_input_shares_nothing():
    torch.manual_seed(0)
    head = MPRAHead(hidden_sizes=8, num_outputs=3, first_layer_rank=4, split_at="input", dropout=None)
    head(torch.randn(4, L, D))[:, 1].sum().backward()
    assert [_grad_sum(sub) > 0 for sub in head.heads] == [False, True, False]
    # The collecting module holds no layers of its own.
    assert set(dict(head.named_children())) == {"heads"}


def test_split_at_input_is_single_output_heads_side_by_side():
    x = torch.randn(4, L, D)
    torch.manual_seed(0)
    head = MPRAHead(hidden_sizes=8, num_outputs=2, first_layer_rank=4, split_at="input").eval()
    out = head(x)
    for k, sub in enumerate(head.heads):
        assert sub.num_outputs == 1 and sub.split_at == "output"
        torch.testing.assert_close(out[:, k], sub(x))


def test_default_head_keeps_state_dict_keys_and_initialisation():
    # split_at="output" is the default and must not change the shared head at all.
    x = torch.randn(2, L, D)
    torch.manual_seed(0)
    a = MPRAHead(hidden_sizes=8, num_outputs=3, first_layer_rank=4).eval()
    out_a = a(x)  # lazy layers initialise on the first call
    torch.manual_seed(0)
    b = MPRAHead(hidden_sizes=8, num_outputs=3, first_layer_rank=4, split_at="output").eval()
    torch.testing.assert_close(out_a, b(x))
    keys = set(a.state_dict())
    assert "output_layer.weight" in keys and "hidden_layers.0.weight" in keys
    assert not any(key.startswith(("branches", "heads")) for key in keys)


@pytest.mark.parametrize(
    ("split_at", "expected"),
    [
        # shared: factor L*D*r + r*H + H + H*K + K
        ("output", lambda r, h, k: L * D * r + r * h + h + h * k + k),
        # shared factor + K * (r*H + H + H + 1)
        ("hidden", lambda r, h, k: L * D * r + k * (r * h + h + h + 1)),
        # K * (L*D*r + r*H + H + H + 1)
        ("input", lambda r, h, k: k * (L * D * r + r * h + h + h + 1)),
    ],
)
def test_split_head_weight_count(split_at: str, expected):
    r, h, k = 4, 8, 3
    head = MPRAHead(hidden_sizes=h, num_outputs=k, first_layer_rank=r, split_at=split_at, norm_type="none")
    _ = head(torch.zeros(1, L, D))
    assert sum(p.numel() for p in head.parameters()) == expected(r, h, k)


def test_config_and_cli_carry_split_at(tmp_path: Path):
    assert HeadConfig().split_at == "output"
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(make_config(tmp_path).to_dict()))
    parser = cli.add_train_arguments(argparse.ArgumentParser())
    config = cli.load_config(parser, parser.parse_args(["--config", str(config_path), "--split_at", "input"]))
    assert config.head.split_at == "input"


@pytest.mark.parametrize("split_at", ["hidden", "input"])
def test_checkpoint_roundtrip_keeps_split_at(tmp_path: Path, split_at: str):
    torch.manual_seed(0)
    model = AlphaGenomeEncoderModel(
        DummyAlphaGenome(),
        MPRAHead(hidden_sizes=8, num_outputs=2, first_layer_rank=4, split_at=split_at),
        construct=Construct(prefix="A", suffix="GT", length=6),
    )
    model.initialize_head(6, device="cpu")
    model.eval()
    before = model.predict_inserts(["cc"])
    assert head_kwargs_of(model.head)["split_at"] == split_at

    path = save_checkpoint(tmp_path / "split.pt", model, save_mode="minimal", config=make_config(tmp_path))
    restored = AlphaGenomeEncoderModel.from_checkpoint(path, device="cpu", backbone_factory=DummyAlphaGenome)
    assert restored.head.split_at == split_at
    np.testing.assert_allclose(restored.predict_inserts(["cc"]).numpy(), before.numpy(), rtol=1e-5, atol=1e-5)


def test_loss_without_missing_labels_is_plain_mse():
    preds, targets = torch.randn(6, 3), torch.randn(6, 3)
    assert torch.equal(_default_loss_fn(preds, targets), F.mse_loss(preds, targets))


def test_loss_skips_missing_labels():
    preds = torch.randn(4, 3, requires_grad=True)
    targets = torch.randn(4, 3)
    targets[0, 1] = targets[2, 0] = targets[3, 2] = float("nan")
    observed = ~torch.isnan(targets)
    loss = _default_loss_fn(preds, targets)
    torch.testing.assert_close(loss, ((preds - targets)[observed] ** 2).mean())
    loss.backward()
    assert preds.grad[~observed].abs().sum() == 0


def test_loss_with_no_labels_is_zero_with_a_graph():
    preds = torch.randn(2, 3, requires_grad=True)
    loss = _default_loss_fn(preds, torch.full((2, 3), float("nan")))
    assert loss.item() == 0.0
    loss.backward()


def _missing_labels():
    rng = np.random.default_rng(0)
    y_true = rng.normal(size=(50, 3))
    y_pred = y_true + rng.normal(scale=0.5, size=(50, 3))
    y_true[rng.random((50, 3)) < 0.4] = np.nan
    return y_true, y_pred


def test_metrics_score_each_track_on_its_labelled_rows():
    y_true, y_pred = _missing_labels()
    metrics = regression_metrics(y_true, y_pred)
    for k, track in enumerate(metrics["per_track"]):
        rows = ~np.isnan(y_true[:, k])
        assert track["n_samples"] == rows.sum()
        assert track["pearsonr"] == pytest.approx(pearsonr(y_true[rows, k], y_pred[rows, k]))
    assert metrics["pearsonr"] == pytest.approx(np.mean([t["pearsonr"] for t in metrics["per_track"]]))
    observed = ~np.isnan(y_true)
    assert metrics["mse"] == pytest.approx(np.mean((y_pred - y_true)[observed] ** 2))
    assert metrics["n_samples"] == 50


def test_training_pearson_is_the_mean_of_the_tracks():
    y_true, y_pred = _missing_labels()
    metrics = _compute_metrics(torch.tensor(y_pred), torch.tensor(y_true), None)
    tracks = [metrics[f"pearson_track{k}"] for k in range(3)]
    assert all(np.isfinite(tracks))
    assert metrics["pearson"] == pytest.approx(np.mean(tracks))
    assert metrics["pearson"] == pytest.approx(regression_metrics(y_true, y_pred)["pearsonr"])


def test_single_output_metrics_unchanged():
    rng = np.random.default_rng(1)
    y_true, y_pred = rng.normal(size=20), rng.normal(size=20)
    metrics = _compute_metrics(torch.tensor(y_pred), torch.tensor(y_true), None)
    assert metrics == {"pearson": pytest.approx(pearsonr(y_true, y_pred))}
