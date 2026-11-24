"""Multi-output heads: per_output_hidden_layers, the masked default loss and NaN-aware metrics."""

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


@pytest.mark.parametrize("pooling_type", ["flatten", "center", "mean"])
@pytest.mark.parametrize("first_layer_rank", [None, 4])
def test_per_output_head_shape(pooling_type: str, first_layer_rank: int | None):
    head = MPRAHead(
        pooling_type=pooling_type, hidden_sizes=8, num_outputs=3,
        first_layer_rank=first_layer_rank, per_output_hidden_layers=True,
    )
    assert head(torch.randn(5, L, D)).shape == (5, 3)
    assert len(head.branches) == 3
    assert not hasattr(head, "output_layer")


def test_deepstarr_head_takes_per_output_hidden_layers():
    head = DeepSTARRHead(hidden_sizes=8, per_output_hidden_layers=True)
    assert head(torch.randn(2, L, D)).shape == (2, 2)
    assert len(head.branches) == 2


def test_each_output_reaches_only_its_own_branch_and_the_shared_factor():
    torch.manual_seed(0)
    head = MPRAHead(hidden_sizes=8, num_outputs=3, first_layer_rank=4, per_output_hidden_layers=True, dropout=None)
    head(torch.randn(4, L, D))[:, 1].sum().backward()
    assert head.first_layer_factor.weight.grad.abs().sum() > 0
    for k, branch in enumerate(head.branches):
        grads = [p.grad for p in branch.parameters()]
        if k == 1:
            assert all(g is not None and g.abs().sum() > 0 for g in grads)
        else:
            assert all(g is None or g.abs().sum() == 0 for g in grads)


def test_shared_head_keeps_state_dict_keys_and_initialisation():
    # The option is off by default and must not change the shared head at all.
    x = torch.randn(2, L, D)
    torch.manual_seed(0)
    a = MPRAHead(hidden_sizes=8, num_outputs=3, first_layer_rank=4).eval()
    out_a = a(x)  # lazy layers initialise on the first call
    torch.manual_seed(0)
    b = MPRAHead(hidden_sizes=8, num_outputs=3, first_layer_rank=4, per_output_hidden_layers=False).eval()
    torch.testing.assert_close(out_a, b(x))
    keys = set(a.state_dict())
    assert "output_layer.weight" in keys and "hidden_layers.0.weight" in keys
    assert not any(key.startswith("branches") for key in keys)


def test_per_output_head_weight_count():
    # shared factor (L*D*r) + K * (r*H + H + H*1 + 1)
    r, h, k = 4, 8, 3
    head = MPRAHead(hidden_sizes=h, num_outputs=k, first_layer_rank=r, per_output_hidden_layers=True, norm_type="none")
    _ = head(torch.zeros(1, L, D))
    assert sum(p.numel() for p in head.parameters()) == L * D * r + k * (r * h + h + h + 1)


def test_config_and_cli_carry_per_output_hidden_layers(tmp_path: Path):
    assert HeadConfig().per_output_hidden_layers is False
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(make_config(tmp_path).to_dict()))
    parser = cli.add_train_arguments(argparse.ArgumentParser())
    config = cli.load_config(parser, parser.parse_args(["--config", str(config_path), "--per_output_hidden_layers"]))
    assert config.head.per_output_hidden_layers is True


def test_checkpoint_roundtrip_keeps_per_output_hidden_layers(tmp_path: Path):
    torch.manual_seed(0)
    model = AlphaGenomeEncoderModel(
        DummyAlphaGenome(),
        MPRAHead(hidden_sizes=8, num_outputs=2, first_layer_rank=4, per_output_hidden_layers=True),
        construct=Construct(prefix="A", suffix="GT", length=6),
    )
    model.initialize_head(6, device="cpu")
    model.eval()
    before = model.predict_inserts(["cc"])
    assert head_kwargs_of(model.head)["per_output_hidden_layers"] is True

    path = save_checkpoint(tmp_path / "branched.pt", model, save_mode="minimal", config=make_config(tmp_path))
    restored = AlphaGenomeEncoderModel.from_checkpoint(path, device="cpu", backbone_factory=DummyAlphaGenome)
    assert restored.head.per_output_hidden_layers is True
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
