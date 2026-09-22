"""Shared regression metrics, and that the training loop and evaluation agree on them."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from alphagenome_encoder_ft.metrics import (
    pearsonr,
    per_track,
    regression_metrics,
    spearmanr,
)
from alphagenome_encoder_ft.train import _compute_metrics


def test_pearson_of_a_perfect_linear_relation_is_one():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert pearsonr(y, 2 * y + 5) == pytest.approx(1.0)
    assert pearsonr(y, -y) == pytest.approx(-1.0)


def test_pearson_matches_a_hand_computed_value():
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([2.0, 1.0, 4.0, 3.0, 5.0])
    expected = np.corrcoef(y_true, y_pred)[0, 1]
    assert pearsonr(y_true, y_pred) == pytest.approx(expected)


def test_spearman_is_one_for_any_monotonic_relation():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    # exponential is monotonic but not linear: Spearman 1, Pearson below it.
    assert spearmanr(y, np.exp(y)) == pytest.approx(1.0)
    assert pearsonr(y, np.exp(y)) < 0.95


def test_spearman_averages_tied_ranks():
    # ties in y_pred: ranks 1, 2.5, 2.5, 4
    assert spearmanr([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 2.0, 3.0]) == pytest.approx(
        spearmanr([1.0, 2.0, 3.0, 4.0], [1.0, 2.5, 2.5, 4.0])
    )


@pytest.mark.parametrize("metric", [pearsonr, spearmanr])
def test_correlations_are_nan_when_undefined(metric):
    assert math.isnan(metric([1.0], [2.0]))  # fewer than two points
    assert math.isnan(metric([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]))  # no variance


def test_correlations_accept_torch_and_numpy_alike():
    y_true = [1.0, 2.0, 3.0, 4.0, 5.0]
    y_pred = [2.0, 1.0, 4.0, 3.0, 5.0]
    from_numpy = pearsonr(np.array(y_true), np.array(y_pred))
    from_torch = pearsonr(torch.tensor(y_true), torch.tensor(y_pred))
    assert from_numpy == pytest.approx(from_torch)


def test_correlations_accept_tensors_that_require_grad():
    y_pred = torch.tensor([2.0, 1.0, 4.0], requires_grad=True)
    assert not math.isnan(pearsonr(torch.tensor([1.0, 2.0, 3.0]), y_pred))


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError, match="Shape mismatch"):
        pearsonr([1.0, 2.0], [1.0, 2.0, 3.0])


def test_per_track_is_empty_for_single_output_and_one_per_column_otherwise():
    assert per_track(pearsonr, np.zeros((4,)), np.zeros((4,))) == []
    scores = per_track(pearsonr, np.array([[1.0, 4.0], [2.0, 3.0], [3.0, 2.0]]), np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]))
    assert len(scores) == 2
    assert scores[0] == pytest.approx(1.0)
    assert scores[1] == pytest.approx(-1.0)


def test_regression_metrics_reports_errors_and_correlations():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = y_true + 1.0

    metrics = regression_metrics(y_true, y_pred)

    assert metrics["n_samples"] == 4
    assert metrics["mse"] == pytest.approx(1.0)
    assert metrics["rmse"] == pytest.approx(1.0)
    assert metrics["mae"] == pytest.approx(1.0)
    assert metrics["pearsonr"] == pytest.approx(1.0)
    assert "per_track" not in metrics


def test_regression_metrics_averages_tracks_and_lists_each():
    y_true = np.array([[1.0, 4.0], [2.0, 3.0], [3.0, 2.0], [4.0, 1.0]])
    y_pred = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])

    metrics = regression_metrics(y_true, y_pred)

    assert [track["pearsonr"] for track in metrics["per_track"]] == pytest.approx([1.0, -1.0])
    assert metrics["pearsonr"] == pytest.approx(0.0)  # the mean of the two tracks


def test_the_training_loop_and_the_evaluation_summary_report_the_same_pearson():
    """One implementation: train.py's per-epoch metric equals the evaluation summary."""

    torch.manual_seed(0)
    targets = torch.randn(64)
    preds = targets * 0.8 + torch.randn(64) * 0.3

    from_training = _compute_metrics(preds, targets, metric_fns=None)["pearson"]
    from_evaluation = regression_metrics(targets.numpy(), preds.numpy())["pearsonr"]

    assert from_training == pytest.approx(from_evaluation)


def test_the_training_loop_reports_one_pearson_per_track():
    torch.manual_seed(0)
    targets = torch.randn(32, 2)
    preds = targets * 0.5 + torch.randn(32, 2) * 0.2

    metrics = _compute_metrics(preds, targets, metric_fns=None)
    summary = regression_metrics(targets.numpy(), preds.numpy())

    assert metrics["pearson_track0"] == pytest.approx(summary["per_track"][0]["pearsonr"])
    assert metrics["pearson_track1"] == pytest.approx(summary["per_track"][1]["pearsonr"])


def test_custom_metric_functions_still_override_the_default():
    metrics = _compute_metrics(
        torch.tensor([1.0, 2.0]), torch.tensor([1.0, 2.0]), metric_fns={"always_7": lambda p, t: 7.0}
    )
    assert metrics == {"always_7": 7.0}
