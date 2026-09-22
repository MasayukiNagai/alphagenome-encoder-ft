"""Regression metrics, shared by the training loop and the evaluation scripts.

One implementation per statistic. The training loop reports them per epoch from torch
tensors; the evaluation scripts report them once from numpy arrays. Every function here
accepts either, so a correlation never depends on which caller computed it.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import torch


def _as_float64(values: Any) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        values = values.detach().float().cpu().numpy()
    return np.asarray(values, dtype=np.float64)


def _as_flat_pair(y_true: Any, y_pred: Any) -> tuple[np.ndarray, np.ndarray]:
    true_values = _as_float64(y_true).reshape(-1)
    pred_values = _as_float64(y_pred).reshape(-1)
    if true_values.shape != pred_values.shape:
        raise ValueError(f"Shape mismatch: {true_values.shape} vs {pred_values.shape}")
    return true_values, pred_values


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Ranks with ties averaged, which is what Spearman needs."""

    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.shape[0], dtype=np.float64)
    start = 0
    while start < sorted_values.shape[0]:
        end = start + 1
        while end < sorted_values.shape[0] and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def pearsonr(y_true: Any, y_pred: Any) -> float:
    """Pearson correlation. ``nan`` when undefined: fewer than two points, or no variance."""

    true_values, pred_values = _as_flat_pair(y_true, y_pred)
    if true_values.size < 2:
        return float("nan")
    true_centered = true_values - true_values.mean()
    pred_centered = pred_values - pred_values.mean()
    denominator = np.linalg.norm(true_centered) * np.linalg.norm(pred_centered)
    if denominator == 0.0:
        return float("nan")
    return float(np.dot(true_centered, pred_centered) / denominator)


def spearmanr(y_true: Any, y_pred: Any) -> float:
    """Spearman correlation: Pearson over average ranks."""

    true_values, pred_values = _as_flat_pair(y_true, y_pred)
    if true_values.size < 2:
        return float("nan")
    return pearsonr(_average_ranks(true_values), _average_ranks(pred_values))


def per_track(metric: Callable[[Any, Any], float], y_true: Any, y_pred: Any) -> list[float]:
    """Apply ``metric`` to each column of ``(N, K)`` targets; empty for single-output heads."""

    true_values = _as_float64(y_true)
    pred_values = _as_float64(y_pred)
    if true_values.ndim != 2 or pred_values.ndim != 2 or true_values.shape[1] < 2:
        return []
    return [metric(true_values[:, k], pred_values[:, k]) for k in range(true_values.shape[1])]


def regression_metrics(y_true: Any, y_pred: Any) -> dict[str, Any]:
    """Full evaluation summary: counts, error magnitudes and both correlations.

    For ``(N, K)`` targets the correlations are the mean across tracks, and each track's
    own values are listed under ``per_track``.
    """

    true_values = _as_float64(y_true)
    pred_values = _as_float64(y_pred)
    if true_values.shape != pred_values.shape:
        raise ValueError(f"Shape mismatch: {true_values.shape} vs {pred_values.shape}")

    residual = pred_values - true_values
    mse = float(np.mean(np.square(residual))) if true_values.size else float("nan")
    metrics: dict[str, Any] = {
        "n_samples": int(true_values.shape[0]) if true_values.ndim else 0,
        "mse": mse,
        "rmse": float(math.sqrt(mse)) if not math.isnan(mse) else float("nan"),
        "mae": float(np.mean(np.abs(residual))) if true_values.size else float("nan"),
    }

    if true_values.ndim == 1 or true_values.shape[1] == 1:
        metrics["pearsonr"] = pearsonr(true_values, pred_values)
        metrics["spearmanr"] = spearmanr(true_values, pred_values)
        return metrics

    tracks = [
        {"pearsonr": p, "spearmanr": s}
        for p, s in zip(
            per_track(pearsonr, true_values, pred_values),
            per_track(spearmanr, true_values, pred_values),
            strict=True,
        )
    ]
    metrics["pearsonr"] = float(np.mean([track["pearsonr"] for track in tracks]))
    metrics["spearmanr"] = float(np.mean([track["spearmanr"] for track in tracks]))
    metrics["per_track"] = tracks
    return metrics
