from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import TensorDataset

from alphagenome_encoder_ft import cli
from alphagenome_encoder_ft.config import build_head
from alphagenome_encoder_ft.model import AlphaGenomeEncoderModel
from conftest import DummyAlphaGenome, make_config


@pytest.fixture
def dummy_backbone(monkeypatch):
    def fake_from_pretrained(cls, weights, head_config, *, device=None, construct=None, **kwargs):
        return cls(DummyAlphaGenome(), build_head("mpra", head_config.__dict__), construct=construct)

    monkeypatch.setattr(AlphaGenomeEncoderModel, "from_pretrained", classmethod(fake_from_pretrained))


def _two_stage_config(tmp_path: Path):
    return make_config(
        tmp_path,
        stage1={"num_epochs": 1},
        stage2={"num_epochs": 1, "encoder_lr": 1e-4, "head_lr": 1e-3},
    )


def _datasets(requested: list[str]):
    def make_dataset(split: str):
        requested.append(split)
        generator = torch.Generator().manual_seed({"train": 0, "val": 1, "test": 2}[split])
        sequences = torch.randn(8, 2, 4, generator=generator)
        return TensorDataset(sequences, sequences.sum(dim=(1, 2)))

    return make_dataset


def test_train_leaves_the_test_split_alone_by_default(tmp_path: Path, dummy_backbone):
    requested: list[str] = []

    results = cli.train(_two_stage_config(tmp_path), construct=None, make_dataset=_datasets(requested))

    assert "test" not in requested
    assert "test_metrics" not in results
    history = json.loads((tmp_path / "history.json").read_text())
    assert history["test_loss"] == [] and history["val_loss"]


def test_train_scores_each_stage_on_test_when_asked(tmp_path: Path, dummy_backbone):
    requested: list[str] = []

    results = cli.train(
        _two_stage_config(tmp_path), construct=None, make_dataset=_datasets(requested), evaluate_test=True
    )

    assert "test" in requested
    assert {"stage1_test_metrics", "stage2_test_metrics", "test_metrics"} <= results.keys()
    assert len(json.loads((tmp_path / "history.json").read_text())["test_loss"]) == 1


def test_evaluate_test_flag_defaults_off():
    import argparse

    parser = cli.add_train_arguments(argparse.ArgumentParser())
    assert parser.parse_args([]).evaluate_test is False
    assert parser.parse_args(["--evaluate_test"]).evaluate_test is True
