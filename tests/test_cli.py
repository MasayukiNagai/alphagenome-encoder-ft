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


def test_stage1_only_drops_the_stage2_section(tmp_path: Path):
    import argparse

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_two_stage_config(tmp_path).to_dict()))
    parser = cli.add_train_arguments(argparse.ArgumentParser())

    two_stage = cli.load_config(parser, parser.parse_args(["--config", str(config_path)]))
    stage1_only = cli.load_config(parser, parser.parse_args(["--config", str(config_path), "--stage1_only"]))

    assert two_stage.stage2 is not None
    assert stage1_only.stage2 is None


def test_stage1_only_trains_no_stage2(tmp_path: Path, dummy_backbone):
    config = _two_stage_config(tmp_path)
    config.stage2 = None

    results = cli.train(config, construct=None, make_dataset=_datasets([]))

    assert "stage2" not in results
    assert (tmp_path / "stage1" / "best.pt").exists() and not (tmp_path / "stage2").exists()
    assert json.loads((tmp_path / "config.json").read_text())["stage2"] is None


def test_train_reports_each_stages_best_validation_to_wandb(tmp_path: Path, dummy_backbone, monkeypatch):
    import sys
    import types

    summary: dict = {}
    init_kwargs: dict = {}
    fake = types.SimpleNamespace(
        init=lambda **kwargs: init_kwargs.update(kwargs),
        log=lambda payload: None,
        define_metric=lambda *args, **kwargs: None,
        finish=lambda: None,
        run=types.SimpleNamespace(summary=summary),
    )
    monkeypatch.setitem(sys.modules, "wandb", fake)
    config = _two_stage_config(tmp_path)
    config.logging.use_wandb = True
    config.stage1.val_evals_per_epoch = 2

    results = cli.train(config, construct=None, make_dataset=_datasets([]))

    history = json.loads((tmp_path / "history.json").read_text())
    stage1_evals = len(results["stage1"]["history"]["val_loss"])
    stage1_losses = history["val_loss"][:stage1_evals]
    i = min(range(stage1_evals), key=stage1_losses.__getitem__)
    assert summary["stage1/best_val_loss"] == stage1_losses[i]
    assert summary["stage1/best_val_pearson"] == history["val_pearson"][i]
    assert summary["stage1/best_epoch"] == history["val_epoch"][i]
    assert summary["stage2/best_val_loss"] == min(history["val_loss"][stage1_evals:])
    assert init_kwargs["config"]["head_hidden_sizes"] == "8"



def test_train_logs_every_validation_pass_with_learning_rates(tmp_path: Path, dummy_backbone, monkeypatch):
    import sys
    import types

    rows: list[dict] = []
    defined: list[tuple] = []
    fake = types.SimpleNamespace(
        init=lambda **kwargs: None,
        log=rows.append,
        define_metric=lambda name, **kwargs: defined.append((name, kwargs.get("step_metric"))),
        finish=lambda: None,
        run=types.SimpleNamespace(summary={}),
    )
    monkeypatch.setitem(sys.modules, "wandb", fake)
    config = _two_stage_config(tmp_path)
    config.logging.use_wandb = True
    config.stage1.val_evals_per_epoch = 2
    config.stage2.val_evals_per_epoch = 2

    results = cli.train(config, construct=None, make_dataset=_datasets([]))

    history = json.loads((tmp_path / "history.json").read_text())
    n_stage1 = len(results["stage1"]["history"]["val_epoch"])
    val_rows = [r for r in rows if "stage1/val_loss" in r or "stage2/val_loss" in r]
    assert len(val_rows) == len(history["val_epoch"])
    assert all("stage1/lr_head" in r for r in val_rows[:n_stage1])
    assert all("stage2/lr_encoder" in r and "stage2/lr_head" in r for r in val_rows[n_stage1:])
    # history: learning rates aligned with val_epoch; no encoder rate in stage 1
    assert len(history["lr_head"]) == len(history["lr_encoder"]) == len(history["val_epoch"])
    assert history["lr_encoder"][:n_stage1] == [None] * n_stage1
    assert history["lr_encoder"][n_stage1:] == [config.stage2.encoder_lr] * (len(history["val_epoch"]) - n_stage1)
    assert ("stage1/*", "epoch") in defined and ("stage2/*", "epoch") in defined
