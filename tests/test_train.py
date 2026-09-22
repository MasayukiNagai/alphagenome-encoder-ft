from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from alphagenome_encoder_ft.config import OptimConfig, TrainConfig
from alphagenome_encoder_ft.constructs import Construct
from alphagenome_encoder_ft.heads import DeepSTARRHead, MPRAHead
from alphagenome_encoder_ft.model import AlphaGenomeEncoderModel
import alphagenome_encoder_ft.train as train_module
from alphagenome_encoder_ft.train import create_scheduler, evaluate, load_checkpoint, run_training_stage, run_two_stage_training, save_checkpoint
from conftest import DummyAlphaGenome


def _make_loader():
    torch.manual_seed(0)
    sequences = torch.randn(12, 2, 4)
    targets = sequences.sum(dim=(1, 2))
    return DataLoader(TensorDataset(sequences, targets), batch_size=4, shuffle=False)


def _make_config(tmp_path: Path) -> TrainConfig:
    return TrainConfig.from_dict(
        {
            "data": {"batch_size": 4},
            "head": {
                "pooling_type": "flatten",
                "hidden_sizes": [8],
                "center_bp": 256,
                "dropout": 0.1,
                "activation": "relu",
            },
            "optim": {
                "optimizer": "adam",
                "learning_rate": 1e-2,
                "weight_decay": 0.0,
                "lr_scheduler": "constant",
                "plateau_factor": 0.5,
                "plateau_patience": 2,
                "plateau_mode": "min",
                "plateau_min_lr": 0.0,
                "gradient_accumulation_steps": 1,
            },
            "stage": {
                "num_epochs": 2,
                "early_stopping_patience": 5,
                "val_evals_per_epoch": 1,
                "second_stage_lr": 1e-3,
                "second_stage_epochs": 1,
            },
            "checkpoint": {
                "pretrained_weights": "/tmp/weights.pt",
                "checkpoint_dir": str(tmp_path),
                "save_mode": "minimal",
            },
            "runtime": {
                "use_amp": False,
                "seed": 0,
            },
        }
    )


def _make_model(construct: Construct | None = None) -> AlphaGenomeEncoderModel:
    model = AlphaGenomeEncoderModel(
        DummyAlphaGenome(),
        MPRAHead(pooling_type="flatten", hidden_sizes=8),
        construct=construct,
    )
    model.initialize_head(sequence_length=2, device="cpu")
    return model


def test_run_training_stage_writes_minimal_checkpoint(tmp_path: Path):
    model = _make_model()
    loader = _make_loader()
    config = _make_config(tmp_path)
    optimizer = torch.optim.Adam(model.head.parameters(), lr=1e-2)

    result = run_training_stage(
        model,
        loader,
        optimizer=optimizer,
        config=config,
        device="cpu",
        num_epochs=2,
        stage="stage1",
        train_encoder=False,
        checkpoint_dir=tmp_path / "stage1",
    )

    assert (tmp_path / "stage1" / "best.pt").exists()
    assert result["best_checkpoint_path"] is not None


def test_two_stage_training_rejects_head_mode(tmp_path: Path):
    model = _make_model()
    loader = _make_loader()
    config = _make_config(tmp_path)
    config.checkpoint.save_mode = "head"
    optimizer = torch.optim.Adam(model.head.parameters(), lr=1e-2)

    try:
        run_two_stage_training(
            model,
            loader,
            stage1_optimizer=optimizer,
            stage2_optimizer_factory=lambda model_obj: torch.optim.Adam(
                model_obj.trainable_parameters(include_encoder=True),
                lr=1e-3,
            ),
            config=config,
            device="cpu",
        )
    except ValueError as exc:
        assert "head save_mode" in str(exc)
    else:
        raise AssertionError("Expected ValueError for head save_mode")


def test_resume_from_stage2_loads_stage1_checkpoint(tmp_path: Path):
    model = _make_model()
    loader = _make_loader()
    config = _make_config(tmp_path)
    optimizer = torch.optim.Adam(model.head.parameters(), lr=1e-2)

    run_training_stage(
        model,
        loader,
        optimizer=optimizer,
        config=config,
        device="cpu",
        num_epochs=1,
        stage="stage1",
        train_encoder=False,
        checkpoint_dir=tmp_path / "stage1",
    )

    stage2_model = _make_model()
    stage2_config = _make_config(tmp_path)
    stage2_config.stage.resume_from_stage2 = True
    result = run_two_stage_training(
        stage2_model,
        loader,
        stage1_optimizer=torch.optim.Adam(stage2_model.head.parameters(), lr=1e-2),
        stage2_optimizer_factory=lambda model_obj: torch.optim.Adam(
            model_obj.trainable_parameters(include_encoder=True),
            lr=1e-3,
        ),
        config=stage2_config,
        device="cpu",
    )

    assert result["stage2"]["best_checkpoint_path"] is not None


def test_two_stage_training_applies_second_stage_dropout_and_scheduler(tmp_path: Path, monkeypatch):
    model = _make_model()
    loader = _make_loader()
    config = _make_config(tmp_path)
    config.stage.second_stage_dropout = 0.25
    config.stage.second_stage_lr_scheduler = "plateau"

    built = []
    real_create_scheduler = train_module.create_scheduler

    def spy_create_scheduler(optim_config, optimizer, total_epochs):
        scheduler = real_create_scheduler(optim_config, optimizer, total_epochs)
        built.append((optim_config, scheduler))
        return scheduler

    monkeypatch.setattr(train_module, "create_scheduler", spy_create_scheduler)

    run_two_stage_training(
        model,
        loader,
        stage1_optimizer=torch.optim.Adam(model.head.parameters(), lr=1e-2),
        stage2_optimizer_factory=lambda model_obj: torch.optim.Adam(
            model_obj.trainable_parameters(include_encoder=True),
            lr=1e-3,
        ),
        config=config,
        device="cpu",
    )

    # The head dropout is raised for the encoder-unfrozen stage, and the stage-2
    # scheduler is built from second_stage_lr_scheduler at the second-stage lr.
    assert model.head.dropout == 0.25
    assert len(built) == 1
    stage2_optim_config, stage2_scheduler = built[0]
    assert stage2_optim_config.lr_scheduler == "plateau"
    assert stage2_optim_config.learning_rate == config.stage.second_stage_lr
    assert isinstance(stage2_scheduler, ReduceLROnPlateau)


def test_stage_config_rejects_invalid_second_stage_dropout():
    try:
        TrainConfig.from_dict(
            {
                "checkpoint": {"pretrained_weights": "/tmp/weights.pt"},
                "stage": {"second_stage_dropout": 1.0},
            }
        )
    except ValueError as exc:
        assert "stage.second_stage_dropout" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid second_stage_dropout")


def test_stage_config_rejects_unknown_second_stage_lr_scheduler():
    try:
        TrainConfig.from_dict(
            {
                "checkpoint": {"pretrained_weights": "/tmp/weights.pt"},
                "stage": {"second_stage_lr_scheduler": "linear"},
            }
        )
    except ValueError as exc:
        assert "stage.second_stage_lr_scheduler" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown second_stage_lr_scheduler")


def test_run_training_stage_runs_validation_within_each_epoch_and_emits_callbacks(tmp_path: Path):
    model = _make_model()
    train_loader = _make_loader()
    val_loader = _make_loader()
    config = _make_config(tmp_path)
    config.stage.num_epochs = 3
    config.stage.val_evals_per_epoch = 2
    optimizer = torch.optim.Adam(model.head.parameters(), lr=1e-2)
    epoch_events = []

    result = run_training_stage(
        model,
        train_loader,
        optimizer=optimizer,
        config=config,
        device="cpu",
        num_epochs=3,
        stage="stage1",
        train_encoder=False,
        val_loader=val_loader,
        checkpoint_dir=tmp_path / "stage1",
        epoch_callback=epoch_events.append,
    )

    assert len(result["history"]["train_loss"]) == 3
    assert result["history"]["val_epoch"] == pytest.approx([1 / 3, 2 / 3, 4 / 3, 5 / 3, 7 / 3, 8 / 3])
    assert result["history"]["test_epoch"] == []
    assert [event["epoch"] for event in epoch_events] == [1.0, 2.0, 3.0]
    assert epoch_events[0]["val_loss"] >= 0.0
    assert "test_loss" not in epoch_events[-1]
    assert epoch_events[1]["val_loss"] >= 0.0


def test_run_training_stage_validates_once_per_epoch_when_requested(tmp_path: Path):
    model = _make_model()
    train_loader = _make_loader()
    val_loader = _make_loader()
    config = _make_config(tmp_path)
    config.stage.num_epochs = 2
    config.stage.val_evals_per_epoch = 1
    optimizer = torch.optim.Adam(model.head.parameters(), lr=1e-2)

    result = run_training_stage(
        model,
        train_loader,
        optimizer=optimizer,
        config=config,
        device="cpu",
        num_epochs=2,
        stage="stage1",
        train_encoder=False,
        val_loader=val_loader,
        checkpoint_dir=tmp_path / "stage1",
    )

    assert result["history"]["val_epoch"] == [1.0, 2.0]


def test_run_training_stage_deduplicates_dense_validation_points(tmp_path: Path):
    model = _make_model()
    train_loader = _make_loader()
    val_loader = _make_loader()
    config = _make_config(tmp_path)
    config.stage.num_epochs = 1
    config.stage.val_evals_per_epoch = 5
    optimizer = torch.optim.Adam(model.head.parameters(), lr=1e-2)

    result = run_training_stage(
        model,
        train_loader,
        optimizer=optimizer,
        config=config,
        device="cpu",
        num_epochs=1,
        stage="stage1",
        train_encoder=False,
        val_loader=val_loader,
        checkpoint_dir=tmp_path / "stage1",
    )

    assert result["history"]["val_epoch"] == [1 / 3, 2 / 3, 1.0]


def test_run_training_stage_early_stopping_counts_validation_events(tmp_path: Path):
    model = _make_model()
    train_loader = _make_loader()
    val_loader = _make_loader()
    config = _make_config(tmp_path)
    config.stage.num_epochs = 10
    config.stage.early_stopping_patience = 2
    config.stage.val_evals_per_epoch = 3
    optimizer = torch.optim.Adam(model.head.parameters(), lr=1e-2)

    original_evaluate = train_module.evaluate
    eval_losses = iter([1.0] + [2.0] * 20)

    def fake_evaluate(*args, **kwargs):
        return {"loss": next(eval_losses), "pearson": 0.0}

    train_module.evaluate = fake_evaluate
    try:
        result = run_training_stage(
            model,
            train_loader,
            optimizer=optimizer,
            config=config,
            device="cpu",
            num_epochs=10,
            stage="stage1",
            train_encoder=False,
            val_loader=val_loader,
            checkpoint_dir=tmp_path / "stage1",
        )
    finally:
        train_module.evaluate = original_evaluate

    assert len(result["history"]["val_epoch"]) == 7
    assert result["best_epoch"] == 1 / 3


def test_load_checkpoint_then_evaluate_best_checkpoint(tmp_path: Path):
    model = _make_model()
    train_loader = _make_loader()
    test_loader = _make_loader()
    config = _make_config(tmp_path)
    optimizer = torch.optim.Adam(model.head.parameters(), lr=1e-2)

    result = run_training_stage(
        model,
        train_loader,
        optimizer=optimizer,
        config=config,
        device="cpu",
        num_epochs=2,
        stage="stage1",
        train_encoder=False,
        checkpoint_dir=tmp_path / "stage1",
    )

    load_checkpoint(result["best_checkpoint_path"], model, map_location="cpu")
    metrics = evaluate(model, test_loader, device="cpu")

    assert metrics["loss"] >= 0.0
    assert "pearson" in metrics


def test_create_scheduler_uses_plateau_config():
    optimizer = torch.optim.Adam([torch.nn.Parameter(torch.tensor(1.0))], lr=1e-2)
    optim_config = OptimConfig(
        lr_scheduler="plateau",
        plateau_factor=0.25,
        plateau_patience=4,
        plateau_mode="min",
        plateau_min_lr=1e-5,
    )

    scheduler = create_scheduler(optim_config, optimizer, total_epochs=5)

    assert isinstance(scheduler, ReduceLROnPlateau)
    assert scheduler.factor == 0.25
    assert scheduler.patience == 4
    assert scheduler.mode == "min"
    assert scheduler.min_lrs == [1e-5]


def test_train_config_rejects_invalid_plateau_settings():
    try:
        TrainConfig.from_dict(
            {
                "checkpoint": {"pretrained_weights": "/tmp/weights.pt"},
                "optim": {"plateau_factor": 1.0},
            }
        )
    except ValueError as exc:
        assert "optim.plateau_factor" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid plateau_factor")


def test_save_checkpoint_persists_head_type_mpra_default(tmp_path: Path):
    model = _make_model()
    config = _make_config(tmp_path)
    path = save_checkpoint(
        tmp_path / "mpra.pt",
        model,
        save_mode="minimal",
        config=config,
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["head_type"] == "mpra"
    assert payload["head_config"]["num_outputs"] == 1


def test_save_checkpoint_persists_the_construct_and_input_length(tmp_path: Path):
    construct = Construct(prefix="AC", suffix="GT", length=281)
    path = save_checkpoint(
        tmp_path / "with_construct.pt",
        _make_model(construct),
        save_mode="minimal",
        config=_make_config(tmp_path),
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)

    assert payload["construct"] == {"prefix": "AC", "suffix": "GT", "length": 281}
    # input_length is what the head was actually built for, set by initialize_head.
    assert payload["input_length"] == 2
    assert "construct_config" not in payload


def test_save_checkpoint_records_a_null_construct(tmp_path: Path):
    path = save_checkpoint(
        tmp_path / "no_construct.pt",
        _make_model(),
        save_mode="minimal",
        config=_make_config(tmp_path),
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)

    assert payload["construct"] is None
    assert payload["input_length"] == 2


def test_head_type_dispatch_defaults_to_mpra_when_absent():
    from alphagenome_encoder_ft.config import build_head

    head = build_head("mpra", {"pooling_type": "flatten", "hidden_sizes": [8]})
    assert isinstance(head, MPRAHead)
    assert not isinstance(head, DeepSTARRHead)


def test_save_checkpoint_persists_head_type_deepstarr(tmp_path: Path):
    # build a deepstarr config and a matching model, assert the saved payload
    # carries the dispatch field.
    config = TrainConfig.from_dict(
        {
            "head": {
                "head_type": "deepstarr",
                "pooling_type": "flatten",
                "hidden_sizes": [8],
                "center_bp": 256,
                "dropout": 0.5,
                "activation": "relu",
                "num_outputs": 2,
            },
            "checkpoint": {
                "pretrained_weights": "/tmp/weights.pt",
                "checkpoint_dir": str(tmp_path),
                "save_mode": "minimal",
            },
            "stage": {"second_stage_lr": 1e-3},
        }
    )
    model = AlphaGenomeEncoderModel(DummyAlphaGenome(), DeepSTARRHead(pooling_type="flatten", hidden_sizes=8))
    model.initialize_head(sequence_length=2, device="cpu")
    path = save_checkpoint(
        tmp_path / "deepstarr.pt",
        model,
        save_mode="minimal",
        config=config,
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["head_type"] == "deepstarr"
    assert payload["head_config"]["num_outputs"] == 2


# -------------------------
# Progress bar (tqdm is optional)
# -------------------------


def test_progress_iterator_passes_through_when_not_asked_for():
    rows = [1, 2, 3]
    iterator, showing = train_module._progress_iterator(rows, total=3, show_progress=False)
    assert iterator is rows
    assert showing is False


def test_progress_iterator_returns_a_bar_when_tqdm_is_available():
    if train_module.tqdm is None:
        pytest.skip("tqdm is not installed")
    iterator, showing = train_module._progress_iterator([1, 2, 3], total=3, show_progress=True)
    assert showing is True
    assert hasattr(iterator, "set_postfix")


def test_progress_iterator_warns_once_and_falls_back_without_tqdm(monkeypatch, capsys):
    # tqdm lives in the optional train group, so --show_progress must say something
    # rather than silently doing nothing.
    monkeypatch.setattr(train_module, "tqdm", None)
    monkeypatch.setattr(train_module, "_warned_about_tqdm", False)
    rows = [1, 2, 3]

    iterator, showing = train_module._progress_iterator(rows, total=3, show_progress=True)
    train_module._progress_iterator(rows, total=3, show_progress=True)

    assert iterator is rows
    assert showing is False
    assert capsys.readouterr().out.count("tqdm is not installed") == 1


def test_training_runs_with_show_progress_and_no_tqdm(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(train_module, "tqdm", None)
    model = _make_model()
    config = _make_config(tmp_path)

    result = run_training_stage(
        model,
        _make_loader(),
        optimizer=torch.optim.Adam(model.head.parameters(), lr=1e-2),
        config=config,
        device="cpu",
        num_epochs=1,
        stage="stage1",
        train_encoder=False,
        checkpoint_dir=tmp_path / "stage1",
        show_progress=True,
    )

    assert result["best_checkpoint_path"] is not None
