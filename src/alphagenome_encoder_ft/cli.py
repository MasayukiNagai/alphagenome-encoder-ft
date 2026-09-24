"""Command-line scaffolding for per-assay train and evaluate scripts.

This is the layer between a shell command and the library. It turns argparse namespaces
and JSON config files into a :class:`TrainConfig` and a :class:`Construct`, lays out a run
directory (``config.json``, ``run.json``, ``history.json``, per-stage checkpoints), wires up
wandb, and writes metrics, predictions and a plot. The training itself is
:mod:`alphagenome_encoder_ft.train`, which knows nothing about any of that; the statistics
are :mod:`alphagenome_encoder_ft.metrics`.

Each assay's script stays short because only two things differ between assays: which
dataset reader to use, and which ``Construct`` to build. Nothing here is needed to use the
package as a library.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import TrainConfig, load_train_config, merge_train_config, parse_hidden_sizes
from .constructs import Construct
from .data import create_dataloader
from .metrics import regression_metrics
from .model import AlphaGenomeEncoderModel
from .train import (
    create_scheduler,
    create_stage1_optimizer,
    create_stage2_optimizer,
    evaluate,
    load_checkpoint,
    run_training_stage,
    run_two_stage_training,
    scheduler_stepper,
)

RUN_METADATA_FILENAME = "run.json"

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_STAGE_FLAGS: list[tuple[str, dict[str, Any]]] = [
    ("num_epochs", {"type": int}),
    ("early_stopping_patience", {"type": int}),
    ("val_evals_per_epoch", {"type": int}),
    ("head_lr", {"type": float}),
    ("dropout", {"type": float}),
    ("lr_scheduler", {"type": str, "choices": ["constant", "cosine", "plateau"]}),
    ("plateau_factor", {"type": float}),
    ("plateau_patience", {"type": int}),
    ("plateau_min_lr", {"type": float}),
]

# Both stage sections have the same fields, so their flags carry the section name:
# ``--stage1_num_epochs``, ``--stage2_encoder_lr``. Other sections' flags are the bare field.
_PREFIXED_SECTIONS = {"stage1", "stage2"}

_SECTION_FLAGS: dict[str, list[tuple[str, dict[str, Any]]]] = {
    "data": [
        ("batch_size", {"type": int}),
        ("num_workers", {"type": int}),
        ("max_shift", {"type": int}),
        ("subset_frac", {"type": float}),
        ("rc_prob", {"type": float}),
        ("shift_prob", {"type": float}),
        ("reverse_complement", {"action": argparse.BooleanOptionalAction}),
        ("random_shift", {"action": argparse.BooleanOptionalAction}),
        ("pin_memory", {"action": argparse.BooleanOptionalAction}),
        ("drop_last", {"action": argparse.BooleanOptionalAction}),
    ],
    "head": [
        ("pooling_type", {"type": str, "choices": ["flatten", "center", "mean", "sum", "max"]}),
        ("center_bp", {"type": int}),
        ("hidden_sizes", {"type": str}),
        ("activation", {"type": str, "choices": ["relu", "gelu"]}),
        ("head_type", {"type": str, "choices": ["mpra", "deepstarr"]}),
        ("num_outputs", {"type": int}),
        ("norm_type", {"type": str, "choices": ["layer", "batch", "group", "none"]}),
    ],
    "optim": [
        ("optimizer", {"type": str, "choices": ["adam", "adamw"]}),
        ("weight_decay", {"type": float}),
        ("gradient_accumulation_steps", {"type": int}),
        ("gradient_clip", {"type": float}),
    ],
    "stage1": _STAGE_FLAGS,
    "stage2": [*_STAGE_FLAGS, ("encoder_lr", {"type": float})],
    "checkpoint": [
        ("pretrained_weights", {"type": str}),
        ("checkpoint_dir", {"type": str}),
        ("save_mode", {"type": str, "choices": ["minimal", "full", "head"]}),
    ],
    "logging": [
        ("use_wandb", {"action": argparse.BooleanOptionalAction}),
        ("wandb_project", {"type": str}),
        ("wandb_name", {"type": str}),
    ],
    "runtime": [
        ("device", {"type": str}),
        ("use_amp", {"action": argparse.BooleanOptionalAction}),
        ("seed", {"type": int}),
    ],
}


def add_train_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add ``--config`` plus one override flag per ``TrainConfig`` field."""

    parser.add_argument("--config", type=str, default=None, help="JSON TrainConfig; flags override it")
    for section, flags in _SECTION_FLAGS.items():
        for name, kwargs in flags:
            parser.add_argument(f"--{_flag_name(section, name)}", default=None, **kwargs)
    parser.add_argument(
        "--resume_from_stage2",
        action="store_true",
        help="skip stage 1 and restart stage 2 from <checkpoint_dir>/stage1/best.pt",
    )
    parser.add_argument("--show_progress", action=argparse.BooleanOptionalAction, default=False)
    return parser


def _flag_name(section: str, name: str) -> str:
    return f"{section}_{name}" if section in _PREFIXED_SECTIONS else name


def add_construct_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Flags that override the driver's default ``Construct``."""

    parser.add_argument("--construct_prefix", type=str, default=None)
    parser.add_argument("--construct_suffix", type=str, default=None)
    parser.add_argument("--construct_length", type=int, default=None)
    parser.add_argument(
        "--no-construct",
        dest="no_construct",
        action="store_true",
        help="Feed the dataset sequences to the model unchanged",
    )
    return parser


def build_overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for section, flags in _SECTION_FLAGS.items():
        values = {name: getattr(args, _flag_name(section, name), None) for name, _ in flags}
        if section == "head" and values.get("hidden_sizes") is not None:
            values["hidden_sizes"] = parse_hidden_sizes(values["hidden_sizes"])
        overrides[section] = {name: value for name, value in values.items() if value is not None}
    return overrides


def load_config(parser: argparse.ArgumentParser, args: argparse.Namespace) -> TrainConfig:
    try:
        config = merge_train_config(load_train_config(args.config), build_overrides(args))
        config.validate()
    except ValueError as exc:
        parser.error(str(exc))
    return config


def resolve_construct(args: argparse.Namespace, default: Construct | None) -> Construct | None:
    """Apply ``--construct_*`` / ``--no-construct`` on top of the driver's default."""

    if getattr(args, "no_construct", False):
        return None
    base = default.to_dict() if default is not None else {"prefix": "", "suffix": "", "length": None}
    if args.construct_prefix is not None:
        base["prefix"] = args.construct_prefix
    if args.construct_suffix is not None:
        base["suffix"] = args.construct_suffix
    if args.construct_length is not None:
        base["length"] = args.construct_length
    return Construct.from_dict(base)


def dataset_kwargs(config: TrainConfig, *, augment: bool) -> dict[str, Any]:
    """``MPRADataset`` keyword arguments from config; augmentation only when ``augment``."""

    return {
        "reverse_complement": config.data.reverse_complement if augment else False,
        "rc_prob": config.data.rc_prob,
        "random_shift": config.data.random_shift if augment else False,
        "shift_prob": config.data.shift_prob,
        "max_shift": config.data.max_shift,
        "subset_frac": config.data.subset_frac,
        "seed": config.runtime.seed,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _resolve_device(config: TrainConfig) -> torch.device:
    return torch.device(config.runtime.device or ("cuda" if torch.cuda.is_available() else "cpu"))


def train(
    config: TrainConfig,
    *,
    construct: Construct | None,
    make_dataset: Callable[[str], Dataset],
    metadata: dict[str, Any] | None = None,
    show_progress: bool = False,
    resume_from_stage2: bool = False,
) -> dict[str, Any]:
    """Two-stage fine-tune driven by ``config``, or stage 1 only when ``config.stage2`` is None.

    ``make_dataset(split)`` returns the dataset for ``"train"``, ``"val"`` or ``"test"``,
    already carrying ``construct`` and the augmentation flags. ``metadata`` is recorded
    alongside the construct in ``run.json`` so evaluation can find the data again.
    ``resume_from_stage2`` skips stage 1 and restarts stage 2 from ``stage1/best.pt``.
    """

    if resume_from_stage2 and config.stage2 is None:
        raise ValueError("resume_from_stage2 needs a stage2 section")

    torch.manual_seed(config.runtime.seed)
    device = _resolve_device(config)
    print(f"Using device: {device}", flush=True)

    run_dir = Path(config.checkpoint.checkpoint_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = make_dataset("train")
    val_dataset = make_dataset("val")
    test_dataset = make_dataset("test")
    if len(train_dataset) == 0:
        raise ValueError("Training split is empty")

    if construct is not None and construct.length is not None:
        input_length = construct.length
    else:
        input_length = int(train_dataset[0][0].shape[0])
    print(f"Model input length: {input_length}")

    with open(run_dir / "config.json", "w") as handle:
        json.dump(config.to_dict(), handle, indent=2)
    with open(run_dir / RUN_METADATA_FILENAME, "w") as handle:
        json.dump(
            {
                **(metadata or {}),
                "construct": construct.to_dict() if construct is not None else None,
                "input_length": input_length,
            },
            handle,
            indent=2,
        )

    print(f"Loading pretrained weights from {config.checkpoint.pretrained_weights}...")
    model = AlphaGenomeEncoderModel.from_pretrained(
        config.checkpoint.pretrained_weights,
        config.head,
        device=device,
        construct=construct,
    )
    model.initialize_head(input_length, device)
    model.eval()

    n_trainable = sum(p.numel() for p in model.head.parameters())
    n_total = sum(p.numel() for p in model.parameters())
    print("AlphaGenomeEncoderModel created.")
    print(f"  Trainable (head)   : {n_trainable:,}")
    print(f"  Frozen (backbone)  : {n_total - n_trainable:,}")
    print(f"  Total parameters   : {n_total:,}")
    print(f"  Trainable fraction : {100 * n_trainable / n_total:.4f}%")
    print(f"  Construct          : {construct}")
    print()
    print("Head architecture:")
    print(model.head)

    loader_kwargs = {
        "batch_size": config.data.batch_size,
        "num_workers": config.data.num_workers,
        "pin_memory": config.data.pin_memory,
    }
    train_loader = create_dataloader(
        train_dataset, shuffle=True, drop_last=config.data.drop_last, **loader_kwargs
    )
    if len(train_loader) == 0:
        raise ValueError(
            f"drop_last leaves no training batch: {len(train_dataset)} rows < batch_size "
            f"{config.data.batch_size}"
        )
    val_loader = create_dataloader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = create_dataloader(test_dataset, shuffle=False, **loader_kwargs)
    print(f"  Train batches : {len(train_loader):,}")
    print(f"  Val batches   : {len(val_loader):,}")
    print(f"  Test batches  : {len(test_loader):,}")

    stage1_optimizer = create_stage1_optimizer(config, model)
    stage1_scheduler = create_scheduler(config.stage1, stage1_optimizer)
    stage1_scheduler_step = scheduler_stepper(config.stage1.lr_scheduler)

    epoch_logger = _wandb_logger(config)

    if config.stage2 is not None:
        results = run_two_stage_training(
            model,
            train_loader,
            stage1_optimizer=stage1_optimizer,
            stage2_optimizer_factory=lambda model_obj: create_stage2_optimizer(config, model_obj),
            config=config,
            device=device,
            val_loader=val_loader,
            stage1_scheduler=stage1_scheduler,
            stage1_scheduler_step=stage1_scheduler_step,
            epoch_callback=epoch_logger,
            show_progress=show_progress,
            resume_from_stage2=resume_from_stage2,
        )
    else:
        results = run_training_stage(
            model,
            train_loader,
            optimizer=stage1_optimizer,
            config=config,
            stage_config=config.stage1,
            device=device,
            stage="stage1",
            train_encoder=False,
            val_loader=val_loader,
            scheduler=stage1_scheduler,
            scheduler_step=stage1_scheduler_step,
            checkpoint_dir=run_dir / "stage1",
            epoch_callback=epoch_logger,
            show_progress=show_progress,
        )

    stages: list[tuple[str, dict[str, Any]]] = [("stage1", results)]
    if "stage2" in results:
        stages = [("stage1", results["stage1"]), ("stage2", results["stage2"])]

    final_metrics: dict[str, float] | None = None
    final_epoch = 0.0
    for stage_name, stage_result in stages:
        load_checkpoint(stage_result["best_checkpoint_path"], model, map_location=device)
        test_metrics = evaluate(model, test_loader, device, use_amp=config.runtime.use_amp)
        test_epoch = float(stage_result.get("best_epoch", 0))
        results[f"{stage_name}_test_metrics"] = test_metrics
        print(
            f"[{stage_name}] final test | epoch {test_epoch:g} | "
            f"test_loss={test_metrics['loss']:.4f} | "
            f"test_pearson={test_metrics.get('pearson', float('nan')):.4f}"
        )
        if epoch_logger is not None:
            epoch_logger(
                {
                    "stage": stage_name,
                    "epoch": test_epoch,
                    "test_loss": test_metrics["loss"],
                    "test_pearson": test_metrics.get("pearson", float("nan")),
                    "event": "final_test",
                }
            )
        final_metrics, final_epoch = test_metrics, test_epoch

    assert final_metrics is not None
    results["test_metrics"] = final_metrics
    results["history"]["test_loss"].append(final_metrics["loss"])
    results["history"]["test_pearson"].append(final_metrics.get("pearson", float("nan")))
    results["history"]["test_epoch"].append(final_epoch)
    with open(run_dir / "history.json", "w") as handle:
        json.dump(results["history"], handle, indent=2)

    if config.logging.use_wandb:
        import wandb

        wandb.finish()
    return results


def _wandb_logger(config: TrainConfig) -> Callable[[dict[str, Any]], None] | None:
    if not config.logging.use_wandb:
        return None
    try:
        import wandb
    except ImportError:
        print("wandb is not installed; continuing without wandb")
        config.logging.use_wandb = False
        return None

    wandb.init(project=config.logging.wandb_project, name=config.logging.wandb_name, config=config.to_dict())

    def _log(metrics: dict[str, Any]) -> None:
        stage = str(metrics["stage"])
        payload = {"epoch": float(metrics["epoch"])}
        for key, value in metrics.items():
            if key not in {"stage", "epoch"}:
                payload[f"{stage}/{key}"] = value
        wandb.log(payload)

    return _log


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def add_evaluate_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument(
        "--input_tsv",
        type=str,
        default=None,
        help="Defaults to the input_tsv recorded in run.json two levels above the checkpoint",
    )
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--use_amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=None)
    return parser


def load_run_metadata(checkpoint_path: Path) -> dict[str, Any]:
    """``run.json`` written by ``train`` (``<run_dir>/<stage>/best.pt`` -> ``<run_dir>/run.json``)."""

    for candidate in (checkpoint_path.parent.parent, checkpoint_path.parent):
        path = candidate / RUN_METADATA_FILENAME
        if path.exists():
            with open(path) as handle:
                return json.load(handle)
    return {}


def resolve_input_tsv(parser: argparse.ArgumentParser, args: argparse.Namespace, checkpoint_path: Path) -> Path:
    input_tsv = args.input_tsv or load_run_metadata(checkpoint_path).get("input_tsv")
    if not input_tsv:
        parser.error("--input_tsv is required (no run.json with input_tsv found next to the checkpoint)")
    return Path(input_tsv)


@torch.no_grad()
def collect_predictions(
    model: AlphaGenomeEncoderModel,
    data_loader,
    *,
    device: torch.device,
    use_amp: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(y_true, y_pred)``: ``(N,)`` for scalar heads, ``(N, K)`` otherwise."""

    model.eval()
    targets_all: list[np.ndarray] = []
    preds_all: list[np.ndarray] = []
    for sequences, targets in data_loader:
        sequences = sequences.to(device)
        organism_idx = torch.zeros(sequences.shape[0], dtype=torch.long, device=device)
        if use_amp and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                preds = model(sequences, organism_idx)
        else:
            preds = model(sequences, organism_idx)
        targets_all.append(targets.float().cpu().numpy())
        preds_all.append(preds.detach().float().cpu().numpy())

    if not targets_all:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)
    y_true = np.concatenate(targets_all, axis=0).astype(np.float32, copy=False)
    y_pred = np.concatenate(preds_all, axis=0).astype(np.float32, copy=False)
    y_pred = y_pred.reshape(y_true.shape)
    return y_true, y_pred


def save_predictions(
    path: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    extra_columns: dict[str, list[Any]] | None = None,
) -> None:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tracks = 1 if y_true.ndim == 1 else y_true.shape[1]
    y_true2 = y_true.reshape(len(y_true), tracks)
    y_pred2 = y_pred.reshape(len(y_pred), tracks)
    extra = extra_columns or {}
    if tracks == 1:
        header = ["y", "y_pred"]
    else:
        header = [f"y_{k}" for k in range(tracks)] + [f"y_pred_{k}" for k in range(tracks)]
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", *extra.keys(), *header])
        for idx in range(len(y_true2)):
            writer.writerow(
                [idx, *(extra[key][idx] for key in extra), *y_true2[idx].tolist(), *y_pred2[idx].tolist()]
            )


def save_scatter_plot(path: Path, y_true: np.ndarray, y_pred: np.ndarray, metrics: dict[str, Any]) -> bool:
    """Write a y vs y_pred scatter. Returns False (with a warning) if matplotlib is absent."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping the scatter plot")
        return False

    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, s=10, alpha=0.6, edgecolors="none")
    finite = np.concatenate([y_true, y_pred])
    finite = finite[np.isfinite(finite)]
    if finite.size:
        lo, hi = float(finite.min()), float(finite.max())
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.0, color="black")
    ax.set_xlabel("y")
    ax.set_ylabel("y_pred")
    ax.set_title("Test set: y vs y_pred")
    ax.text(
        0.03,
        0.97,
        "\n".join(
            [
                f"n = {metrics['n_samples']}",
                f"Pearson r = {metrics['pearsonr']:.4f}",
                f"Spearman rho = {metrics['spearmanr']:.4f}",
                f"RMSE = {metrics['rmse']:.4f}",
            ]
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return True


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    *,
    make_test_dataset: Callable[[Construct | None], Dataset],
    output_dir: str | Path | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
    pin_memory: bool | None = None,
    device: str | None = None,
    use_amp: bool | None = None,
    extra_columns: dict[str, list[Any]] | None = None,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Score a checkpoint on ``make_test_dataset(model.construct)`` and write metrics/predictions/plot.

    Loader and runtime arguments default to the values in the checkpoint's saved config.
    ``extra_columns`` (e.g. sequence ids, aligned with the dataset order) are written into
    ``test_predictions.csv``. Returns ``(metrics, y_true, y_pred)`` for callers that want to
    add their own stratified metrics.
    """

    checkpoint_path = Path(checkpoint_path).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    # The config is provenance a checkpoint need not carry; without it the loader and runtime
    # settings fall back to TrainConfig defaults, which the arguments below override anyway.
    saved_config = checkpoint.get("config")
    config = TrainConfig.from_dict(saved_config) if saved_config else TrainConfig()

    batch_size = batch_size if batch_size is not None else config.data.batch_size
    num_workers = num_workers if num_workers is not None else config.data.num_workers
    pin_memory = pin_memory if pin_memory is not None else config.data.pin_memory
    use_amp = use_amp if use_amp is not None else config.runtime.use_amp
    resolved_device = torch.device(device or config.runtime.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {resolved_device}")

    out_dir = Path(output_dir) if output_dir is not None else checkpoint_path.parent / f"{checkpoint_path.stem}_test_eval"
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    model = AlphaGenomeEncoderModel.from_checkpoint(checkpoint_path, device=resolved_device)
    test_dataset = make_test_dataset(model.construct)
    test_loader = create_dataloader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
    )
    y_true, y_pred = collect_predictions(model, test_loader, device=resolved_device, use_amp=use_amp)

    metrics = regression_metrics(y_true, y_pred)
    metrics.update(
        {
            "checkpoint_path": str(checkpoint_path),
            "output_dir": str(out_dir),
            "construct": model.construct.to_dict() if model.construct is not None else None,
            "input_length": model.input_length,
            "save_mode": checkpoint.get("save_mode"),
        }
    )

    if extra_columns is not None:
        for key, values in extra_columns.items():
            if len(values) != len(y_true):
                raise ValueError(f"extra column {key!r} has {len(values)} rows but there are {len(y_true)} predictions")

    save_predictions(out_dir / "test_predictions.csv", y_true, y_pred, extra_columns)
    save_scatter_plot(out_dir / "y_vs_y_pred.png", y_true, y_pred, metrics)
    return metrics, y_true, y_pred


def write_metrics(output_dir: str | Path, metrics: dict[str, Any]) -> Path:
    path = Path(output_dir) / "test_metrics.json"
    with open(path, "w") as handle:
        json.dump(metrics, handle, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics to {path}")
    return path
