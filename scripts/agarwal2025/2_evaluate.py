#!/usr/bin/env python
"""Score a lentiMPRA checkpoint on its validation or test split.

The construct and the holdout come from the checkpoint and its run.json, so the input is
assembled, and the rows selected, exactly as during training — whether the run held out
chromosomes or the publication's folds. A run.json with no ``split_by`` predates the fold
option and is read as a chromosome split.

Predictions average the forward and reverse-complement inputs (the whole assembled
construct is flipped, as in RC augmentation); ``--forward_only`` scores the forward strand
alone. The forward-only metrics are recorded under ``forward_only`` either way.
**For a number comparable with the JAX implementation pass ``--forward_only``**:
``alphagenome_FT_MPRA/scripts/test_ft_model_mpra.py`` scores the forward strand alone, so
the RC average is not the like-for-like number.

Writes ``<split>_metrics.json``, ``<split>_predictions.csv`` and ``<split>_y_vs_y_pred.png``
into ``--output_dir`` (default: ``evaluation/`` next to the checkpoint).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from alphagenome_encoder_ft import Construct
from alphagenome_encoder_ft.cli import (
    add_evaluate_arguments,
    evaluate_checkpoint,
    load_run_metadata,
    resolve_input_tsv,
    write_metrics,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import (  # noqa: E402
    LentiMPRAAgarwal2025SplitDataset,
    Splitter,
    foreign_split_flags,
)


def check_insert_length(construct: Construct | None, dataset: LentiMPRAAgarwal2025SplitDataset) -> None:
    """The construct must close exactly around the insert, with nothing trimmed or padded."""

    if construct is None or construct.length is None or construct.window_start is not None or not len(dataset):
        return
    expected = construct.length - len(construct.prefix) - len(construct.suffix)
    actual = len(dataset.inserts[0])
    if actual != expected:
        raise ValueError(
            f"This checkpoint expects {expected} bp inserts but the dataset yields {actual} bp. "
            "Retrain the checkpoint against the current construct."
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a lentiMPRA checkpoint")
    add_evaluate_arguments(parser)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--forward_only", action="store_true",
                        help="score the forward strand only instead of the forward/RC average "
                             "(this is what the JAX evaluation does)")
    parser.add_argument("--val_chroms", nargs="+", default=None,
                        help="a chromosome-split run: defaults to the val_chroms in run.json")
    parser.add_argument("--test_chroms", nargs="+", default=None,
                        help="a chromosome-split run: defaults to the test_chroms in run.json")
    parser.add_argument("--train_folds", nargs="+", type=int, default=None,
                        help="a fold-split run: defaults to the train_folds in run.json")
    parser.add_argument("--val_folds", nargs="+", type=int, default=None,
                        help="a fold-split run: defaults to the val_folds in run.json")
    parser.add_argument("--test_folds", nargs="+", type=int, default=None,
                        help="a fold-split run: defaults to the test_folds in run.json")
    return parser


def check_split_flags(parser: argparse.ArgumentParser, args: argparse.Namespace, split_by: str) -> None:
    """Refuse an override belonging to the other holdout: this run's split is already fixed."""

    given = foreign_split_flags(split_by, vars(args))
    if given:
        parser.error(
            f"{', '.join(given)} {'has' if len(given) == 1 else 'have'} no meaning for this "
            f"checkpoint: run.json records a {split_by} split"
        )


def main() -> dict[str, Any]:
    parser = build_arg_parser()
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint_path).resolve()
    if not checkpoint_path.exists():
        parser.error(f"Checkpoint not found: {checkpoint_path}")
    input_tsv = resolve_input_tsv(parser, args, checkpoint_path)
    run = load_run_metadata(checkpoint_path)
    # A run.json with no split_by predates the fold option, so it is a chromosome split.
    split_by = str(run.get("split_by") or "chrom")
    check_split_flags(parser, args, split_by)
    try:
        splitter = Splitter.from_run_metadata(
            run,
            val_chroms=args.val_chroms,
            test_chroms=args.test_chroms,
            train_folds=args.train_folds,
            val_folds=args.val_folds,
            test_folds=args.test_folds,
        )
    except ValueError as exc:
        parser.error(str(exc))
    # Targets are read, and standardized, as in training; metrics are on that scale.
    target_columns = run.get("target_columns", ["mean_value"])
    standardization = run.get("target_standardization")

    def make_dataset(construct: Construct | None) -> LentiMPRAAgarwal2025SplitDataset:
        dataset = LentiMPRAAgarwal2025SplitDataset(
            input_tsv, split=args.split, splitter=splitter,
            target_columns=target_columns, target_standardization=standardization, construct=construct,
        )
        check_insert_length(construct, dataset)
        return dataset

    metrics, _, _ = evaluate_checkpoint(
        checkpoint_path,
        make_test_dataset=make_dataset,
        output_dir=args.output_dir or checkpoint_path.parent / "evaluation",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        device=args.device,
        use_amp=args.use_amp,
        split=args.split,
        rc_average=not args.forward_only,
    )
    metrics["input_tsv"] = str(input_tsv)
    metrics["target_columns"] = list(target_columns)
    metrics["target_standardization"] = standardization
    metrics["split_by"] = splitter.kind
    metrics.update(splitter.metadata())
    write_metrics(metrics["output_dir"], metrics, split=args.split)
    return metrics


if __name__ == "__main__":
    main()
