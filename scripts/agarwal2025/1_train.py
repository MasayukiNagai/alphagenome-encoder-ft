#!/usr/bin/env python
"""Two-stage fine-tune of the AlphaGenome encoder on lentiMPRA (Agarwal et al. 2025).

``--split_by`` chooses the holdout:

``chrom`` (default)
    Reads ``<celltype>_with_chrom.tsv`` from ``0_add_coordinates.py``. ``--val_chroms``
    and ``--test_chroms`` choose the held-out chromosomes (default: val chr19, 21, X;
    test chr7, 13, the holdout Malinois used on Gosai 2024).

``fold``
    Reads the published table directly (``seq_id, seq, mean_value, fold, rev``), so no
    coordinate lift-over is needed, and holds out the folds the publication ships:
    ``--train_folds`` 2-9, ``--val_folds`` 1, ``--test_folds`` 10. **This is the split
    the released lentiMPRA checkpoints were trained and scored on**, so it is the one to
    use for a number comparable with theirs; a chromosome-split run is not comparable.

The split is recorded in run.json and the evaluator reads it back, so a checkpoint is
always scored on the rows it was trained against.

The insert is the 200 bp element (the table's adapters are stripped) and
``LentiMPRAAgarwal2025Library.construct()`` rebuilds the 281 bp reporter around it.
Hyperparameters come from ``--config``; any flag overrides it for a single run.

For one model over several cell types, train on the table ``0_merge_cell_types.py`` writes
with ``--target_columns K562 HepG2 WTC11 --num_outputs 3``. Elements not measured in a cell
type are missing labels that the loss skips. ``--standardize_targets`` z-scores each column
with its training-split mean and SD, so a cell type with a wider spread does not dominate
the loss; the statistics are recorded in run.json and the evaluator reuses them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from alphagenome_encoder_ft import LentiMPRAAgarwal2025Library
from alphagenome_encoder_ft.cli import (
    add_construct_arguments,
    add_train_arguments,
    dataset_kwargs,
    load_config,
    resolve_construct,
    train,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import (  # noqa: E402
    DEFAULT_FOLD_SPLITS,
    MALINOIS_TEST_CHROMS,
    MALINOIS_VAL_CHROMS,
    SPLIT_BY,
    LentiMPRAAgarwal2025SplitDataset,
    foreign_split_flags,
    make_splitter,
    target_standardization,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune the AlphaGenome encoder on Agarwal 2025 lentiMPRA")
    parser.add_argument("--input_tsv", type=str, required=True,
                        help="<celltype>_with_chrom.tsv for --split_by chrom, else the published table")
    parser.add_argument("--split_by", choices=SPLIT_BY, default="chrom",
                        help="hold out chromosomes (default) or the publication's folds")
    parser.add_argument("--val_chroms", nargs="+", default=None,
                        help=f"--split_by chrom: validation chromosomes, with or without 'chr' "
                             f"(default: {' '.join(MALINOIS_VAL_CHROMS)})")
    parser.add_argument("--test_chroms", nargs="+", default=None,
                        help=f"--split_by chrom: test chromosomes (default: {' '.join(MALINOIS_TEST_CHROMS)})")
    parser.add_argument("--train_folds", nargs="+", type=int, default=None,
                        help=f"--split_by fold: training folds "
                             f"(default: {' '.join(str(f) for f in DEFAULT_FOLD_SPLITS['train'])})")
    parser.add_argument("--val_folds", nargs="+", type=int, default=None,
                        help=f"--split_by fold: validation folds "
                             f"(default: {' '.join(str(f) for f in DEFAULT_FOLD_SPLITS['val'])})")
    parser.add_argument("--test_folds", nargs="+", type=int, default=None,
                        help=f"--split_by fold: test folds "
                             f"(default: {' '.join(str(f) for f in DEFAULT_FOLD_SPLITS['test'])})")
    parser.add_argument("--target_columns", nargs="+", default=["mean_value"],
                        help="target column(s), one output each (default: mean_value)")
    parser.add_argument("--standardize_targets", action="store_true",
                        help="z-score each target column with its training-split mean and SD")
    add_train_arguments(parser)
    add_construct_arguments(parser)
    return parser


def check_split_flags(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Refuse a flag belonging to the other holdout rather than ignoring it silently.

    Passing ``--val_chroms`` to a fold run is a mistake worth stopping on: accepting and
    ignoring it trains on a split the caller did not ask for.
    """

    given = foreign_split_flags(args.split_by, vars(args))
    if given:
        parser.error(
            f"{', '.join(given)} {'has' if len(given) == 1 else 'have'} no meaning with "
            f"--split_by {args.split_by}"
        )


def main() -> dict[str, Any]:
    parser = build_arg_parser()
    args = parser.parse_args()

    config = load_config(parser, args)
    check_split_flags(parser, args)
    try:
        splitter = make_splitter(
            args.split_by,
            val_chroms=args.val_chroms,
            test_chroms=args.test_chroms,
            train_folds=args.train_folds,
            val_folds=args.val_folds,
            test_folds=args.test_folds,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if len(args.target_columns) != config.head.num_outputs:
        parser.error(f"{len(args.target_columns)} target column(s) but head.num_outputs is {config.head.num_outputs}")
    construct = resolve_construct(args, LentiMPRAAgarwal2025Library.construct())
    standardization = (
        target_standardization(args.input_tsv, args.target_columns, splitter)
        if args.standardize_targets
        else None
    )

    def make_dataset(split: str) -> LentiMPRAAgarwal2025SplitDataset:
        return LentiMPRAAgarwal2025SplitDataset(
            args.input_tsv,
            split=split,
            splitter=splitter,
            target_columns=args.target_columns,
            target_standardization=standardization,
            construct=construct,
            **dataset_kwargs(config, augment=split == "train"),
        )

    return train(
        config,
        construct=construct,
        make_dataset=make_dataset,
        metadata={
            "dataset": "agarwal2025",
            "input_tsv": str(args.input_tsv),
            "split_by": splitter.kind,
            **splitter.metadata(),
            "target_columns": args.target_columns,
            "target_standardization": standardization,
        },
        show_progress=args.show_progress,
        resume_from_stage2=args.resume_from_stage2,
        evaluate_test=args.evaluate_test,
    )


if __name__ == "__main__":
    main()
