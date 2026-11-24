#!/usr/bin/env python
"""Two-stage fine-tune of the AlphaGenome encoder on Gosai 2024 episomal MPRA.

Reads the TSV ``0_prepare.py`` writes. The insert is the assayed sequence (73-200 bp) and
``EpisomalMPRAGosai2024Library.construct()`` wraps it in 29 bp of upstream vector plus the
155 bp polylinker/minP/5'UTR-Kozak block for a 384 bp model input; ``--window malinois``
selects the 600 bp window Malinois trained on instead. ``--val_chroms`` and ``--test_chroms``
choose the held-out chromosomes (default: val chr19, 21, X; test chr7, 13), recorded in
run.json for the evaluator. Hyperparameters come from ``--config``; any flag overrides it for
a single run.

``--target_columns`` trains one output per column of a table prepared for several cell types
(``0_prepare.py --cell_type K562 HepG2 SKNSH``); set ``--num_outputs`` to the same count.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

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
    MALINOIS_TEST_CHROMS,
    MALINOIS_VAL_CHROMS,
    EpisomalMPRAGosai2024Dataset,
    check_holdout,
)
from library import EpisomalMPRAGosai2024Library  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune the AlphaGenome encoder on Gosai 2024 episomal MPRA")
    parser.add_argument("--input_tsv", type=str, required=True, help="the TSV 0_prepare.py writes")
    parser.add_argument(
        "--window",
        choices=["regulatory", "malinois"],
        default="regulatory",
        help="regulatory (default): 384 bp, insert -> minP -> Kozak. "
             "malinois: the 600 bp window Malinois trained on, both flanks, insert centred.",
    )
    parser.add_argument("--val_chroms", nargs="+", default=list(MALINOIS_VAL_CHROMS),
                        help="validation chromosomes, with or without 'chr' (default: 19 21 X)")
    parser.add_argument("--test_chroms", nargs="+", default=list(MALINOIS_TEST_CHROMS),
                        help="test chromosomes, with or without 'chr' (default: 7 13)")
    parser.add_argument("--target_columns", nargs="+", default=["mean_value"],
                        help="target column(s), one output each (default: mean_value)")
    add_train_arguments(parser)
    add_construct_arguments(parser)
    return parser


def main() -> dict[str, Any]:
    parser = build_arg_parser()
    args = parser.parse_args()

    config = load_config(parser, args)
    try:
        val_chroms, test_chroms = (sorted(c) for c in check_holdout(args.val_chroms, args.test_chroms))
    except ValueError as exc:
        parser.error(str(exc))
    if len(args.target_columns) != config.head.num_outputs:
        parser.error(f"{len(args.target_columns)} target column(s) but head.num_outputs is {config.head.num_outputs}")
    default_construct = (
        EpisomalMPRAGosai2024Library.construct()
        if args.window == "regulatory"
        else EpisomalMPRAGosai2024Library.malinois_construct()
    )
    construct = resolve_construct(args, default_construct)

    def make_dataset(split: str) -> EpisomalMPRAGosai2024Dataset:
        return EpisomalMPRAGosai2024Dataset(
            args.input_tsv,
            split=split,
            val_chroms=val_chroms,
            test_chroms=test_chroms,
            target_columns=args.target_columns,
            construct=construct,
            **dataset_kwargs(config, augment=split == "train"),
        )

    return train(
        config,
        construct=construct,
        make_dataset=make_dataset,
        metadata={
            "dataset": "gosai2024",
            "input_tsv": str(args.input_tsv),
            "window": args.window,
            "val_chroms": val_chroms,
            "test_chroms": test_chroms,
            "target_columns": args.target_columns,
        },
        show_progress=args.show_progress,
        resume_from_stage2=args.resume_from_stage2,
        evaluate_test=args.evaluate_test,
    )


if __name__ == "__main__":
    main()
