#!/usr/bin/env python
"""Two-stage fine-tune of the AlphaGenome encoder on lentiMPRA (Agarwal et al. 2025).

Reads ``<celltype>_with_chrom.tsv`` from ``0_add_coordinates.py``. The insert is the 200 bp
element (the table's adapters are stripped) and ``LentiMPRAAgarwal2025Library.construct()``
rebuilds the 281 bp reporter around it. ``--val_chroms`` and ``--test_chroms`` choose the
held-out chromosomes (default: val chr19, 21, X; test chr7, 13); both are recorded in
run.json, where the evaluator reads them. Hyperparameters come from ``--config``; any flag
overrides it for a single run.
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
    MALINOIS_TEST_CHROMS,
    MALINOIS_VAL_CHROMS,
    LentiMPRAAgarwal2025ChromDataset,
    check_holdout,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune the AlphaGenome encoder on Agarwal 2025 lentiMPRA")
    parser.add_argument("--input_tsv", type=str, required=True, help="<celltype>_with_chrom.tsv")
    parser.add_argument("--val_chroms", nargs="+", default=list(MALINOIS_VAL_CHROMS),
                        help="validation chromosomes, with or without 'chr' (default: 19 21 X)")
    parser.add_argument("--test_chroms", nargs="+", default=list(MALINOIS_TEST_CHROMS),
                        help="test chromosomes, with or without 'chr' (default: 7 13)")
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
    construct = resolve_construct(args, LentiMPRAAgarwal2025Library.construct())

    def make_dataset(split: str) -> LentiMPRAAgarwal2025ChromDataset:
        return LentiMPRAAgarwal2025ChromDataset(
            args.input_tsv,
            split=split,
            val_chroms=val_chroms,
            test_chroms=test_chroms,
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
            "val_chroms": val_chroms,
            "test_chroms": test_chroms,
        },
        show_progress=args.show_progress,
        resume_from_stage2=args.resume_from_stage2,
        evaluate_test=args.evaluate_test,
    )


if __name__ == "__main__":
    main()
