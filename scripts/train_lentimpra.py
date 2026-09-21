#!/usr/bin/env python
"""Fine-tune the AlphaGenome encoder on lentiMPRA (Agarwal et al. 2025).

The `seq` column of those TSVs is 230 bp and already carries the 15 bp adapters, so the
default construct appends only minP + barcode, giving a 281 bp model input. Hyperparameters
come from --config; any flag overrides it for a single run.
"""

from __future__ import annotations

import argparse
from typing import Any

from alphagenome_encoder_ft import LentiMPRADataset, lentimpra_construct
from alphagenome_encoder_ft.cli import (
    add_construct_arguments,
    add_train_arguments,
    dataset_kwargs,
    load_config,
    resolve_construct,
    train,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune an AlphaGenome encoder on lentiMPRA")
    parser.add_argument("--input_tsv", type=str, required=True)
    parser.add_argument(
        "--strip_adapters",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Take the 15 bp adapters off seq so the insert is the bare element (default)",
    )
    add_train_arguments(parser)
    add_construct_arguments(parser)
    return parser


def main() -> dict[str, Any]:
    parser = build_arg_parser()
    args = parser.parse_args()
    config = load_config(parser, args)
    # These two go together: the reader takes the 15 bp adapters off the published seq
    # column, and the construct puts them back. The insert is then the bare 200 bp element,
    # which is what a designed sequence looks like and what attribution should cover.
    construct = resolve_construct(args, lentimpra_construct())

    def make_dataset(split: str) -> LentiMPRADataset:
        return LentiMPRADataset(
            args.input_tsv,
            split=split,
            construct=construct,
            strip_adapters=args.strip_adapters,
            **dataset_kwargs(config, augment=split == "train"),
        )

    return train(
        config,
        construct=construct,
        make_dataset=make_dataset,
        metadata={
            "dataset": "lentimpra",
            "input_tsv": str(args.input_tsv),
            "strip_adapters": args.strip_adapters,
        },
        show_progress=args.show_progress,
    )


if __name__ == "__main__":
    main()
