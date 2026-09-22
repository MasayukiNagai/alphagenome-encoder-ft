#!/usr/bin/env python
"""Fine-tune the AlphaGenome encoder on lentiMPRA (Agarwal et al. 2025).

The `seq` column of those TSVs is 230 bp and already carries the 15 bp adapters, so the
default construct appends only minP + barcode, giving a 281 bp model input. Hyperparameters
come from --config; any flag overrides it for a single run.
"""

from __future__ import annotations

import argparse
from typing import Any

from alphagenome_encoder_ft import LentiMPRAAgarwal2025Dataset, LentiMPRAAgarwal2025Library
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
    add_train_arguments(parser)
    add_construct_arguments(parser)
    return parser


def main() -> dict[str, Any]:
    parser = build_arg_parser()
    args = parser.parse_args()
    config = load_config(parser, args)
    construct = resolve_construct(args, LentiMPRAAgarwal2025Library.construct())

    def make_dataset(split: str) -> LentiMPRAAgarwal2025Dataset:
        return LentiMPRAAgarwal2025Dataset(
            args.input_tsv,
            split=split,
            construct=construct,
            **dataset_kwargs(config, augment=split == "train"),
        )

    return train(
        config,
        construct=construct,
        make_dataset=make_dataset,
        metadata={"dataset": "lentimpra", "input_tsv": str(args.input_tsv)},
        show_progress=args.show_progress,
    )


if __name__ == "__main__":
    main()
