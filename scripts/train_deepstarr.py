#!/usr/bin/env python
"""Fine-tune the AlphaGenome encoder on Drosophila DeepSTARR (dev + hk enhancer activity).

The default construct puts the STARR-seq adapters around the insert and windows the result
to 256 bp. Use --head_type deepstarr --num_outputs 2 (or set them in the config).
"""

from __future__ import annotations

import argparse
from typing import Any

from alphagenome_encoder_ft import DeepSTARRDeAlmeida2022Dataset, DeepSTARRDeAlmeida2022Library
from alphagenome_encoder_ft.cli import (
    add_construct_arguments,
    add_train_arguments,
    dataset_kwargs,
    load_config,
    resolve_construct,
    train,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune an AlphaGenome encoder on DeepSTARR")
    parser.add_argument("--input_tsv", type=str, required=True)
    parser.add_argument("--split_column", type=str, default="set")
    parser.add_argument("--sequence_column", type=str, default="sequence")
    add_train_arguments(parser)
    add_construct_arguments(parser)
    return parser


def main() -> dict[str, Any]:
    parser = build_arg_parser()
    args = parser.parse_args()
    config = load_config(parser, args)
    construct = resolve_construct(args, DeepSTARRDeAlmeida2022Library.construct())

    def make_dataset(split: str) -> DeepSTARRDeAlmeida2022Dataset:
        return DeepSTARRDeAlmeida2022Dataset(
            args.input_tsv,
            split=split,
            split_column=args.split_column,
            sequence_column=args.sequence_column,
            construct=construct,
            **dataset_kwargs(config, augment=split == "train"),
        )

    return train(
        config,
        construct=construct,
        make_dataset=make_dataset,
        metadata={
            "dataset": "deepstarr",
            "input_tsv": str(args.input_tsv),
            "split_column": args.split_column,
            "sequence_column": args.sequence_column,
        },
        show_progress=args.show_progress,
        resume_from_stage2=args.resume_from_stage2,
        evaluate_test=args.evaluate_test,
    )


if __name__ == "__main__":
    main()
