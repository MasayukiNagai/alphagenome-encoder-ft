#!/usr/bin/env python
"""Score a DeepSTARR checkpoint on its test split; metrics are reported per track (dev, hk)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from alphagenome_encoder_ft import Construct, DeepSTARRDeAlmeida2022Dataset
from alphagenome_encoder_ft.cli import (
    add_evaluate_arguments,
    evaluate_checkpoint,
    load_run_metadata,
    resolve_input_tsv,
    write_metrics,
)


def main() -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Evaluate a DeepSTARR checkpoint")
    add_evaluate_arguments(parser)
    parser.add_argument("--split", type=str, default="test")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint_path).resolve()
    if not checkpoint_path.exists():
        parser.error(f"Checkpoint not found: {checkpoint_path}")
    input_tsv = resolve_input_tsv(parser, args, checkpoint_path)
    metadata = load_run_metadata(checkpoint_path)

    def make_test_dataset(construct: Construct | None) -> DeepSTARRDeAlmeida2022Dataset:
        return DeepSTARRDeAlmeida2022Dataset(
            input_tsv,
            split=args.split,
            split_column=metadata.get("split_column", "set"),
            sequence_column=metadata.get("sequence_column", "sequence"),
            construct=construct,
        )

    metrics, _, _ = evaluate_checkpoint(
        checkpoint_path,
        make_test_dataset=make_test_dataset,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        device=args.device,
        use_amp=args.use_amp,
    )
    metrics["input_tsv"] = str(input_tsv)
    write_metrics(metrics["output_dir"], metrics)
    return metrics


if __name__ == "__main__":
    main()
