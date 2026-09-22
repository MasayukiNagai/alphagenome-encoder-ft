#!/usr/bin/env python
"""Score a lentiMPRA checkpoint on its held-out test fold.

The construct comes from the checkpoint, so the evaluation input is assembled exactly as it
was during training.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from alphagenome_encoder_ft import Construct, LentiMPRAAgarwal2025Dataset
from alphagenome_encoder_ft.cli import (
    add_evaluate_arguments,
    evaluate_checkpoint,
    resolve_input_tsv,
    write_metrics,
)


def check_insert_length(construct: Construct | None, dataset: LentiMPRAAgarwal2025Dataset) -> None:
    """The construct must close exactly around the element, with nothing trimmed or padded."""

    if construct is None or construct.length is None or not len(dataset):
        return
    expected = construct.length - len(construct.prefix) - len(construct.suffix)
    actual = len(dataset.inserts[0])
    if actual != expected:
        raise ValueError(
            f"This checkpoint expects {expected} bp inserts but the dataset yields {actual} bp. "
            "Re-convert or retrain the checkpoint against the current construct."
        )


def main() -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Evaluate a lentiMPRA checkpoint")
    add_evaluate_arguments(parser)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint_path).resolve()
    if not checkpoint_path.exists():
        parser.error(f"Checkpoint not found: {checkpoint_path}")
    input_tsv = resolve_input_tsv(parser, args, checkpoint_path)

    def make_test_dataset(construct: Construct | None) -> LentiMPRAAgarwal2025Dataset:
        dataset = LentiMPRAAgarwal2025Dataset(input_tsv, split="test", construct=construct)
        check_insert_length(construct, dataset)
        return dataset

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
