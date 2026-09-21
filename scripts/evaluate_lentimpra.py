#!/usr/bin/env python
"""Score a lentiMPRA checkpoint on its held-out test fold.

The construct comes from the checkpoint, so the evaluation input is assembled exactly as it
was during training.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from alphagenome_encoder_ft import Construct, LentiMPRADataset
from alphagenome_encoder_ft.cli import (
    add_evaluate_arguments,
    evaluate_checkpoint,
    load_run_metadata,
    resolve_input_tsv,
    write_metrics,
)


def main() -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Evaluate a lentiMPRA checkpoint")
    add_evaluate_arguments(parser)
    parser.add_argument(
        "--strip_adapters",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Defaults to what the run recorded; a checkpoint with no run.json is assumed "
        "to have been trained on seq with the adapters inline",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint_path).resolve()
    if not checkpoint_path.exists():
        parser.error(f"Checkpoint not found: {checkpoint_path}")
    input_tsv = resolve_input_tsv(parser, args, checkpoint_path)

    # Stripping has to match how the model was trained, or the construct rebuilds a
    # different molecule. Runs record it; older checkpoints predate the option.
    strip_adapters = args.strip_adapters
    if strip_adapters is None:
        strip_adapters = bool(load_run_metadata(checkpoint_path).get("strip_adapters", False))
    print(f"strip_adapters: {strip_adapters}")

    def make_test_dataset(construct: Construct | None) -> LentiMPRADataset:
        return LentiMPRADataset(
            input_tsv, split="test", construct=construct, strip_adapters=strip_adapters
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
    metrics["strip_adapters"] = strip_adapters
    write_metrics(metrics["output_dir"], metrics)
    return metrics


if __name__ == "__main__":
    main()
