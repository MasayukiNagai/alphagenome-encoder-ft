#!/usr/bin/env python
"""Score a Gosai 2024 episomal MPRA checkpoint on its validation or test chromosomes.

The construct and the held-out chromosomes come from the checkpoint and its run.json, so the
input is assembled, and the rows selected, exactly as during training. The test rows are
Malinois's reported test set (see ``dataset.py``).

Predictions average the forward and reverse-complement inputs, as Malinois's do (Methods,
"Correlation of empirical and predicted MPRA activity"); the whole assembled construct is
flipped, as in RC augmentation. ``--forward_only`` scores the forward strand alone. The
forward-only metrics are recorded under ``forward_only`` either way.

Metrics are also reported by insert length. 4.3% of inserts are shorter than 200 bp, so the
window runs further into EGFP for them; a pooled metric dilutes any effect there about 23x.

Writes ``<split>_metrics.json``, ``<split>_predictions.csv`` and ``<split>_y_vs_y_pred.png``
into ``--output_dir`` (default: ``evaluation/`` next to the checkpoint).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

from alphagenome_encoder_ft import Construct
from alphagenome_encoder_ft.cli import (
    add_evaluate_arguments,
    evaluate_checkpoint,
    load_run_metadata,
    resolve_input_tsv,
    write_metrics,
)
from alphagenome_encoder_ft.metrics import regression_metrics

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import EpisomalMPRAGosai2024Dataset  # noqa: E402


def check_insert_lengths(construct: Construct | None, dataset: EpisomalMPRAGosai2024Dataset) -> None:
    """The window must hold the longest insert whole, or the reporter silently changes.

    A construct with ``window_start`` checks this itself. A centred construct (the Malinois
    window) is checked here: an insert longer than the window leaves would be trimmed into
    the flanks.
    """

    if construct is None or construct.length is None or not len(dataset):
        return
    longest = max(len(insert) for insert in dataset.inserts)
    if construct.window_start is not None:
        construct.check_insert_window(longest)
        return
    budget = construct.length - len(construct.prefix) - len(construct.suffix)
    if longest > budget:
        raise ValueError(
            f"This checkpoint leaves {budget} bp for the insert but the dataset yields one of "
            f"{longest} bp, which would be trimmed into the construct's flanks."
        )


def stratify_by_insert_length(y_true: np.ndarray, y_pred: np.ndarray, insert_lens: list[int]) -> dict[str, Any]:
    """Metrics split into full-length (200 bp) and short (< 200 bp) inserts."""

    full = EpisomalMPRAGosai2024Dataset.LIBRARY.INSERT_LENGTH
    lengths = np.asarray(insert_lens)
    if lengths.size == 0 or (lengths < 0).any():
        return {}
    out: dict[str, Any] = {}
    for name, mask in {f"full_{full}bp": lengths == full, f"short_lt{full}bp": lengths < full}.items():
        if not mask.any():
            continue
        out[name] = regression_metrics(y_true[mask], y_pred[mask])
        out[name]["insert_len_min"] = int(lengths[mask].min())
        out[name]["insert_len_max"] = int(lengths[mask].max())
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a Gosai 2024 episomal MPRA checkpoint")
    add_evaluate_arguments(parser)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--forward_only", action="store_true",
                        help="score the forward strand only instead of the forward/RC average")
    parser.add_argument("--val_chroms", nargs="+", default=None,
                        help="defaults to the val_chroms recorded in run.json")
    parser.add_argument("--test_chroms", nargs="+", default=None,
                        help="defaults to the test_chroms recorded in run.json")
    return parser


def main() -> dict[str, Any]:
    parser = build_arg_parser()
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint_path).resolve()
    if not checkpoint_path.exists():
        parser.error(f"Checkpoint not found: {checkpoint_path}")
    input_tsv = resolve_input_tsv(parser, args, checkpoint_path)
    run = load_run_metadata(checkpoint_path)
    val_chroms = args.val_chroms or run.get("val_chroms")
    test_chroms = args.test_chroms or run.get("test_chroms")
    if not val_chroms or not test_chroms:
        parser.error("no val_chroms/test_chroms in run.json; pass --val_chroms and --test_chroms")
    target_columns = run.get("target_columns", ["mean_value"])

    def make_dataset(construct: Construct | None) -> EpisomalMPRAGosai2024Dataset:
        dataset = EpisomalMPRAGosai2024Dataset(
            input_tsv, split=args.split, val_chroms=val_chroms, test_chroms=test_chroms,
            target_columns=target_columns, construct=construct,
        )
        check_insert_lengths(construct, dataset)
        return dataset

    # The same reader as the dataset, so the annotations line up with the predictions.
    rows = EpisomalMPRAGosai2024Dataset.read_rows(input_tsv, args.split, val_chroms, test_chroms)
    seq_ids = [row.get("seq_id", "") for row in rows]
    insert_lens = [int(row["insert_len"]) if "insert_len" in row else -1 for row in rows]

    metrics, y_true, y_pred = evaluate_checkpoint(
        checkpoint_path,
        make_test_dataset=make_dataset,
        output_dir=args.output_dir or checkpoint_path.parent / "evaluation",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        device=args.device,
        use_amp=args.use_amp,
        extra_columns={"seq_id": seq_ids, "insert_len": insert_lens},
        split=args.split,
        rc_average=not args.forward_only,
    )

    by_length = stratify_by_insert_length(y_true, y_pred, insert_lens)
    if by_length:
        metrics["by_insert_length"] = by_length
    metrics["input_tsv"] = str(input_tsv)
    metrics["target_columns"] = list(target_columns)
    metrics["val_chroms"] = list(val_chroms)
    metrics["test_chroms"] = list(test_chroms)
    write_metrics(metrics["output_dir"], metrics, split=args.split)
    return metrics


if __name__ == "__main__":
    main()
