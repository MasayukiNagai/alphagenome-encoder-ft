#!/usr/bin/env python
"""Convert a v0 checkpoint (ConstructSpec + construct_mode) to the v1 Construct format.

v0 stored four named reporter pieces plus a five-way `construct_mode`; v1 stores one
`Construct` (prefix, suffix, length) and the model `input_length`. This rewrites the payload
and leaves the weights untouched.

    python scripts/convert_checkpoint_v0.py old.pt new.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

# What each v0 mode actually concatenated, in order.
_MODE_PIECES = {
    "none": ((), ()),
    "adapters": (("left_adapter",), ("right_adapter",)),
    "promoter": ((), ("promoter_seq",)),
    "promoter_barcode": ((), ("promoter_seq", "barcode_seq")),
    "all": (("left_adapter",), ("right_adapter", "promoter_seq", "barcode_seq")),
}


def convert_payload(checkpoint: dict[str, Any], *, construct_mode: str | None = None) -> dict[str, Any]:
    if "construct" in checkpoint and "input_length" in checkpoint:
        raise ValueError("Checkpoint is already in the v1 format")

    construct_config = dict(checkpoint.get("construct_config") or {})
    config = checkpoint.get("config") or {}
    data_config = dict(config.get("data") or {})

    mode = construct_mode or construct_config.get("construct_mode") or data_config.get("construct_mode")
    if mode is None:
        # Some v0 checkpoints (e.g. the autotune reference runs) record the reporter pieces
        # but not which of them were actually concatenated. Guessing would silently build a
        # construct the model was never trained with, so ask instead.
        available = sorted(k for k in ("left_adapter", "right_adapter", "promoter_seq", "barcode_seq") if construct_config.get(k))
        raise ValueError(
            "Checkpoint records no construct_mode, so the assembled layout is ambiguous; "
            f"pass --construct_mode (one of: {', '.join(sorted(_MODE_PIECES))}). "
            f"Pieces present: {', '.join(available) or 'none'}. "
            f"sequence_length: {construct_config.get('sequence_length') or data_config.get('sequence_length')}"
        )
    if mode not in _MODE_PIECES:
        raise ValueError(f"Unknown v0 construct_mode: {mode!r}")
    prefix_keys, suffix_keys = _MODE_PIECES[mode]

    def _join(keys: tuple[str, ...]) -> str:
        parts = []
        for key in keys:
            piece = construct_config.get(key) or data_config.get(f"{key}_seq") or data_config.get(key)
            if not piece:
                raise ValueError(f"Mode {mode!r} needs {key!r} but the checkpoint does not carry it")
            parts.append(str(piece).strip().upper())
        return "".join(parts)

    input_length = construct_config.get("sequence_length") or data_config.get("sequence_length")
    if input_length is None:
        raise ValueError(
            "Checkpoint carries no sequence_length; pass --input_length with the model input size"
        )

    converted = dict(checkpoint)
    converted["construct"] = {
        "prefix": _join(prefix_keys),
        "suffix": _join(suffix_keys),
        "length": int(input_length),
    }
    converted["input_length"] = int(input_length)
    converted.pop("construct_config", None)

    # v1 DataConfig no longer has these fields, so a v0 config would fail to load.
    for key in (
        "input_tsv",
        "sequence_length",
        "construct_mode",
        "left_adapter_seq",
        "right_adapter_seq",
        "promoter_seq",
        "barcode_seq",
    ):
        data_config.pop(key, None)

    # TrainConfig.from_dict rejects sections it does not know, and pipeline-specific ones
    # (cell_type, origin, source_ckpt, ...) are common in v0 payloads. Keep them under an
    # underscore key, which from_dict ignores, so provenance survives without breaking the
    # schema.
    known = {"data", "head", "optim", "stage", "checkpoint", "logging", "runtime"}
    kept = {key: value for key, value in config.items() if key in known or str(key).startswith("_")}
    extra = {key: value for key, value in config.items() if key not in known and not str(key).startswith("_")}
    if extra:
        kept["_v0_config_sections"] = extra

    converted["config"] = {**kept, "data": data_config}
    converted["converted_from"] = "v0"
    return converted


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a v0 checkpoint to the v1 Construct format")
    parser.add_argument("input_path", type=str)
    parser.add_argument("output_path", type=str)
    parser.add_argument(
        "--construct_mode",
        type=str,
        default=None,
        choices=sorted(_MODE_PIECES),
        help="Which pieces the v0 run concatenated, when the checkpoint does not record it",
    )
    parser.add_argument(
        "--input_length",
        type=int,
        default=None,
        help="Model input length, when the v0 checkpoint does not record one",
    )
    args = parser.parse_args()

    input_path = Path(args.input_path).resolve()
    output_path = Path(args.output_path).resolve()
    if output_path.exists():
        parser.error(f"Refusing to overwrite {output_path}")

    checkpoint = torch.load(input_path, map_location="cpu", weights_only=False)
    if args.input_length is not None:
        checkpoint.setdefault("construct_config", {})["sequence_length"] = args.input_length
    try:
        converted = convert_payload(checkpoint, construct_mode=args.construct_mode)
    except ValueError as exc:
        parser.error(str(exc))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(converted, output_path)
    print(f"construct   : {converted['construct']}")
    print(f"input_length: {converted['input_length']}")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
