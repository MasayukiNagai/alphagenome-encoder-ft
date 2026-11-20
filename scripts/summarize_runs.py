#!/usr/bin/env python
"""Average scored runs over seeds and print one markdown table.

Reads ``<runs_dir>/<name>-s<seed>/<stage>/evaluation/{val,test}_metrics.json`` as written by
the dataset ``2_evaluate.py`` scripts, groups runs by ``<name>`` and stage, and prints the
mean Pearson over seeds with the seed range (max - min) in brackets. ``Pearson`` is the
reported prediction (strand-averaged unless the run was scored ``--forward_only``);
``forward`` is the forward strand alone. A stage directory that is a symlink belongs to
another run and is skipped.

    python scripts/summarize_runs.py results/agarwal2025_K562_locon_ema
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

RUN_NAME = re.compile(r"^(?P<name>.+)-s(?P<seed>\d+)$")
SPLITS = ("val", "test")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs_dir", type=Path, help="directory holding <name>-s<seed> run directories")
    return parser


def collect(runs_dir: Path) -> dict[tuple[str, str], dict[str, dict[str, list[float]]]]:
    """``(name, stage) -> split -> {"pearson": [...], "forward": [...], "seeds": [...]}``."""

    table: dict[tuple[str, str], dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: {split: defaultdict(list) for split in SPLITS}
    )
    for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        match = RUN_NAME.match(run_dir.name)
        if match is None:
            continue
        for stage_dir in sorted(run_dir.glob("stage*")):
            if stage_dir.is_symlink():
                continue
            for split in SPLITS:
                path = stage_dir / "evaluation" / f"{split}_metrics.json"
                if not path.is_file():
                    continue
                metrics = json.loads(path.read_text())
                entry = table[(match["name"], stage_dir.name)][split]
                entry["pearson"].append(metrics["pearsonr"])
                entry["forward"].append(metrics.get("forward_only", metrics)["pearsonr"])
                entry["seeds"].append(int(match["seed"]))
    return table


def cell(values: list[float]) -> str:
    if not values:
        return "-"
    return f"{np.mean(values):.4f} [{np.max(values) - np.min(values):.4f}]"


def main() -> None:
    args = build_arg_parser().parse_args()
    if not args.runs_dir.is_dir():
        raise SystemExit(f"not a directory: {args.runs_dir}")
    table = collect(args.runs_dir)
    if not table:
        raise SystemExit(f"no scored runs under {args.runs_dir}")
    print("| run | stage | seeds | val Pearson | val forward | test Pearson | test forward |")
    print("|---|---|---|---|---|---|---|")
    for (name, stage), splits in sorted(table.items(), key=lambda item: (item[0][1], item[0][0])):
        seeds = sorted(set(splits["val"]["seeds"]) | set(splits["test"]["seeds"]))
        print(f"| {name} | {stage} | {', '.join(map(str, seeds))} | "
              f"{cell(splits['val']['pearson'])} | {cell(splits['val']['forward'])} | "
              f"{cell(splits['test']['pearson'])} | {cell(splits['test']['forward'])} |")


if __name__ == "__main__":
    main()
