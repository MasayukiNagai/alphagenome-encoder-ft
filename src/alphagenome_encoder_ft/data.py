"""MPRA datasets: a format-agnostic base plus one reader per assay."""

from __future__ import annotations

import csv
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from alphagenome_pytorch.utils.sequence import sequence_to_onehot

from .constructs import Construct, LENTIMPRA_LEFT_ADAPTER, LENTIMPRA_RIGHT_ADAPTER


def _reverse_complement_onehot(onehot: np.ndarray) -> np.ndarray:
    return onehot[::-1, :][:, [3, 2, 1, 0]]


class MPRADataset(Dataset[tuple[Tensor, Tensor]]):
    """Inserts and targets in memory; the construct and augmentation live here.

    Each item is ``(onehot, target)`` where ``onehot`` is the assembled model input.
    Jitter is a random window offset passed to the construct (needs ``construct.length``);
    reverse complement is applied to the whole assembled one-hot. Subclasses are readers
    that parse a file and call ``super().__init__``. The base can be used directly for
    inserts built in memory, e.g. from a variant table.
    """

    def __init__(
        self,
        inserts: Sequence[str],
        targets: np.ndarray | Sequence[float] | Sequence[Sequence[float]],
        *,
        construct: Construct | None = None,
        reverse_complement: bool = False,
        rc_prob: float = 0.5,
        random_shift: bool = False,
        shift_prob: float = 0.5,
        max_shift: int = 15,
        subset_frac: float = 1.0,
        seed: int = 42,
    ) -> None:
        if not 0 < subset_frac <= 1:
            raise ValueError("subset_frac must be in (0, 1]")
        if not 0 <= rc_prob <= 1:
            raise ValueError("rc_prob must be in [0, 1]")
        if not 0 <= shift_prob <= 1:
            raise ValueError("shift_prob must be in [0, 1]")
        if max_shift < 0:
            raise ValueError("max_shift must be >= 0")
        if random_shift and (construct is None or construct.length is None):
            raise ValueError("random_shift requires a Construct with length")

        inserts = [str(insert).strip().upper() for insert in inserts]
        targets = np.asarray(targets, dtype=np.float32)
        if targets.shape[0] != len(inserts):
            raise ValueError(f"targets has {targets.shape[0]} rows but there are {len(inserts)} inserts")

        self.construct = construct
        self.reverse_complement = reverse_complement
        self.rc_prob = rc_prob
        self.random_shift = random_shift
        self.shift_prob = shift_prob
        self.max_shift = max_shift
        self._rng = np.random.default_rng(seed)

        if subset_frac < 1.0 and inserts:
            sample_size = max(1, int(round(len(inserts) * subset_frac)))
            sample_indices = sorted(self._rng.choice(len(inserts), size=sample_size, replace=False).tolist())
            inserts = [inserts[int(idx)] for idx in sample_indices]
            targets = targets[sample_indices]

        self.inserts = inserts
        self.targets = targets

    def __len__(self) -> int:
        return len(self.inserts)

    def _draw_offset(self) -> int:
        if self.random_shift and self.max_shift > 0 and self._rng.random() < self.shift_prob:
            return int(self._rng.integers(-self.max_shift, self.max_shift + 1))
        return 0

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        insert = self.inserts[index]
        if self.construct is not None:
            sequence = self.construct.assemble_sequence(insert, offset=self._draw_offset())
        else:
            sequence = insert
        onehot = sequence_to_onehot(sequence).astype(np.float32, copy=False)
        if self.reverse_complement and self._rng.random() < self.rc_prob:
            onehot = _reverse_complement_onehot(onehot)
        target = np.asarray(self.targets[index], dtype=np.float32)
        return torch.from_numpy(np.ascontiguousarray(onehot)), torch.from_numpy(target)


def strip_flanks(
    sequences: Sequence[str],
    prefix: str,
    suffix: str,
    *,
    labels: Sequence[str] | None = None,
) -> list[str]:
    """Remove a known constant ``prefix`` and ``suffix`` from every sequence.

    Verified, never assumed: a sequence that does not carry both raises, naming the row, so
    an unexpected file fails loudly instead of silently losing real bases off the ends.
    """

    stripped: list[str] = []
    for index, raw in enumerate(sequences):
        sequence = str(raw).strip().upper()
        label = labels[index] if labels is not None and index < len(labels) else str(index)
        if len(sequence) <= len(prefix) + len(suffix):
            raise ValueError(
                f"Sequence {label!r} is {len(sequence)} bp, too short to carry "
                f"{len(prefix)} + {len(suffix)} bp of flanks"
            )
        if not sequence.startswith(prefix) or not sequence.endswith(suffix):
            raise ValueError(
                f"Sequence {label!r} does not carry the expected flanks: it starts "
                f"{sequence[: len(prefix)]!r} and ends {sequence[-len(suffix):]!r}, expected "
                f"{prefix!r} and {suffix!r}. Pass strip_adapters=False if this file stores "
                "inserts without them."
            )
        stripped.append(sequence[len(prefix) : len(sequence) - len(suffix)])
    return stripped


def read_tsv_rows(path: str | Path, keep: Callable[[dict[str, str]], bool] | None = None) -> list[dict[str, str]]:
    """Read a TSV with a header into dicts, optionally keeping only rows where ``keep`` is true."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return [row for row in reader if keep is None or keep(row)]


class LentiMPRADataset(MPRADataset):
    """Reader for lentiMPRA TSVs (``seq``, ``mean_value``, ``fold``, ``rev``).

    Keeps ``rev == 0`` rows (the ``rev == 1`` partners are exact reverse complements with the
    same target; RC is applied as an augmentation instead) and selects folds by split.

    ``strip_adapters`` removes the 15 bp cloning adapters that the published Agarwal et al.
    2025 TSVs carry inline, leaving the bare 200 bp element as the insert. Pair it with
    :func:`~alphagenome_encoder_ft.constructs.lentimpra_construct`, which puts them back;
    leave it off and pair with
    :func:`~alphagenome_encoder_ft.constructs.lentimpra_promoter_barcode_construct`, which
    expects them already present. Either way the model input is the same 281 bp.

    It is off by default because this reader is also used for files with the same columns
    but no adapters, such as a pre-assembled construct.
    """

    DEFAULT_FOLD_SPLITS = {
        "train": [2, 3, 4, 5, 6, 7, 8, 9],
        "val": [1],
        "test": [10],
    }

    def __init__(
        self,
        input_tsv: str | Path,
        split: str = "train",
        *,
        train_folds: Sequence[int] | None = None,
        valid_folds: Sequence[int] | None = None,
        test_folds: Sequence[int] | None = None,
        sequence_column: str = "seq",
        target_column: str = "mean_value",
        strip_adapters: bool = False,
        left_adapter: str = LENTIMPRA_LEFT_ADAPTER,
        right_adapter: str = LENTIMPRA_RIGHT_ADAPTER,
        **kwargs,
    ) -> None:
        if split not in self.DEFAULT_FOLD_SPLITS:
            raise ValueError(f"Unknown split: {split!r}")
        folds = {
            "train": list(train_folds) if train_folds is not None else self.DEFAULT_FOLD_SPLITS["train"],
            "val": list(valid_folds) if valid_folds is not None else self.DEFAULT_FOLD_SPLITS["val"],
            "test": list(test_folds) if test_folds is not None else self.DEFAULT_FOLD_SPLITS["test"],
        }[split]
        fold_set = set(int(fold) for fold in folds)

        self.input_tsv = Path(input_tsv)
        self.split = split
        rows = read_tsv_rows(
            self.input_tsv,
            keep=lambda row: int(row["rev"]) == 0 and int(row["fold"]) in fold_set,
        )
        inserts = [row[sequence_column] for row in rows]
        if strip_adapters:
            # Only the kept rows are checked, and that is the point: the rev == 1 rows are
            # the reverse complement of the whole molecule, so they carry the reverse
            # complement of each adapter at the opposite end. They are already filtered out
            # above, and every remaining row carries the adapters in forward orientation.
            inserts = strip_flanks(
                inserts,
                left_adapter,
                right_adapter,
                labels=[row.get("seq_id", str(index)) for index, row in enumerate(rows)],
            )
        self.strip_adapters = strip_adapters

        super().__init__(
            inserts,
            [float(row[target_column]) for row in rows],
            **kwargs,
        )


class DeepSTARRDataset(MPRADataset):
    """Reader for Drosophila DeepSTARR TSVs with a split column and two log2 targets (dev, hk)."""

    DEFAULT_TARGET_COLUMNS = ("Dev_log2_enrichment", "Hk_log2_enrichment")

    def __init__(
        self,
        input_tsv: str | Path,
        split: str = "train",
        *,
        split_column: str = "set",
        sequence_column: str = "sequence",
        target_columns: Sequence[str] = DEFAULT_TARGET_COLUMNS,
        **kwargs,
    ) -> None:
        if len(target_columns) < 1:
            raise ValueError("target_columns must have at least one column")

        self.input_tsv = Path(input_tsv)
        self.split = split
        self.target_columns = tuple(target_columns)
        rows = read_tsv_rows(
            self.input_tsv,
            keep=lambda row: split_column not in row or row[split_column] == split,
        )
        super().__init__(
            [row[sequence_column] for row in rows],
            [[float(row[col]) for col in self.target_columns] for row in rows],
            **kwargs,
        )


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    *,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    """Create a standard PyTorch DataLoader."""

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
