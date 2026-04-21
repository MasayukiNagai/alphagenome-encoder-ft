"""MPRA dataset utilities."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from alphagenome_pytorch.utils.sequence import sequence_to_onehot

from .constructs import ConstructSpec


# Drosophila DeepSTARR library adapters (from the original STARR-seq protocol,
# Arnold et al. / de Almeida et al.); exported so downstream users can assemble
# the same 256 bp construct the encoder was trained on.
DEEPSTARR_ADAPTER_UP = "TCCCTACACGACGCTCTTCCGATCT"
DEEPSTARR_ADAPTER_DOWN = "AGATCGGAAGAGCACACGTCTGAACT"


def _reverse_complement_onehot(onehot: np.ndarray) -> np.ndarray:
    return onehot[::-1, :][:, [3, 2, 1, 0]]


class LentiMPRADataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """PyTorch Dataset for lentiMPRA TSV files."""

    DEFAULT_FOLD_SPLITS = {
        "train": [2, 3, 4, 5, 6, 7, 8, 9],
        "val": [1],
        "test": [10],
    }

    def __init__(
        self,
        input_tsv: str | Path,
        split: str = "train",
        train_folds: list[int] | None = None,
        valid_folds: list[int] | None = None,
        test_folds: list[int] | None = None,
        construct_spec: ConstructSpec | None = None,
        construct_mode: str = "all",
        reverse_complement: bool = False,
        rc_prob: float = 0.5,
        random_shift: bool = False,
        shift_prob: float = 0.5,
        max_shift: int = 15,
        sequence_length: int | None = None,
        subset_frac: float = 1.0,
        seed: int = 42,
    ) -> None:
        if split not in self.DEFAULT_FOLD_SPLITS:
            raise ValueError(f"Unknown split: {split!r}")
        if sequence_length is not None and sequence_length <= 0:
            raise ValueError("sequence_length must be > 0")
        if not 0 < subset_frac <= 1:
            raise ValueError("subset_frac must be in (0, 1]")
        if not 0 <= rc_prob <= 1:
            raise ValueError("rc_prob must be in [0, 1]")
        if not 0 <= shift_prob <= 1:
            raise ValueError("shift_prob must be in [0, 1]")
        if max_shift < 0:
            raise ValueError("max_shift must be >= 0")

        self.input_tsv = Path(input_tsv)
        self.split = split
        if construct_spec is None:
            raise ValueError("construct_spec must be provided")
        self.construct_spec = construct_spec
        self.construct_mode = self.construct_spec.validate_mode(construct_mode)
        self.promoter_seq = self.construct_spec.promoter_seq
        self.barcode_seq = self.construct_spec.barcode_seq
        self.left_adapter_seq = self.construct_spec.left_adapter
        self.right_adapter_seq = self.construct_spec.right_adapter
        self.reverse_complement = reverse_complement
        self.rc_prob = rc_prob
        self.random_shift = random_shift
        self.shift_prob = shift_prob
        self.max_shift = max_shift
        self.sequence_length = sequence_length
        self._rng = np.random.default_rng(seed)
        self.train_folds = (
            list(train_folds) if train_folds is not None else list(self.DEFAULT_FOLD_SPLITS["train"])
        )
        self.valid_folds = (
            list(valid_folds) if valid_folds is not None else list(self.DEFAULT_FOLD_SPLITS["val"])
        )
        self.test_folds = (
            list(test_folds) if test_folds is not None else list(self.DEFAULT_FOLD_SPLITS["test"])
        )

        if not self.input_tsv.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.input_tsv}")

        rows = self._read_tsv()

        if subset_frac < 1.0 and rows:
            sample_size = max(1, int(round(len(rows) * subset_frac)))
            sample_indices = self._rng.choice(len(rows), size=sample_size, replace=False)
            rows = [rows[int(idx)] for idx in sorted(sample_indices.tolist())]

        self._payloads = [str(row["seq"]) for row in rows]
        self._targets = np.asarray([float(row["mean_value"]) for row in rows], dtype=np.float32)
        self._construct_lengths = [
            len(self.construct_spec.assemble_sequence(payload, mode=self.construct_mode))
            for payload in self._payloads
        ]
        if self.sequence_length is not None:
            too_long = [
                (idx, construct_length)
                for idx, construct_length in enumerate(self._construct_lengths)
                if construct_length > self.sequence_length
            ]
            if too_long:
                sample_idx, sample_length = too_long[0]
                raise ValueError(
                    "sequence_length is shorter than the assembled construct length "
                    f"for sample {sample_idx}: {self.sequence_length} < {sample_length}"
                )

    def _read_tsv(self) -> list[dict[str, str]]:
        split_folds = {
            "train": self.train_folds,
            "val": self.valid_folds,
            "test": self.test_folds,
        }[self.split]

        rows: list[dict[str, str]] = []
        with open(self.input_tsv, newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                if int(row["rev"]) != 0:
                    continue
                if int(row["fold"]) not in split_folds:
                    continue
                rows.append(row)
        return rows

    def __len__(self) -> int:
        return len(self._payloads)

    def _pad_to_length(self, onehot: np.ndarray) -> np.ndarray:
        if self.sequence_length is None:
            return onehot.astype(np.float32, copy=False)
        length = onehot.shape[0]
        if length == self.sequence_length:
            return onehot.astype(np.float32, copy=False)

        padded = np.zeros((self.sequence_length, 4), dtype=np.float32)
        padded[:length] = onehot.astype(np.float32, copy=False)
        return padded

    def _augment(self, onehot: np.ndarray) -> np.ndarray:
        out = onehot
        if self.reverse_complement and self._rng.random() < self.rc_prob:
            out = _reverse_complement_onehot(out)
        if self.random_shift and self.max_shift > 0 and self._rng.random() < self.shift_prob:
            shift = int(self._rng.integers(-self.max_shift, self.max_shift + 1))
            out = np.roll(out, shift, axis=0)
        return out

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        construct = self.construct_spec.assemble_sequence(
            self._payloads[index],
            mode=self.construct_mode,
        )
        onehot = sequence_to_onehot(construct).astype(np.float32, copy=False)
        onehot = self._augment(onehot)
        onehot = self._pad_to_length(onehot)
        target = np.float32(self._targets[index])
        return torch.from_numpy(onehot), torch.tensor(target, dtype=torch.float32)


class DeepSTARRDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """PyTorch Dataset for Drosophila DeepSTARR (dev + hk enhancer activity).

    Assembles each reporter construct as ``ADAPTER_UP + insert + ADAPTER_DOWN`` and
    pads/trims to ``sequence_length`` (default 256 bp). Returns ``(onehot, target(2,))``
    where ``target = [dev, hk]``. Targets are read unchanged from the TSV (assumed
    already log2-transformed per the original DeepSTARR dataset convention).
    """

    DEFAULT_TARGET_COLUMNS = ("Dev_log2_enrichment", "Hk_log2_enrichment")

    def __init__(
        self,
        input_tsv: str | Path,
        split: str = "train",
        split_column: str = "set",
        sequence_column: str = "sequence",
        target_columns: tuple[str, ...] = DEFAULT_TARGET_COLUMNS,
        use_adapters: bool = True,
        left_adapter: str = DEEPSTARR_ADAPTER_UP,
        right_adapter: str = DEEPSTARR_ADAPTER_DOWN,
        sequence_length: int = 256,
        reverse_complement: bool = False,
        rc_prob: float = 0.5,
        random_shift: bool = False,
        shift_prob: float = 0.5,
        max_shift: int = 25,
        subset_frac: float = 1.0,
        seed: int = 42,
    ) -> None:
        if sequence_length <= 0:
            raise ValueError("sequence_length must be > 0")
        if not 0 < subset_frac <= 1:
            raise ValueError("subset_frac must be in (0, 1]")
        if not 0 <= rc_prob <= 1:
            raise ValueError("rc_prob must be in [0, 1]")
        if not 0 <= shift_prob <= 1:
            raise ValueError("shift_prob must be in [0, 1]")
        if max_shift < 0:
            raise ValueError("max_shift must be >= 0")
        if len(target_columns) < 1:
            raise ValueError("target_columns must have at least one column")

        self.input_tsv = Path(input_tsv)
        self.split = split
        self.split_column = split_column
        self.sequence_column = sequence_column
        self.target_columns = tuple(target_columns)
        self.use_adapters = bool(use_adapters)
        self.left_adapter = left_adapter if self.use_adapters else ""
        self.right_adapter = right_adapter if self.use_adapters else ""
        self.sequence_length = int(sequence_length)
        self.reverse_complement = reverse_complement
        self.rc_prob = rc_prob
        self.random_shift = random_shift
        self.shift_prob = shift_prob
        self.max_shift = max_shift
        self._rng = np.random.default_rng(seed)

        if not self.input_tsv.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.input_tsv}")

        rows = self._read_tsv()
        if subset_frac < 1.0 and rows:
            sample_size = max(1, int(round(len(rows) * subset_frac)))
            sample_indices = self._rng.choice(len(rows), size=sample_size, replace=False)
            rows = [rows[int(idx)] for idx in sorted(sample_indices.tolist())]

        self._inserts = [str(row[self.sequence_column]) for row in rows]
        self._targets = np.asarray(
            [[float(row[col]) for col in self.target_columns] for row in rows],
            dtype=np.float32,
        )

    def _read_tsv(self) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        with open(self.input_tsv, newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                if self.split_column in row and row[self.split_column] != self.split:
                    continue
                rows.append(row)
        return rows

    def __len__(self) -> int:
        return len(self._inserts)

    def _pad_or_trim(self, onehot: np.ndarray) -> np.ndarray:
        length = onehot.shape[0]
        if length == self.sequence_length:
            return onehot.astype(np.float32, copy=False)
        if length > self.sequence_length:
            return onehot[: self.sequence_length].astype(np.float32, copy=False)
        padded = np.zeros((self.sequence_length, 4), dtype=np.float32)
        padded[:length] = onehot.astype(np.float32, copy=False)
        return padded

    def _augment(self, onehot: np.ndarray) -> np.ndarray:
        out = onehot
        if self.reverse_complement and self._rng.random() < self.rc_prob:
            out = _reverse_complement_onehot(out)
        if self.random_shift and self.max_shift > 0 and self._rng.random() < self.shift_prob:
            shift = int(self._rng.integers(-self.max_shift, self.max_shift + 1))
            out = np.roll(out, shift, axis=0)
        return out

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        construct = f"{self.left_adapter}{self._inserts[index]}{self.right_adapter}"
        onehot = sequence_to_onehot(construct).astype(np.float32, copy=False)
        onehot = self._augment(onehot)
        onehot = self._pad_or_trim(onehot)
        target = self._targets[index]
        return torch.from_numpy(onehot), torch.from_numpy(np.asarray(target, dtype=np.float32))


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
