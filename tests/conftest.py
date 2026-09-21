from __future__ import annotations

from pathlib import Path

import torch

from alphagenome_encoder_ft.config import TrainConfig


class DummyAlphaGenome(torch.nn.Module):
    """Stand-in backbone: per-position Linear(4 -> 1536), same interface as AlphaGenome."""

    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(4, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1536),
        )

    def forward(self, sequences, organism_idx, encoder_only=False):
        del organism_idx
        if not encoder_only:
            raise ValueError("Dummy model only supports encoder_only=True")
        batch, length, channels = sequences.shape
        encoded = self.encoder(sequences.reshape(batch * length, channels)).reshape(batch, length, 1536)
        return {"encoder_output": encoded}


def make_config(tmp_path: Path, *, save_mode: str = "minimal", **sections) -> TrainConfig:
    raw = {
        "data": {"batch_size": 4},
        "head": {
            "pooling_type": "flatten",
            "hidden_sizes": [8],
            "center_bp": 256,
            "dropout": 0.1,
            "activation": "relu",
        },
        "checkpoint": {
            "pretrained_weights": "/tmp/weights.pt",
            "checkpoint_dir": str(tmp_path),
            "save_mode": save_mode,
        },
    }
    for section, values in sections.items():
        raw[section] = {**raw.get(section, {}), **values}
    return TrainConfig.from_dict(raw)
