"""Wrapped AlphaGenome encoder + regression head, with the reporter construct attached."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

from alphagenome_pytorch import AlphaGenome
from alphagenome_pytorch.extensions.finetuning.transfer import load_trunk, remove_all_heads
from alphagenome_pytorch.utils.sequence import sequence_to_onehot_tensor

from .config import HeadConfig, TrainConfig, build_head
from .constructs import Construct


class AlphaGenomeEncoderModel(nn.Module):
    """AlphaGenome backbone + head.

    ``forward`` takes the final model input (one-hot of the assembled reporter).
    ``predict_inserts`` / ``forward_inserts`` take *inserts* and let ``self.construct`` add
    the fixed flanks, so a loaded checkpoint scores new inserts with no extra bookkeeping.
    """

    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module,
        *,
        construct: Construct | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.construct = construct
        self.input_length: int | None = None

    @property
    def encoder(self) -> nn.Module:
        if not hasattr(self.backbone, "encoder"):
            raise AttributeError("Backbone does not expose an 'encoder' module")
        return self.backbone.encoder

    # -------------------------
    # Forward paths
    # -------------------------

    def encode(self, sequences: Tensor, organism_idx: Tensor | None = None) -> Tensor:
        if organism_idx is None:
            organism_idx = torch.zeros(sequences.shape[0], dtype=torch.long, device=sequences.device)
        outputs = self.backbone(sequences, organism_idx, encoder_only=True)
        return outputs["encoder_output"]

    def predict_from_encoder(self, encoder_output: Tensor) -> Tensor:
        return self.head(encoder_output)

    def forward(self, sequences: Tensor, organism_idx: Tensor | None = None) -> Tensor:
        """Final model input ``(B, L, 4)`` in, predictions out."""

        return self.predict_from_encoder(self.encode(sequences, organism_idx))

    def forward_inserts(self, inserts: Tensor, organism_idx: Tensor | None = None) -> Tensor:
        """Insert one-hots ``(B, L, 4)`` in; flanks attached inside the graph.

        Gradients flow back to ``inserts``, so this is the entry point for attribution.
        """

        if inserts.ndim != 3 or inserts.shape[-1] != 4:
            raise ValueError(f"Expected inserts of shape (B, L, 4), got {tuple(inserts.shape)}")
        sequences = self.construct.assemble_onehot(inserts) if self.construct is not None else inserts
        return self(sequences, organism_idx)

    def predict_inserts(self, inserts: Sequence[str], organism_idx: Tensor | None = None) -> Tensor:
        """Insert strings in, predictions out, under ``no_grad``."""

        batch = [insert.strip().upper() for insert in inserts]
        if not batch:
            raise ValueError("predict_inserts requires at least one insert")
        if self.construct is not None:
            batch = self.construct.assemble_sequences(batch)
        if len({len(seq) for seq in batch}) != 1:
            raise ValueError("All assembled sequences must have the same length")

        device = next(self.parameters()).device
        onehot = torch.stack(
            [sequence_to_onehot_tensor(seq, dtype=torch.float32, device=device) for seq in batch],
            dim=0,
        )
        with torch.no_grad():
            return self(onehot, organism_idx)

    # -------------------------
    # Parameter handling
    # -------------------------

    def initialize_head(self, sequence_length: int, device: torch.device | str) -> None:
        """Run one dummy forward so lazily-shaped head layers materialize, and remember the length."""

        with torch.no_grad():
            device = torch.device(device)
            dummy_sequence = torch.zeros(1, sequence_length, 4, device=device)
            dummy_organism_idx = torch.zeros(1, dtype=torch.long, device=device)
            _ = self.predict_from_encoder(self.encode(dummy_sequence, dummy_organism_idx))
        self.input_length = int(sequence_length)

    def set_encoder_trainable(self, trainable: bool) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = trainable

    def trainable_parameters(self, include_encoder: bool) -> list[nn.Parameter]:
        params = list(self.head.parameters())
        if include_encoder:
            params = list(self.encoder.parameters()) + params
        deduped: list[nn.Parameter] = []
        seen: set[int] = set()
        for param in params:
            if param.requires_grad and id(param) not in seen:
                deduped.append(param)
                seen.add(id(param))
        return deduped

    # -------------------------
    # Constructors
    # -------------------------

    @staticmethod
    def _resolve_device(device: torch.device | str | None) -> torch.device:
        if device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_weights: str | Path,
        head_config: HeadConfig,
        *,
        device: torch.device | str | None = None,
        construct: Construct | None = None,
        backbone_factory=AlphaGenome,
        head_type: str | None = None,
    ) -> "AlphaGenomeEncoderModel":
        device = cls._resolve_device(device)
        backbone = backbone_factory()
        backbone = load_trunk(backbone, pretrained_weights, exclude_heads=True)
        backbone = remove_all_heads(backbone)
        resolved_head_type = head_type or getattr(head_config, "head_type", "mpra")
        head = build_head(resolved_head_type, head_config.__dict__)
        model = cls(backbone, head, construct=construct)
        model.set_encoder_trainable(False)
        model.to(device)
        return model

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        device: torch.device | str | None = None,
        backbone_factory=AlphaGenome,
    ) -> "AlphaGenomeEncoderModel":
        device = cls._resolve_device(device)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        save_mode = checkpoint.get("save_mode", "minimal")
        if save_mode == "head":
            raise ValueError("Head-only checkpoints cannot be loaded standalone")

        if "input_length" not in checkpoint or "construct" not in checkpoint:
            raise ValueError(
                f"{checkpoint_path} predates the Construct checkpoint format (missing 'construct' / "
                "'input_length'); convert it with scripts/convert_checkpoint_v0.py"
            )

        head_config_dict = dict(checkpoint.get("head_config", {}))
        head_type = checkpoint.get("head_type", head_config_dict.get("head_type", "mpra"))
        construct = Construct.from_dict(checkpoint["construct"])

        model = cls(backbone_factory(), build_head(head_type, head_config_dict), construct=construct)
        model.to(device)
        model.initialize_head(int(checkpoint["input_length"]), device)

        if save_mode == "minimal":
            model.encoder.load_state_dict(checkpoint["encoder_state_dict"])
        elif save_mode == "full":
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        else:
            raise ValueError(f"Unknown checkpoint save_mode: {save_mode}")

        model.head.load_state_dict(checkpoint["head_state_dict"])
        model.set_encoder_trainable(False)
        model.to(device)
        model.eval()
        return model

    # -------------------------
    # Checkpointing
    # -------------------------

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        save_mode: str = "minimal",
        config: TrainConfig | None = None,
    ) -> Path:
        """Write a checkpoint ``from_checkpoint`` can restore, with no ``TrainConfig`` needed.

        ``config`` is recorded as provenance when given; :mod:`cli` passes the one it trained
        with. The import is local because :mod:`train` imports this module.
        """

        from .train import save_checkpoint

        return save_checkpoint(path, self, save_mode=save_mode, config=config)
