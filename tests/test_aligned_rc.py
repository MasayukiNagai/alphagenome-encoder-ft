"""data.align_rc: reverse-complement inside the model and flip the encoder bins back."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from alphagenome_encoder_ft import cli
from alphagenome_encoder_ft.config import DataConfig
from alphagenome_encoder_ft.heads import MPRAHead
from alphagenome_encoder_ft.model import AlphaGenomeEncoderModel
from alphagenome_encoder_ft.train import train_epoch
from conftest import DummyAlphaGenome, make_config

L = 256  # two 128 bp bins


def _model() -> AlphaGenomeEncoderModel:
    torch.manual_seed(0)
    model = AlphaGenomeEncoderModel(DummyAlphaGenome(), MPRAHead(hidden_sizes=8, dropout=None))
    model.initialize_head(L, device="cpu")
    return model.eval()


def _onehot(batch: int, length: int = L) -> torch.Tensor:
    idx = torch.randint(0, 4, (batch, length))
    return torch.nn.functional.one_hot(idx, 4).float()


def _rc(x: torch.Tensor) -> torch.Tensor:
    return x.flip(1)[..., [3, 2, 1, 0]]


def test_masked_samples_are_reverse_complemented_and_flipped_back():
    model, x = _model(), _onehot(4)
    mask = torch.tensor([True, False, True, False])
    with torch.no_grad():
        out = model.encode_aligned_rc(x, mask)
        forward = model.encode(x)
        flipped = model.encode(_rc(x)).flip(1)  # the dummy encoder emits (B, L, 1536)
    assert torch.allclose(out[mask], flipped[mask])
    assert torch.allclose(out[~mask], forward[~mask])


def test_input_length_must_be_a_multiple_of_the_bin():
    with pytest.raises(ValueError, match="multiple of 128"):
        _model().encode_aligned_rc(_onehot(2, 200), torch.ones(2, dtype=torch.bool))


def test_align_rc_needs_reverse_complement():
    with pytest.raises(ValueError, match="align_rc needs data.reverse_complement"):
        DataConfig(align_rc=True)
    assert DataConfig(reverse_complement=True, align_rc=True).align_rc


def test_dataset_does_not_reverse_complement_when_the_model_does(tmp_path: Path):
    aligned = make_config(tmp_path, data={"reverse_complement": True, "align_rc": True})
    plain = make_config(tmp_path, data={"reverse_complement": True})
    assert cli.dataset_kwargs(aligned, augment=True)["reverse_complement"] is False
    assert cli.dataset_kwargs(plain, augment=True)["reverse_complement"] is True
    assert cli.dataset_kwargs(plain, augment=False)["reverse_complement"] is False


def test_cli_flag_sets_align_rc(tmp_path: Path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(make_config(tmp_path).to_dict()))
    parser = cli.add_train_arguments(argparse.ArgumentParser())
    args = parser.parse_args(["--config", str(config_path), "--reverse_complement", "--align_rc"])
    config = cli.load_config(parser, args)
    assert config.data.reverse_complement and config.data.align_rc


@pytest.mark.parametrize("train_encoder", [False, True])
@pytest.mark.parametrize("prob", [None, 1.0])
def test_train_epoch_uses_aligned_rc_only_when_asked(monkeypatch, train_encoder: bool, prob):
    model = _model()
    calls = []
    original = model.encode_aligned_rc

    def spy(sequences, rc_mask, organism_idx=None):
        calls.append(rc_mask.clone())
        return original(sequences, rc_mask, organism_idx)

    monkeypatch.setattr(model, "encode_aligned_rc", spy)
    loader = DataLoader(TensorDataset(_onehot(8), torch.randn(8)), batch_size=4)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    train_epoch(model, loader, optimizer, "cpu", use_amp=False, train_encoder=train_encoder, aligned_rc_prob=prob)
    if prob is None:
        assert calls == []
    else:
        assert len(calls) == 2 and all(mask.all() for mask in calls)
