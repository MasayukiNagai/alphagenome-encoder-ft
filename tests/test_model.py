"""Model-level construct handling: insert entry points and checkpoint round trips."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from alphagenome_pytorch.utils.sequence import sequence_to_onehot

from alphagenome_encoder_ft.constructs import Construct
from alphagenome_encoder_ft.heads import MPRAHead
from alphagenome_encoder_ft.model import AlphaGenomeEncoderModel
from alphagenome_encoder_ft.train import save_checkpoint
from conftest import DummyAlphaGenome, make_config


def _onehot(sequence: str) -> torch.Tensor:
    return torch.from_numpy(sequence_to_onehot(sequence).astype(np.float32))


def _make_model(construct: Construct | None, *, input_length: int) -> AlphaGenomeEncoderModel:
    torch.manual_seed(0)
    model = AlphaGenomeEncoderModel(
        DummyAlphaGenome(),
        MPRAHead(pooling_type="flatten", hidden_sizes=8),
        construct=construct,
    )
    model.initialize_head(input_length, device="cpu")
    model.eval()
    return model


def test_predict_inserts_matches_forward_on_the_assembled_onehot():
    construct = Construct(prefix="A", suffix="GT", length=6)
    model = _make_model(construct, input_length=6)

    assembled = _onehot(construct.assemble_sequence("cc")).unsqueeze(0)
    direct = model(assembled, torch.zeros(1, dtype=torch.long))
    predicted = model.predict_inserts(["cc"])

    np.testing.assert_allclose(predicted.numpy(), direct.detach().numpy(), rtol=1e-5, atol=1e-5)


def test_forward_inserts_matches_predict_inserts():
    construct = Construct(prefix="A", suffix="GT", length=6)
    model = _make_model(construct, input_length=6)

    from_tensor = model.forward_inserts(_onehot("CC").unsqueeze(0))
    from_strings = model.predict_inserts(["CC"])

    np.testing.assert_allclose(
        from_tensor.detach().numpy(), from_strings.numpy(), rtol=1e-5, atol=1e-5
    )


def test_forward_inserts_backpropagates_to_the_insert():
    model = _make_model(Construct(prefix="AAA", suffix="GGG", length=10), input_length=10)
    insert = _onehot("ACGT").unsqueeze(0).requires_grad_(True)

    model.forward_inserts(insert).sum().backward()

    assert insert.grad is not None
    assert insert.grad.shape == (1, 4, 4)
    assert torch.count_nonzero(insert.grad) > 0


def test_predict_inserts_does_not_build_a_graph():
    model = _make_model(Construct(length=4), input_length=4)
    assert model.predict_inserts(["ACGT"]).requires_grad is False


def test_insert_entry_points_pass_through_without_a_construct():
    model = _make_model(None, input_length=4)
    onehot = _onehot("ACGT").unsqueeze(0)

    direct = model(onehot, torch.zeros(1, dtype=torch.long))
    np.testing.assert_allclose(
        model.predict_inserts(["acgt"]).numpy(), direct.detach().numpy(), rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        model.forward_inserts(onehot).detach().numpy(), direct.detach().numpy(), rtol=1e-5, atol=1e-5
    )


def test_predict_inserts_rejects_ragged_batches_without_a_length():
    model = _make_model(Construct(prefix="A"), input_length=3)
    with pytest.raises(ValueError, match="same length"):
        model.predict_inserts(["AC", "ACG"])


def test_predict_inserts_rejects_an_empty_batch():
    model = _make_model(None, input_length=2)
    with pytest.raises(ValueError, match="at least one insert"):
        model.predict_inserts([])


def test_variant_effect_is_a_difference_of_predictions():
    construct = Construct(prefix="AA", suffix="GG", length=8)
    model = _make_model(construct, input_length=8)

    ref, alt = "ACGT", "ACCT"
    delta = model.predict_inserts([alt]) - model.predict_inserts([ref])
    both = model.predict_inserts([ref, alt])

    np.testing.assert_allclose(delta.numpy(), (both[1] - both[0]).numpy(), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("save_mode", ["minimal", "full"])
def test_checkpoint_roundtrip_restores_construct_and_input_length(tmp_path: Path, save_mode: str):
    construct = Construct(prefix="A", suffix="GT", length=6)
    model = _make_model(construct, input_length=6)
    before = model.predict_inserts(["cc"])

    path = save_checkpoint(
        tmp_path / f"{save_mode}.pt",
        model,
        save_mode=save_mode,
        config=make_config(tmp_path, save_mode=save_mode),
    )
    restored = AlphaGenomeEncoderModel.from_checkpoint(
        path, device="cpu", backbone_factory=DummyAlphaGenome
    )

    assert restored.construct == construct
    assert restored.input_length == 6
    np.testing.assert_allclose(
        restored.predict_inserts(["cc"]).numpy(), before.numpy(), rtol=1e-5, atol=1e-5
    )


def test_checkpoint_roundtrip_without_a_construct(tmp_path: Path):
    model = _make_model(None, input_length=4)
    path = save_checkpoint(
        tmp_path / "no_construct.pt",
        model,
        save_mode="minimal",
        config=make_config(tmp_path),
    )
    restored = AlphaGenomeEncoderModel.from_checkpoint(
        path, device="cpu", backbone_factory=DummyAlphaGenome
    )
    assert restored.construct is None
    assert restored.input_length == 4


def test_model_save_checkpoint_round_trips_without_a_config(tmp_path: Path):
    construct = Construct(prefix="A", suffix="GT", length=6)
    model = _make_model(construct, input_length=6)
    before = model.predict_inserts(["cc"])

    path = model.save_checkpoint(tmp_path / "no_config.pt")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert not {"config", "stage", "epoch", "metrics"} & payload.keys()

    restored = AlphaGenomeEncoderModel.from_checkpoint(
        path, device="cpu", backbone_factory=DummyAlphaGenome
    )
    assert restored.construct == construct
    np.testing.assert_allclose(
        restored.predict_inserts(["cc"]).numpy(), before.numpy(), rtol=1e-5, atol=1e-5
    )


def test_head_config_is_read_from_the_head_not_the_config(tmp_path: Path):
    """A config that disagrees with the built head must not decide what is restored."""

    model = _make_model(None, input_length=4)  # the head was built with hidden_sizes [8]
    path = save_checkpoint(
        tmp_path / "mismatched.pt",
        model,
        save_mode="minimal",
        config=make_config(tmp_path, head={"hidden_sizes": [1024]}),
    )

    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["head_config"]["hidden_sizes"] == [8]

    restored = AlphaGenomeEncoderModel.from_checkpoint(
        path, device="cpu", backbone_factory=DummyAlphaGenome
    )
    assert restored.head.hidden_sizes == [8]


def test_from_checkpoint_rejects_a_v0_payload(tmp_path: Path):
    model = _make_model(Construct(length=4), input_length=4)
    path = save_checkpoint(
        tmp_path / "v0.pt",
        model,
        save_mode="minimal",
        config=make_config(tmp_path),
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload.pop("construct")
    payload.pop("input_length")
    payload["construct_config"] = {"construct_mode": "none", "sequence_length": 4}
    torch.save(payload, path)

    with pytest.raises(ValueError, match="convert_checkpoint_v0"):
        AlphaGenomeEncoderModel.from_checkpoint(
            path, device="cpu", backbone_factory=DummyAlphaGenome
        )


def test_from_checkpoint_rejects_head_only(tmp_path: Path):
    model = _make_model(None, input_length=2)
    path = save_checkpoint(
        tmp_path / "head_only.pt",
        model,
        save_mode="head",
        config=make_config(tmp_path, save_mode="head"),
    )
    with pytest.raises(ValueError, match="Head-only checkpoints"):
        AlphaGenomeEncoderModel.from_checkpoint(
            path, device="cpu", backbone_factory=DummyAlphaGenome
        )


def test_save_checkpoint_requires_an_initialized_head(tmp_path: Path):
    model = AlphaGenomeEncoderModel(DummyAlphaGenome(), MPRAHead(pooling_type="flatten", hidden_sizes=8))
    with pytest.raises(ValueError, match="input_length is unset"):
        save_checkpoint(
            tmp_path / "uninitialized.pt",
            model,
            save_mode="minimal",
            config=make_config(tmp_path),
        )
