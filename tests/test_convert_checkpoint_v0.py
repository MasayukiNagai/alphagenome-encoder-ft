"""v0 -> v1 checkpoint conversion: mode mapping and the ambiguity guard."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from alphagenome_encoder_ft.config import TrainConfig

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "convert_checkpoint_v0.py"
_spec = importlib.util.spec_from_file_location("convert_checkpoint_v0", SCRIPT)
convert_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(convert_module)
convert_payload = convert_module.convert_payload

PIECES = {
    "left_adapter": "AAA",
    "right_adapter": "CCC",
    "promoter_seq": "GGG",
    "barcode_seq": "TT",
    "sequence_length": 281,
}


def _payload(**construct_config) -> dict:
    return {
        "save_mode": "minimal",
        "construct_config": {**PIECES, **construct_config},
        "config": {"data": {"sequence_length": 281, "construct_mode": None}},
    }


@pytest.mark.parametrize(
    "mode,prefix,suffix",
    [
        ("none", "", ""),
        ("adapters", "AAA", "CCC"),
        ("promoter", "", "GGG"),
        ("promoter_barcode", "", "GGGTT"),
        ("all", "AAA", "CCCGGGTT"),
    ],
)
def test_each_mode_maps_to_the_pieces_it_concatenated(mode: str, prefix: str, suffix: str):
    converted = convert_payload(_payload(construct_mode=mode))

    assert converted["construct"] == {"prefix": prefix, "suffix": suffix, "length": 281}
    assert converted["input_length"] == 281
    assert "construct_config" not in converted
    assert converted["converted_from"] == "v0"


def test_an_explicit_mode_supplies_one_the_checkpoint_lacks():
    converted = convert_payload(_payload(), construct_mode="promoter_barcode")
    assert converted["construct"]["suffix"] == "GGGTT"


def test_a_missing_mode_is_refused_rather_than_guessed():
    # The autotune reference checkpoints carry every piece but no construct_mode.
    # Defaulting to "none" would build an empty construct and silently mispredict.
    with pytest.raises(ValueError) as exc:
        convert_payload(_payload())

    message = str(exc.value)
    assert "no construct_mode" in message
    assert "promoter_barcode" in message  # the error lists the valid modes
    assert "281" in message  # and the recorded length, which identifies the right one


def test_the_removed_v1_data_fields_are_stripped_from_the_config():
    payload = _payload(construct_mode="promoter")
    payload["config"]["data"].update(
        {"input_tsv": "/x.tsv", "promoter_seq": "GGG", "batch_size": 32}
    )

    data = convert_payload(payload)["config"]["data"]

    assert "input_tsv" not in data and "promoter_seq" not in data and "sequence_length" not in data
    assert data["batch_size"] == 32  # fields v1 still has survive


def test_pipeline_specific_config_sections_are_quarantined_not_dropped():
    """TrainConfig.from_dict rejects unknown sections; the reference checkpoints have four."""

    payload = _payload(construct_mode="promoter_barcode")
    payload["config"].update({"cell_type": "K562", "origin": "autotune", "source_ckpt": "x.pt"})

    config = convert_payload(payload)["config"]

    # the converted payload loads
    TrainConfig.from_dict(config)
    # and the provenance survives under a key from_dict ignores
    assert config["_v0_config_sections"]["cell_type"] == "K562"
    assert config["_v0_config_sections"]["origin"] == "autotune"
    assert "cell_type" not in config


def test_an_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="Unknown v0 construct_mode"):
        convert_payload(_payload(construct_mode="core"))


def test_a_v1_payload_is_not_converted_twice():
    with pytest.raises(ValueError, match="already in the v1 format"):
        convert_payload({"construct": None, "input_length": 281})


def test_a_mode_missing_its_pieces_is_reported():
    payload = _payload(construct_mode="promoter")
    payload["construct_config"]["promoter_seq"] = None

    with pytest.raises(ValueError, match="promoter_seq"):
        convert_payload(payload)


def test_old_training_layout_is_quarantined_so_the_converted_config_loads():
    payload = _payload(construct_mode="promoter_barcode")
    payload["config"].update(
        {
            "optim": {"optimizer": "adam", "learning_rate": 1e-3, "weight_decay": 1e-6},
            "stage": {"num_epochs": 100, "second_stage_lr": 1e-5},
        }
    )

    config = convert_payload(payload)["config"]

    TrainConfig.from_dict(config)
    assert config["optim"] == {"optimizer": "adam", "weight_decay": 1e-6}
    assert config["_v0_config_sections"]["optim"] == {"learning_rate": 1e-3}
    assert config["_v0_config_sections"]["stage"] == {"num_epochs": 100, "second_stage_lr": 1e-5}
