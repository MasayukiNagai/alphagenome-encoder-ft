"""Encoder-only AlphaGenome fine-tuning utilities for MPRA."""

__all__ = [
    "DataConfig",
    "HeadConfig",
    "OptimConfig",
    "StageConfig",
    "CheckpointConfig",
    "LoggingConfig",
    "RuntimeConfig",
    "TrainConfig",
    "build_head",
    "load_train_config",
    "merge_train_config",
    "parse_hidden_sizes",
    "MPRADataset",
    "LentiMPRADataset",
    "DeepSTARRDataset",
    "read_tsv_rows",
    "create_dataloader",
    "Construct",
    "lentimpra_construct",
    "lentimpra_full_construct",
    "deepstarr_construct",
    "AlphaGenomeEncoderModel",
    "MPRAHead",
    "DeepSTARRHead",
    "train_epoch",
    "evaluate",
    "run_training_stage",
    "run_two_stage_training",
    "save_checkpoint",
    "load_checkpoint",
    "set_encoder_trainable",
    "create_optimizer",
    "create_scheduler",
    "scheduler_stepper",
]

_MODULE_BY_NAME = {
    **dict.fromkeys(
        ["MPRADataset", "LentiMPRADataset", "DeepSTARRDataset", "read_tsv_rows", "create_dataloader"],
        "data",
    ),
    # The individual reporter pieces (LENTIMPRA_PROMOTER and friends) stay in
    # alphagenome_encoder_ft.constructs rather than the top-level API: they are building
    # blocks for composing a custom layout, not part of the curated surface.
    **dict.fromkeys(
        ["Construct", "lentimpra_construct", "lentimpra_full_construct", "deepstarr_construct"],
        "constructs",
    ),
    **dict.fromkeys(
        [
            "DataConfig",
            "HeadConfig",
            "OptimConfig",
            "StageConfig",
            "CheckpointConfig",
            "LoggingConfig",
            "RuntimeConfig",
            "TrainConfig",
            "build_head",
            "load_train_config",
            "merge_train_config",
            "parse_hidden_sizes",
        ],
        "config",
    ),
    "AlphaGenomeEncoderModel": "model",
    **dict.fromkeys(["MPRAHead", "DeepSTARRHead"], "heads"),
    **dict.fromkeys(
        [
            "train_epoch",
            "evaluate",
            "run_training_stage",
            "run_two_stage_training",
            "save_checkpoint",
            "load_checkpoint",
            "set_encoder_trainable",
            "create_optimizer",
            "create_scheduler",
            "scheduler_stepper",
        ],
        "train",
    ),
}


def __getattr__(name: str):
    module_name = _MODULE_BY_NAME.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(f".{module_name}", __name__), name)
