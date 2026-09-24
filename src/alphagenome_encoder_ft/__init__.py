"""Encoder-only AlphaGenome fine-tuning utilities for MPRA."""

__all__ = [
    "DataConfig",
    "HeadConfig",
    "OptimConfig",
    "StageConfig",
    "Stage2Config",
    "CheckpointConfig",
    "LoggingConfig",
    "RuntimeConfig",
    "TrainConfig",
    "build_head",
    "load_train_config",
    "merge_train_config",
    "parse_hidden_sizes",
    "MPRADataset",
    "LentiMPRAAgarwal2025Dataset",
    "DeepSTARRDeAlmeida2022Dataset",
    "read_tsv_rows",
    "strip_flanks",
    "create_dataloader",
    "Construct",
    "LentiMPRAAgarwal2025Library",
    "DeepSTARRDeAlmeida2022Library",
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
    "create_stage1_optimizer",
    "create_stage2_optimizer",
    "encoder_head_param_groups",
    "create_scheduler",
    "scheduler_stepper",
]

_MODULE_BY_NAME = {
    **dict.fromkeys(
        [
            "MPRADataset",
            "LentiMPRAAgarwal2025Dataset",
            "DeepSTARRDeAlmeida2022Dataset",
            "read_tsv_rows",
            "strip_flanks",
            "create_dataloader",
        ],
        "data",
    ),
    **dict.fromkeys(
        ["Construct", "LentiMPRAAgarwal2025Library", "DeepSTARRDeAlmeida2022Library"],
        "constructs",
    ),
    **dict.fromkeys(
        [
            "DataConfig",
            "HeadConfig",
            "OptimConfig",
            "StageConfig",
            "Stage2Config",
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
            "create_stage1_optimizer",
            "create_stage2_optimizer",
            "encoder_head_param_groups",
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
