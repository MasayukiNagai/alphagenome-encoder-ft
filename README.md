# AlphaGenome Encoder Fine-tuning

`alphagenome-encoder-ft` is a PyTorch implementation of the encoder-only fine-tuning workflow from [`alphagenome_FT_MPRA`](https://github.com/Al-Murphy/alphagenome_FT_MPRA).

This repository is built on top of [`alphagenome-pytorch`](https://github.com/genomicsxai/alphagenome-pytorch/tree/main) and focuses on a smaller scope than the original JAX-based project. In particular, it currently targets lentiMPRA-style scalar regression with AlphaGenome encoder features, reusable encoder-only training utilities, and shared construct assembly utilities for inference.

Note: The current codebase does not yet include the full feature surface of `alphagenome_FT_MPRA`, such as attribution pipelines, cached embedding workflows, or the full collection of benchmarking and downstream analysis scripts.

## Installation

`alphagenome-encoder-ft` requires Python 3.12+ and depends on
[`alphagenome-pytorch`](https://github.com/genomicsxai/alphagenome-pytorch).

Recommended with `uv`:

```bash
cd alphagenome-encoder-ft
uv sync

# Or to include libraries for training
uv sync --group train
```

With `pip`:

```bash
cd alphagenome-encoder-ft
pip install -e .
# For training, evaluation, test
pip install tqdm wandb matplotlib pytest
```

## Repository Layout

```text
alphagenome-encoder-ft/
├── src/alphagenome_encoder_ft/
│   ├── __init__.py
│   ├── config.py     # default configs for each cell type
│   ├── constructs.py # ConstructSpec assembly rules
│   ├── data.py       # lentiMPRA dataset and dataloader helpers
│   ├── heads.py      # MPRAHead
│   ├── model.py      # EncoderMPRAModel wrapper (AG Encoder + MPRAHead)
│   ├── train.py      # reusable encoder-only training utilities
│   └── ...
├── configs/
│   ├── lentimpra_HepG2.json
│   ├── lentimpra_K562.json
│   └── lentimpra_WTC11.json
├── scripts/
│   ├── train_mpra.py    # config/CLI entrypoint for training
│   ├── evaluate_mpra.py # evaluate a saved checkpoint on the test split
│   └── ...
└── tests/
```

## Train

```bash
cd alphagenome-encoder-ft
python scripts/train_mpra.py \
  --config configs/lentimpra_HepG2.json \
  --input_tsv /path/to/HepG2.tsv \
  --pretrained_weights /path/to/alphagenome.pt
```

- Input TSV for lentiMPRA: https://github.com/autosome-ru/human_legnet
- Pretrained weights: https://huggingface.co/gtca/alphagenome_pytorch

For local runs without installation, you can use `PYTHONPATH=src`.

`data.construct_mode` now uses the modes implemented by `ConstructSpec`:
`none`, `adapters`, `promoter`, `promoter_barcode`, and `all`.
The default in config is `promoter_barcode`.

## Evaluate

```bash
cd alphagenome-encoder-ft
python scripts/evaluate_mpra.py \
  --checkpoint_path /path/to/best.pt \
  --output_dir /path/to/eval_outputs
```

The evaluator reconstructs the model from the checkpoint config, runs the `test` split, computes Pearson and Spearman over the full concatenated test set, and saves:

- `test_metrics.json`
- `test_predictions.csv`
- `y_vs_y_pred.png`

## Load checkpoint

```python
from alphagenome_encoder_ft import EncoderMPRAModel

model = EncoderMPRAModel.from_checkpoint("/path/to/best.pt", device="cuda")
construct = model.construct_spec.assemble_sequence("ACGT", mode="promoter_barcode")
```

- Standalone loading supports `save_mode="minimal"` and `save_mode="full"`.
- `save_mode="head"` does not include enough backbone state to reconstruct an `EncoderMPRAModel` by itself.
- The checkpoint from `train_mpra.py` already contains the construct definition under `construct_config`, so `model.construct_spec` is typically ready to use after `from_checkpoint(...)`. For `construct_spec`, see below.

## Construct MPRA reporters

In MPRA, the assayed sequence is not just the variable insert itself. The
full reporter often includes fixed backbone pieces such as cloning adapters, a promoter, and a barcode. The exact construct depends on the assay design, but a typical reporter looks like:

```text
left_adapter + insert + right_adapter + promoter + barcode
```

During training or inference, you might want to control which of those fixed pieces are included around the insert sequence. `ConstructSpec` provides that assembly logic in one place.

If your data always arrives in the same final reporter shape, `ConstructSpec` is not necessary. It is an optional convenience for flexible assembly, mainly included for downstream applications that need to switch construct layouts.

### `ConstructSpec`

[`ConstructSpec`](/grid/koo/home/nagai/projects/ag_mpra_torch/alphagenome-encoder-ft/src/alphagenome_encoder_ft/constructs.py) defines the fixed reporter pieces:
`left_adapter`, `right_adapter`, `promoter_seq`, and `barcode_seq`.

Use the default lentiMPRA construct:

```python
from alphagenome_encoder_ft import ConstructSpec

construct_spec = ConstructSpec.lentimpra_default()
```

Or override the pieces for your assay:

```python
from alphagenome_encoder_ft import ConstructSpec

construct_spec = ConstructSpec(
    left_adapter="AAA",
    right_adapter="TTT",
    promoter_seq="GGGG",
    barcode_seq="CCCC",
)
```

### Assembly modes

`ConstructSpec` supports the following modes:

- `none`: insert only (i.e., nothing will be added)
- `adapters`: left adapter + insert + right adapter
- `promoter`: insert + promoter
- `promoter_barcode`: insert + promoter + barcode
- `all`: left adapter + insert + right adapter + promoter + barcode

Example:

```python
construct_spec = ConstructSpec.lentimpra_default()
reporter = construct_spec.assemble_sequence("MYSEQUENCE", mode="promoter_barcode")
# reporter: MYSEQUENCE + PROMOTER + BARCODE
```

For one-hot inputs, use `assemble_onehot(...)` with the same modes.
