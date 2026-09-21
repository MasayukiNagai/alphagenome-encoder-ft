# AlphaGenome Encoder Fine-tuning

`alphagenome-encoder-ft` is a PyTorch implementation of the encoder-only fine-tuning workflow from [`alphagenome_FT_MPRA`](https://github.com/Al-Murphy/alphagenome_FT_MPRA), built on [`alphagenome-pytorch`](https://github.com/genomicsxai/alphagenome-pytorch).

It fine-tunes the AlphaGenome encoder on massively parallel reporter assays (MPRA) and predicts regulatory activity for new inserts, so a trained model can be used to score variants and run attribution.

Note: this does not cover the full feature surface of `alphagenome_FT_MPRA`, such as cached embedding workflows or the full collection of benchmarking scripts.

## The core idea: the insert is the only variable part

In an MPRA the assayed molecule is a reporter construct. Only the insert changes between rows; the adapters, minimal promoter, barcode and vector backbone are fixed for a library. A `Construct` holds that fixed context, so the same rule builds the model input during training and at inference, and it is saved into the checkpoint.

That is what makes this work after loading a checkpoint:

```python
from alphagenome_encoder_ft import AlphaGenomeEncoderModel

model = AlphaGenomeEncoderModel.from_checkpoint("best.pt")

# variant effect: the insert is all you pass
effect = model.predict_inserts([alt_insert]) - model.predict_inserts([ref_insert])
```

## Supported heads

| `head_type` | Class | Outputs | Default use |
|-------------|-------|---------|-------------|
| `mpra` (default) | `MPRAHead` | 1 scalar per sequence | lentiMPRA-style scalar regression |
| `deepstarr` | `DeepSTARRHead` | 2 scalars per sequence (dev, hk) | Drosophila STARR-seq dual-output regression |

Both heads share the same pooling modes (`flatten`, `center`, `mean`, `sum`, `max`) and the same `norm → MLP → Linear` layout; `DeepSTARRHead` is a subclass of `MPRAHead` whose only functional difference is `num_outputs=2`. Checkpoints persist `head_type` so `from_checkpoint(...)` dispatches to the right class.

`norm_type` selects the normalization applied to the encoder output before the MLP:

| `norm_type` | Module | Normalizes over |
|---|---|---|
| `layer` (default) | `LayerNorm(1536)` | channels, per position; independent of batch composition |
| `batch` | `BatchNorm1d(1536)` | batch and position, per channel; keeps running statistics |
| `group` | `GroupNorm(8, 1536)` | channel groups, per position |
| `none` | `Identity` | nothing |

`batch` and `group` normalize over a channel dimension, so the head transposes to `(B, D, L)` for them and back afterwards; `layer` and `none` act on the last dimension directly. The default stays `layer`, which is what earlier checkpoints were trained with and keeps their `norm.weight` / `norm.bias` keys loading unchanged.

## Installation

Requires Python 3.12+.

```bash
uv add "alphagenome-encoder-ft @ git+https://github.com/MasayukiNagai/alphagenome-encoder-ft.git"
```

For local development:

```bash
git clone https://github.com/MasayukiNagai/alphagenome-encoder-ft.git
cd alphagenome-encoder-ft
uv pip install -e .
uv pip install wandb matplotlib pytest   # training, evaluation, tests
```

## `Construct`

```python
from alphagenome_encoder_ft import Construct

construct = Construct(prefix="", suffix="TCCATT...GCAATG" + "AGAGACTGAGGCCAC", length=281)
construct.assemble_sequence(insert)          # -> str, the model input
construct.assemble_sequences([a, b, c])      # -> list[str]
construct.assemble_onehot(insert_onehot)     # -> Tensor, differentiable
```

Three fields, and that is the whole abstraction:

| field | meaning |
|---|---|
| `prefix` | fixed sequence before the insert |
| `suffix` | fixed sequence after the insert |
| `length` | optional fixed model input length |

**The window rule.** With `length` set, the assembled sequence is windowed to exactly that many bases. Longer sequences are trimmed from both ends; shorter ones are padded with `N` (an all-zero one-hot) on both ends. When the amount is odd, the extra base goes on the suffix side.

```
prefix + insert + suffix = 13 bp, length=10  ->  trim 1 left, 2 right
prefix + insert + suffix = 10 bp, length=13  ->  pad  1 left, 2 right
```

**Jitter.** `offset` slides that window and is the training-time shift augmentation. The dataset draws it per item; inference leaves it at 0, so predictions use the exact centered layout the model was trained on. Because the window (not a roll) does the shifting, sequence never wraps from one end to the other.

`Construct` is optional everywhere. `None` means the sequences are already model inputs.

### Presets

```python
from alphagenome_encoder_ft import (
    lentimpra_construct,
    lentimpra_promoter_barcode_construct,
    deepstarr_construct,
)

lentimpra_construct()                     # the whole reporter around a bare 200 bp element
lentimpra_promoter_barcode_construct()    # minus the adapters, for inserts that already carry them
deepstarr_construct()                     # STARR-seq adapters around the insert -> 256 bp
```

Both lentiMPRA presets describe the same 281 bp reporter; they differ in where the insert
starts:

| preset | insert it expects | what it adds |
|---|---|---|
| `lentimpra_construct()` | bare element, 200 bp | left adapter, right adapter, minP, barcode |
| `lentimpra_promoter_barcode_construct()` | 230 bp with adapters inline | minP, barcode |

The published Agarwal et al. 2025 TSVs are the second case: their `seq` column is 230 bp
with the adapters inline. Two ways to handle that, and they produce byte-identical model
input:

```python
# take the adapters off, so the insert is the bare element (what the drivers do)
LentiMPRADataset(tsv, strip_adapters=True, construct=lentimpra_construct())

# or leave seq alone and add only what is missing
LentiMPRADataset(tsv, construct=lentimpra_promoter_barcode_construct())
```

Prefer the first. The insert is then the 200 bp element, so `predict_inserts` takes the
sequence you designed with no adapter bookkeeping, and attribution returns one gradient row
per element base instead of 30 constant adapter rows you have to remember to ignore.

`strip_adapters` verifies rather than assumes: a row that does not carry the expected
flanks raises and names itself. It defaults to off, because this reader is also used for
files with the same columns but no adapters.

The individual pieces live in `alphagenome_encoder_ft.constructs` rather than the top-level API, for composing a layout of your own, such as an ablation that drops the barcode:

```python
from alphagenome_encoder_ft import Construct
from alphagenome_encoder_ft.constructs import LENTIMPRA_PROMOTER

promoter_only = Construct(suffix=LENTIMPRA_PROMOTER, length=281)
```

Available: `LENTIMPRA_PROMOTER`, `LENTIMPRA_BARCODE`, `LENTIMPRA_LEFT_ADAPTER`, `LENTIMPRA_RIGHT_ADAPTER`, `DEEPSTARR_ADAPTER_UP`, `DEEPSTARR_ADAPTER_DOWN`.

## Scoring inserts

| method | input | gradients |
|---|---|---|
| `predict_inserts(inserts)` | `Sequence[str]` | no, runs under `no_grad` |
| `forward_inserts(onehot)` | `Tensor (B, L, 4)` | yes |
| `forward(onehot)` | `Tensor (B, L, 4)`, already assembled | yes |

Attribution over the insert, with the flanks attached inside the graph:

```python
x = insert_onehot.unsqueeze(0).requires_grad_(True)   # (1, L, 4), insert only
model.forward_inserts(x).sum().backward()
saliency = x.grad                                      # (1, L, 4), aligned to the insert
```

## Datasets

`MPRADataset` takes inserts and targets in memory and owns the construct and the augmentation. It is not tied to a file format, so a variant table assembled in Python works directly:

```python
from alphagenome_encoder_ft import MPRADataset, lentimpra_construct

# inserts here are bare elements, so the full reporter preset applies
ds = MPRADataset(inserts, targets, construct=lentimpra_construct(), reverse_complement=True)
```

Readers subclass it and parse one assay's layout:

- `LentiMPRADataset(input_tsv, split=...)` — `seq` / `mean_value` / `fold` / `rev`; keeps `rev == 0` and selects folds per split.
- `DeepSTARRDataset(input_tsv, split=...)` — a split column plus two log2 targets.

Reverse complement applies to the whole assembled sequence, matching a double-stranded plasmid.

## Train and evaluate

One driver per assay, because the dataset layout and the construct are assay-specific:

```bash
python scripts/train_lentimpra.py \
  --config configs/lentimpra_K562.json \
  --input_tsv /path/to/K562.tsv \
  --pretrained_weights /path/to/alphagenome.safetensors

python scripts/evaluate_lentimpra.py --checkpoint_path results/mpra_K562/stage2/best.pt
```

`--construct_prefix`, `--construct_suffix`, `--construct_length` and `--no-construct` override the driver's default construct for a run. Config files hold training hyperparameters only; the input file and the construct are the driver's arguments. Training writes `config.json`, `run.json` (construct, input length, input TSV) and `history.json` into the run directory, and evaluation reads the construct back from the checkpoint.

- Input TSV for lentiMPRA: https://github.com/autosome-ru/human_legnet
- Pretrained weights: https://huggingface.co/gtca/alphagenome_pytorch

`cli.py` holds that scaffolding: argparse, config files, run-directory layout and wandb.
It exists so a new assay's script is short, and it is installed so scripts outside this
repository can use it too. Nothing in it is needed to use the package as a library. The
training itself is `train.py`, which takes a model and data loaders you built and knows
nothing about config files or run directories, and the statistics are `metrics.py`.

## Layout

```text
src/alphagenome_encoder_ft/
├── constructs.py # Construct + assay presets
├── data.py       # MPRADataset base + per-assay readers
├── heads.py      # MPRAHead, DeepSTARRHead
├── model.py      # AlphaGenomeEncoderModel (backbone + head + construct)
├── train.py      # epoch loop, evaluation, checkpointing, two-stage schedule
├── metrics.py    # Pearson, Spearman, the evaluation summary
├── config.py     # TrainConfig and friends
└── cli.py        # argparse/config/run-directory scaffolding for the scripts
scripts/
├── train_lentimpra.py / evaluate_lentimpra.py
├── train_deepstarr.py / evaluate_deepstarr.py
└── convert_checkpoint_v0.py
```

## Upgrading from 0.x

Version 1.0 replaced `ConstructSpec` (four named reporter pieces plus a five-way `construct_mode`) with `Construct`, and the loaders do not read the old format.

- **Old checkpoints**: convert once, then load normally.

  ```bash
  python scripts/convert_checkpoint_v0.py old.pt new.pt
  ```

  The converter maps each old `construct_mode` to the prefix and suffix it concatenated, and carries `sequence_length` over as `input_length`. Weights are untouched.

- **Old code**: pin the previous release.

  ```bash
  uv add "alphagenome-encoder-ft @ git+https://github.com/MasayukiNagai/alphagenome-encoder-ft.git@v0.1.0"
  ```

Other renames: `EncoderMPRAModel` is gone (use `AlphaGenomeEncoderModel`), `predict_sequences` is replaced by `predict_inserts` and `forward_inserts`, and `DataConfig` no longer carries `input_tsv`, `sequence_length`, `construct_mode` or the adapter/promoter/barcode fields.
