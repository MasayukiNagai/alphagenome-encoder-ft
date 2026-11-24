# AlphaGenome Encoder Fine-tuning

`alphagenome-encoder-ft` fine-tunes the AlphaGenome encoder on massively parallel reporter assays (MPRA) and predicts regulatory activity for new inserts. It is a PyTorch implementation of the encoder-only workflow from [`alphagenome_FT_MPRA`](https://github.com/Al-Murphy/alphagenome_FT_MPRA), built on [`alphagenome-pytorch`](https://github.com/genomicsxai/alphagenome-pytorch).

Input is a table of inserts and measured activities; output is a checkpoint that scores new inserts, so a trained model can be used for variant effect prediction and attribution. Cached embedding workflows and the benchmarking scripts of `alphagenome_FT_MPRA` are not covered.

## Installation

Requires Python 3.12+.

As a dependency of another project:

```bash
uv add "alphagenome-encoder-ft @ git+https://github.com/MasayukiNagai/alphagenome-encoder-ft.git"
```

Pretrained backbone weights: https://huggingface.co/gtca/alphagenome_pytorch

## Usage

### Initialize

`AlphaGenomeEncoderModel` is the AlphaGenome backbone with a regression head attached. `from_pretrained` loads the backbone trunk, drops the AlphaGenome output heads and freezes the encoder:

```python
from alphagenome_encoder_ft import AlphaGenomeEncoderModel, HeadConfig

weights_path = "/path/to/alphagenome.safetensors"
input_length = 256   # length of the sequences the model will see
device = "cuda"

model = AlphaGenomeEncoderModel.from_pretrained(
    weights_path,
    HeadConfig(pooling_type="flatten", hidden_sizes=[1024], num_outputs=1),
    device=device,
)
model.initialize_head(input_length, device)   # materializes the lazily-shaped head layers
```

`initialize_head` runs one dummy forward at the given input length and records that length on the model. It must run before training or before loading head weights.

`HeadConfig` defines the head architecture only. Dropout is a training setting: each training stage sets `model.head.dropout` from its own config section.

### Train and evaluate

Each published dataset has a folder under `scripts/` with its data preparation, reader, training and evaluation scripts, because the dataset layout and the construct are dataset-specific:

| folder | dataset | steps |
|---|---|---|
| `scripts/agarwal2025/` | lentiMPRA, Agarwal et al. 2025 | `0_add_coordinates.py` (genomic coordinates for a chromosome split), `1_train.py`, `2_evaluate.py` |
| `scripts/gosai2024/` | episomal MPRA, Gosai et al. 2024 | `0_prepare.py` (Supplementary Table S2 to a training TSV), `1_train.py`, `2_evaluate.py` |
| `scripts/dealmeida2022/` | DeepSTARR, de Almeida et al. 2022 | `1_train.py`, `2_evaluate.py` |

```bash
python scripts/agarwal2025/1_train.py \
  --config configs/agarwal2025_locon_ema.json \
  --input_tsv data/K562_with_chrom.tsv \
  --pretrained_weights /path/to/alphagenome.safetensors \
  --checkpoint_dir results/agarwal2025_K562_locon_ema

python scripts/agarwal2025/2_evaluate.py --split test \
  --checkpoint_path results/agarwal2025_K562_locon_ema/stage2/best.pt
```

`run_train.sh CONFIG` and `run_evaluate.sh RUN_DIR` in each folder wrap those two (the data and weights paths come from `INPUT_TSV` and `ALPHAGENOME_WEIGHTS`); both forward extra flags to the Python script. The lentiMPRA and Gosai evaluators average forward and reverse-complement predictions unless `--forward_only` is passed, and score `--split val` or `test`.

Config files in `configs/` hold training hyperparameters only, and every field has a matching command-line flag that overrides it. They are named `<dataset>_<stage-2 setting>.json` and serve every cell type of that dataset (the data file decides the cell type); lentiMPRA and Gosai have four each, differing only in stage 2: `fullft` (full fine-tuning), `fullft_ema`, `locon` and `locon_ema` (rank-128 LoCon adapters). Their values were tuned on K562. `docs/LoCon_EMA.md` compares them and gives the commands that reproduce the comparison.

Training runs in two stages: stage 1 trains the head with the encoder frozen, and stage 2 unfreezes the encoder and continues from the best stage-1 checkpoint. `--stage1_only` skips stage 2, `--resume_from_stage2` starts stage 2 from an existing `stage1/best.pt`, and the test split is scored only with `--evaluate_test` or by the evaluate script.

A run directory:

```text
results/agarwal2025_K562_locon_ema/
├── config.json          # the resolved TrainConfig
├── run.json             # construct, input length, input TSV, held-out chromosomes
├── history.json
├── stage1/best.pt
└── stage2/
    ├── best.pt
    └── evaluation/      # <split>_metrics.json, <split>_predictions.csv, <split>_y_vs_y_pred.png
```

`scripts/summarize_runs.py RUNS_DIR` averages scored `<name>-s<seed>` runs over seeds into one table.

Two dataset readers ship with the package, `LentiMPRAAgarwal2025Dataset` and `DeepSTARRDeAlmeida2022Dataset`, each parsing one published table. Any other data source subclasses `MPRADataset` or uses it directly — it takes inserts and targets in memory and owns the construct and the augmentation:

```python
from alphagenome_encoder_ft import MPRADataset

ds = MPRADataset(inserts, targets, construct=construct, reverse_complement=True, random_shift=True)
```

Reverse complement applies to the whole assembled sequence, matching a double-stranded plasmid; `random_shift` draws a per-item window offset and needs a construct with `length`.

- lentiMPRA input TSVs: https://github.com/autosome-ru/human_legnet

### Save and load a checkpoint

A checkpoint carries the head configuration, the input length and the construct, so loading needs nothing else:

```python
model.save_checkpoint("best.pt", save_mode="minimal")

model = AlphaGenomeEncoderModel.from_checkpoint("best.pt", device="cuda")
model.construct     # the Construct the model was trained with
model.input_length  # 281
```

`save_mode="minimal"` stores the encoder and head weights, `"full"` the whole model, `"head"` the head alone (which cannot be loaded standalone). The loaded model comes back in eval mode with the encoder frozen.

Pass `config=` to record the `TrainConfig` a run used; the training driver does, and nothing in loading reads it.

### Predict

| method | input | gradients |
|---|---|---|
| `predict_inserts(inserts)` | `Sequence[str]` | no, runs under `no_grad` |
| `forward_inserts(onehot)` | `Tensor (B, L, 4)`, insert only | yes |
| `forward(onehot)` | `Tensor (B, L, 4)`, already assembled | yes |

The insert is all that is passed; the construct adds the flanks:

```python
y_pred = model.predict_inserts([ref_insert, alt_insert])
y_effect = y_pred[1] - y_pred[0]
```

For attribution, `forward_inserts` attaches the flanks inside the graph, so gradients come back aligned to the insert:

```python
x = insert_onehot.unsqueeze(0).requires_grad_(True)   # (1, L, 4), insert only
model.forward_inserts(x).sum().backward()
saliency = x.grad                                      # (1, L, 4), aligned to the insert
```

## Construct

In an MPRA the assayed molecule is a reporter construct, and only the insert changes between rows; the adapters, minimal promoter, barcode and vector backbone are fixed for a library. `Construct` holds that fixed context so the same rule builds the model input during training and at inference, and it is saved into the checkpoint.

`Construct` is optional everywhere. `None` means the sequences are already model inputs.

```python
from alphagenome_encoder_ft import Construct

construct = Construct(prefix=LEFT_ADAPTER, suffix=RIGHT_ADAPTER + DOWNSTREAM, length=281, window_start=0)
construct.assemble_sequence(insert)          # -> str, the model input
construct.assemble_sequences([a, b, c])      # -> list[str]
construct.assemble_onehot(insert_onehot)     # -> Tensor, differentiable
```

| field | meaning |
|---|---|
| `prefix` | fixed sequence before the insert |
| `suffix` | fixed sequence after the insert |
| `length` | optional fixed model input length |
| `window_start` | optional index into `prefix + insert + suffix` where the window begins; default centred |

With `length` set, the assembled sequence is windowed to exactly that many bases. Longer sequences are trimmed from both ends; shorter ones are padded with `N` (an all-zero one-hot) on both ends. When the amount is odd, the extra base goes on the suffix side:

```
prefix + insert + suffix = 13 bp, length=10  ->  trim 1 left, 2 right
prefix + insert + suffix = 10 bp, length=13  ->  pad  1 left, 2 right
```

`offset` slides that window and is the training-time shift augmentation. The dataset draws it per item; inference leaves it at 0, so predictions use the exact centered layout the model was trained on. Because a window rather than a roll does the shifting, sequence never wraps from one end to the other.

With a centred window, a shift past the ends of the flanks brings in `N`. When more of the real reporter is known than the window shows, pass it all as flanks and set `window_start` to fix where the window begins. At offset 0 the window then covers `window_start` to `window_start + length`, and a shift slides over the real flank. `N` appears only where the window runs past the flanks:

```python
# 300 bp flanks; the window keeps 29 bp of upstream vector before a 200 bp insert
Construct(prefix=UPSTREAM, suffix=DOWNSTREAM, length=384, window_start=300 - 29)
```

`window_start` must lie within the prefix. The window must also contain the whole insert at every offset used; `MPRADataset` checks the longest insert against `max_shift` at load time. The left edge stays fixed, so a shorter insert keeps the same upstream bases and takes more of the suffix. A construct without `window_start` behaves and serializes exactly as before.

### Presets

Two published libraries ship as classes that hold their reporter pieces and build the matching construct:

```python
from alphagenome_encoder_ft import LentiMPRAAgarwal2025Library, DeepSTARRDeAlmeida2022Library

# left adapter + insert + right adapter + minP and spacer -> 281 bp; the flank continues to the barcode
LentiMPRAAgarwal2025Library.construct()

# up adapter + insert + down adapter, windowed to 256 bp
DeepSTARRDeAlmeida2022Library.construct()
```

The pieces (`LEFT_ADAPTER`, `DOWNSTREAM`, …) are attributes of those classes, so a layout of your own can reuse them.

The lentiMPRA `DOWNSTREAM` is read from the library's cloning primers (Supplementary Table S13), on the reporter strand: the minimal promoter in sense orientation, then the spacer, the barcode readout primer site and the barcode (`N`). Checkpoints trained before this was corrected, and checkpoints converted from the first JAX release, carry the construct they were trained with, because the construct is saved into the checkpoint. That construct had the reverse complement of the primer's tail after the right adapter (`TCCATTATATACCCTCTAGTGTCGGTTCACGCAATG` + a fixed barcode). It still loads and scores as trained.

## Repository layout

```text
src/alphagenome_encoder_ft/
├── constructs.py # Construct + library presets
├── data.py       # MPRADataset base + per-assay readers
├── heads.py      # MPRAHead, DeepSTARRHead
├── model.py      # AlphaGenomeEncoderModel (backbone + head + construct)
├── train.py      # epoch loop, evaluation, checkpointing, two-stage schedule
├── metrics.py    # Pearson, Spearman, the evaluation summary
├── config.py     # TrainConfig and friends
└── cli.py        # argparse/config/run-directory scaffolding for the scripts
configs/          # training hyperparameters, <dataset>_<stage-2 setting>.json
docs/             # LoCon_EMA.md (stage-2 options), low_rank_head.md (first_layer_rank)
scripts/
├── agarwal2025/  # lentiMPRA: coordinates, chromosome-split reader, train, evaluate, reproduce
├── gosai2024/    # episomal MPRA: prepare, construct, reader, train, evaluate, reproduce
├── dealmeida2022/  # DeepSTARR: train, evaluate
├── summarize_runs.py         # seed-averaged table of scored runs
└── convert_checkpoint_v0.py  # pre-Construct checkpoints -> the current format
```

`cli.py` holds the scaffolding — argparse, config files, run-directory layout, wandb — so a new assay's script is short, and it is installed so scripts outside this repository can use it too. Nothing in it is needed to use the package as a library: `train.py` takes a model and data loaders you built and knows nothing about config files or run directories.
