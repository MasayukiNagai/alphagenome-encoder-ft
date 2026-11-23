# Low-rank first head layer

`head.first_layer_rank: r` factorizes the first layer of the MPRA head. With flatten pooling that layer maps every encoder bin (bins × 1536 features) to `hidden_sizes[0]`, and it holds almost all of the head's weights. The factorized layer is `W ≈ U V`, with `U` (in × r, no bias) followed by `V` (r × hidden, with bias), and nothing between them: no activation, no dropout. `null` (`--first_layer_rank none`) keeps the full layer.

On a 3-bin input (281 bp lentiMPRA, 384 bp Gosai) with `hidden_sizes [1024]`:

| First layer | Head weights |
|---|---|
| Full | 4.72M |
| Rank 256 | 1.45M |
| Rank 128 (default) | 0.73M |
| Rank 64 | 0.37M |
| Rank 32 | 0.19M |

At rank 128 a head-only checkpoint shrinks from 19 MB to 2.8 MB. Run time does not change: the encoder dominates it.

## Result

Pearson r, strand-averaged, mean of seeds 42, 43 and 44, after both stages. Stage 2 is full fine-tuning + EMA (`configs/<dataset>_fullft_ema.json`) with dropout 0.5 in both stages; the data and splits are those of `docs/LoCon_EMA.md`.

| First layer | lentiMPRA K562 val | lentiMPRA K562 test | Gosai K562 val | Gosai K562 test |
|---|---|---|---|---|
| Full | .9026 | .8775 | .9406 | .9321 |
| Rank 256 | .9030 | .8778 | .9410 | .9325 |
| Rank 128 | .9030 | .8780 | .9408 | .9318 |
| Rank 64 | .9030 | .8782 | .9408 | .9327 |
| Rank 32 | .9029 | .8780 | – | – |

Seed ranges are .0001 to .0009 on validation.

- Every rank matches the full layer after stage 2, on both libraries.
- The rank matters only in stage 1, where the encoder is frozen and the head does all the work. There, lentiMPRA holds from rank 256 to 64 (.8821, .8822, .8816; full .8803) and drops at rank 32 (.8796), while Gosai is already .0006 lower at rank 64 (.9205 against .9211 for rank 128 and the full layer). Rank 128 is the smallest rank lossless in both stages on both libraries, so it is the default.

## Dropout

lentiMPRA K562, validation, same settings with the head dropout set in both stages:

| Dropout | Full, stage 1 | Full, stage 2 | Rank 128, stage 1 | Rank 128, stage 2 |
|---|---|---|---|---|
| 0.1 | .8792 | .9010 | .8814 | .9026 |
| 0.3 | .8798 | .9019 | .8819 | .9030 |
| 0.5 | .8803 | .9026 | .8822 | .9030 |

- Rank 128 is ahead of the full layer at every dropout in stage 1 (by about .002) and is barely affected by dropout after stage 2 (.0004 across the three values, against .0016 for the full layer). Its seed ranges are also tighter (.0002 against up to .0009 after stage 2).
- The factorization regularizes the head, so the rank-128 head needs less tuning of dropout.

## What did not work

Earlier stage-1 comparisons, one seed, against a factorized layer of similar size:

- **A narrower full layer** (`hidden_sizes [64]`, 0.30M): .8717 to .8770 on lentiMPRA depending on dropout, against .8790 for rank 64.
- **A shared per-bin projection** (`head.proj_dim 64`, a linear map applied to each bin before flattening): .8775 on lentiMPRA, against .8790 for rank 64.
- **A ReLU bottleneck** (`hidden_sizes [128, 1024]`, the size of rank 128): .0015 below rank 128 on Gosai at dropout 0.1, and unstable at dropout 0.5. The activation and the dropout then act on the 128-wide layer, which the factorized layer avoids.

## Reproduce

The table and the dropout comparison are two-stage runs of the dataset scripts with `--first_layer_rank` and the dropout flags on top of the `fullft_ema` config (see `docs/LoCon_EMA.md` for preparing the data). For lentiMPRA K562:

```bash
export INPUT_TSV=data/K562_with_chrom.tsv ALPHAGENOME_WEIGHTS=/path/to/model.safetensors
for seed in 42 43 44; do for rank in none 256 128 64 32; do
  run=results/agarwal2025_K562_head/rank_$rank-s$seed
  CHECKPOINT_DIR=$run bash scripts/agarwal2025/run_train.sh agarwal2025_fullft_ema \
    --seed $seed --first_layer_rank $rank
  bash scripts/agarwal2025/run_evaluate.sh $run
done; done
python scripts/summarize_runs.py results/agarwal2025_K562_head
```

Add `--stage1_dropout D --stage2_dropout D` for the dropout table. Gosai is the same with `scripts/gosai2024/` and `gosai2024_fullft_ema`. On one H100 a lentiMPRA run takes about 35 minutes and a Gosai run about 2 hours.
