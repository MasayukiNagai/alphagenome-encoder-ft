# LoCon and EMA

Two stage-2 options, both off unless set: low-rank convolution adapters on the encoder (LoCon, `stage2.encoder_locon_rank`) and an exponential moving average of the trained weights (EMA, `ema_decay`). The results below come from fine-tuning on two K562 MPRA libraries. They can be rerun with the scripts in `scripts/agarwal2025/` and `scripts/gosai2024/` (see [Reproduce](#reproduce)).

## Data

Both libraries are K562 and hold out the same chromosomes as Malinois: validation chr19, 21 and X; test chr7 and 13; training all other chromosomes.

| Library | Train | Val | Test |
|---|---|---|---|
| lentiMPRA (Agarwal et al. 2025) | 165,714 | 16,500 | 14,450 |
| Gosai et al. 2024 episomal MPRA | 668,929 | 62,405 | 62,582 |

- lentiMPRA: the table's reverse-complement rows (`rev == 1`) are dropped; RC augmentation is applied on the fly instead. Coordinates come from the ENCODE BED of each library (`scripts/agarwal2025/0_add_coordinates.py`). The model input is the 281 bp construct (adapters, insert, minP, barcode).
- Gosai: the test split keeps only rows with lfcSE ≤ 1 in all three cell types, which reproduces Malinois's test set. Train and val are not filtered on lfcSE. The model input is a 384 bp window: 29 bp of vector, the insert, and the reporter down to the GFP start codon (`scripts/gosai2024/library.py`).

## EMA (`ema_decay`)

The EMA keeps an exponential moving average of the weights a stage trains, updated after every optimizer step: `avg = d * avg + (1 - d) * w`. Training continues on the live weights. Validation, the LR scheduler, early stopping and `best.pt` use the average. Averaging removes the step-to-step noise of the last several hundred steps, which is the job LR decay otherwise does.

- Set per stage: `stage1.ema_decay`, `stage2.ema_decay`; `null` (or `--stage2_ema_decay none`) turns it off. The runs below use 0.9995 in stage 2. EMA in stage 1 was not tested.
- The decay warms up: after `n` steps it is `min(d, (1 + n) / (10 + n))`, so `d` itself applies only after about `9 / (1 - d)` steps (18,000 for 0.9995). lentiMPRA has about 1,300 steps per epoch and reaches its best stage-2 checkpoint around epoch 6, before 0.999, 0.9995 or 0.9999 take effect, so the three settings average identically there. On Gosai (about 5,200 steps per epoch) they also scored the same.

## LoCon (`stage2.encoder_locon_rank`)

LoCon freezes the encoder and adds a trainable low-rank path to each of its 13 block convolutions (the DNA embedder's block and both blocks of the six down blocks): `y = conv(x) + up(down(x))`. `down` is a width-5 convolution to `r` channels and `up` a 1x1 convolution back. `up` starts at zero, so stage 2 starts exactly from the stage-1 model. The pair adds a rank-`r` term to each convolution's effective kernel.

- Rank 128 trains 11.3M encoder weights, 13% of the encoder's 90M (88k per unit of rank). `--stage2_encoder_locon_rank none` gives full fine-tuning.
- It saves optimizer state, not activation memory or compute: gradients still pass through the whole encoder. Checkpoints store the full encoder plus the adapters.
- The embedder's first convolution has no adapter. `stage2.encoder_unfreeze_first_conv` trains it in full; it did not help.

## Performance

Pearson r on the splits above, strand-averaged (forward and reverse-complement predictions averaged), mean of seeds 42, 43 and 44. All runs use RC augmentation and the rank-128 head, and each seed's four stage-2 runs start from the same stage-1 checkpoint. The change column is against full fine-tuning on the same checkpoints.

| Stage 2 | Config | lentiMPRA val | lentiMPRA test | Gosai val | Gosai test |
|---|---|---|---|---|---|
| Full fine-tuning | `*_fullft` | .9020 | .8765 | .9401 | .9310 |
| Full fine-tuning + EMA | `*_fullft_ema` | **.9030** (+.0010) | **.8780** (+.0015) | **.9407** (+.0006) | **.9322** (+.0012) |
| LoCon r128 | `*_locon` | .9008 (-.0012) | .8754 (-.0011) | .9394 (-.0007) | .9299 (-.0011) |
| LoCon r128 + EMA (chosen) | `*_locon_ema` | .9017 (-.0003) | .8768 (+.0003) | .9402 (+.0001) | .9307 (-.0003) |

Seed ranges are .0002 to .0013 on validation.

- **EMA** adds .0006 to .0015 and won all 21 paired seed comparisons across both libraries, including RC settings not shown here. It helps most where the model overfits most: +.004 val on lentiMPRA without RC augmentation. With EMA, the best checkpoint always came before the plateau scheduler's first LR cut.
- **LoCon** scores about .001 below full fine-tuning in every setting, and the EMA gain adds on top. Nothing closed that gap: encoder LR 5e-5 to 5e-4 (1e-4 best, higher worse, 1e-3 diverged), rank 256, also training the first convolution or the per-channel scales and biases, or NoRA. Truncating the full fine-tuning update shows it needs about rank 512 to keep its score.
- LoCon shrinks the train/val gap (Gosai .031 to .020) without improving validation.
- **Against the recipe before these options.** The lentiMPRA hyperparameter-search baseline (full-width head, full fine-tuning, no EMA, scored forward-only) reached .8979 val / .8718 test. Scored the same way, LoCon r128 + EMA reaches .8983 / .8733. Strand averaging adds another .0034 val / .0035 test.

Full fine-tuning + EMA is the most accurate. LoCon + EMA was chosen anyway: it gives up about .001 while training 13% of the encoder's weights.

## On a new dataset

- **EMA.** Count the decay in optimizer steps, not epochs. Compare `9 / (1 - d)` with the steps to the expected best checkpoint; if the run is shorter, the choice of `d` hardly matters.
- **LoCon.** The adapter path is not scaled by rank (no LoRA alpha), so the encoder LR and the rank interact. 1e-4 was tuned at rank 128; retune the LR if you change the rank.

## Reproduce

Each `reproduce_locon_ema.sh SEED` trains stage 1 once, runs the four stage-2 configs from it, and scores every stage on validation and test with strand averaging. `ALPHAGENOME_WEIGHTS` is the pretrained backbone (https://huggingface.co/gtca/alphagenome_pytorch).

lentiMPRA, from `datasets/original/K562.tsv` of https://github.com/autosome-ru/human_legnet:

```bash
python scripts/agarwal2025/0_add_coordinates.py --celltype K562 \
    --input_tsv data/K562.tsv --output data/K562_with_chrom.tsv
for seed in 42 43 44; do
  INPUT_TSV=data/K562_with_chrom.tsv ALPHAGENOME_WEIGHTS=/path/to/model.safetensors \
    bash scripts/agarwal2025/reproduce_locon_ema.sh "$seed"
done
python scripts/summarize_runs.py results/agarwal2025_K562_locon_ema_reproduce
```

Gosai, from Supplementary Table S2 of Gosai et al. 2024:

```bash
bash scripts/gosai2024/run_prepare.sh DATA-Table_S2__MPRA_dataset.txt data/gosai_K562_inserts.tsv
for seed in 42 43 44; do
  INPUT_TSV=data/gosai_K562_inserts.tsv ALPHAGENOME_WEIGHTS=/path/to/model.safetensors \
    bash scripts/gosai2024/reproduce_locon_ema.sh "$seed"
done
python scripts/summarize_runs.py results/gosai2024_K562_locon_ema_reproduce
```

On one H100, a seed takes roughly 1.5 hours on lentiMPRA and 6 hours on Gosai. The seeds are independent and can run in parallel.

What to expect:

- Stage 1 with a fixed seed is deterministic. Stage 2 is not bitwise reproducible across runs, so a rerun matches the table as a 3-seed mean, to within about the seed range.
- The Gosai runs in the table were trained on a prepared table from before `0_prepare.py` dropped Malinois's outlier rows (19 rows in all). Its validation and test row counts are the same as the current table's.
