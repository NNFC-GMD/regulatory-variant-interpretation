# Cherimoya, trained beside ChromBPNet

[Cherimoya](https://github.com/jmschrei/cherimoya) (Jacob Schreiber; pinned at
0.3.0 in `pixi.toml`) is a compact sequence-to-function model that builds on
ChromBPNet. It predicts a base-resolution profile and a total count from 2,114 bp
of sequence, like ChromBPNet, but it has no separate bias model: it learns the
observed signal, Tn5 preference included.

This workflow trains it on exactly what the
[igvf_tf_chrombpnet](https://github.com/IGVF/igvf_tf_chrombpnet) pipeline
trains ChromBPNet on, with the same five chromosome folds, and scores both
models on the same test peaks with the same code. The aim is a fold-by-fold
comparison of the two methods.

## What a model sees

`make_jobs.py` reads each dataset's igvf_tf_chrombpnet `config.yaml` and
writes one Cherimoya `fit.json` per fold:

| Cherimoya input | taken from | note |
|---|---|---|
| `signals` | 00.0's `preprocessing/signal/data_unstranded.bw` | already corrected to +4/-4; a bigwig input is never shifted again |
| `loci` | 00.1's `<dataset>_<peak_type>_peaks_no_blacklist.narrowPeak` | 1,000-bp summit windows; the midpoint is the summit, so `summits` stays false |
| `negatives` | 01.0's `output_<peak_type>_fold_<k>_negatives.bed` | GC-matched, 2,114 bp, centred |
| `training/validation/test_chroms` | igvf_tf_chrombpnet's `folds/fold_<k>.json` (`train`/`valid`/`test`) | fold 0 equals Cherimoya's own default split |
| `exclusion_lists` | none | the peaks and negatives are already blacklist-filtered |
| `random_state` | 1234 + fold | the default 0 would start all five folds identically |

Everything else is Cherimoya's default (`cherimoya_cli/defaults.py`): 9 layers,
128 filters, batch 64, float32, at least 20 epochs and 20,000 steps, no early
stopping, `torch.compile` with max-autotune. `--set KEY=VALUE` overrides a fit
parameter, but not one that would change the data being compared. Cherimoya's
own ENCODE atlas (CATv1) trained in bfloat16 (`--set dtype=bfloat16`), which is
the obvious knob to try if training time matters.

## How the two are compared

The tools report different numbers on different region sets. ChromBPNet's
`chrombpnet_metrics.json` gives the median JSD over its filtered test peaks, and
Cherimoya's `*.test.performance.tsv` gives the mean JSD over all test-chromosome
peaks after its own outlier filter. `compare_chrombpnet.py` scores both models
with one definition on one set of regions:

- the regions are ChromBPNet's test peaks for the fold, from its
  `evaluation/chrombpnet_predictions.h5`;
- observed signal is the shared bigwig over each region's central 1,000 bp;
- the measures are ChromBPNet's own (`chrombpnet/training/metrics.py`):
  - counts: Pearson, Spearman and MSE of log(1 + counts);
  - profile: median Jensen-Shannon distance, and the median of its min-max
    normalised form.

It first recomputes ChromBPNet's numbers from its predictions and stops if they
differ from `chrombpnet_metrics.json` by more than 1e-6. A mismatch would mean
the regions or the signal are not what ChromBPNet was scored on.
`tests/test_compare_chrombpnet.py` checks the vectorised measures against a
transcription of ChromBPNet's code.

`summarize.py` pools every fold into three files:

- `comparison.tsv`: both models' measures, with Cherimoya's training time and
  memory;
- `comparison_paired.tsv`: per fold, both models and the difference;
- `comparison_summary.txt`: medians, median differences, and fold-by-fold
  win counts.

## Running it

```bash
pixi install -e cherimoya          # once, from this repo (login node is fine)

python workflows/cherimoya/make_jobs.py \
    --chrombpnet-configs <chrombpnet configs>/*/config.yaml \
    --folds-dir <igvf_tf_chrombpnet>/folds \
    --genome <the genome FASTA ChromBPNet used> \
    --out-root <out-root>

# a whole node: 8 GPUs x 2 tasks each (DCAI: SBATCH_ACCOUNT set; one box at a time)
CHERI_ROOT=<out-root> sbatch [--dependency=afterany:<running box>] workflows/cherimoya/run_box.sbatch

# or inside an allocation that is already running other work, on the GPUs and cores given
CHERI_ROOT=<out-root> CHERI_GPUS=7 CHERI_SLOTS_PER_GPU=1 CHERI_CORE_OFFSET=200 \
    srun --jobid <job> --overlap bash workflows/cherimoya/run_box.sh

# once ChromBPNet's models exist for folds whose comparison was waiting
CHERI_ROOT=<out-root> CHERI_STAGES=compare ... run_box.sh

pixi run -e cherimoya python workflows/cherimoya/summarize.py --root <out-root> --out-dir <dir>
pixi run -e cherimoya python workflows/cherimoya/plot_comparison.py \
    --paired <dir>/comparison_paired.tsv --out <dir>/comparison_counts
```

`plot_comparison.py` draws one scatter panel per measure, ChromBPNet on x and
Cherimoya on y. The default measures are counts Pearson and Spearman. There is
one dot per dataset and fold, so a fully scored dataset shows five dots. Each
dataset gets an Okabe-Ito colour plus a marker shape that changes every eight
datasets.

Each fold's outputs land in `<out-root>/<dataset>/fold_<k>/`:

- `cherimoya.fold_<k>.torch`: the best-validation EMA checkpoint;
- `.final.torch`, `.log`, `.detailed.log`;
- `.validation.performance.tsv` and `.test.performance.tsv`: Cherimoya's own
  measures;
- `compare.tsv` and `compare_counts.npz`;
- `fit.time` and `compare.time`: wall seconds and peak RSS.

The box log, `status.tsv` and `resources.tsv` go to `<out-root>/box/<job>/`.
`run_task.sh` skips a finished stage, so resubmitting resumes. A comparison
whose ChromBPNet fold is not trained yet exits 3 ("waiting") instead of
failing.

## Downstream (not run yet)

Cherimoya's own `pipeline` stops after TF-MoDISco and motif marginalisation,
with no Fi-NeMo. Its outputs are compatible with the tools the ChromBPNet
pipeline runs after training, though:

- `cherimoya attribute` writes the one-hot and hypothetical-contribution
  `.npz` pair that TF-MoDISco reads;
- Fi-NeMo accepts regions in that same format.

So the downstream can match igvf_tf_chrombpnet step for step:

1. DeepLIFT/SHAP on the counts head over all peaks, per fold;
2. the fold average;
3. TF-MoDISco (the NNFC-GMD/tfmodisco fork 08.0 uses);
4. Fi-NeMo.

None of it is wired up here yet.
