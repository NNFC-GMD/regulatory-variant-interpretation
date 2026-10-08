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

## Downstream: attribution, and ChromBPNet's own steps after it

After training, every fold goes through the steps igvf_tf_chrombpnet runs
after ChromBPNet's: DeepLIFT/SHAP on the counts head over every peak, the
fold average (06.0), the bigwig (07.0), TF-MoDISco (08.0), the cross-dataset
compendium (09.0), Fi-NeMo (10.0) and its report (11.0). The attributions come
from `attribute_fast.py` (`cherimoya attribute` computed by `cheri_dls`, each
run audited against the stock `deep_lift_shap` on 8 peaks); 06.0-11.0 are
igvf_tf_chrombpnet's own step files, run on the attributions converted to
its 05.0 layout.

```bash
# once: each fold's attribute.json, each dataset's igvf config, the compendium lists
python workflows/cherimoya/make_downstream.py \
    --cheri-root <out-root> \
    --chrombpnet-configs <chrombpnet configs>/*/config.yaml \
    --shared <datasets compared with ChromBPNet, one per line> \
    --compendia <compendia root>

# sign the engine off once (after a full fold is checked): the sha256 of its three files
python workflows/cherimoya/attr_status.py --write-pin workflows/cherimoya --root <out-root>

# attribute every trained fold; repeat passes until every fold is done or failed
CHERI_ROOT=<out-root> CHERI_STAGES=attribute CHERI_SLOTS_PER_GPU=1 CHERI_CORES_PER_SLOT=4 \
    srun --jobid <job> --overlap bash workflows/cherimoya/run_box.sh
python workflows/cherimoya/attr_status.py --root <out-root> --summary [--settled]
```

What `make_downstream.py` writes, per fold, in `<out-root>/<dataset>/fold_<k>/attribute.json`:

- the fold's model (`cherimoya.fold_<k>.torch`) and its peaks file (fit.json's
  `loci`, the narrowPeak ChromBPNet's 05.0 scores too);
- every chromosome of the fold's training, validation and test lists
  (Cherimoya's default is training and validation only), and every peak must
  lie on one, so the idx mask has one entry per narrowPeak row;
- a window of `attr_window` = 1000 bp around the summit, the narrowest that
  holds the bases 08.0 (`modisco -w 500`) and 10.0 (`--region-width 1000`)
  read;
- 20 shuffles (checked against ChromBPNet's `interpret.args.json`), the counts
  head, `random_state` 1234 + 100 x fold: tangermeme seeds reference k of
  every peak with `random_state + k`, so folds closer than 20 apart would share
  references and the fold average would not average reference noise out;
- `batch_size` 64, which only sets the engine's head rows (2 x 64, as
  validated); the audit's stock run is capped at 20. A stock
  `cherimoya attribute` rerun of the JSON must set `batch_size` 20: stock at 64
  needs about 15.5 GB of GPU memory;
- the outputs `attr.ohe.npz`, `attr.attr.npz` and `attr.idx.npy` in the fold
  directory, plus attribute_fast.py's `attr.attr.deltas.npy` and
  `attr.attr.meta.json`;
- the `fast` block: reference precision, `mem_budget_gb` 7.5 (about 7 GB of GPU
  memory), 3 reference workers (4 cores per process), audit 8.

It also writes `<out-root>/configs/<dataset>/config.yaml`, the minimal igvf
config for 06.0-11.0 (`output_dir` `<out-root>/<dataset>/downstream`, counts
only, no bias suffix), and `<compendia>/inputs/`: the shared and other dataset
lists and the `label<TAB>config.yaml` lists of each compendium (`cbp_` and
`cheri_` labels).

`run_task.sh <fold> attribute` (not part of `all`) waits (exit 3) until the
fit is done, skips a fold `attr_status.py` calls DONE, refuses an engine whose
files do not hash to the pin, clears the fold's earlier attribution files and
runs `attribute_fast.py`. A failed audit exits 1 and keeps the outputs; later
runs leave that fold alone unless `CHERI_ATTR_RERUN_FAILED=1`. From a worktree
without installed environments, set `CHERI_PIXI_MANIFEST` to the main
checkout's `pixi.toml`; `CHERI_ATTRIBUTE_FAST` names the engine to run and
`CHERI_ATTR_ARGS` adds flags (e.g. `--seqs-per-step 4` for a second process on
a GPU).

`attr_status.py` is the one definition of DONE: the four outputs and the meta,
the meta newer than the model (a refit makes the attribution stale), a passed
audit, finite outputs of shape (kept, 4, W) that agree with the idx mask, one
mask entry per peak, the parameters and seeds of `attribute.json`, and the
engine's sha256 equal to the pin `<out-root>/attribution_engine.json`. States:
done, running, waiting, stale, failed, missing. A fold attributed by the stock
command instead is accepted through `attr.attr.stock.json` (see its doc).

`to_chrombpnet_scores.py` converts a DONE fold for 06.0-11.0. Run it in the
cherimoya environment, inside a box, under the fold's lock
(`flock -n <fold>/.task.lock ...`, or pass `--lock`):

```bash
python workflows/cherimoya/to_chrombpnet_scores.py \
    --attribute-json <out-root>/<dataset>/fold_<k>/attribute.json
```

It writes into the directory 06.0 reads,
`<out-root>/<dataset>/downstream/full_models/<dataset>_<peak_type>_fold_<k>/interpretation/`
(output_dir and peak_type from the dataset's config; `--out-dir` names it
instead):

- `interpretation.counts_scores.h5` in ChromBPNet's 05.0 layout: `/raw/seq`
  int8, `/shap/seq` (the attributions) and `/projected_shap/seq` (ohe x
  attributions in float32, then cast) float16, each (N, 4, W), chunks of 16
  rows, Blosc blosclz level 9 with shuffle as 05.0 writes them;
- `interpretation.interpreted_regions.bed`: the peaks-file lines the idx mask
  keeps, byte for byte, so h5 row j is peaks-file row `flatnonzero(idx)[j]`.
  Cherimoya drops windows with a base other than A, C, G or T and ChromBPNet
  keeps them, so join the two models by peak name; ChromBPNet's column
  c + 1057 - W/2 is this file's column c;
- `interpretation.provenance.json`: the sources (paths, sizes, mtimes), the idx
  sha256, the meta's audit and code blocks, the dropped peaks, and what the
  float16 cast did (non-zero values that became 0, flagged above 1e-3;
  subnormals; max |value|).

It refuses a fold that is not DONE, an ohe that is not one-hot, attributions
that are not finite or exceed float16's 65504, and rows that differ from those
of a sibling fold already converted. Each file goes through `.partial` and a
rename, the provenance last; a rerun whose sources and outputs still match the
provenance does nothing, and `--force` converts again. It streams the npz files
in blocks: about 2 GB of RAM and 20 s per 40,000 rows at W = 1000.

Two checks of a whole dataset come with it, each writing a JSON report to
`<output_dir>/validation/` and exiting 0 only when everything passes:

- `--check-dataset <dataset> --root <out-root> [--against chrombpnet|fasta]`,
  before 06.0: every fold converted from its current attribution, one idx mask
  and one bed (byte for byte) across the folds, then the centring, base order
  and rows. Against ChromBPNet's own 05.0 files of the same fold (found through
  the fold's `compare.json`), joined by peak name: the ohe equals ChromBPNet's
  `raw/seq[:, :, 557:1557]` at W = 1000, on every row of the first fold and
  2,000 rows of the others, and the rows only ChromBPNet has are exactly those
  with a column of zeros. Against the genome, for datasets ChromBPNet did not
  interpret: on 2,000 rows of every fold, the one-hot of the window's central
  W bases is the ohe, and every dropped row's window holds a letter other than
  A, C, G or T.
- `--check-average <dataset> --root <out-root>`, after 06.0: the average has
  the first fold's rows and raw/seq, and on 1,000 rows its float16 values are
  the average recomputed from the npz files with 06.0's arithmetic, bit for bit.

`downstream.sh <list.txt> [...]` runs a box's datasets through all of it and
then through igvf_tf_chrombpnet's own 06.0, 07.0 and 08.0 step files, unchanged
(`REPO_ROOT` names the checkout; `CHERI_ROOT` and `BOX_DIR` are required too):
once `attr_status.py` calls every fold DONE, it converts the folds, runs the
dataset check, 06.0 and the average check, then 07.0 beside 08.0 in the
background (at most `CHERI_08_MAX` 08.0 runs at once on the node, taken in list
order) and goes on with the next dataset. Each step is pinned to the least busy
CPUs it may use (`pick_cores.py`), with its thread counts set and no GPU, and
leaves a marker in `<output_dir>/.done/` and a row (exit, seconds, max RSS) in
`${BOX_DIR}/cheri_downstream.tsv`. A rerun skips what is marked, moves an
output without its marker aside (`.incomplete.<time>`) and redoes that step
and the ones after it. Datasets waiting for their attribution wait while an
attribution runner holds the box; the script's header lists every knob.

`qc_modisco_folds.sh [<list.txt> ...]` adds the per-fold motif QC at
ChromBPNet's own scale (igvf_tf_chrombpnet's 04.4/04.5: 30,000 peaks, 5,000
seqlets, a 400-bp window), for the datasets ChromBPNet interpreted (default
`${BOX_DIR}/attribution_datasets.txt`). For each fold once `attr_status.py`
calls it DONE, `subset_scores.py` takes the fold's attribution on the peaks of
ChromBPNet's QC subsample of the same fold (`auxiliary/30K_subsample_peaks.bed`
of the ChromBPNet fold named by `compare.json`), in that order and joined by
peak name, and writes
`<output_dir>/full_models/<dataset>_<peak_type>_fold_<k>/auxiliary/interpret_subsample/cherimoya.counts_scores.h5`
(raw int8, shap and projected_shap float16, gzip; each row bit for bit the
converted fold's row of that peak), its bed (the subsample's lines Cherimoya
kept) and a provenance record. It checks the rows against ChromBPNet's own QC
scores of the fold: the ohe equals their `raw/seq[:, :, 557:1557]`, and the
peaks Cherimoya dropped have a column of zeros there. The igvf checkout's
`src/motif_qc.py` then runs on it exactly as 04.5 runs it (`--prefix
cherimoya`, into `.../evaluation/motif_qc_counts/`), at most `CHERI_QC_MAX`
(4) folds at once with 16 threads each under `nice 19`. A fold is done when
`cherimoya_qc.done.json` there names the current scores and their sources are
unchanged; a fold attributed again has its QC directory moved aside and is
redone. Rows go to `${BOX_DIR}/cheri_qc.tsv`; the header lists every knob.
