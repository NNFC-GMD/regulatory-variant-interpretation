# Downstream: attribution, and ChromBPNet's own steps after it

After training, every fold goes through the steps igvf_tf_chrombpnet runs
after ChromBPNet's: DeepLIFT/SHAP on the counts head over every peak, the
fold average (06.0), the bigwig (07.0), TF-MoDISco (08.0), the cross-dataset
compendium (09.0), Fi-NeMo (10.0) and its report (11.0). 06.0-11.0 are
igvf_tf_chrombpnet's own step files, run on the attributions converted to its
05.0 layout. The attributions come from one of two engines, chosen by
`make_downstream.py --engine` and run by `run_task.sh` with the same
`CHERI_ATTR_ENGINE`:

- `cherimoya` (new projects): `cherimoya attribute`, Cherimoya 0.3.1's fast
  DeepLIFT/SHAP path (this repository's engine, moved upstream). It checks
  itself against tangermeme's `deep_lift_shap` on its first sequences and falls
  back to it when they disagree; `../attribution/stock_record.py` records
  which path ran. Runs in the `cherimoya` environment.
- `rvi` (the default; the AMSC campaign): `attribute_fast.py`, computed by
  `cheri_dls` and each run audited against the stock `deep_lift_shap` on 8
  peaks. Runs in the `cherimoya-030` environment, since it refuses other
  Cherimoya and tangermeme versions, and needs the engine pin.

`../pipeline/` runs all of this for a project, with the `cherimoya` engine
(`docs/pipeline.md`).

```bash
# once: each fold's attribute.json, each dataset's igvf config, the compendium lists
python workflows/downstream/make_downstream.py \
    --cheri-root <out-root> \
    --chrombpnet-configs <chrombpnet configs>/*/config.yaml \
    --shared <datasets compared with ChromBPNet, one per line; may be empty> \
    --compendia <compendia root> \
    --engine cherimoya            # or rvi (the default): see above

# attribute every trained fold; repeat passes until every fold is done or failed
CHERI_ROOT=<out-root> CHERI_STAGES=attribute CHERI_ATTR_ENGINE=cherimoya \
    CHERI_SLOTS_PER_GPU=4 CHERI_CORES_PER_SLOT=4 \
    srun --jobid <job> --overlap bash workflows/train/run_box.sh

# with the rvi engine instead: sign it off once (after a full fold is checked), then CHERI_ATTR_ENGINE=rvi
python workflows/attribution/attr_status.py --write-pin workflows/attribution --root <out-root>
python workflows/attribution/attr_status.py --root <out-root> --summary [--settled]
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
  directory, plus, with the rvi engine, attribute_fast.py's
  `attr.attr.deltas.npy` and `attr.attr.meta.json`, and with the cherimoya
  engine, `stock_record.py`'s `attr.attr.stock.json`;
- `fast`: with `--engine cherimoya`, `true` (0.3.1 refuses anything but a
  bool); with `--engine rvi`, attribute_fast.py's block: reference precision,
  `mem_budget_gb` 7.5 (about 7 GB of GPU memory), 3 reference workers (4 cores
  per process), audit 8. `output` stays `counts` for both: 0.3.1 reads it as
  `logcounts` with a FutureWarning, and the converted files are named by it.

It also writes `<out-root>/configs/<dataset>/config.yaml`, the minimal igvf
config for 06.0-11.0 (`output_dir` `<out-root>/<dataset>/downstream`, counts
only, no bias suffix), and `<compendia>/inputs/`: the shared and other dataset
lists and the `label<TAB>config.yaml` lists of each compendium (`cbp_` and
`cheri_` labels).

`run_task.sh <fold> attribute` (not part of `all`) waits (exit 3) until the
fit is done, skips a fold `attr_status.py` calls DONE, clears the fold's
earlier attribution files, and runs the engine `CHERI_ATTR_ENGINE` names:

- `cherimoya`: `cherimoya attribute -p attribute.json`, its output kept in
  the fold's `attribute.log`, then `stock_record.py --fold . --log
  attribute.log --exit <status>`. That writes `attr.attr.stock.json`: the
  attribute.json as written, the package versions, and `fast_path` (false when
  the log has Cherimoya's fallback warning). It refuses non-finite
  attributions. No pin is needed.
- `rvi`: it refuses an engine whose files do not hash to the pin, then runs
  `attribute_fast.py`. A failed audit exits 1 and keeps the outputs; later runs
  leave that fold alone unless `CHERI_ATTR_RERUN_FAILED=1`.
  `CHERI_ATTRIBUTE_FAST` names the engine to run and `CHERI_ATTR_ARGS` adds
  flags (e.g. `--seqs-per-step 4` for a second process on a GPU).

The attribute stage runs in `CHERI_ATTR_PIXI_ENV` (default `cherimoya-030`
for rvi, `CHERI_PIXI_ENV`, itself `cherimoya` by default, for cherimoya). From
a worktree without installed environments, set `CHERI_PIXI_MANIFEST` to the
main checkout's `pixi.toml`.

`attr_status.py` is the one definition of DONE: the four outputs and the meta,
the meta newer than the model (a refit makes the attribution stale), a passed
audit, finite outputs of shape (kept, 4, W) that agree with the idx mask, one
mask entry per peak, the parameters and seeds of `attribute.json`, and the
engine's sha256 equal to the pin `<out-root>/attribution_engine.json`. States:
done, running, waiting, stale, failed, missing. A fold attributed by the stock
command instead (the cherimoya engine, or a stock rerun after a failed audit)
is accepted through `attr.attr.stock.json`, which `stock_record.py` writes (see
both docs).

`to_chrombpnet_scores.py` converts a DONE fold for 06.0-11.0. Run it in the
cherimoya environment, inside a box, under the fold's lock
(`flock -n <fold>/.task.lock ...`, or pass `--lock`):

```bash
python workflows/downstream/to_chrombpnet_scores.py \
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

The h5 is a copy of the fold's npz files, needed only while 06.0 and the
checks run: a finished project's prune deletes it and keeps the bed and the
provenance (docs/results.md, "What prune removes"). To redo 06.0 later,
convert the folds again first; the provenance no longer matches a missing h5,
so the converter writes it again. downstream.sh would skip the steps whose
`.done/` markers are there.

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

`bias_control.sh <chrombpnet config.yaml> <fold> <out_dir>` is the bias
control: ChromBPNet attributes its bias-free model, Cherimoya has no bias
model, so Tn5 patterns can appear only on Cherimoya's side. It attributes
ChromBPNet's bias-included `models/chrombpnet.h5` of that fold on the fold's
QC subsample with 05.0's `chrombpnet contribs_bw` (counts head, `--shap-seed
1234`) on a GPU claimed as `full_chain.sh` claims one (a claim file under
`/dev/shm/$USER/box.<job>/gpu_claims/`, JAX capped at 0.09 of the GPU), then
runs `motif_qc.py` on the scores as 04.5 does (`--prefix chrombpnet`). The
chrombpnet DeepSHAP has no rule for the logsumexp joining a bias-included
model's two count heads and refuses that head before reading any peak; the
script records the refusal in `<out_dir>/status.json`, skips MoDISco and exits
3. `CHERI_BIAS_HEAD=profile` runs the profile head, which it accepts.
