# Using a project's results

What a finished Mímir project leaves under its root, what each file holds, and how to read it. For people who
want to use the models, attributions, motifs, hits and cooperativity tables, on Gefion or from a copy elsewhere.
`docs/pipeline.md` explains how each file is made; each project's own `README.md` says what is particular to it
(its datasets, its compendium, known issues).

`<root>` is the project directory, `<ds>` a dataset (the project's prefix + a sample stem), `<k>` a fold (0–4) and
`<tag>` the project's compendium.

## Where to start

| you want | read |
|---|---|
| how good each dataset's models are | `<root>/STATUS.md`; per fold `cherimoya/<ds>/fold_<k>/cherimoya.fold_<k>.test.performance.tsv` |
| predictions for new sequences or variants | the models, below |
| which bases drive accessibility, on a genome browser | `cherimoya/<ds>/downstream/contrib_scores/<ds>/<ds>_average_shaps.counts.bw` |
| the same, as arrays | the fold-averaged attribution h5 beside it |
| the motifs a dataset's model learned | TF-MoDISco: `contrib_scores/<ds>/modisco/` |
| the project's motifs, named | the compendium: `compendia/<tag>/modisco_compendium_meta.tsv` |
| where those motifs sit, peak by peak | Fi-NeMo hits: `cherimoya/<ds>/downstream/finemo_<tag>/<ds>_all/hits_unique.tsv` |
| which TF pairs cooperate, and how that changes | DeepISA: `cherimoya/cooperativity/combined/` and `official/` |

## Getting the files

- **On Gefion:** read them in place. Run Python in this repository's pixi environments (`docs/cluster.md`): `pixi
  run -e cherimoya python` has Cherimoya, torch, numpy, pandas, h5py and hdf5plugin. Use a box for anything that
  reads a whole dataset; the login node is too small.
- **Elsewhere:** someone with access packs a tar for you:
  ```bash
  pixi run -e default python workflows/pipeline/project.py <root>/project.yaml share --out <file.tar> [--datasets <ds>,<ds>]
  ```
  It holds every dataset's models, fold-averaged attributions and bigWig, TF-MoDISco, Fi-NeMo hits and reports,
  peaks, signal QC, the DeepISA tables, the compendium, the project's README and STATUS, and this page
  (`docs/results.md`). `MANIFEST.md5` lists every file's md5: check it with `md5sum -c <project>/MANIFEST.md5` from
  the directory you unpacked it in. `SHARE.txt` lists anything that was missing when it was packed.
  `--with-signal` adds the observed signal bigWigs. Ask before sending a project's files outside the group: its
  data may be unpublished (its README says).

## Peaks and the row index

- `peaks/<ds>/<ds>.narrowPeak.gz`: the called peaks (MACS3 on pseudoreplicates, docs/pipeline.md's recipe).
- `chrombpnet/<ds>/preprocessing/peaks/<ds>_all_peaks_no_blacklist.narrowPeak`: the same peaks as 1,000-bp windows
  centred on each summit, blacklist removed. This file is the **row index of everything downstream**: the models
  were trained on it, and row *j* of an attribution is a row of this file (see below).
- `chrombpnet/<ds>/preprocessing/signal/data_unstranded.bw`: the observed signal, Tn5 cut sites shifted +4/−4, per
  base. `<root>/bigwigs/` links to it.

## The models

One Cherimoya model per dataset and fold: `cherimoya/<ds>/fold_<k>/cherimoya.fold_<k>.torch` (2.5 MB).
- **Folds:** ChromBPNet's five chromosome folds (`<igvf>/folds/fold_<k>.json`). Fold 0, for example, is tested on
  chr1, chr3 and chr6, validated on chr8 and chr20, and trained on the rest. Each fold's `fit.json` lists its exact
  inputs and settings.
- **The checkpoint** is the best-validation (EMA) one. `cherimoya.fold_<k>.final.torch` is the last epoch's; it is
  not used.
- **Input:** 2,114 bp of one-hot sequence (A, C, G, T; channels first). **Output:** a 1,000-bp profile and the log
  of the total counts over those 1,000 bp. One unstranded signal track, no control track.
- **Loading** (Cherimoya 0.3.0 and 0.3.1 load every model; the `cherimoya` environment has 0.3.1):
  ```python
  import cherimoya                                  # also switches on TF32 matmuls
  from cherimoya import Cherimoya
  from cherimoya.wrappers import LogCountWrapper

  model = Cherimoya.load(path, device="cuda", compile=False).eval()
  counts = LogCountWrapper(model)                   # log(1 + predicted counts), the count head
  ```
  `workflows/variant_scoring/predict_cherimoya.py` (`load_model`, `CountPredictor`) does this for batches of
  sequences on both strands.
- **How we use them:** predict both strands and average the counts, log((e^fwd + e^rev) / 2); then average the 5
  folds. For a peak, the **held-out** model is the fold whose test chromosomes hold it: the one model that never
  saw it. The fold mean includes 4 models that trained on (or validated on) the peak.

## Attributions

DeepLIFT/SHAP on the count head, with 20 dinucleotide-shuffled references per peak, for every peak with every
fold's model. Only the central 1,000 bp (summit ± 500) are attributed.

**Per fold, the canonical copy:** `cherimoya/<ds>/fold_<k>/`
- `attr.attr.npz`: `arr_0`, float32, shape (N, 4, 1000): **hypothetical** scores (the contribution each base would
  have at that position).
- `attr.ohe.npz`: `arr_0`, int8, (N, 4, 1000): the one-hot sequence. The *actual* (projected) scores are `ohe ×
  attr`.
- `attr.idx.npy`: bool, one entry per row of `<ds>_all_peaks_no_blacklist.narrowPeak`, True for the N rows
  attributed. Row *j* of the npz is peaks row `numpy.flatnonzero(idx)[j]`. A window with a base other than A, C, G
  or T is left out.
- `attr.attr.stock.json` (or `.meta.json` and `.deltas.npy` for the older engine): the run's parameters, versions
  and convergence check.

```python
import numpy as np
attr = np.load("attr.attr.npz")["arr_0"]          # (N, 4, 1000) float32, ~4 GB for 290k peaks
ohe = np.load("attr.ohe.npz")["arr_0"]
idx = np.load("attr.idx.npy")
actual = (attr * ohe).sum(axis=1)                 # (N, 1000): each base's contribution
```

**Averaged over the 5 folds:** `cherimoya/<ds>/downstream/contrib_scores/<ds>/<ds>_average_shaps.counts.h5`
(igvf_tf_chrombpnet's 06.0), in ChromBPNet's layout:
- `raw/seq` int8, `shap/seq` float16 (hypothetical) and `projected_shap/seq` float16 (actual), each (N, 4, 1000);
- rows in the order of fold 0's `.../full_models/<ds>_all_fold_0/interpretation/interpretation.interpreted_regions.bed`
  (the attributed peaks rows, in file order);
- gzip-compressed, so plain `h5py` reads it. (The per-fold 05.0 files, when they are there, use Blosc: `import
  hdf5plugin` first.)

This is the file TF-MoDISco and Fi-NeMo read; use it unless you need one fold. Its bigWig
(`..._average_shaps.counts.bw`, 07.0) holds the projected scores summed per base, inside the peak windows only.

## TF-MoDISco and the compendium

- `contrib_scores/<ds>/modisco/modisco_counts_results.h5`: TF-MoDISco (08.0) on the fold-averaged counts scores,
  window 500 bp around each summit. `counts_report/motifs.html` shows each pattern, its seqlet count and its best
  database matches.
- `compendia/<tag>/`: every dataset's patterns clustered into one non-redundant set (MotifCompendium, 09.0).
  `modisco_compendium_meta.tsv` has one row per input pattern with its `cluster_id` and annotations;
  `modisco_compendium.meme` the motifs. Cluster *n* is the motif called `pos_patterns.<n>` (or `neg_patterns.<n>`)
  in the hits. `cherimoya/cooperativity/motifs.tsv` gives each one a single name (`cluster_annotation`: its most
  frequent annotation above the compendium's minimum score; an unannotated cluster keeps its motif name).
- One compendium per project, so `pos_patterns.12` means the same motif in every dataset. A compendium is never
  rebuilt in place, and the hits record its md5 (`compendium.md5`).

## Fi-NeMo hits

`cherimoya/<ds>/downstream/finemo_<tag>/<ds>_all/` (10.0 and 11.0), on the fold-averaged scores, against the
project's compendium:
- `hits_unique.tsv`: one row per motif instance. **Use this one.** Columns: `chr`, `start`, `end` (the trimmed
  motif), `start_untrimmed`, `end_untrimmed`, `motif_name` (`pos_patterns.<n>`), `hit_coefficient` (the fitted
  strength), `hit_coefficient_global`, `hit_similarity`, `hit_correlation` (to the motif's CWM), `hit_importance`,
  `hit_importance_sq`, `strand`, `peak_name`, `peak_id`.
- `hits.tsv`: the same, before overlapping peaks' duplicates are removed (a site in two overlapping windows appears
  twice). `hits.bed.gz` (+ `.tbi`) is for genome browsers.
- `peaks_qc.tsv`: per peak, how well the hits reconstruct the attribution (`nll`, `dual_gap`, `global_scale`).
- `finemo_report/`: 11.0's report: `report.html`, `motif_report.tsv` (one row per motif), `motif_occurrences.tsv`
  (per peak, the count of each motif), the CWMs and the distributions as figures.

## Cooperativity (DeepISA)

In-silico ablation of motif sites with the 5 fold models (`docs/pipeline.md`, Cooperativity;
`workflows/cooperativity/README.md` has the definitions). Everything is in log2 counts.
- `cherimoya/cooperativity/<ds>/`: `sites.tsv.gz` (each ablated site: `motif`, `tf`, `e` its effect, `a` the
  intact prediction) and `pairs.tsv.gz` (each site pair: `distance`, `gap`, the single effects `e_a`, `e_b`, the
  interaction `i` = a − a₋A − a₋B + a₋AB, and `i_f0`..`i_f4` under each fold's model alone).
- `combined/` (our definitions): `pairs_by_dataset.tsv`, `tfs_by_dataset.tsv` (synergy = Σ(i > 0.1) / Σ(|i| > 0.1),
  with a bootstrap interval and a label); `changes_pairs.tsv`, `changes_tfs.tsv` per comparison: both groups' values,
  `delta` = b − a with `delta_lo`/`delta_hi`, `p`, `q` (Benjamini–Hochberg within the comparison), `significant` (q
  < 0.05), `ci_excludes_0` (the older, uncorrected criterion) and `robust` (significant and the same sign under all
  5 folds' models); with a series, `trend_pairs.tsv`, `trend_tfs.tsv` (slope along the series, its q, a
  heterogeneity test); `distance.tsv`; `summary.json`.
- `official/`: the same comparisons by the deepISA package's own rules (its `calc_interaction` and
  `calc_coop_score`), from the same ablations.
- `matrix/` (when the project ran coop_matrix): every dataset's sites under every dataset's models;
  `matrix/combined/` the paired changes on the windows open under both models (`open.tsv`).

A model that predicts smaller effects has fewer pairs above |i| > 0.1: read a synergy beside the model's accuracy
(STATUS.md has both).

## Variant scores (projects that scored variants)

`variant_scoring/<run>/scores/<ds>/<model>.variants.summary.tsv.gz` (and `.npz`, the same columns): one row per
variant, over the 5 folds: `logfc.mean` (log2 of alt / ref predicted counts, both strands, fold mean),
`logfc.heldout` (the held-out fold's), `abs_logfc.*`, the active-allele quantile and its change, and each one's
empirical p against the run's shuffled-window null (`*.pval`). `workflows/variant_scoring/scores.py` defines every
column.

## What prune removes

`project.py <project.yaml> prune` deletes, from a manifest a person has read first, the files a later step derived
and nothing reads again (`workflows/pipeline/prune.py` has every rule; each project's `prune/` folder has its
manifests and what was deleted). How to get each kind back:

| removed | what it was | to get it back |
|---|---|---|
| `downstream/full_models/<ds>_all_fold_<k>/interpretation/interpretation.counts_scores.h5` | each fold's attribution as float16, in ChromBPNet's 05.0 layout: a copy of the fold's `attr.*` files | `workflows/downstream/to_chrombpnet_scores.py --attribute-json <fold>/attribute.json --config <root>/cherimoya/configs/<ds>/config.yaml`, about 2 minutes per fold; it needs h5py, so in a box |
| `finemo_<tag>/<ds>_all/intermediate_inputs.npz` | Fi-NeMo's input arrays: the fold-averaged scores and the sequence, again | 10.0 writes it again when rerun; nothing else reads it |
| `finemo_<tag>/<ds>_all/hits.bed` | the uncompressed copy of `hits.bed.gz` | `zcat hits.bed.gz` |
| `peaks/<ds>/reps/` | MACS3's per-pseudoreplicate calls | rerun 00.0 |
| `chrombpnet/<ds>/preprocessing/<ds>/output_all_fold_<k>_auxiliary/` | 01.0's genome-wide GC and candidate regions | rerun 01.0 |

Projects with trained ChromBPNet models or variant scores lose more (the `chrombpnet` and `variant_scoring` rule
sets): ChromBPNet's profile-head attributions, bigWigs, genome-wide predictions and bias-model intermediates, and
the per-fold variant scores, raw predictions and windows. The models, the metrics, the files a comparison reads and
the per-dataset summaries stay. Nothing a stage's `project.py check` reads is ever removed, so a pruned project
still checks as done.

A rerun of the downstream stage needs the 05.0 files first: `downstream.sh` skips a step whose `.done/` marker is
there, so convert the folds again (above) before redoing 06.0.
