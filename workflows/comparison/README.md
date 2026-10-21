# The comparison, level by level

`compare.sh <levels> [--datasets <list.txt>] [--only <ds,...>] [--force]`
compares the two models on the datasets both were interpreted on (default
`${COMPENDIA_ROOT}/inputs/shared10.txt`), one level at a time, each level a
script in this folder run in the cherimoya environment, writing under
`${COMPARISON_ROOT}`. A unit (a dataset, or a level's summary) never waits:
one whose inputs are not there yet exits 3 and says what it lacks, so the run
goes on and a later run picks up what has appeared. Each unit keeps a done
marker of its inputs and settings (`<level dir>/.done/<unit>.json`), so a rerun
redoes only what changed. Rows go to `${BOX_DIR}/compare_steps.tsv` (level,
unit, exit, seconds, max RSS, note) and logs to `${COMPARISON_ROOT}/logs/`; the
script's header lists every knob.

| Level | Script | Needs | Writes |
|---|---|---|---|
| 1 | `compare_performance.py` | every fold's `compare.tsv` and `compare_counts.npz` | `performance/`: `summarize.py --datasets`'s tables; per measure both models' medians, the median of the dataset mean differences with a bootstrap 95 % CI over datasets, Wilcoxon over folds and over dataset means; residuals by observed-count decile on each dataset's pooled test peaks; `plot_comparison.py`'s scatter per measure, the paired differences and the residual curves |
| 2 | `compare_attributions.py` | per dataset: every fold DONE (`attr_status.py`), ChromBPNet's 05.0 folds and 06.0 average, Cherimoya's 06.0 average with downstream.sh's V2 marker | `attributions/<ds>/`: per peak (joined by name; Cherimoya's column c is ChromBPNet's c + 557, and the one-hot of every joined row of every fold must equal ChromBPNet's), Pearson and Spearman of the projected scores, cosine of the hypothetical maps, top-50 Jaccard, sums and norms, within ±250 and ±500 bp of the summit, for each fold, the averages, each model's leave-one-out ceiling and the held-out folds; `summary.tsv` by chromosome role, signal and GC decile. The summary tests cross-model agreement against both ceilings; the converted float16 folds stand in for the npz as the float16 check |
| 3 | `compare_modisco.py` | per dataset: both models' 08.0 results (Cherimoya's with downstream.sh's 08.0 marker); both models' per-fold QC runs | `modisco/<ds>/`: every pattern annotated (tomtom-lite against MotifCompendium's database) and flagged as Tn5/DNase bias (chrombpnet's motifs.meme), matched across models by joint-compendium cluster and by tomtom-lite, seqlets placed in the genome through each model's own bed and overlapped; per fold at the QC scale the same, with each model's fold-to-fold matching as the ceiling; the bias control's patterns (or any `--bias-modisco` reference) against Cherimoya's unmatched ones |
| 4 | `compare_compendia.py` | the cbp10, cheri10 and joint20 compendia (cheri39 and a 0.90 rebuild when there) | `compendium/`: reciprocal best matches of the like-for-like compendia's cluster CWMs (normalized, so the attribution scales do not count), annotation and TF-family overlap; each joint cluster's patterns, datasets and seqlets per model and its class: shared, ChromBPNet-only, Cherimoya-only |
| 5 | `compare_finemo.py` | per dataset: both models' Fi-NeMo hits on joint20 (same compendium md5) | `finemo/<ds>/`: on the peaks both beds hold, hits matched by motif, strand and centre within 5 bp (0 and 10 also), Cherimoya's precision and recall against ChromBPNet per motif, peak-set Jaccard, coefficient Spearman, per-peak motif-count correlation, 11.0's reports side by side; with `COMPARE_PRODUCTION=1`, the production hits (cbp10 against cheri39) through level 4's matches |

`tests/test_compare_*.py` run each level on synthetic inputs whose answers are
known (CPU only).
