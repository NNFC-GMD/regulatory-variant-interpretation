# TF cooperativity from Cherimoya by in-silico ablation (DeepISA)

This folder runs DeepISA's in-silico ablation (ISA) cooperativity analysis, with Cherimoya models in place of
DeepISA's own CNN:
- **Paper:** He et al. 2025, "Genome-wide rules of transcription factor cooperativity revealed through in silico
  binding site ablation", bioRxiv 10.1101/2025.06.19.660093.
- **Code:** the paper's scripts are in github.com/anderssonlab/DeepCompARE (`Scripts_python/tf_cooperativity.py`).
  The official package is github.com/anderssonlab/deepISA (deepISA 0.1.0, pinned at `179091c` in `pixi.toml`'s
  cherimoya environment). `combine.py --official` applies the package's current rules to these ablations with the
  package's own functions (section 4).

Each motif site in a peak is ablated alone and together with every other site, and each pair's interaction is the
departure from additivity. The scripts:
- `coop_isa.py` handles one dataset: it builds the sequences, predicts them with every fold model, and writes the site
  and pair tables.
- `lane.sh` runs `coop_isa.py` on one GPU for several datasets in turn.
- `combine.py` works across datasets. It computes synergy scores with intervals, the changes between cell states
  (with BH-corrected p-values and a fold check), the trends along a series of groups, and the distance profile; with
  `--official`, the official package's scores and labels too.
- `motif_names.py` names each Fi-NeMo motif (`pos_patterns.<cluster>`) from the compendium the hits were called with.
- `matrix.py` and `paired.py` score every dataset's sites under every dataset's models and compare the models on the
  same site pairs (section 9).
- `check_official.py` runs the official package's aggregate stage again on the tables `combine.py --official`
  wrote, and recomputes its rules from `coop_isa.py`'s tables (section 10).

To run it on a new project, use the pipeline (`workflows/pipeline/`, stage `cooperativity`; docs/pipeline.md): it
writes the inputs below from the project's own Fi-NeMo hits and compendium (section 8).

The rest of this page documents the one run so far, on 20 October 2026 (ET). It covers the AMSC adipocyte ATAC-seq
datasets, on Gefion. It lists every input file and says why each choice was made.

`$AMSC` stands for `/dcai/projects/iu_0109/datasets/amsc`. The run's outputs, logs and the exact script copies that
ran are in `$AMSC/cherimoya/cooperativity/`.

## 1. The models

One Cherimoya model per dataset and fold: 10 datasets × 5 folds = 50 models.

```
$AMSC/cherimoya/<dataset>/fold_<k>/cherimoya.fold_<k>.torch        k = 0..4
```

- **Checkpoint:** `cherimoya.fold_<k>.torch` holds the EMA weights from the epoch with the highest mean validation
  count Pearson: the last epoch marked `True` in the `Saved?` column of `cherimoya.fold_<k>.log`.
  - `cherimoya.fold_<k>.final.torch`, the end-of-training weights, sits beside it. It was not used.
  - Each file is 2.5 MB.
- **Version:** cherimoya 0.3.0 (jmschrei/cherimoya), torch 2.14 with CUDA 13.
  - Cherimoya's README warns that a checkpoint may not load with another version, so load these with 0.3.0.
  - This repo's pixi environment `cherimoya` pins it. On Gefion it is installed at
    `/dcai/users/mateug/git/regulatory-variant-interpretation/.pixi/envs/cherimoya/bin/python`.
- **Loading:**

  ```python
  from cherimoya import Cherimoya

  model = Cherimoya.load(path, device="cuda", compile=False).eval()
  ```

  - Each model has one unstranded signal track (`signal_groups == [1]`) and no control track.
  - Input: a 2,114-bp one-hot sequence.
  - Architecture: Cherimoya's defaults (9 layers, 128 filters).
- **Training:**
  - Trained by `../train/` (see its README).
  - Each model's exact inputs are in `fold_<k>/fit.json`: igvf_tf_chrombpnet's peaks, GC-matched negatives and
    +4/−4-shifted signal bigwig.
  - The folds are ChromBPNet's chromosome folds.
  - Fold 0, for example: test chr1, chr3, chr6; validation chr8, chr20; training everything else.
  - Each model's test-set performance is in `fold_<k>/cherimoya.fold_<k>.test.performance.tsv`.

### The output we ablate

Only the count head is ablated; the profile head was not used. We take:

- the count output, `cherimoya.wrappers.LogCountWrapper(model)`, which is log(1 + predicted counts) over the
  model's central 1,000-bp output;
- on the forward and the reverse-complement sequence, averaged as log((e^fwd + e^rev) / 2);
- for each of the 5 fold models, then the mean over folds;
- divided by ln 2, so that every quantity below is in **log2 counts**.

The forward pass was `CountPredictor`'s default on a GPU, in batches of 2,048. That is Cherimoya's inference
megakernel under `torch.no_grad()`:
- it has two bf16 steps: the MLP uses bf16 copies of the weights (made by `eval()`), and each block stores its
  convolution output in bf16;
- everything else is float32, with the TF32 matmuls that `import cherimoya` switches on.

A rerun on CPU, or through the training (grad) path, gives slightly different numbers. The docstring of
`../variant_scoring/predict_cherimoya.py` has the details.

The code is `load_model` and `CountPredictor` in `../variant_scoring/predict_cherimoya.py`, and `strand_mean` in
`../qtl/peak_scores.py`.

Why the count head, the strand mean and the fold mean:
- The count head is the region-level output, the closest analogue of DeepISA's per-region activity.
- The strand mean and fold mean match how we score variants (`../variant_scoring/scores.py`) and reduce fold-to-fold
  noise.

**Caveat:** every chromosome is in the test set of one fold and in the training set of three or four others. So the
fold mean includes models that trained on the peak.
- Held-out-only ISA would use one model per peak: the fold that has the peak's chromosome in its test set.
- That was not done here.

## 2. The datasets

All 39 AMSC datasets have both ChromBPNet and Cherimoya models. These are the 10 that both models were interpreted
on (attribution, TF-MoDISco, and Fi-NeMo with the joint20 compendium), and the inputs of that compendium:

| dataset | condition |
|---|---|
| amsc_village1_d0 | day 0 (preadipocytes) |
| amsc_village2_d0 | day 0 |
| amsc_village3_d0_1 | day 0 |
| amsc_village4_d0_1 | day 0 |
| amsc_village4_d0_2 | day 0 |
| amsc_village1_d14_CON | day 14 adipocytes, basal |
| amsc_village2_d14_CON | day 14, basal |
| amsc_village2_d14_FFA | day 14, free fatty acids |
| amsc_village4_d14_LG_1 | day 14, low glucose |
| amsc_village2_d24_CON | day 24, basal |

## 3. The sites ablated: Cherimoya's own motif instances

Per dataset, the Fi-NeMo output on that dataset's Cherimoya attributions:

```
$AMSC/cherimoya/<dataset>/downstream/finemo_joint20/<dataset>_all/peaks_qc.tsv      peak_id, chr, peak_region_start
$AMSC/cherimoya/<dataset>/downstream/finemo_joint20/<dataset>_all/hits_unique.tsv   chr, start, end, motif_name, hit_coefficient, peak_id
```

- **Attributions:** the count head's DeepLIFT/SHAP with 20 shuffles (`../attribution/`).
  - Fi-NeMo ran once per dataset, on the 5-fold average of the attributions, for all peaks (`_all`).
  - The scan used igvf_tf_chrombpnet's 10.0 step, on the attributions that `../downstream/` converts. Each peak
    was scanned over the 1,000-bp region centred on its summit.
- **Motif vocabulary (`joint20`):** one TF-MoDISco motif compendium shared by all datasets and both models, at
  `$AMSC/compendia/joint20/`. It was built from 10 ChromBPNet and 10 Cherimoya datasets
  (`$AMSC/compendia/inputs/joint20.tsv`).
  - Why: a motif name (`pos_patterns.N`) means the same motif in every dataset, so TF pairs can be compared across
    datasets and cell states.
- **TF names:** from `$AMSC/comparison/finemo/motifs.tsv`, columns `motif` and `cluster_annotation`, with the trailing
  `_<n>` removed (e.g. `CEBP_2` → `CEBP`).
  - This merges the variants of one TF family into one name.
  - The table has one annotation per motif, the same in all 10 datasets.
  - 9 of the 341 `pos_` motifs have no annotation (NA): pos_patterns.59, .114, .206, .295, .299, .313, .354, .360
    and .381. They keep their own name as their TF name; see section 6 for how the 20 October outputs got this wrong.

Selection, per dataset:
1. **Activating motifs only:** hits whose `motif_name` starts with `pos_`.
   - Why: DeepISA's synergy/redundancy reading assumes that removing a site lowers the prediction. Its code drops
     pairs where it does not (section 4). Negative-contribution (repressive) patterns would flip that.
2. **Peaks with at least 2 such hits:** about 257k–263k of ~291k peaks per dataset.
3. **50,000 of them at random** (`numpy.random.default_rng(0).choice`, without replacement).
   - Why: cost. 50,000 peaks are about 0.7 M sequences and about 6 minutes on one H100 for all 5 folds.
4. **At most 12 sites per peak:** the strongest by `hit_coefficient`, skipping any hit that overlaps one already kept.
   - Why: the pairs grow as k(k−1)/2. 12 sites give 66 pairs, so 79 sequences per peak. Overlapping hits would make
     "ablate A, keep B" undefined.
   - A peak left with fewer than 2 sites is dropped. That leaves 48.6k–49.1k peaks per dataset.
5. **Window:** 2,114 bp centred on the Fi-NeMo region:
   `[peak_region_start + 500 − 1057, peak_region_start + 500 + 1057)` on the genome below.
   - A window that runs off its chromosome is skipped.

**Genome:** `/dcai/projects/iu_0109/annotations/hg38/Sequence/IGVFFI0653VCGH.fasta` with its `.fai`. This is
GRCh38 no-alt (IGVF), the genome the models were trained on.

## 4. Ablation and the definitions

An ablated site's bases are set to N, an all-zero one-hot column. The site is `[start, end)` of the hit, in Fi-NeMo's
trimmed coordinates. DeepISA's `seq_ops.ablate_motifs` also writes N over the motif.

Per peak, three kinds of sequence are predicted: the intact window, each site ablated alone, and each pair ablated
together.

```
a       log2 counts, intact window        a_-A  site A ablated        a_-AB  A and B ablated
e_A     = a − a_-A                         the site's effect
i_AB    = e_A − e_A|-B = a − a_-A − a_-B + a_-AB    > 0 synergy, < 0 redundancy (symmetric in A and B)
```

- **Distance:** centre to centre of the two hits, in bp.
- **TF pair:** the two TF names, sorted (unordered). A pair's sides follow that order: `motif_a`, `tf_a` and `e_a`
  are one site.

**Cooperativity** (`combine.py`; these are the reported numbers):
- **Site pairs:**
  - cooperative when |i| > 0.1;
  - synergistic when i > 0.1, redundant when i < −0.1;
  - independent otherwise (|i| ≤ 0.1).
- **Per TF pair, and per TF:**
  - a TF's score counts every pair the TF is in, each unordered pair counted for both its TFs;
  - synergy score = Σ i over synergistic pairs / Σ |i| over cooperative pairs;
  - i_sum = Σ |i| over cooperative pairs;
  - independence = share of pairs with |i| ≤ 0.1.
- **Labels**, as DeepISA's `assign_cooperativity` sets them:
  - TF pairs: i_sum ≥ 1, independence > 0.9;
  - TFs: i_sum ≥ 5, independence > 0.95;
  - synergistic when the synergy score is > 0.7, redundant when < 0.3, intermediate in between;
  - independent when i_sum is below its bar and independence above its bar; unlabelled otherwise;
  - more than 10 pairs are needed for a label.
- **Intervals:** a Poisson bootstrap over peaks, 200 replicates. All of a peak's pairs get the same weight, because
  pairs in one peak share the intact prediction and are not independent.
- **Comparisons between cell states**, pooling the datasets of each group (`groups.tsv` and `comparisons.tsv` in the
  run folder):

  | comparison | group a | group b |
  |---|---|---|
  | differentiation | day 0 (the 5 day-0 datasets) | day 14 basal (village 1 and 2) |
  | FFA | village 2 basal | village 2 FFA (the same village) |
  | low glucose | day 14 basal (village 1 and 2) | village 4 low glucose |

  - There is no village-4 basal dataset among the 10, so the low-glucose comparison crosses villages.
  - A change is reported for a TF pair or TF with at least 50 pairs in each group. It comes with the 95% bootstrap
    interval of the difference in synergy score (b − a).

**Thresholds and scale:**
- 0.1 (and 0.01 below) are DeepISA's thresholds, which it applied to its own outputs. Here they are applied to log2
  Cherimoya counts, so the two scales are not the same.
- A 0.1 change in log2 counts is a 7% change in predicted accessibility.

### Where this differs from DeepISA's paper code

Checked against the paper's `Scripts_python/tf_cooperativity.py` in anderssonlab/DeepCompARE (`read_cooperativity`,
`calculate_tf_pair_synergy_score`). The official package's current rules differ again; the next table compares those.

| | DeepISA paper code | here |
|---|---|---|
| interaction i | isa2 − (isa_both − isa1) | the same, a − a_-A − a_-B + a_-AB |
| independent | \|i\| ≤ 0.01; 0.01 < \|i\| ≤ 0.1 is a "gray zone", left out of both counts | \|i\| ≤ 0.1 (gray zone counted as independent) |
| pairs dropped | either single ablation raises the prediction (isa < 0), or isa_both − isa1 < 0; overlapping motifs; pairs whose tracks disagree (synergy in one, redundancy in another); exact duplicates | none: `pos_` motifs only, overlaps removed before pairing (section 3), every pair kept |
| distance | \|start1 − start2\| ≤ 255 bp (DeepISA's receptive field) | centre to centre, no cap (up to ~1 kb; Cherimoya's input is 2,114 bp) |
| tracks | i averaged over the cooperative tracks of a cell type | one track: one model per dataset, fold mean |
| ablated bases | `ablate_motifs` adds 1 to the end, so [start, end] inclusive | [start, end) |
| pair order | ordered: both (A, B) and (B, A) (`get_permutations`), grouped by protein1, protein2; the isa_both − isa1 drop treats the two sides differently | unordered: each site pair once |
| rounding | predictions and ISA values to 3 decimals, the TF-pair table to 2 | none |
| TF-pair synergy score, i_sum, labels | as in the code | the same |
| TF scores | from the TF-pair table after `assign_cooperativity`: only labelled TF pairs with more than 10 pairs, and the i sums leave out TF pairs labelled Independent | from every site pair the TF is in |

### The official deepISA package's rules (`combine.py --official`)

Read in deepISA 0.1.0 at `179091c` (`score/aggregate_isa.py`, `score/combi_isa.py`, `quickstart.py`). The package
runs its own model, sites and predictions; `--official` gives its aggregate stage this folder's ablations instead.

| | official package | here (ours) |
|---|---|---|
| interaction i | isa1 + isa2 − isa_both | the same quantity |
| ablated site | N (an all-zero column) | the same |
| model output | any scalar the model returns, one model, one strand | log2 counts, both strands, the 5-fold mean |
| sites | every Fi-NeMo hit (its own discovery run) | at most 12 non-overlapping `pos_` sites per peak, strongest first, in 50k sampled peaks |
| pair filter | both single effects ≥ the 10th percentile of the non-negative ones, and both effects given the other ablated ≥ 0 | none |
| distance | edge to edge, 1–255 bp (`make_pairs_for_region`, receptive field 255) | centre to centre, no cap |
| score | coop_score = Σi / Σ\|i\| over the qualified pairs | synergy = Σ(i > 0.1) / Σ(\|i\| > 0.1) |
| labels | Mann–Whitney test against the pairs > 150 bp apart, BH (q < 0.1), then Synergistic above the 70th percentile of the significant scores, Redundant below the 30th, Intermediate between, Independent otherwise (coop_score set to NaN); min_count 10 | absolute: the bars above |
| across datasets | nothing | changes per comparison, Poisson bootstrap over peaks |

What `--official` does, per dataset (`<root>/official/<dataset>/`):
1. writes the package's two input tables: `motif_single_isa.csv` (each site's `isa_t0` = e) and
   `motif_combi_isa.csv` (`tf1`, `tf2`, `distance` = the edge-to-edge gap, `isa1_t0`, `isa2_t0`, `isa_both_t0`),
   keeping only pairs with a gap of 1–255 bp, as `make_pairs_for_region` does;
2. calls `calc_interaction(isa_thresh_quantile=0.1, normalize=False)` and `calc_coop_score` at `tf_pair` and `tf`
   (`min_count=10`, `q_val_thresh=0.1`), exactly as QuickStart's aggregate stage does, writing
   `coop_tf_pair_t0.csv` and `coop_tf_t0.csv`.

Then, across datasets (`<root>/official/`): the changes per comparison use coop = Σi / Σ|i| recomputed from the
qualified pairs' `interaction_t0`, with the same bootstrap as ours. The package's labels are relative within a
dataset and its coop_score is NaN wherever the label is Independent, so neither is compared across datasets. The 10th
percentile comes from our sites, not from every hit. The package was written for pandas 2, so its functions run with
pandas' object strings (`future.infer_string` off).

`coop_isa.py` also writes quick per-dataset tables: `tf_pairs.tsv`, `tfs.tsv`, and `independence_all` in `run.json`.
- There, independence is the share of all pairs with |i| ≤ 0.01.
- The reported numbers are `combine.py`'s.

## 5. How it ran

The run used Slurm job 528519 on Gefion: one whole node with 8 H100s, on GPUs 2–6.
- **Start:** five lanes started at 20:22 ET on 20 October, two datasets each.
- **End:** the last dataset finished at 20:35:55 ET. `combine.py` was done at 20:38 ET.
- **First attempt:** five lane steps started at 20:21:55 ET and failed within 4 s (exit code 2). They were started
  again at 20:22.

The scripts that ran are the copies in `$AMSC/cherimoya/cooperativity/` (`coop_isa.py`, `lane.sh`, `combine.py`).
- They had their inputs built in: `--dataset` named a dataset under `$AMSC`, and `combine.py` held the groups.
- The scripts in this folder take the same inputs as arguments, and they have been reformatted.
- Their computation is the same, with one exception: they name TFs correctly where the run did not (section 6).
- The run imported `peak_scores.py` and `predict_cherimoya.py` from the `variant-scoring` worktree
  (`/dcai/users/mateug/git/rvi-variant-scoring/workflows/variant_scoring/`). Those copies differ from
  `../qtl/peak_scores.py` and `../variant_scoring/predict_cherimoya.py` only in paths and docstrings.

```bash
OUT=$AMSC/cherimoya/cooperativity
srun --jobid 528519 --overlap -N1 -n1 $OUT/lane.sh <gpu> <cpus> <dataset 1> <dataset 2> > $OUT/logs/lane_<gpu>.log 2>&1 &
python $OUT/combine.py --root $OUT        # --t-coop 0.1 --n-boot 200 --min-n 50
```

| GPU | datasets |
|---|---|
| 2 | amsc_village1_d0, amsc_village2_d24_CON |
| 3 | amsc_village1_d14_CON, amsc_village3_d0_1 |
| 4 | amsc_village2_d0, amsc_village4_d0_1 |
| 5 | amsc_village2_d14_CON, amsc_village4_d14_LG_1 |
| 6 | amsc_village2_d14_FFA, amsc_village4_d0_2 |

`coop_isa.py` ran with these settings: `--folds 0 1 2 3 4 --n-peaks 50000 --max-sites 12 --seed 0 --batch 2048
--t-ind 0.01 --t-coop 0.1 --min-n 11`. Each dataset's `run.json` records its arguments.

Size and time per dataset:
- 48.6k–49.1k peaks;
- 0.69–0.79 M sequences, each predicted by 5 models on both strands;
- 208k–228k sites and 430k–517k site pairs;
- 363–405 s on one H100.

Over the 10 datasets: 7.3 M sequences and 4.6 M site pairs.

### To run it again with this folder's scripts

`$AMSC/cherimoya/cooperativity/args/<dataset>.args` lists each dataset's inputs, one argument per line. They are the
inputs above: `--models`, `--finemo-dir`, `--motifs` and `--fasta`. Their `--out-dir` is a new folder, `rerun/`, so the
run's outputs are not touched.

```bash
RUN=$AMSC/cherimoya/cooperativity
export COOP_PYTHON=<the cherimoya environment's python>
# one dataset on GPU 0
bash workflows/cooperativity/lane.sh 0 0-31 $RUN/rerun/logs $RUN/args/amsc_village1_d0.args
# all ten, after the coop_isa.py runs
python workflows/cooperativity/combine.py --root $RUN/rerun --motifs $AMSC/comparison/finemo/motifs.tsv \
    --groups $RUN/groups.tsv --comparisons $RUN/comparisons.tsv
```

## 6. Outputs

Per dataset, in `<dataset>/`:
- **`sites.tsv.gz`:** one row per site: `peak_id, chr, start, end, motif, tf, e` (the site's effect), `a` (the intact
  prediction).
- **`pairs.tsv.gz`:** one row per site pair: `peak_id, chr, motif_a, motif_b, tf_a, tf_b, distance, e_a, e_b, i`.
  This is the table to start from, but in the 20 October outputs name the sites from `motif_a` and `motif_b`, not
  from `tf_a` and `tf_b` (see below).
- **`tf_pairs.tsv`, `tfs.tsv`:** the quick per-dataset tables (independence at 0.01).
- **`run.json`:** arguments, counts, run time.

In `combined/`:
- **`pairs_by_dataset.tsv`, `tfs_by_dataset.tsv`:** per dataset: n, synergy score with 95% interval, independence,
  i_sum, mean i, label.
- **`changes_pairs.tsv`, `changes_tfs.tsv`:** per comparison: both groups' values, the difference (b − a) with its 95%
  interval, and `significant`.
- **`distance.tsv`:** shares of synergistic, redundant and independent pairs by distance bin, per dataset.
- **`summary.json`:** per-dataset totals and label counts, and the number of significant changes.

Logs: `logs/<dataset>.log`, `logs/lane_<gpu>.log`, `logs/combine.log`.

### Known issue in the 20 October outputs: TF names

The run's copy of `coop_isa.py` named the TFs in two wrong ways. The scripts in this folder fix both.
1. **Unannotated motifs got an empty name.** The 9 `pos_` motifs without an annotation (section 3) were given the
   TF name "" instead of their own name. In each dataset, 5–8 of them were ablated, at 125–298 sites.
2. **The TF columns don't line up with the motif columns.** In `pairs.tsv.gz`, `tf_a` and `tf_b` were sorted by
   name, while `motif_a`, `motif_b`, `e_a` and `e_b` stayed in site order. So `tf_a` is `motif_b`'s TF in 42% of
   amsc_village1_d0's rows. `i` and `distance` are symmetric and unaffected.

The run's copy of `combine.py` filled an empty name from `motif_a` or `motif_b` by column. Because of point 2,
that took the partner's motif ID about half the time. For example, a CEBP + pos_patterns.59 pair is counted as
"CEBP + pos_patterns.16", pos_patterns.16 being CEBP's own motif. In `combined/`:
- **`tfs_by_dataset.tsv`:** 18 annotated motif IDs appear as if they were TFs. They are pos_patterns.0, 3, 4, 5, 7,
  11, 13, 16, 19, 21, 22, 31, 38, 79, 94, 155, 275 and 376, and they hold the unannotated motifs' pairs.
- **`changes_tfs.tsv` and `changes_pairs.tsv`:** these have 2 / 1 / 2 such rows for differentiation / FFA / low
  glucose: pos_patterns.16, pos_patterns.21, "CEBP + pos_patterns.16" and "NFI + pos_patterns.21". None is
  significant.
  - The significant counts in section 7 stand.
  - The numbers tested include these rows.
- **Annotated TFs:** each pair is still counted once under the annotated TF's own name, so their rows are right.
- **Unannotated motifs:** their own rows hold only part of their pairs.

To correct the combined tables without new predictions, run this folder's `combine.py` on the run's outputs. It
names every site from its motif with `--motifs`. It needs CPU only.

```bash
python workflows/cooperativity/combine.py --root $RUN --motifs $AMSC/comparison/finemo/motifs.tsv \
    --groups $RUN/groups.tsv --comparisons $RUN/comparisons.tsv
```

This overwrites `$RUN/combined/`. To keep the original, copy the per-dataset folders and run it on the copy.

## 7. Results in brief

From `combined/summary.json` and `combined/distance.tsv` of the 20 October run. TF names don't enter these
numbers, except the counts tested in the last table (section 6):

- **Most pairs are independent:** 72–81% per dataset. Synergistic pairs are 8–12%, redundant 10–16%.
- **Synergy is short-range.** Over all 10 datasets:

  | distance | synergistic | redundant | independent |
  |---|---|---|---|
  | < 20 bp | 41% | 22% | 37% |
  | 20–50 bp | 29% | 9% | 62% |
  | 50–100 bp | 12% | 11% | 77% |
  | 100–200 bp | 1% | 16% | 83% |
  | 200–500 bp | 0.3% | 16% | 84% |
  | 500–1,000 bp | 0.1% | 7% | 93% |

  Redundancy reaches several hundred bp.
- **Changes in synergy score outside the 95% interval:**

  | comparison | TFs | TF pairs |
  |---|---|---|
  | differentiation | 25 of 55 | 67 of 369 |
  | FFA | 19 of 55 | 51 of 279 |
  | low glucose | 14 of 57 | 52 of 314 |

  - The intervals are per TF and are not corrected for multiple testing; about 5% would fall outside by chance.
  - The low-glucose comparison also crosses villages (section 4).

## 8. Running it in a project

The pipeline's `cooperativity` stage (`workflows/pipeline/run_project.sh`) runs this folder on any project whose
Fi-NeMo hits were called against one compendium (its `motifs` stage), so that `pos_patterns.<n>` is the same motif
in every dataset:
1. `motif_names.py --compendium <root>/compendia/<tag> --out <root>/cherimoya/cooperativity/motifs.tsv`: each cluster
   named by the most frequent annotation at or above the compendium's `min_annotation_score`, as the comparison
   names them. (The AMSC run above took this table from the ChromBPNet comparison instead.)
2. `project.py <project.yaml> coop-inputs`: one args file per dataset (models, Fi-NeMo directory, motifs, genome,
   folds), and `groups.tsv` and `comparisons.tsv` from project.yaml's `cooperativity` block (by default one group per
   dataset).
3. `lane.sh`, one dataset per GPU, then `combine.py --official`.

`pairs.tsv.gz` from this version of `coop_isa.py` also holds `gap` and `isa_both`, which `--official` needs, and
`start_a`, `start_b`, `a` and the per-fold `i_f<k>`, which the fold check and section 9 need. The AMSC run's tables
predate them all; `combine.py` reads them without the fold check.

What `combine.py` reports has changed since the AMSC run (section 7):
- **`significant`** is now Benjamini–Hochberg q < 0.05 within each comparison. Each change's p comes from its
  bootstrap SE (normal approximation): 200 replicates are too few to count p-values from. The AMSC run's criterion,
  the 95% interval excluding 0 without correction, is kept as `ci_excludes_0`.
- **The fold check:** each change computed again from each fold's models alone (fold k of both groups; the folds are
  the same chromosome splits in every dataset). `folds_agree` counts the folds whose change has the fold-mean
  change's sign, and `robust` = significant and all of them agree. The bootstrap resamples peaks only, so it misses
  the noise of training a separate model per dataset; the fold check covers part of it.
- **Trends** (`--series`, project.yaml's `cooperativity.series`): per TF pair and TF with at least `--min-n` pairs in
  every group, the score in each group, the least-squares slope along the series with its CI, p and q, and a
  heterogeneity test for any change along it (a Wald test of the consecutive differences, their covariance from the
  bootstrap, chi-square with G − 1 df). `trend_pairs.tsv` and `trend_tfs.tsv`, in `combined/` and `official/`.
- **Effect sizes** in `summary.json`: per dataset the median intact prediction a and the median |e| and |i|. A weaker
  model (a dataset with fewer reads) predicts smaller effects, so fewer of its pairs pass 0.1 and it looks more
  independent; read the synergy beside them and beside the model's test accuracy (STATUS.md has both).

## 9. The same sites under every model (`coop_matrix`)

A change between two datasets in sections 4–8 compares each dataset's own sites under its own models. It mixes two
things: a change in how the model reads a pair of sites (the grammar), and a change in which sites are bound, in
which peaks. The models are inputs to an in-silico experiment, so every dataset's models can score every dataset's
sites. The pipeline's optional `coop_matrix` stage does this (`matrix.py`):
1. `coop_isa.py` with every dataset's sites (`--finemo-dir`) under every dataset's models (`--models`), into
   `<coop>/matrix/<sites>/<model>/`. The sites depend only on the Fi-NeMo directory, `--n-peaks`, `--max-sites` and
   `--seed`, so one dataset's sites give the same rows under every model. D datasets make D × D runs.
2. With a series, `combine.py --series --official` on the datasets' own sites, for the trends above.
3. `paired.py` on the matrix:
   - **Rows:** the union of every dataset's site pairs. A pair in several datasets' sites (same chromosome, starts
     and motifs) is kept once.
   - **Open windows:** a window counts for a model when its intact prediction a reaches the 10th percentile of a over
     that model's own windows (`--open-quantile`). In a window the model sees as closed every effect is small, and the
     pair would look independent whatever the grammar. A comparison uses the windows open under all of its models;
     the series, under all of the series' models. `open.tsv` has each model's threshold and counts.
   - **Paired bootstrap:** one set of Poisson weights per window and replicate, the same for every model. The pairs
     are the same, so most of their noise cancels in a difference.
   - **Per comparison** (`changes_pairs.tsv`, `changes_tfs.tsv`): the paired change in synergy (as in section 4), and
     delta i, the mean over the pairs of i(b) − i(a). delta i has no threshold, so smaller effects under a weaker
     model do not push pairs under 0.1. Both have p, q, the fold check and `robust`.
   - **Per series** (`trend_pairs.tsv`, `trend_tfs.tsv`): the paired trend of the synergy, and the slope of the mean
     i, each with p, q and the fold check.

Out: `<coop>/matrix/combined/`, and `summary.json` with the counts per comparison and for the series. Cost: about
6–7 GPU-minutes per run; 5 datasets are 25 runs, ~25 min on 8 GPUs.

## 10. Checking the official tables (`check_official.py`)

```
rvi_run cherimoya python workflows/cooperativity/check_official.py --root <coop> --motifs <coop>/motifs.tsv \
    --groups <coop>/groups.tsv
```

It checks `combine.py --official` four ways per dataset:
- the package's aggregate stage, run again by hand on the CSVs `combine.py` wrote, gives the same tables;
- the CSVs hold the pairs' own effects (`tf1` is `motif1`'s TF; `isa1_t0`, `isa2_t0`, `isa_both_t0` are the pair's
  `e_a`, `e_b`, `isa_both`);
- the package's rules, recomputed here from `coop_isa.py`'s tables, give the package's tables;
- the group scores of the changes are sum(i) / sum(|i|) over each group's qualified pairs.

It prints one line per check and dataset and exits 1 if any fails. CPU only, about a minute for 5 datasets.
