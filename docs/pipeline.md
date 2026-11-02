# The Mímir pipeline

This is the reference for how a project goes from fragment files to Cherimoya models, motifs and TF cooperativity:
what each step does and why, what it reads and writes, the command that runs it, its environment, and how long it
takes. `docs/new_project.md` is the how-to; `docs/cluster.md` has the Gefion (DCAI) rules; `AGENTS.md` is for coding
agents.

## Overview

| stage | what | where the code is |
|---|---|---|
| fetch | the input fragment files, md5-checked | `workflows/pipeline/fetch_igvf.py` (login node) |
| **Prepare** (`prep`) | Tn5 shift, peaks, signal track, background regions, QC | igvf_tf_chrombpnet 00.0–02.0 |
| **Train** (`train`) | Cherimoya, one model per chromosome fold | `workflows/train/` |
| **Attribute** (`attribute`) | DeepLIFT/SHAP on the counts head, every peak, every fold | `cherimoya attribute`, `workflows/attribution/` |
| **Motifs** (`downstream`, `motifs`) | fold average, TF-MoDISco, one compendium, Fi-NeMo hits | `workflows/downstream/`, igvf_tf_chrombpnet 06.0–11.0 |
| **Cooperativity** (`cooperativity`) | DeepISA: in-silico ablation of motif pairs | `workflows/cooperativity/` |
| summary | `<root>/STATUS.md` | `workflows/pipeline/status.py` |

Outside the driver, for projects that need them:
- **Score:** variant effects (`workflows/variant_scoring/`, `workflows/nulls/`) and QTL checks (`workflows/qtl/`).
- **Compare:** ChromBPNet against Cherimoya, level by level (`workflows/train/compare_chrombpnet.py`,
  `workflows/comparison/`).

One project is one directory, `<root>`, and one `project.yaml` in it. `workflows/pipeline/project.py` derives every
path from root, prefix and the sample stems; a dataset is `<ds>` = prefix + stem. Its docstring has the full layout.

### Two repositories, one pinned commit

- **This repository (RVI)** holds everything Cherimoya-specific and the driver.
- **[igvf_tf_chrombpnet](https://github.com/IGVF/igvf_tf_chrombpnet)**, branch `chrombpnet2`, holds the
  preprocessing (00.0–02.0) and the motif steps (06.0–11.0). The driver runs its step files unchanged, from the
  checkout `project.yaml`'s `igvf` names, at the commit in `workflows/pipeline/igvf.pin` (`2446d6f`). The preflight
  refuses another commit or local changes.
- **[chrombpnet](https://github.com/NNFC-GMD/chrombpnet)** (`stable-modern-gpu-support`) is needed only for its
  environment: 01.0, 06.0 and 07.0 run in it, and `detect_shifts.py` uses its shift detector as a cross-check. No
  ChromBPNet model is trained.

### The box

`workflows/pipeline/submit.sh` submits one whole-node box (`run_project.sbatch`), which runs
`workflows/pipeline/run_project.sh`:
- **Stages run one after another:** preflight, then `project.yaml`'s stages in pipeline order, then summary. Within a
  stage the work runs in parallel on the node.
- **The preflight** checks that both checkouts are clean and igvf is at its pin, that every environment and input
  exists, and runs the tests: `pixi run check` (lint, shellcheck, the CPU tests) and `pixi run -e cherimoya
  test-pipeline` (the tests of what the driver runs) are required; every other test in the cherimoya environment
  (`test-cherimoya`) runs beside the stages and is reported in STATUS.md, not required.
- **Outputs decide what is done.** A stage is skipped when `project.py check <stage>` finds its outputs complete,
  and counts as done only when the check passes after it ran. Exit codes are not trusted: `train/run_box.sh` always
  exits 0, and the runners report per task. So a resubmitted box resumes where the last one stopped.
- **A failed stage holds the box** for `hold_minutes` (default 60). Fix it from the login node, or inside the box
  with `srun --jobid <job> --overlap -N1 -n1 <cmd>`. Then `touch <root>/box/<job>/retry` reruns the stage, and `touch
  <root>/box/<job>/release` ends the box. If nobody acts, the box ends when the hold runs out.
- **Records:**
  - `<root>/box/<job>/stages.tsv`: one row per stage (state, exit, seconds, start in ET);
  - `<root>/box/<job>/logs/`: one log per command;
  - `<root>/box/slurm_<job>.log`: the driver's own log;
  - `<root>/STATUS.md`: the summary.

### Environments

| checkout | env | runs |
|---|---|---|
| RVI | `default` | lint, the CPU tests, `project.py`, `fetch_igvf.py`, `make_jobs.py`, `make_downstream.py`, `status.py` |
| RVI | `cherimoya` | Cherimoya 0.3.1, tangermeme 1.6.0, torch 2.14 (CUDA 13), deepISA 0.1.0: fit, `cherimoya attribute`, the conversion, `motif_names.py`, `coop_isa.py`, `combine.py`, `test-cherimoya` |
| RVI | `cherimoya-030` | Cherimoya 0.3.0 and tangermeme 1.5.0: only the frozen `attribute_fast.py` engine and its tests (`test-engine`) |
| igvf | `peaks` | MACS3 PR #756, `detect_shifts.py`, 00.0 call_peaks |
| igvf | `preprocess` | 00.0 prepare_signal (figwig), 00.1, 02.0, `make_configs.py` |
| igvf | `modisco` | 08.0, TF-MoDISco from NNFC-GMD/tfmodisco |
| igvf | `motif-compendium` | 09.0, MotifCompendium 1.0.19 |
| igvf | `finemo` | 10.0 and 11.0, Fi-NeMo 0.41 |
| igvf | `qc` | igvf_tf_chrombpnet's own tests |
| chrombpnet | `cuda13` | 01.0, 06.0, 07.0; `detect_shifts.py --chrombpnet-python` (`cuda12` on a driver below 580) |

Cherimoya's model code (`cherimoya.py`, `cheri.py`, `io.py`, `wrappers.py`, `training.py`) is the same in 0.3.0 and
0.3.1, so both environments load every model.

## Fetch

`fetch_igvf.py <project.yaml>` runs on the login node (I/O only). For each sample whose `fragments` is an IGVF file
accession, it reads the portal's metadata (md5, size, link), streams the file to a temporary name, compares the md5
with the portal's (and with `project.yaml`'s, if given), and only then renames it into place:
- the fragments go to `<root>/fragments/<stem>.fragments.tsv.gz`, and their `.tbi` beside them;
- the extra files go to `<root>/portal/`, with a `manifest.tsv` of every file.

Why the name matters:
- igvf's config check accepts `.fragments.tsv.gz` and not the portal's `.bed.gz`;
- `make_configs.py` resolves symlinks, so the real file must carry the name.

The fragments must be BGZF; the peak caller and figwig read them as such. A local file named
`<stem>.fragments.tsv.gz` is linked in instead of fetched. All fragments are used, with no barcode filter: a
pseudobulk's fragment file already holds only its cells.

## Prepare (`prep`)

`stage_prep` in `run_project.sh`, all datasets at once. These steps are CPU only.

### Tn5 shift

**Command:**
```
igvf_run peaks python workflows/dcai/detect_shifts.py --fragments-dir <root>/fragments --genome <genome> \
    --out <root>/chrombpnet/shifts.json --jobs <n> --every <shift_every> --chrombpnet-python <chrombpnet env python>
```

**What it does.** It finds the Tn5 shift already present in each file. Each file is thinned genome-wide (every Nth
fragment; `shift_every`, default 100) and fed to the pipeline's detector (scPrinter's base composition at the cut
sites against reference Tn5 bias matrices). ChromBPNet's own detector is run on the same sample.

**Why:**
- **One shift for both signal and peaks.** The configs record the shift present in the data (`plus_shift`,
  `minus_shift`). Every later step converts it to ChromBPNet's +4/−4, so the bigwig and the MACS insertions sit at
  the same bases.
- **10x fragments come out +4/−5.**
- **It refuses rather than guesses.** The detectors must agree and the fit's MSE must be at most 0.002, or it exits
  2 and the stage fails. Nothing pins a shift nobody believes.
- **Why thin rather than sample the start of the file:** the stock detector samples only the first million lines,
  which in a sorted file is the start of chr1.
- **Why every 100th and not igvf's default 2000:** the default left too few fragments for files of 10–30 M
  fragments. The AMSC files of 5–9 GB thinned to about 400k.

### Configs

**Command:**
```
igvf_run preprocess python workflows/dcai/make_configs.py --fragments-dir ... --dataset-root <root> \
    --shifts ... --prefix <prefix>
```

One igvf config per file: `<root>/chrombpnet/configs/<ds>/config.yaml`. It records:
- the pinned shift;
- `call_peaks: true` and `macs_input: bed`;
- `bigwig_engine: figwig`;
- `peak_summit_window: 1000`;
- folds 0–4;
- `output_dir` `<root>/chrombpnet/<ds>`.

### The chain

`run_chain.sh <config>` runs one dataset's chain; all datasets' chains run at once:

```
00.0.call_peaks  ||  00.0.prepare_signal  ->  00.1.preprocess_peaks  ->  01.0.preprocess_nonpeaks  ->  02.0.qc_training_data
```

**00.0.call_peaks** (`peaks` env) follows the kundajelab/igvf_pseudobulking_pipeline v2.0.1 recipe, on MACS3
PR #756 (jmschrei/MACS@fa52988, byte-identical to the MACS3 3.0.5 release):
- each fragment's two Tn5 insertions become 1-bp records (`-f BED`);
- each fragment goes to pseudoreplicate 1 or 2 from one stream seeded 42;
- `macs3 callpeak -f BED -g hs -p 0.01 --shift -75 --extsize 150 --nomodel --keep-dup all --call-summits` runs on
  rep1, rep2 and the pooled repT;
- the top 300k rows per replicate are kept by column 8 (−log10 p);
- a repT row stays only if it overlaps both rep1 and rep2 (`-f 0.5 -F 0.5 -e`);
- rows touching the ENCODE blacklist ENCFF356LFX are dropped.

Departures from v2.0.1, each recorded in `call_peaks.json`:
- **No double shift.** The insertions sit where the bigwig puts them (+4/−4 from the detected shift). The original
  adds +4/−5 to fragments 10x has already shifted.
- **Deterministic pseudoreplicates:** one seeded stream.
- **No `-B --SPMR`:** they only write the bedGraphs behind a p-value track, which nothing here reads.

Why `-f BED` and not `-f FRAG`: FRAG switches MACS to paired-end mode, which ignores the shift and extension and
piles up whole fragments, so it is not the same recipe.

Output: `<root>/peaks/<ds>/<ds>.narrowPeak.gz`, `call_peaks.json` (the counts after each filter) and `reps/`.

**The other steps:**
- **00.0.prepare_signal** (`preprocess`): `figwig bam2bw` writes the unstranded bigwig of +4/−4 cut sites,
  `<root>/chrombpnet/<ds>/preprocessing/signal/data_unstranded.bw`, with a sidecar. It gives the same per-base values
  as the numpy path, on several cores.
- **00.1** (`preprocess`): each called peak becomes a 1,000-bp window centred on its own summit, and blacklisted
  windows are removed: `preprocessing/peaks/<ds>_all_peaks_no_blacklist.narrowPeak`.
- **01.0** (chrombpnet env): `chrombpnet prep nonpeaks` writes GC-matched background regions, one set per fold:
  `preprocessing/<ds>/output_all_fold_<k>_negatives.bed`. It deliberately runs ChromBPNet's own code; the negatives
  are half of the training data.
- **02.0** (`preprocess`): QC of the training set as a whole:
  - TSS enrichment;
  - signal at peak summits;
  - peaks against background (AUROC);
  - the GC match.

  Output: `<root>/chrombpnet/<ds>/plots/signal_qc/<ds>_signal_qc.json` and plots.

**Measured (AMSC, 39 files of 5–9 GB, several chains at once):**

| step | time |
|---|---|
| call_peaks | ~1,750 s |
| prepare_signal | ~580 s |
| 01.0 | ~1,000 s |
| 02.0 | ~230 s |

Up to ~63 GB of RAM per chain. Peak calling and the signal scale with the number of fragments, and 01.0 is mostly a
fixed genome scan. Files 30–60× smaller take minutes.

## Train (`train`)

**Command:**
```
rvi_run default python workflows/train/make_jobs.py --chrombpnet-configs <root>/chrombpnet/configs/*/config.yaml \
    --folds-dir <igvf>/folds --genome <genome> --out-root <root>/cherimoya
env CHERI_ROOT=<root>/cherimoya CHERI_STAGES=fit CHERI_SLOTS_PER_GPU=<train_per_gpu> CHERI_CORES_PER_SLOT=<train_cores> \
    bash workflows/train/run_box.sh
```

`make_jobs.py` writes one `fit.json` per dataset and fold (`<root>/cherimoya/<ds>/fold_<k>/`) and `tasks.tsv`.
`run_box.sh` keeps every GPU busy: several workers per GPU, each pinned to its own cores. It starts them slot by
slot across the GPUs, so a short queue still uses every GPU. Each worker runs `run_task.sh <fold> fit`, which is
`cherimoya fit -p fit.json`.

The inputs are igvf's:
- the 00.0 bigwig, already +4/−4;
- the 00.1 summit windows;
- the 01.0 negatives of the fold;
- the chromosome split of `<igvf>/folds/fold_<k>.json`. These are the same five chromosome folds ChromBPNet uses, so
  any later comparison is fold for fold.

The settings: Cherimoya 0.3.1's defaults, except `random_state` = 1234 + fold.
- **Architecture:** 9 layers, 128 filters; 2,114 bp in, 1,000 bp out.
- **Training:** batch 64, jitter 500, negative ratio 0.25, float32, `torch.compile` with max-autotune,
  `num_threads` 4.
- **Length:** `max_epochs` 20, with `min_total_steps` 20,000: below about 1,000 steps an epoch, Cherimoya raises the
  epoch count until 20,000 steps. So fewer peaks means more epochs, not a proportionally shorter fit.
- **The model kept:** `cherimoya.fold_<k>.torch` is the best-validation (EMA) checkpoint. The fit then evaluates it
  on the validation and test chromosomes; `cherimoya.fold_<k>.test.performance.tsv` is the done marker.
- **No control track.** The scPrinter Tn5-bias control (track plus CNN) was piloted on one AMSC fold and failed its
  pre-registered test: it did not remove Tn5 patterns from the profile head.
- **Why several fits per GPU:** training is CPU- and launch-bound (the main Python thread is near 100%). The default
  4 per H100 is above what AMSC measured (3 per GPU, beside ChromBPNet training).

The compare stage (`compare.json`) is written but not run; it needs a trained ChromBPNet.

**Measured (AMSC, ~290k peaks):**
- a median of 7,702 s per fold at 3 fits per GPU beside ChromBPNet;
- about 50 min alone on a GPU;
- ~21 GB RSS.

Time scales with steps, which is the peaks times 20 epochs, but never fewer than 20,000.

## Attribute (`attribute`)

**Command:**
```
rvi_run default python workflows/downstream/make_downstream.py --cheri-root <root>/cherimoya \
    --chrombpnet-configs ... --shared <root>/cherimoya/shared.txt --compendia <root>/compendia --engine cherimoya
env CHERI_ROOT=... CHERI_STAGES=attribute CHERI_ATTR_ENGINE=cherimoya CHERI_SLOTS_PER_GPU=<attribute_per_gpu> \
    CHERI_CORES_PER_SLOT=<attribute_cores> bash workflows/train/run_box.sh
```

`make_downstream.py` writes three things:
- **each fold's `attribute.json`:**
  - the fold's model and its peaks;
  - every chromosome of the fold's training, validation and test lists, so every peak is attributed;
  - `in_window` 2114, `attr_window` 1000, `algorithm` deep_lift_shap, `output` counts;
  - 20 dinucleotide-shuffled references, `random_state` 1234 + 100 × fold;
  - `"fast": true`.
- **each dataset's downstream config:** `<root>/cherimoya/configs/<ds>/config.yaml`, with `output_dir`
  `<root>/cherimoya/<ds>/downstream`, counts head only.
- **the compendium input lists** in `<root>/compendia/inputs/`. With no ChromBPNet, `--shared` is an empty file, so
  `cheri<D>.tsv` holds every dataset and `rest<D>.txt` lists them.

**Why these settings:**
- The 1,000-bp window is the narrowest that holds what 08.0 (`-w 500`) and 10.0 (region width 1000) read.
- The fold seeds are 100 apart: tangermeme seeds reference k with `random_state + k`, so folds closer than 20 apart
  would share references, and the fold average would not average reference noise out.
- `output` stays "counts": 0.3.1 takes it as "logcounts" with a FutureWarning, and the converted files are named by
  it.

**The engine.** `run_task.sh <fold> attribute` with `CHERI_ATTR_ENGINE=cherimoya` runs `cherimoya attribute -p
attribute.json`. This is Cherimoya 0.3.1's fast DeepLIFT/SHAP path (cherimoya PR #104), this repository's engine
moved upstream. It checks itself against tangermeme's `deep_lift_shap` on its first sequences, and when they
disagree it warns and finishes on tangermeme's path. Then `workflows/attribution/stock_record.py` writes
`attr.attr.stock.json` beside the outputs:
- `parameters`: the attribute.json as written;
- `versions`;
- `fast_path`: false if the log has the fallback warning;
- the count of convergence warnings.

It refuses non-finite attributions. `attr_status.py` is the one definition of done; it accepts such a fold through
that record. The driver prints any fallback; one is still a valid attribution, only slower.

**The AMSC campaign** used `workflows/attribution/attribute_fast.py` (`CHERI_ATTR_ENGINE=rvi`, the default), which
audits every run against stock on 8 peaks. It runs in the `cherimoya-030` environment, because it refuses other
Cherimoya and tangermeme versions. It is kept unchanged so those results can be reproduced. Its three files are
hashed into an engine pin (`<root>/attribution_engine.json`) and must never be edited.

**Measured (AMSC, ~290k peaks, the same engine):**
- about 1 h per fold at one process per GPU;
- ~7 GB of GPU memory;
- up to ~24 GB RSS.

The default 4 per H100 fits: the engine plans at most ~12 GB each. Time scales with peaks × 20 references.

## Motifs (`downstream`, then `motifs`)

### downstream

**Command:**
```
env CHERI_ROOT=... BOX_DIR=<box> CHERI_SHARED_LIST=<root>/cherimoya/shared.txt CHERI_08_MAX=<modisco_max> \
    CHERI_08_THREADS=<modisco_threads> bash workflows/downstream/downstream.sh <root>/compendia/inputs/rest<D>.txt
```

Per dataset, once every fold's attribution is done:
- **convert:** `to_chrombpnet_scores.py` writes each fold into ChromBPNet's 05.0 layout, so igvf's own steps run
  unchanged. The output is `.../downstream/full_models/<ds>_all_fold_<k>/interpretation/interpretation.counts_scores.h5`,
  with its bed and a provenance file.
- **check:** `--check-dataset --against fasta`. All folds share one mask and one bed, and on 2,000 rows per fold the
  one-hot is the genome's.
- **06.0:** the five folds averaged.
- **V2:** `--check-average` recomputes the average on 1,000 rows, bit for bit.
- **07.0:** the averaged scores as a bigwig.
- **08.0:** TF-MoDISco (`modisco` env) on the counts average:
  - NNFC-GMD/tfmodisco's fork of modisco 2.5.2, with an inverted-index neighbour search; it gives identical results,
    faster;
  - `-w 500`, a budget of 500,000 seqlets (`modisco_max_seqlets`, igvf's default);
  - output `contrib_scores/<ds>/modisco/modisco_counts_results.h5`.

Each step leaves a marker in `<root>/cherimoya/<ds>/downstream/.done/`. A rerun skips marked steps and redoes an
output without a marker.

**Measured (AMSC):**

| step | time |
|---|---|
| convert | ~137 s per fold |
| check | ~30 s |
| 06.0 | ~380 s |
| V2 | ~125 s |
| 07.0 | ~370 s |
| 08.0 | median 13,284 s at 24 threads and 500k seqlets (11,181–23,175 s), ~55 GB RSS with its Leiden workers |

Leiden, most of an 08.0, is single-threaded, so more threads mostly sit idle. Run several 08.0 at once instead
(`modisco_max`). 08.0 time grows faster than linearly with the seqlet count.

### motifs

**Command:**
```
env BOX_DIR=<box> CHAIN_FINEMO_SLOTS=<finemo_slots> FINEMO_BATCH_SIZE=<finemo_batch> MOTIF_MIN_AGE_S=0 MOTIF_POLL_S=60 \
    bash <igvf>/workflows/dcai/motif_chain.sh <inputs cheri<D>.tsv> <root>/compendia/<tag> <inputs cheri<D>.tsv>
```

- **09.0** (`motif-compendium`): MotifCompendium 1.0.19 in list mode. It clusters every dataset's TF-MoDISco
  patterns into one non-redundant compendium, `<root>/compendia/<tag>/`:
  - `modisco_compiled.h5`, `modisco_compendium_meta.tsv`, `compendium.json`;
  - clustering cutoff 0.95 (`MOTIF_COMPENDIUM_THRESHOLD`);
  - annotation against the MotifCompendium human database at a minimum similarity of 0.7.

  **Why one compendium per project:** Fi-NeMo then calls every dataset's hits against the same motifs, so
  `pos_patterns.<n>` means the same motif in every dataset. That is what lets datasets (for example time points) be
  compared. A compendium directory is immutable; rebuild into a new tag.
- **10.0** (`finemo`, one GPU per dataset): Fi-NeMo hits on the fold-averaged counts scores against that compendium,
  in a 1,000-bp region (`finemo_region_width`) and batch 200. Output:
  - `<root>/cherimoya/<ds>/downstream/finemo_<tag>/<ds>_all/`, holding `hits.tsv`, `hits_unique.tsv`,
    `peaks_qc.tsv` and `hits.bed.gz`;
  - `compendium.md5`, which ties the hits to their compendium.

  A compendium of more than ~450 patterns needs a batch below 200 (10.0's header has the GPU memory rule).
- **11.0** (`finemo`): `finemo report` into `finemo_report/`.

**Measured:**
- 09.0: 1–2 min (AMSC's 39 datasets: 2,924 patterns into 583 clusters);
- 10.0: about 80 min per dataset on a GPU of its own (AMSC, 290k peaks; much longer on shared GPUs);
- 11.0: about 8–30 min (AMSC).

## Cooperativity (`cooperativity`)

**Command:**
```
rvi_run cherimoya python workflows/cooperativity/motif_names.py --compendium <root>/compendia/<tag> --out <coop>/motifs.tsv
rvi_run default python workflows/pipeline/project.py <project.yaml> coop-inputs
COOP_PYTHON=<cherimoya env python> bash workflows/cooperativity/lane.sh <gpu> <cpus> <coop>/logs <coop>/args/<ds>.args   # one per dataset
rvi_run cherimoya python workflows/cooperativity/combine.py --root <coop> --motifs <coop>/motifs.tsv \
    --groups <coop>/groups.tsv --comparisons <coop>/comparisons.tsv --official
```
with `<coop>` = `<root>/cherimoya/cooperativity`.

**What it is:** DeepISA (He et al. 2025), in-silico ablation with Cherimoya:
- In each of 50,000 sampled peaks, take at most 12 non-overlapping Fi-NeMo `pos_` sites.
- Set each site, then each pair, to N.
- Predict the log2 counts with all 5 fold models on both strands.
- A pair's interaction is i = a − a₋A − a₋B + a₋AB.

**The motif names.** `motif_names.py` names each compendium cluster by its most frequent annotation at or above the
compendium's minimum score, as the comparison names clusters. A cluster with none keeps its motif name.

**Groups and comparisons** come from `project.yaml`'s `cooperativity` block. By default each dataset is its own group.

**Two sets of definitions**, both computed:
- **Ours** (`combined/`): synergy = Σ(i > 0.1) / Σ(|i| > 0.1), with labels, a Poisson bootstrap over peaks, and the
  change per comparison.
- **The official deepISA package's** (`official/`): its own `calc_interaction` and `calc_coop_score`, on our
  ablations, exactly as its QuickStart calls them:
  - the q10 single-effect filter;
  - gaps of 1–255 bp;
  - Σi/Σ|i|;
  - the Mann–Whitney test against pairs more than 150 bp apart, then BH;
  - labels at the 70th and 30th percentiles.

  Changes across datasets use Σi/Σ|i| from its qualified pairs, because its labels are relative within a dataset.

`workflows/cooperativity/README.md` has the definitions and how they differ (section 4), and how the pipeline runs it
(section 8).

**Measured (AMSC):** about 6–7 min per dataset on one GPU (~0.7 M sequences, 5 folds), ~8 GB RSS. The per-dataset
runs go one per GPU at once.

## Changing a pinned version

The pins:

| what | where |
|---|---|
| igvf_tf_chrombpnet | `workflows/pipeline/igvf.pin` |
| Cherimoya, tangermeme, torch, deepISA | `pixi.toml` and `pixi.lock` |
| the frozen engine | `workflows/attribution/` and each project's `attribution_engine.json` |

To change one:
1. **Change it on a branch,** with the docs that describe it (this file, the stage READMEs).
2. **igvf_tf_chrombpnet:** push the new commit to `chrombpnet2` and put its full hash in `igvf.pin`. Install any
   igvf environment it changes before a box needs it: several first activations at once would each try to install
   it.
3. **pixi:** `pixi lock`, then check that environments you did not mean to change did not move. Diff each
   environment's package list in the lock before and after: `cherimoya-030` must stay as it was solved. Then `pixi
   install --locked -e <env>` on the login node.
4. **Test in a box,** by the preflight: it requires `pixi run check` and `pixi run -e cherimoya test-pipeline`, and
   runs `pixi run -e cherimoya test-cherimoya` beside the stages (in STATUS.md); run `pixi run -e cherimoya-030
   test-engine` too when the engine's environment is touched. Then run one project's stages on the new
   version before relying on it.
5. **Never edit** `attribute_fast.py`, `cheri_dls.py` or `dls_refs.py`. A changed byte makes every attribution the
   engine pin covers refuse to run.
