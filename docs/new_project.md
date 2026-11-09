# Starting a new project

How to take a set of fragment files through the whole pipeline: preprocessing, Cherimoya (5 folds), attribution,
TF-MoDISco, a compendium, Fi-NeMo and DeepISA. `docs/pipeline.md` explains each step; this page is the order to do
things in. Names below are placeholders (`example_course`, stems `t0`, `t1`, …).

Every command runs from this repository's checkout on Gefion. The login node is for these commands only; the work
runs in one box (`docs/cluster.md`).

## 1. Make the project directory and its project.yaml

One project is one directory, and nothing of it goes into this repository:

```bash
mkdir -p /dcai/projects/iu_0109/datasets/example_course
cp workflows/pipeline/project.example.yaml /dcai/projects/iu_0109/datasets/example_course/project.yaml
```

Edit it. `project.example.yaml` documents every key; the ones that matter first:
- **The names:** `project`, `root` (that directory) and `prefix`. A dataset is called prefix + stem, e.g. `ex_t0`.
- **`compendium`:** the tag of the project's one motif compendium.
- **`samples`:** for each, a `stem` and its `fragments`, either an IGVF file accession or the absolute path of a
  BGZF file named `<stem>.fragments.tsv.gz`. Optional: `md5`, `index` (the .tbi's accession), `extra` (other portal
  files to keep) and `label`.
- **`cooperativity`:** `groups` (by stem; by default one per sample) and `comparisons`, `[name, group_a, group_b]`,
  where a change is b − a. Write out every comparison you want, e.g. each time point against the first and each
  against the next. For a time course, add `series`: the groups in order, for the trends along them.
- **`stages`:** leave it out to run the default stages. Add `coop_matrix` to the list to also score every dataset's
  DeepISA sites under every dataset's models (paired changes and trends; docs/pipeline.md).

Check that it loads:
```bash
pixi run -e default python workflows/pipeline/project.py <root>/project.yaml show
```

## 2. Fetch the inputs (login node)

```bash
pixi run -e default python workflows/pipeline/fetch_igvf.py <root>/project.yaml
```

What it does:
- It downloads each accession from the IGVF portal, checks it against the portal's md5 and size, and checks that a
  fragments file is BGZF.
- The fragments go to `<root>/fragments/<stem>.fragments.tsv.gz` (with `.tbi`), the other files to `<root>/portal/`,
  and every file gets a row in `<root>/portal/manifest.tsv`.
- A file already in place with the right md5 is kept, so it can be rerun. `--dry-run` only checks.
- A local file is checked and linked in.
- Controlled-access files are refused: fetch those yourself and give their paths.

## 3. Dry run (login node)

```bash
bash workflows/pipeline/run_project.sh <root>/project.yaml --dry-run
```

It prints:
- each stage's state;
- the preflight's findings;
- every command the box would run, with its paths.

It writes nothing under the project. Fix whatever it reports. An uncommitted change in this checkout is reported
as an error: the box refuses to run from a dirty checkout, so the run can be reproduced.

## 4. Submit the box

```bash
bash workflows/pipeline/submit.sh <root>/project.yaml
```

Before `sbatch`, it checks:
1. this checkout has no uncommitted changes, and its HEAD is on a remote branch (push first);
2. `igvf_tf_chrombpnet` is at `workflows/pipeline/igvf.pin`, unmodified (also checked in the preflight);
3. the inputs are in place (`project.py check fetch`) and the dry run passes;
4. no other box of yours is running or pending (one box at a time).

It then submits `run_project.sbatch` (one whole node, `--time=2-00:00:00`). It prints the job ID, the log path and
Slurm's expected start in ET. That start is optimistic: with no idle node a box can wait hours. A box costs about
$24 per hour it runs; ask the people paying before you submit.

`--stages "train attribute"` runs only those stages; done stages are skipped anyway.

## 5. Watch it

- **The driver's log:** `tail -f <root>/box/slurm_<job>.log`. Every stage logs its start, a one-line summary, and
  done or not done.
- **`<root>/box/<job>/stages.tsv`:** a row per stage (state, exit, seconds, start in ET).
- **`<root>/box/<job>/logs/`:** one log per command, e.g. `prep.log`, `train.log`, `downstream.log`, `motifs.log`,
  `combine.log`, and `check.<stage>.log`, the outputs still missing.
- **The runners' own records:**
  - `<root>/cherimoya/box/<job>.<n>/` for training and attribution (`status.tsv`, per-fold logs, `resources.tsv`);
  - `<root>/box/<job>/cheri_downstream.tsv` for the downstream steps;
  - `<root>/box/<job>/motif_chain.tsv` for 09.0–11.0.
- **Where each stage stands, from the outputs**, at any time:
  ```bash
  pixi run -e default python workflows/pipeline/project.py <root>/project.yaml status
  pixi run -e default python workflows/pipeline/status.py <root>/project.yaml       # the STATUS.md text
  ```
- **The queue:** `squeue -u $USER` and `scontrol show job <job>`.

## 6. When a stage fails

The box holds for `hold_minutes` (60 by default) and logs what is missing. Then:
- **Look** at `<root>/box/<job>/logs/` and `check.<stage>.log`.
- **Fix it.** On the login node, edit a config or remove a broken output. Inside the box, run a command there:
  `srun --jobid <job> --overlap -N1 -n1 <cmd>` (pin it with `taskset -c <cpus>` beside running work).
- **Rerun the stage:** `touch <root>/box/<job>/retry`. Its done parts are skipped.
- **Or end the box:** `touch <root>/box/<job>/release`. It ends anyway when the hold runs out.

Never edit a script the running box is executing; bash reads scripts as it goes. Fix the code in the repository,
commit and push, and resubmit.

## 7. Resume, rerun

- **Resume:** submit again. Each stage whose outputs are complete is skipped, and the runners skip finished tasks
  inside a stage. So a box stopped halfway costs only what was not done.
- **Redo one dataset's step:** move its outputs aside, then resubmit with that stage, e.g. `--stages downstream`.
  Never delete them: the downstream steps rename an output that has no marker to `.incomplete.<time>` themselves.
- **A new compendium:** use a new `compendium` tag. A compendium directory is never rebuilt in place, and its hits
  are tied to it by `compendium.md5`.

## 8. The results

Under `<root>`, as `workflows/pipeline/project.py`'s docstring lays them out; `docs/results.md` says what each
file holds and how to read it. What to look at first:

| what | where |
|---|---|
| the summary | `STATUS.md`: shift, peaks, test counts Pearson and profile JSD per dataset, attribution (fast path), Fi-NeMo hits, DeepISA |
| peaks | `peaks/<ds>/<ds>.narrowPeak.gz`, and `call_peaks.json` (counts after each filter) |
| training-set QC | `chrombpnet/<ds>/plots/signal_qc/` (TSS enrichment, peaks vs background) |
| models | `cherimoya/<ds>/fold_<k>/cherimoya.fold_<k>.torch` and `.test.performance.tsv` |
| attributions | `cherimoya/<ds>/fold_<k>/attr.*` and, averaged over folds, `cherimoya/<ds>/downstream/contrib_scores/<ds>/` (h5 and bigwig) |
| TF-MoDISco | `cherimoya/<ds>/downstream/contrib_scores/<ds>/modisco/` (results h5 and report) |
| the compendium | `compendia/<tag>/` (`modisco_compendium_meta.tsv` names each cluster) |
| motif hits | `cherimoya/<ds>/downstream/finemo_<tag>/<ds>_all/` (`hits_unique.tsv`, `finemo_report/`) |
| cooperativity | `cherimoya/cooperativity/`: per dataset `<ds>/`; ours `combined/` (`changes_pairs.tsv`, `changes_tfs.tsv`, with a series `trend_*.tsv`); the official rules `official/` |
| coop_matrix | `cherimoya/cooperativity/matrix/`: `<sites>/<model>/` per run; `combined/` the paired changes and trends |

Write down anything particular to the project, such as decisions, problems and dates (in ET), in a `NOTES.md` in
its root, not in this repository.

## 9. When the project is finished

1. **Write its `README.md`** in the root, for the people who will use it: what the datasets are, the compendium,
   where any report is, known issues, and what is particular to it. `docs/results.md` covers what every project
   has, so link to it rather than repeat it.
2. **Prune it** (docs/pipeline.md, "When a project is finished"): the dry run writes a manifest; read it, check
   the totals per rule and the unclassified list, then apply it.
   ```bash
   nice -n 19 ionice -c3 pixi run -e default python workflows/pipeline/project.py <root>/project.yaml prune
   pixi run -e default python workflows/pipeline/project.py <root>/project.yaml prune --apply <root>/prune/manifest.<stamp>.tsv
   ```
   `project.py <root>/project.yaml status` should still show every stage done afterwards.
3. **Note it in `NOTES.md`:** the date, the manifest and what it freed.
4. **To share it** with someone without cluster access: `project.py <root>/project.yaml share --out <file.tar>`
   (ask first if the data are unpublished).
