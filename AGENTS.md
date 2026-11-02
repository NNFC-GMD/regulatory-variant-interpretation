# Instructions for coding agents

This repository (NNFC-GMD/regulatory-variant-interpretation) is the reference repository of **Mímir**, the NNFC
initiative to prepare chromatin datasets, train sequence-to-function models, score variants and interpret them. All
of its framework lives here: the end-to-end driver, every step that is not igvf_tf_chrombpnet's, and the
documentation. If something about how to run the pipeline is not written down here, write it down here.

## Where to start

- `docs/pipeline.md`: every step, what it does and why, its command, environment and cost.
- `docs/new_project.md`: starting and running a project, step by step.
- `docs/cluster.md`: the rules of the Gefion (DCAI) cluster.
- `workflows/pipeline/`: the driver: `project.py` (the layout and the stage checks), `fetch_igvf.py`, `submit.sh`,
  `run_project.sh`, `status.py`, `project.example.yaml`.

## Map

| folder | what |
|---|---|
| `workflows/pipeline/` | one project end to end, in one box |
| `workflows/train/` | Cherimoya training on igvf_tf_chrombpnet's inputs and folds; the ChromBPNet comparison per fold; the GPU runner (`run_box.sh`, `run_task.sh`) |
| `workflows/attribution/` | the attribution status (`attr_status.py`), the stock record (`stock_record.py`), and the frozen AMSC engine (`attribute_fast.py`, `cheri_dls.py`, `dls_refs.py`) with its validation harness |
| `workflows/downstream/` | attributions converted for igvf_tf_chrombpnet's 06.0–11.0, the driver of 06.0–08.0, per-fold motif QC |
| `workflows/comparison/` | ChromBPNet against Cherimoya, level by level, and the report |
| `workflows/variant_scoring/`, `workflows/nulls/`, `workflows/qtl/` | variant effects, their nulls, and the QTL checks |
| `workflows/cooperativity/` | DeepISA with Cherimoya, ours and the official rules |
| `docs/` | the documentation above |
| `tests/` | the test suite (no data needed; some tests need torch, pandas or a GPU) |

The preprocessing (00.0–02.0) and the motif steps (06.0–11.0) are igvf_tf_chrombpnet's, run unchanged at the commit
in `workflows/pipeline/igvf.pin`.

## Checking your work

- **On the login node, lint only:** `pixi run lint`, `pixi run ruff format --check workflows tests`, `pixi run
  check-bash`. Also the dry run, `bash workflows/pipeline/run_project.sh <project.yaml> --dry-run`.
- **Tests run only in a box:** `pixi run check` (default env), `pixi run -e cherimoya test-pipeline` (the tests of
  what the driver runs), `pixi run -e cherimoya test-cherimoya` (every test that needs torch, pandas or deepISA), and
  `pixi run -e cherimoya-030 test-engine` (the frozen engine's). Every box's preflight requires the first two and
  reports the third. Never run tests, numpy, pandas or torch on the login node.
- **New code comes with tests,** and with docs updated in the same change (`docs/pipeline.md`, the stage's
  README).

## Rules

- **No dataset names** (project names, sample names) in code, docs, tests or commit messages. They belong in each
  project's root (`project.yaml`, `NOTES.md`, `STATUS.md`). The one exception is `workflows/cooperativity/README.md`,
  the record of the AMSC DeepISA run.
- **Never edit `workflows/attribution/attribute_fast.py`, `cheri_dls.py` or `dls_refs.py`**, not even a comment.
  An engine pin hashes them, and a changed byte makes every attribution it covers refuse to run.
- **Results never go in the repository.** Data, configs and outputs live under project roots; scripts take paths
  as arguments.
- **Git:**
  - never `git add -A`; add files by name;
  - commit as `Eugenio Mattei <emattei@users.noreply.github.com>`, the message ending with `Co-Authored-By: Claude
    Opus 5.5 <noreply@anthropic.com>`;
  - push only to NNFC-GMD repositories, and to IGVF/igvf_tf_chrombpnet's branch `chrombpnet2`.
- **Ask the user first** before submitting a box, before opening a pull request upstream (e.g. jmschrei/*), and
  before anything that publishes.
- **Prefer the official tools** (upstream packages and their own commands) over rewrites. A rewrite needs a reason,
  written down.
- **Cluster rules:** see `docs/cluster.md` (one box at a time, the login node's limits, never edit a running
  script, times in ET).
- **Writing style for docs and reports:** plain, short sentences; concrete paths and commands; no overclaiming (a
  small difference is "about the same").
