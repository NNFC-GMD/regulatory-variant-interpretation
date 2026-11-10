# regulatory-variant-interpretation

Sequence-to-function models of chromatin accessibility, trained to interpret
regulatory variants. This is the reference repository of Mímir, the NNFC
initiative to prepare datasets, train models, score variants and interpret them:
its framework, driver and documentation live here.

## Start here

- [docs/install.md](docs/install.md): setting up a machine: pixi, one setup command (the environments and the
  references), with or without Slurm. This repository alone runs a project from fragments to DeepISA.
- [docs/pipeline.md](docs/pipeline.md): every step from fragment files to
  models, motifs and TF cooperativity: what it does and why, its command,
  environment and cost.
- [docs/new_project.md](docs/new_project.md): starting and running a project.
- [docs/results.md](docs/results.md): using a project's results: what each file holds and how to read it,
  pruning a finished project, and packing its results for someone outside the cluster.
- [docs/cluster.md](docs/cluster.md): the rules of the Gefion (DCAI) cluster.
- [AGENTS.md](AGENTS.md): instructions for coding agents.

## Layout

`workflows/` holds one folder per stage. Each script's header says what it reads and writes.

- `pipeline/`: the end-to-end driver for a project: its `project.yaml`, the
  machine's setup (`setup.sh`), the fetch from the IGVF portal, one box that runs every stage, and `STATUS.md`.
- `train/` trains [Cherimoya](https://github.com/jmschrei/cherimoya) models on
  the inputs and chromosome folds the
  [igvf_tf_chrombpnet](https://github.com/IGVF/igvf_tf_chrombpnet) pipeline uses
  for ChromBPNet. It then scores both models on the same test peaks with the
  same code. See its README.
- `attribution/`: DeepLIFT/SHAP for Cherimoya. New projects attribute with
  `cherimoya attribute` (Cherimoya 0.3.1's fast path), recorded by
  `stock_record.py`; the AMSC campaign's engine (`attribute_fast.py`,
  `cheri_dls.py`, `dls_refs.py`) is kept frozen beside it. Also the status and
  engine pin of each fold's attribution (`attr_status.py`), and the engine's
  validation harness
  (`dls_*.py`). The three engine files are kept byte for byte as signed off, because
  the engine pin compares their sha256: `attribute_fast.py`'s usage line still
  names its old folder, `workflows/cherimoya/`.
- `downstream/`: each fold's attribution converted for igvf_tf_chrombpnet's own
  steps after 05.0 (06.0-11.0: averages, bigwigs, TF-MoDISco, compendia,
  Fi-NeMo), the drivers that run them, and the per-fold motif QC. See its README.
- `comparison/`: ChromBPNet and Cherimoya compared level by level (performance,
  attributions, TF-MoDISco, compendia, Fi-NeMo), and the report. See its README.
- `variant_scoring/`: variant lists, windows and both models' predictions, the
  count-based variant scores, and the pilot's checks.
- `nulls/`: the genomic null for variant scores and the report on both nulls.
- `qtl/`: variant scores against measured QTL effects, scored at the variant
  and at the QTL's peak.
- `cooperativity/`: TF cooperativity by in-silico ablation of motif sites
  (DeepISA's definitions) with Cherimoya models. See its README.
- `vendor/igvf_tf_chrombpnet/`: igvf_tf_chrombpnet's preprocessing (00.0-02.0)
  and motif steps (06.0-11.0), copied unchanged from one commit, with their
  own `pixi.toml`/`pixi.lock` and the chromosome folds. Its `VENDORED.md` says
  what is copied and how to refresh it; `MANIFEST.tsv` hashes every file.
- `docs/`: the install guide, the pipeline reference, the new-project guide,
  the results, the cluster rules.
- `tests/`: the tests. They need no data; some need torch, pandas or a GPU, so
  they run in a box (`docs/cluster.md`).
- `pixi.toml` / `pixi.lock`: environments.
  - `default`: lint, the CPU tests, and the driver's helpers.
  - `cherimoya`: Cherimoya 0.3.1 (training, `cherimoya attribute`, scoring),
    the official deepISA package, CUDA 13 torch.
  - `chrombpnet`: chrombpnet (NNFC-GMD) on CPU, for the four pieces of it the
    vendored steps use (the shift cross-check, 01.0's background regions,
    06.0, 07.0).
  - `cherimoya-030`: Cherimoya 0.3.0 and tangermeme 1.5.0, only for the frozen
    `attribute_fast.py` engine and its tests.
  - The vendored steps' own: `peaks`, `preprocess`, `modisco`,
    `motif-compendium`, `finemo` (`vendor/igvf_tf_chrombpnet/pixi.toml`).

## Commands

```bash
bash workflows/pipeline/setup.sh --references <dir>   # every environment, and the references (docs/install.md)
bash workflows/pipeline/setup.sh --with-engine        # also cherimoya-030, the frozen attribution engine

# login node: lint only
pixi run lint && pixi run check-bash && pixi run ruff format --check workflows tests
# in a box: the tests
pixi run check                            # bash -n + shellcheck, ruff, pytest (default env)
pixi run -e cherimoya test-pipeline      # the tests of what workflows/pipeline/ runs (a box's preflight)
pixi run -e cherimoya test-cherimoya      # the tests that need torch, pandas or deepISA
pixi run -e cherimoya-030 test-engine     # the frozen engine's tests
```

No data or results are tracked. Datasets, their configurations and every
output live outside the repository, and the scripts take their paths as
arguments.
