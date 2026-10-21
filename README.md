# regulatory-variant-interpretation

Sequence-to-function models of chromatin accessibility, trained to interpret
regulatory variants.

## Layout

`workflows/` holds one folder per stage. Each script's header says what it reads and writes.

- `train/` trains [Cherimoya](https://github.com/jmschrei/cherimoya) models on
  the inputs and chromosome folds the
  [igvf_tf_chrombpnet](https://github.com/IGVF/igvf_tf_chrombpnet) pipeline uses
  for ChromBPNet. It then scores both models on the same test peaks with the
  same code. See its README.
- `attribution/`: DeepLIFT/SHAP for Cherimoya (`attribute_fast.py` and its
  engine, `cheri_dls.py` and `dls_refs.py`), the status and engine pin of each
  fold's attribution (`attr_status.py`), and the engine's validation harness
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
- `tests/`: `pixi run test`. These tests need no GPU and no data.
- `pixi.toml` / `pixi.lock`: environments.
  - `default`: lint and tests.
  - `cherimoya`: training and scoring, CUDA 13 torch.

## Commands

```bash
pixi install                 # default environment: lint and tests
pixi install -e cherimoya    # training (NVIDIA driver >= 580)
pixi run check               # bash -n + shellcheck, ruff, pytest
```

No data or results are tracked. Datasets, their configurations and every
output live outside the repository, and the scripts take their paths as
arguments.
