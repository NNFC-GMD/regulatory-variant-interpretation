# regulatory-variant-interpretation

Sequence-to-function models of chromatin accessibility, trained to interpret
regulatory variants.

## Layout

- `workflows/cherimoya/` trains [Cherimoya](https://github.com/jmschrei/cherimoya)
  models on the inputs and chromosome folds the
  [igvf_tf_chrombpnet](https://github.com/IGVF/igvf_tf_chrombpnet) pipeline uses
  for ChromBPNet. It then scores both models on the same test peaks with the
  same code. See its README.
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
