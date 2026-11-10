# vendor/igvf_tf_chrombpnet: igvf_tf_chrombpnet's steps, copied unchanged

The pipeline's preprocessing (steps 00.0-02.0) and its motif steps after Cherimoya's attributions (06.0-11.0) are
[IGVF/igvf_tf_chrombpnet](https://github.com/IGVF/igvf_tf_chrombpnet)'s, the same steps its ChromBPNet runs use. This
folder is a byte-for-byte copy of the files those steps need, from one commit of its branch `chrombpnet2`, so that
this repository runs from fragments to DeepISA on its own: no igvf_tf_chrombpnet checkout and no chrombpnet checkout.

- **The commit** is in `MANIFEST.tsv`, with the sha256 of every file. The driver's preflight
  (`workflows/pipeline/run_project.sh`), `submit.sh` and `tests/test_vendor.py` refuse a file that differs from
  its row.
- **The environments** are the copy's own `pixi.toml` and `pixi.lock`, the ones its steps were run with:
  `bash workflows/pipeline/setup.sh` installs `peaks`, `preprocess`, `modisco`, `motif-compendium` and `finemo` from
  them. chrombpnet, which 01.0, 06.0, 07.0 and `detect_shifts.py` run in, is this repository's `chrombpnet`
  environment: NNFC-GMD/chrombpnet installed from git at the commit `lib/bash/common.sh` names (`CHROMBPNET_REV`),
  with the versions that shape those steps' outputs pinned to that commit's own lock.
- **The references** come from `project.yaml`'s `reference_root`, which the driver exports as `REFERENCE_ROOT`. The
  copy's `src/cli.py download-references` fills it (`setup.sh --references <dir>`), and the driver takes the
  genome Cherimoya trains on from the same layout (`lib/python/utils/references.py`). The steps and the driver
  therefore read one genome.
- **The folds** are the copy's `folds/fold_<k>.json`, for 01.0 and for Cherimoya alike.

## What is copied, and what is not

Copied:
- **The steps:** `workflows/SLURM/` 00.0 (call_peaks, prepare_signal), 00.1, 01.0, 02.0 and 06.0-11.0. The driver
  runs them with `bash`, so their `#SBATCH` lines (Stanford's Sherlock) are never read.
- **What they source, call or import:** `lib/bash/`, `lib/python/utils/` and five scripts of `src/`.
- **The scripts that run them on one node:** `workflows/dcai/` `env.sh`, `detect_shifts.py`, `make_configs.py`,
  `run_chain.sh` and `motif_chain.sh`.
- **The folds, `pixi.toml` and `pixi.lock`.**

`workflows/pipeline/vendor_igvf.sh` lists every file.

Not copied:
- the ChromBPNet steps (03.x bias models, 04.x full models, 05.0 attributions);
- igvf_tf_chrombpnet's other box scripts;
- `utils/folds.py` and `utils/regions.py`, which only those steps import;
- its tests, configs and results;
- the `molab/`, `nextflow/` and `igvf3_cardiomyocyte/` folders.

`tests/test_vendor.py` checks that every file the copied scripts name or import is here. The only exceptions are
the ones the driver cannot reach:
- `run_chain.sh`'s bias branch, which `CHAIN_STAGES=prep` skips;
- the default `config/` and `results/` paths, which the driver always overrides;
- an error message's `status.sh`.

The environment `qc` of the copy's `pixi.toml` is not installed: nothing here uses it. `download-references` runs
in `preprocess`.

## Never edit these files here

A fix to a step goes to igvf_tf_chrombpnet (branch `chrombpnet2`), and the copy is then refreshed from the new
commit:

```bash
bash workflows/pipeline/vendor_igvf.sh <igvf_tf_chrombpnet clone> <commit on origin/chrombpnet2>
bash workflows/pipeline/setup.sh       # environments the new lock changed
```

Then run the verification box (docs/pipeline.md, "Changing a pinned version") before merging. A new commit
can change the steps' outputs; the box compares them with a finished project's. Commit `MANIFEST.tsv` and the files
together.

## Licence

igvf_tf_chrombpnet has no LICENSE file. Every commit that touched the copied files, up to the commit in
`MANIFEST.tsv`, is by this repository's maintainer, who chose to copy them here (2026-11-10). If igvf_tf_chrombpnet
adds a licence, copy it in with the next refresh (add it to `vendor_igvf.sh`'s list).
