# Setting up a machine

This repository runs a project from fragment files to DeepISA on its own. The preprocessing and motif steps it
shares with igvf_tf_chrombpnet are copied into `vendor/igvf_tf_chrombpnet/` (its `VENDORED.md`). chrombpnet is one
of its pixi environments. No other checkout is needed. This page sets up a machine once; `docs/new_project.md`
then runs a project on it. On Gefion, `docs/cluster.md` has the local values; everything below applies there too.

## What the machine needs

- **Linux on x86-64.** The environments are built for `linux-64` only.
- **One or more NVIDIA GPUs, on a driver ≥ 580.** Cherimoya's torch and Fi-NeMo's are CUDA 13 builds. The pipeline
  was run on H100 80 GB. Training and attribution run 4 processes per GPU, each attribution at most ~12 GB; on a
  smaller GPU, lower `slots.train_per_gpu` and `slots.attribute_per_gpu`. The preflight refuses a machine with no
  GPU or an older driver.
- **CPUs and memory.** Nothing is sized for one machine. The box uses the CPUs it may use and the memory it has (the
  cgroup limit, else all of it), and scales the slots `project.yaml` leaves out to them (`project.py
  resolve_slots`).
  - The largest single needs: a prep chain, about 63 GB (on a 5-9 GB fragments file; smaller files need less), and
    a TF-MoDISco run, about 55 GB.
  - A machine of 16 CPUs and 64 GB runs one of each at a time, so it works, slowly. 128 GB or more is better.
  - A whole node of 8 H100, 216 CPUs and 2 TB (Gefion) ran a 5-dataset project, every stage, in 5 h 23 min.
- **Disk:**
  - about 20 GB for the environments;
  - 3.9 GB for the references;
  - per project, about 28 GB per dataset while it runs (139 GB for 5 datasets), about half that after `prune`.
- **System tools:**
  - bash 4 or later;
  - `/usr/bin/time` (GNU time: the package `time`);
  - `taskset` and `flock` (util-linux);
  - `pgrep` (procps);
  - `sha256sum`, `gzip`, `awk`, `git`, `curl`;
  - `nvidia-smi` (from the driver).

  No system Python is needed: the default environment's is used.
- **`/dev/shm`, writable.** The steps keep node-local scratch there (`/raid` too, if it exists and is writable). In a
  container, give it room, e.g. `docker run --shm-size=32g`.
- **Network for setting up, not for running:**
  - for the environments: conda-forge, bioconda, PyPI, and GitHub (git dependencies: deepISA, chrombpnet, the MACS3
    and TF-MoDISco forks, MotifCompendium);
  - for the references: the IGVF portal and ENCODE.
  - `fetch_igvf.py` needs the IGVF portal for inputs given as accessions. The box itself needs no network.

## Install

1. **pixi** (tested with 0.81):
   ```bash
   curl -fsSL https://pixi.sh/install.sh | bash      # into ~/.pixi/bin; the scripts look there
   ```
2. **This repository** (private: your GitHub account needs access to NNFC-GMD):
   ```bash
   git clone git@github.com:NNFC-GMD/regulatory-variant-interpretation.git
   cd regulatory-variant-interpretation
   ```
3. **The environments and the references,** each from its lock file and md5-checked. It installs and downloads only,
   so a login node can run it. It is safe to rerun.
   ```bash
   bash workflows/pipeline/setup.sh --references /data/mimir/references
   ```
   - It installs eight environments:
     - this repository's `default`, `cherimoya` and `chrombpnet`;
     - from the copied steps' `vendor/igvf_tf_chrombpnet/pixi.toml`: `peaks`, `preprocess`, `modisco`,
       `motif-compendium` and `finemo`.
   - It then runs the copied steps' `download-references` into the directory: the IGVF GRCh38 genome
     (IGVFFI0653VCGH) and its `.fai`, the chrom sizes, the ENCODE blacklist (ENCFF356LFX), the MotifCompendium 1.0.19
     database and the GENCODE v43 TSSs.
   - Its last lines say what, if anything, is still missing. `--dry-run` prints the commands.
   - `--with-engine` adds `cherimoya-030`, which only the frozen attribution engine (`workflows/attribution/`) uses.

   That directory is every project's `reference_root`. One copy serves every project on the machine.
4. **Check** (on the machine that will run boxes, or inside a first box): `pixi run check` and `pixi run -e cherimoya
   test-pipeline`. Every box's preflight runs them anyway.

## Running

`docs/new_project.md` walks through a project. In short:
- **The project:** `project.yaml` (from `workflows/pipeline/project.example.yaml`) names the samples, the root and
  `reference_root`. Leave `slots:` out so they scale to the machine.
- **With Slurm:** add a `slurm:` block (partition, account, GPUs, CPUs, time limit), then `bash
  workflows/pipeline/submit.sh <project.yaml>`. One job is one whole node. Give `gpus:` wherever Slurm hands out
  GPUs: a job that asks for none sees none, and the preflight stops.
- **Without Slurm:** `bash workflows/pipeline/run_project.sh <project.yaml>` on the machine, under tmux or nohup.
  `CUDA_VISIBLE_DEVICES=0,1` keeps it to those GPUs; `taskset -c <cpus>` to those CPUs.
- **First,** `bash workflows/pipeline/run_project.sh <project.yaml> --dry-run` prints every command, safe anywhere.

The preflight logs the machine and the slots it chose. On one GPU, 16 CPUs and 64 GB that is, for example:

```
machine: GPUs 0 (driver 580.126.09); CPUs 16; memory 64 GB; scratch /dev/shm/<user>/pipeline.local.<pid>
slots: train 4/GPU x 4 cores; attribute 4/GPU x 4; prep 1 at once; TF-MoDISco 1 x 16 threads; Fi-NeMo 1 (batch 200);
  DeepISA 16 cores per lane
```

## Not covered

- **NVIDIA drivers below 580.** There is no CUDA 12 build of the `cherimoya` or `finemo` environments here.
- **Genomes other than GRCh38.** The references layout and the folds are igvf_tf_chrombpnet's, for hg38.
- **macOS and ARM.**
- **Fragments in a bucket.** Give IGVF accessions (`fetch_igvf.py` downloads them) or local paths.
- **ChromBPNet itself.** The comparison with ChromBPNet (`workflows/comparison/`) needs ChromBPNet models trained by
  igvf_tf_chrombpnet, and its per-fold motif QC needs that repository's checkout.
