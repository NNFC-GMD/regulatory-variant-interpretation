# Running on Gefion (DCAI)

The rules every run here follows, and why. They apply to people and agents alike.

## A job is a whole node

- **Every Slurm job rents a whole dgx node.** That is 216 usable CPUs, about 2 TB of RAM and 8× H100 80 GB, billed
  per node-hour (about $24). Partition `defq`; `SBATCH_ACCOUNT=iu_0109` is required.
- **So work is packed into one box.** Never submit many small jobs. The driver (`workflows/pipeline/run_project.sh`)
  runs a project's stages inside one allocation, many tasks at once.
- **One box at a time.** Don't submit a second box, not even a held or chained one, while one is running or
  pending. `workflows/pipeline/submit.sh` refuses to. Feed more work into the running box instead.
- **Set a long `--time`.** Billing is the time the box actually runs, and a box killed at its limit loses the work
  in flight. `defq`'s default is 8 h. `run_project.sbatch` asks for 2 days and ends when the work does.
- **Users cannot raise a running job's time limit.** A pending job's limit can still be changed with `scontrol
  update`.

## The login node is tiny

8 cores and 15 GB of RAM. A heavy command there can kill the session for everyone on it.
- **Fine there:**
  - lint and format checks (`pixi run lint`, `pixi run ruff format --check workflows tests`, `pixi run check-bash`);
  - `git`;
  - small I/O: `fetch_igvf.py`, `pixi install`, `download-references`;
  - the dry run;
  - `project.py`, `status.py`.
- **Not there:**
  - tests (`pytest`, `pixi run check`, `test-cherimoya`, `test-engine`);
  - numpy, pandas or torch work, JAX;
  - anything that reads a whole dataset.

  The tests run in each box's preflight.

## Working inside a running box

- **Run a command in the box:** `srun --jobid <job> --overlap -N1 -n1 <cmd>`. The step sees all of the job's CPUs and
  GPUs, so pin it with `taskset -c <cpus>` (and `nice -n 19`) beside running work. `CUDA_VISIBLE_DEVICES` picks its
  GPU.
- **Never edit a bash script a running box is executing.** Bash reads a script as it goes, so an edit in place can
  run half-old, half-new code. Change it in the repository, commit, and resubmit; or, for a box-local script, write
  a copy and `mv` it over.
- **A box ends when its batch script does.** Scripts that hold a box use hold files (`<box>/.hold.*`).
  `run_project.sh` holds after a failed stage, for `hold_minutes`; see `docs/new_project.md`.

## Queue

- **The cluster is often full.** With no idle node a box can wait many hours. Priority is age plus fair share, so
  heavy recent use lowers it, and resubmitting resets the age.
- **Slurm's estimates are optimistic.** `scontrol show job <job>` (`StartTime`) and `sbatch --test-only` give a
  start that ignores other pending jobs. Tell people the expected start, in ET.

## Software

- **pixi** is at `~/.pixi/bin/pixi`, not on the default PATH. There are no bioinformatics modules: every tool comes
  from a pixi environment (`docs/pipeline.md`, Environments).
- **Install environments on the login node,** before a box needs them, with `pixi install --locked -e <env>`.
  Several first activations at once inside a box would each try to install.
- **NVIDIA driver ≥ 580:** the CUDA 13 environments (`cherimoya`, chrombpnet `cuda13`); `CONDA_OVERRIDE_CUDA=13.0`
  when solving on the login node.

## Times

People here work in US Eastern time. Report times in ET. The cluster clock is CET/CEST, six hours ahead most of the
year, so convert (`TZ=America/New_York date`). The pipeline's logs are already in ET.
