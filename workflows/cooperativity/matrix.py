#!/usr/bin/env python3
"""The pipeline's coop_matrix stage: every dataset's DeepISA sites scored by every dataset's models, then the trends
along the project's series and the paired comparisons (paired.py).

After the cooperativity stage, <root> (coop_isa.py's --out-dir) holds args/<dataset>.args (project.py coop-inputs),
motifs.tsv, groups.tsv, comparisons.tsv and, when project.yaml's cooperativity block has a series, series.tsv. This
script:
  1. writes <root>/matrix/args/<sites>__<model>.args for every pair of datasets: <sites>'s args with <model>'s
     --models, --dataset <model> and --out-dir <root>/matrix/<sites>; coop_isa.py's sites depend only on the
     Fi-NeMo directory and the sampling settings, so the rows of one dataset's sites line up under every model;
  2. runs those not done yet (no run.json) with lane.sh, one lane per GPU, the runs dealt round-robin, each lane on
     its own --cores CPUs (logs in <root>/matrix/logs/);
  3. with a series: combine.py again on the datasets' own sites, now with --series (and --official), for the trends;
  4. paired.py --matrix <root>/matrix: the paired changes and trends, in <root>/matrix/combined/.
Exits 0 when every run has its run.json and steps 3 and 4 succeeded.

    matrix.py --root <coop_isa.py --out-dir> [--cores 16] [--official]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import combine as cb  # noqa: E402


def read_args(path: Path) -> list[str]:
    """An args file: one argument per line, blank lines and #-comments skipped (as lane.sh reads it)."""
    return [x.strip() for x in path.read_text().splitlines() if x.strip() and not x.strip().startswith("#")]


def option(args: list[str], name: str) -> str:
    return args[args.index(name) + 1]


def with_option(args: list[str], name: str, value: str) -> list[str]:
    out = list(args)
    out[out.index(name) + 1] = value
    return out


def write_matrix_args(root: Path, datasets: list[str]) -> list[tuple[str, str, Path]]:
    """(sites, model, args file) for every pair of datasets."""
    own = {d: read_args(root / "args" / f"{d}.args") for d in datasets}
    (root / "matrix" / "args").mkdir(parents=True, exist_ok=True)
    runs = []
    for x in datasets:
        for m in datasets:
            a = with_option(own[x], "--models", option(own[m], "--models"))
            a = with_option(with_option(a, "--dataset", m), "--out-dir", str(root / "matrix" / x))
            f = root / "matrix" / "args" / f"{x}__{m}.args"
            f.write_text("\n".join(a) + "\n")
            runs.append((x, m, f))
    return runs


def gpu_count() -> int:
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=False).stdout
    except OSError:
        return 1
    return max(1, sum(1 for line in out.splitlines() if line.startswith("GPU ")))


def cpu_blocks(n: int, size: int) -> list[str]:
    """n CPU lists of `size` CPUs each, from the CPUs this process may use (taskset -c syntax); fewer CPUs than n x
    size: the blocks share them."""
    cpus = sorted(os.sched_getaffinity(0))
    blocks = []
    for k in range(n):
        start = (k * size) % len(cpus)
        blocks.append(",".join(map(str, (cpus[start:] + cpus[:start])[:size])))
    return blocks


def run_lanes(root: Path, todo: list[tuple[str, str, Path]], n_gpus: int, cores: int) -> None:
    logs = root / "matrix" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    n = min(n_gpus, len(todo))
    lanes = [todo[k::n] for k in range(n)]
    env = {**os.environ, "COOP_PYTHON": sys.executable}
    procs = []
    for gpu, (lane, cpus) in enumerate(zip(lanes, cpu_blocks(n, cores))):
        files = [str(f) for _, _, f in lane]
        log = open(logs / f"lane_{gpu}.log", "w")  # closed after the wait
        print(f"lane {gpu}: CPUs {cpus}: {', '.join(Path(f).stem for f in files)}", flush=True)
        procs.append((subprocess.Popen(["bash", str(HERE / "lane.sh"), str(gpu), cpus, str(logs), *files],
                                       stdout=log, stderr=subprocess.STDOUT, env=env), log))  # fmt: skip
    for p, log in procs:
        p.wait()
        log.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True, help="coop_isa.py's --out-dir of the cooperativity stage")
    ap.add_argument("--cores", type=int, default=16, help="CPUs per lane")
    ap.add_argument("--gpus", type=int, help="lanes (default: the GPUs nvidia-smi lists)")
    ap.add_argument("--official", action="store_true", help="combine.py's --official in step 3")
    args = ap.parse_args(argv)
    root = args.root
    groups = {r["group"]: [d.strip() for d in r["datasets"].split(",")] for r in cb.read_tsv(root / "groups.tsv")}
    datasets = list(dict.fromkeys(d for g in groups.values() for d in g))
    runs = write_matrix_args(root, datasets)
    todo = [r for r in runs if not (root / "matrix" / r[0] / r[1] / "run.json").is_file()]
    print(f"{len(runs)} runs ({len(datasets)} datasets' sites x their models), {len(todo)} to do", flush=True)
    if todo:
        run_lanes(root, todo, args.gpus or gpu_count(), args.cores)
    missing = [f"{x} under {m}" for x, m, _ in runs if not (root / "matrix" / x / m / "run.json").is_file()]
    if missing:
        print(f"ERROR: {len(missing)} runs have no run.json: {', '.join(missing)}; see {root / 'matrix' / 'logs'}")
        return 1
    files = ["--motifs", str(root / "motifs.tsv"), "--groups", str(root / "groups.tsv"),
             "--comparisons", str(root / "comparisons.tsv")]  # fmt: skip
    series = ["--series", str(root / "series.tsv")] if (root / "series.tsv").is_file() else []
    rc = 0
    if series:
        cmd = [sys.executable, str(HERE / "combine.py"), "--root", str(root), *files, *series]
        print("$", " ".join(cmd + (["--official"] if args.official else [])), flush=True)
        rc |= subprocess.run(cmd + (["--official"] if args.official else []), check=False).returncode
    cmd = [sys.executable, str(HERE / "paired.py"), "--matrix", str(root / "matrix"), *files, *series]
    print("$", " ".join(cmd), flush=True)
    rc |= subprocess.run(cmd, check=False).returncode
    return 1 if rc else 0


if __name__ == "__main__":
    sys.exit(main())
