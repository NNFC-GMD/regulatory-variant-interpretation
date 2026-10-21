#!/usr/bin/env python3
"""Pick the CPUs a step should be pinned to on a shared node: the least busy of those this process may use.

downstream.sh runs igvf_tf_chrombpnet's 06.0-08.0 inside a box where ChromBPNet's trainings, the attribution runners
and the motif chain already run, each pinned to its own cores (taskset). A step pinned to a fixed slice of the node
would land on theirs; this picks, among the CPUs this process may use (its affinity, so a caller confined by taskset
picks inside its own mask), the n with the lowest score

    busy fraction over a short sample of /proc/stat
    + 0.5 if a process of this user is pinned to a mask that holds the CPU and is narrower than this process's own
      (someone's step owns it, even if it is idle right now)
    + 2   if another step picked it through this script and is still running (a reservation, below)

ties broken by CPU number (all of them when n exceeds what this process may use). With --reserve, the pick is written to <dir>/<owner pid> under a lock held while sampling,
so two steps picking at once do not get the same CPUs. A reservation counts while its owner process is alive, and is
removed by the next pick once it is not. The owner is the process that then runs the step (downstream.sh's step
subshell, which execs it).

    pick_cores.py <n> [--reserve <dir> --owner <pid>] [--sample 1.0]

Prints the CPUs, comma-separated. Stdlib only.
"""

from __future__ import annotations

import argparse
import fcntl
import os
import sys
import time
from pathlib import Path

CLAIMED = 0.5
RESERVED = 2.0


def parse_cpus(text: str) -> set[int]:
    """'0-3,8,10-11' -> {0, 1, 2, 3, 8, 10, 11}."""
    out: set[int] = set()
    for part in text.strip().split(","):
        if not part:
            continue
        lo, _, hi = part.partition("-")
        out.update(range(int(lo), int(hi or lo) + 1))
    return out


def cpu_times(proc: Path) -> dict[int, tuple[int, int]]:
    """{cpu: (idle + iowait, total)} jiffies from /proc/stat."""
    out = {}
    for line in (proc / "stat").read_text().splitlines():
        name, *values = line.split()
        if name.startswith("cpu") and name[3:].isdigit():
            ticks = [int(v) for v in values[:8]]
            out[int(name[3:])] = (ticks[3] + ticks[4], sum(ticks))
    return out


def busy_fractions(proc: Path, seconds: float) -> dict[int, float]:
    before = cpu_times(proc)
    time.sleep(seconds)
    after = cpu_times(proc)
    out = {}
    for cpu, (idle1, total1) in after.items():
        idle0, total0 = before.get(cpu, (idle1, total1))
        span = total1 - total0
        out[cpu] = 1.0 - (idle1 - idle0) / span if span > 0 else 0.0
    return out


def claimed_cpus(proc: Path, uid: int, width: int) -> set[int]:
    """CPUs inside the mask of a process of `uid` pinned to fewer than `width` CPUs."""
    out: set[int] = set()
    for status in proc.glob("[0-9]*/status"):
        try:
            text = status.read_text()
        except OSError:
            continue  # the process ended
        owner, mask = None, None
        for line in text.splitlines():
            if line.startswith("Uid:"):
                owner = int(line.split()[1])
            elif line.startswith("Cpus_allowed_list:"):
                mask = parse_cpus(line.split(":", 1)[1])
        if owner == uid and mask and len(mask) < width:
            out |= mask
    return out


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def reserved_cpus(directory: Path) -> set[int]:
    """CPUs other running steps picked; reservations of ended owners are removed."""
    out: set[int] = set()
    for path in directory.iterdir():
        if not path.name.isdigit():
            continue
        if not alive(int(path.name)):
            path.unlink(missing_ok=True)
            continue
        try:
            out |= parse_cpus(path.read_text())
        except (OSError, ValueError):
            continue
    return out


def choose(n: int, allowed: set[int], busy: dict[int, float], claimed: set[int], reserved: set[int]) -> list[int]:
    """The n allowed CPUs of lowest score (see the module docstring), in CPU order; all of them if n is more."""
    n = min(n, len(allowed))

    def score(cpu: int) -> tuple[float, int]:
        return busy.get(cpu, 1.0) + CLAIMED * (cpu in claimed) + RESERVED * (cpu in reserved), cpu

    return sorted(sorted(allowed, key=score)[:n])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("n", type=int, help="CPUs to pick")
    ap.add_argument("--reserve", type=Path, help="reservation directory shared by the steps of one node")
    ap.add_argument("--owner", type=int, help="with --reserve: the pid that runs the step")
    ap.add_argument("--sample", type=float, default=1.0, help="seconds of /proc/stat to sample (default 1)")
    ap.add_argument("--proc", type=Path, default=Path("/proc"), help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.n <= 0:
        ap.error("n must be positive")
    if (args.reserve is None) != (args.owner is None):
        ap.error("--reserve and --owner go together")
    allowed = os.sched_getaffinity(0)
    lock = None
    if args.reserve is not None:
        args.reserve.mkdir(parents=True, exist_ok=True)
        lock = os.open(args.reserve / ".lock", os.O_WRONLY | os.O_CREAT, 0o644)
        fcntl.flock(lock, fcntl.LOCK_EX)
    try:
        busy = busy_fractions(args.proc, args.sample)
        claimed = claimed_cpus(args.proc, os.getuid(), len(allowed))
        reserved = reserved_cpus(args.reserve) if args.reserve is not None else set()
        cpus = choose(args.n, allowed, busy, claimed, reserved)
        text = ",".join(map(str, cpus))
        if args.reserve is not None:
            (args.reserve / str(args.owner)).write_text(text + "\n")
    finally:
        if lock is not None:
            os.close(lock)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
