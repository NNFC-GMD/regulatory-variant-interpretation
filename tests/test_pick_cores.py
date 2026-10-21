"""pick_cores.py: the CPUs a step is pinned to, from a fake /proc (stdlib only)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

WORKFLOWS = Path(__file__).resolve().parents[1] / "workflows"
sys.path[:0] = [str(WORKFLOWS / d) for d in ("train", "attribution", "downstream")]

import pick_cores as pc  # noqa: E402


def _stat(proc: Path, idle: dict[int, int], total: int = 1000) -> None:
    lines = ["cpu  1 2 3 4 5 6 7 8"]
    for cpu, value in sorted(idle.items()):
        lines.append(f"cpu{cpu} {total - value} 0 0 {value} 0 0 0 0")
    (proc / "stat").write_text("\n".join(lines) + "\n")


def _process(proc: Path, pid: int, uid: int, cpus: str) -> None:
    (proc / str(pid)).mkdir()
    (proc / str(pid) / "status").write_text(f"Name:\tx\nUid:\t{uid}\t{uid}\t{uid}\t{uid}\nCpus_allowed_list:\t{cpus}\n")


def test_parse_cpus():
    assert pc.parse_cpus("0-3,8,10-11\n") == {0, 1, 2, 3, 8, 10, 11}
    assert pc.parse_cpus("") == set()


def test_choose_prefers_idle_then_unclaimed_then_unreserved():
    allowed = set(range(8))
    busy = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.9, 4: 0.1, 5: 0.0, 6: 0.0, 7: 0.0}
    # 0-2 idle and free; 5 idle but in someone's mask (0.5); 6 reserved by a running step (2.0); 7 both
    assert pc.choose(4, allowed, busy, claimed={5, 7}, reserved={6, 7}) == [0, 1, 2, 4]
    assert pc.choose(5, allowed, busy, claimed={5, 7}, reserved={6, 7}) == [0, 1, 2, 4, 5]
    assert pc.choose(20, allowed, busy, set(), set()) == list(range(8))  # capped at what is allowed


def test_busy_fractions_and_claims_from_proc(tmp_path, monkeypatch):
    proc = tmp_path / "proc"
    proc.mkdir()
    _stat(proc, {0: 100, 1: 900})
    assert pc.cpu_times(proc) == {0: (100, 1000), 1: (900, 1000)}  # (idle + iowait, total); the "cpu" line skipped
    samples = iter([{0: (100, 1000), 1: (900, 1000)}, {0: (105, 1100), 1: (995, 1100)}])
    monkeypatch.setattr(pc, "cpu_times", lambda p: next(samples))
    busy = pc.busy_fractions(proc, 0)
    assert abs(busy[0] - 0.95) < 1e-9 and abs(busy[1] - 0.05) < 1e-9
    uid = os.getuid()
    _process(proc, 10, uid, "4-7")
    _process(proc, 11, uid, "0-63")  # as wide as the caller: not a claim
    _process(proc, 12, uid + 1, "8-9")  # someone else's
    assert pc.claimed_cpus(proc, uid, width=64) == {4, 5, 6, 7}


def test_reservations_of_ended_steps_are_dropped(tmp_path):
    (tmp_path / str(os.getpid())).write_text("1,2\n")
    dead = subprocess.Popen(["true"])
    dead.wait()
    (tmp_path / str(dead.pid)).write_text("3,4\n")
    assert pc.reserved_cpus(tmp_path) == {1, 2}
    assert not (tmp_path / str(dead.pid)).exists()


def test_the_command_line_reserves_what_it_picks(tmp_path):
    allowed = sorted(os.sched_getaffinity(0))
    cmd = [sys.executable, str(WORKFLOWS / "downstream" / "pick_cores.py"), "1", "--sample", "0.01",
           "--reserve", str(tmp_path), "--owner", str(os.getpid())]  # fmt: skip
    first = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    assert int(first) in allowed
    assert (tmp_path / str(os.getpid())).read_text().strip() == first
    if len(allowed) > 1:
        other = subprocess.run([*cmd[:-1], str(os.getppid())], capture_output=True, text=True, check=True)
        assert other.stdout.strip() != first  # the first pick is reserved while its owner lives
