"""workflows/train/run_box.sh hands each task the GPU with the fewest tasks running, so a short queue, or the tail of a
long one, spreads over every GPU. The tasks are a stub (CHERI_TASK_CMD) that sleeps and records its GPU and times."""

from __future__ import annotations

import csv
import os
import subprocess
from pathlib import Path

import pytest

RUN_BOX = Path(__file__).resolve().parents[1] / "workflows" / "train" / "run_box.sh"

STUB = """#!/bin/bash
# <workdir> <stages>: sleep the workdir's duration, recording the GPU and the start and end times
w="$1"
echo "${CUDA_VISIBLE_DEVICES} $(date +%s.%N)" > "${w}/start"
sleep "$(cat "${w}/duration")"
date +%s.%N > "${w}/end"
"""


def _run(tmp_path: Path, durations: list[float], gpus: str, slots: int) -> tuple[list[dict], list[dict]]:
    if len(os.sched_getaffinity(0)) < len(gpus.split(",")) * slots:
        pytest.skip("too few CPUs for one core per slot")
    rows = ["dataset\tfold\tworkdir\tinputs_ready\tchrombpnet_ready"]
    for k, d in enumerate(durations):
        w = tmp_path / "ds" / f"fold_{k}"
        w.mkdir(parents=True)
        (w / "duration").write_text(f"{d}\n")
        rows.append(f"ds\t{k}\t{w}\tyes\tno")
    (tmp_path / "tasks.tsv").write_text("\n".join(rows) + "\n")
    stub = tmp_path / "stub.sh"
    stub.write_text(STUB)
    env = {**os.environ, "CHERI_ROOT": str(tmp_path), "CHERI_GPUS": gpus, "CHERI_SLOTS_PER_GPU": str(slots),
           "CHERI_CORES_PER_SLOT": "1", "CHERI_STAGES": "fit", "CHERI_STAGGER": "0", "CHERI_TASK_CMD": str(stub),
           "CHERI_SCRATCH_ROOT": str(tmp_path / "scratch")}  # fmt: skip
    r = subprocess.run(["bash", str(RUN_BOX)], env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    box = next((tmp_path / "box").iterdir())
    claims = list(csv.DictReader(open(box / "claims.tsv"), delimiter="\t"))
    status = list(csv.DictReader(open(box / "status.tsv"), delimiter="\t"))
    return claims, status


def _intervals(tmp_path: Path, n: int) -> list[tuple[str, float, float]]:
    out = []
    for k in range(n):
        w = tmp_path / "ds" / f"fold_{k}"
        gpu, t0 = (w / "start").read_text().split()
        out.append((gpu, float(t0), float((w / "end").read_text())))
    return out


def _each_claim_took_a_least_loaded_gpu(claims: list[dict]) -> None:
    for c in claims:
        loads = dict(x.split(":") for x in c["loads"].split(","))
        assert int(loads[c["gpu"]]) == min(int(v) for v in loads.values()), c


def test_a_short_queue_spreads_over_every_gpu(tmp_path):
    """5 tasks, 2 GPUs of 4 slots: before, all 4 slots of one GPU could take 4 of them; now 3 and 2."""
    claims, status = _run(tmp_path, [2.0] * 5, "0,1", 4)
    assert len(status) == 5 and all(s["exit"] == "0" for s in status)
    _each_claim_took_a_least_loaded_gpu(claims)
    used = [g for g, _, _ in _intervals(tmp_path, 5)]
    assert sorted(used.count(g) for g in "01") == [2, 3]


def test_the_tail_goes_to_the_least_loaded_gpu_and_no_gpu_exceeds_its_slots(tmp_path):
    """Unequal tasks: whenever one is taken, its GPU has the fewest running, and never more than the slots."""
    durations = [3.0, 0.3, 3.0, 0.3, 1.0, 1.0, 0.5, 0.5]
    claims, status = _run(tmp_path, durations, "0,1,2", 2)
    assert len(status) == len(durations) and {s["gpu"] for s in status} <= {"0", "1", "2"}
    _each_claim_took_a_least_loaded_gpu(claims)
    spans = _intervals(tmp_path, len(durations))
    for gpu in "012":
        events = sorted([(t0, 1) for g, t0, _ in spans if g == gpu] + [(t1, -1) for g, _, t1 in spans if g == gpu])
        running = peak = 0
        for _, step in events:
            running += step
            peak = max(peak, running)
        assert peak <= 2, (gpu, peak)
