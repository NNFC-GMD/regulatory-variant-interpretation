"""The A/B benchmark driver workflows/cherimoya/dls_ab.py and the runner it shares with dls_e2e.py (CPU only; no torch).

The benchmark's numbers rest on these pieces, checked here:
- OutputTee copies a child's output byte for byte and dates every tqdm update, including updates split across reads
  and bars mixed with warnings; tqdm's elapsed field is read in both its formats.
- steady_rate is the least-squares slope of the pair counter after the first 5 batches; on the real E9 stock log it
  gives the rate agreement.py reads from the same log.
- /proc/schedstat parsing and the CPU accounting; the projections (the end-to-end one gives back the run's own wall
  time at the run's own size); the speed-ups and G1's shared-GPU verdict.
- prepare writes JSONs that differ only in loci and outputs; run and report work end to end with fake tools in place
  of the stock command and attribute_fast.py.
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
import threading
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workflows" / "cherimoya"))

import dls_ab  # noqa: E402
import dls_e2e  # noqa: E402

E9_STOCK_LOG = Path("/dcai/projects/iu_0109/datasets/amsc/cherimoya/analyses/attr_pilot/e9/stock.log")


def _bar(n: int, total: int, elapsed: str = "00:01", desc: str = "") -> str:
    pct = int(100 * n / total)
    head = f"{desc}: " if desc else ""
    return f"\r{head}{pct:3d}%|####      | {n}/{total} [{elapsed}<00:02, 19.6it/s]"


def test_elapsed_field_in_both_formats():
    assert dls_e2e._elapsed_s("00:35<49:06:27,  1.77s/it") == 35.0
    assert dls_e2e._elapsed_s("1:02:03<00:00, 109.39it/s") == 3723.0
    assert dls_e2e._elapsed_s("?<?, ?it/s") is None


def test_feed_dates_updates_split_across_reads_and_mixed_with_warnings():
    tee = dls_e2e.OutputTee(None, None)
    warning = (
        "/x/deep_lift_shap.py:745: RuntimeWarning: Convergence deltas too high: tensor([1.2e-04,\n"
        "        2.0e-03], device='cuda:0')\n"
        '  warnings.warn("Convergence deltas too high: " +\n'
    )
    loading = _bar(5, 5, "00:00", desc="Loading Loci") + "\n"
    first = _bar(0, 100, "00:00") + _bar(20, 100, "00:35")
    tee.feed(loading + first[:-9], 1.0)  # the 20/100 update is cut inside its statistics
    tee.feed(first[-9:] + warning, 2.0)
    tee.feed(_bar(40, 100, "00:36") + _bar(40, 100, "00:36") + _bar(100, 100, "00:37"), 3.0)
    assert tee.updates == [
        (1.0, "Loading Loci", 5, 5, 0.0),
        (1.0, "", 0, 100, 0.0),
        (2.0, "", 20, 100, 35.0),
        (3.0, "", 40, 100, 36.0),
        (3.0, "", 40, 100, 36.0),
        (3.0, "", 100, 100, 37.0),
    ]
    assert len(tee._buffer) < 10  # only the open line is kept


def test_tee_copies_the_pipe_byte_for_byte(tmp_path):
    read_fd, write_fd = os.pipe()
    payload = ("Loading Loci: 100%|█| 3/3 [00:00<00:00]\n" + _bar(0, 60) + _bar(20, 60) + "é ok\n").encode()
    times = iter([10.0, 11.0, 12.0, 13.0, 14.0])
    with open(read_fd, "rb", buffering=0) as pipe, open(tmp_path / "log", "wb") as log:
        tee = dls_e2e.OutputTee(pipe, log, clock=lambda: next(times))
        tee.start()
        # split inside a multi-byte character: the decoder must hold it over
        cut = payload.index("█".encode()) + 1
        os.write(write_fd, payload[:cut])
        for _ in range(500):  # until the tee has read and logged the first chunk, so the reads stay apart
            if os.path.getsize(tmp_path / "log") == cut:
                break
            threading.Event().wait(0.01)
        os.write(write_fd, payload[cut:])
        os.close(write_fd)
        tee.join(timeout=10)
    assert (tmp_path / "log").read_bytes() == payload
    assert [(desc, n, total) for _, desc, n, total, _ in tee.updates] == [
        ("Loading Loci", 3, 3),
        ("", 0, 60),
        ("", 20, 60),
    ]
    assert [t for t, *_ in tee.updates] == [11.0, 11.0, 11.0]  # all three completed in the second read


def test_steady_rate_skips_the_first_five_batches():
    # bar at t = 10, first batch at 45 (35 s warm-up), then 125 pairs/s; the last value printed twice
    points = [(10.0, 0), (45.0, 20)] + [(45.0 + (n - 20) / 125, n) for n in range(40, 2001, 20)]
    points.append((points[-1][0] + 0.5, 2000))
    s = dls_ab.steady_rate(points, batch_size=20)
    assert s["bar_start_s"] == 10.0 and s["first_batch_s"] == pytest.approx(35.0)
    assert s["window_start_n"] == 100 and s["n_end"] == 2000 and s["window_pairs"] == 1900
    assert s["rate_ls"] == pytest.approx(125.0) and s["rate_two_point"] == pytest.approx(125.0)
    # jitter of the arrival times moves the slope by far less than it moves single intervals
    rng = np.random.default_rng(0)
    noisy = [(t + (rng.uniform(-0.05, 0.05) if n >= 40 else 0.0), n) for t, n in points]
    assert dls_ab.steady_rate(noisy, batch_size=20)["rate_ls"] == pytest.approx(125.0, rel=2e-3)
    assert dls_ab.steady_rate([(0.0, 0), (1.0, 20)], batch_size=20)["rate_ls"] is None


@pytest.mark.skipif(not E9_STOCK_LOG.exists(), reason="the E9 stock log is not on this machine")
def test_stock_steady_on_the_e9_stock_log():
    tee = dls_e2e.OutputTee(None, None)
    tee.feed(E9_STOCK_LOG.read_text(errors="replace"), 0.0)
    total, bar = dls_ab.attribution_bar([list(u) for u in tee.updates])
    assert total == 99980
    assert sorted({n for _, n, _ in bar}) == list(range(0, 99981, 20))  # every batch of 20 pairs was printed
    s = dls_ab.stock_steady([list(u) for u in tee.updates], batch_size=20)
    assert s["pairs_total"] == 99980
    e = s["tqdm_elapsed"]
    assert e["first_batch_s"] == 35.0 and e["n_end"] == 99980
    assert e["rate_ls"] == pytest.approx(114.3, abs=0.1)  # agreement.py's figure for this log
    assert e["rate_two_point"] == pytest.approx(e["rate_ls"], rel=0.02)


def test_schedstat_and_cpu_accounting(tmp_path):
    path = tmp_path / "schedstat"
    path.write_text(
        "version 15\ntimestamp 1\ncpu215 0 0 0 0 0 0 9 9 9\n"
        "cpu216 0 0 0 0 0 0 1000000000 500000000 7\ncpu217 0 0 0 0 0 0 2000000000 0 7\ndomain0 ff 1 2 3\n"
    )
    before = dls_ab.read_schedstat([216, 217], str(path))
    assert before == {216: (1_000_000_000, 500_000_000), 217: (2_000_000_000, 0)}
    after = {216: (11_000_000_000, 1_500_000_000), 217: (12_000_000_000, 1_000_000_000)}
    c = dls_ab.cpu_account(before, after, tool_cpu_s=15.0, driver_cpu_s=1.0, wall_s=10.0)
    assert c["busy_s"] == pytest.approx(20.0) and c["run_delay_s"] == pytest.approx(2.0)
    assert c["other_cpu_s"] == pytest.approx(4.0) and c["other_cpu_cores"] == pytest.approx(0.4)
    assert c["run_delay_cores"] == pytest.approx(0.2) and c["tool_cpu_cores"] == pytest.approx(1.5)


def test_projection_extends_the_measured_run():
    row = {"steady_pairs_per_s": 1000.0, "pairs": 99980, "peaks": 4999, "wall_s": 136.0, "per_peak_s": 1e-3}
    p = dls_ab.project(row, fold_peaks=290_000, models=195, n_shuffles=20)
    assert p["fold_pairs"] == 5_800_000
    assert p["steady_min"] == pytest.approx(5_800_000 / 1000 / 60)
    e2e_s = 136.0 + (5_800_000 - 99980) / 1000 + 1e-3 * (290_000 - 4999)
    assert p["end_to_end_min"] == pytest.approx(e2e_s / 60)
    assert p["end_to_end_gpu_h"] == pytest.approx(e2e_s * 195 / 3600)
    assert p["steady_gpu_h"] == pytest.approx(5800 * 195 / 3600)
    # at the run's own size the end-to-end projection is the run's wall time
    row_full = dict(row, pairs=4999 * 20)
    assert dls_ab.project(row_full, 4999, 1, 20)["end_to_end_min"] == pytest.approx(136.0 / 60)


def test_speedups_and_the_g1_verdict():
    rows = [
        {"tool": "stock", "rep": 1, "steady_pairs_per_s": 100.0},
        {"tool": "fast", "rep": 1, "steady_pairs_per_s": 900.0},
        {"tool": "stock", "rep": 2, "steady_pairs_per_s": 125.0},
        {"tool": "fast", "rep": 2, "steady_pairs_per_s": 1000.0},
    ]
    s = dls_ab.speedups(rows)
    assert s["stock_mean"] == pytest.approx(112.5) and s["fast_mean"] == pytest.approx(950.0)
    assert s["ratio_of_means"] == pytest.approx(950 / 112.5)
    assert s["per_rep"] == {1: pytest.approx(9.0), 2: pytest.approx(8.0)}
    assert s["min"] == pytest.approx(7.2) and s["max"] == pytest.approx(10.0)
    assert dls_ab.g1_shared(s)
    rows[3]["steady_pairs_per_s"] = 600.0  # rep 2 at 4.8x: the mean passes (6.67x) but a repetition does not
    assert dls_ab.speedups(rows)["ratio_of_means"] > 5 and not dls_ab.g1_shared(dls_ab.speedups(rows))
    assert dls_ab.speedups(rows[:1]) == {} and not dls_ab.g1_shared({})
    rows[0]["steady_pairs_per_s"] = None  # a run without a steady rate is left out
    assert dls_ab.speedups(rows)["per_rep"] == {2: pytest.approx(4.8)}


# ---------------------------------------------------------------------------------------------------------------------
# prepare, run and report with fake tools

FAKE_STOCK = textwrap.dedent(
    """
    import json, sys, time
    import numpy as np
    p = json.load(open(sys.argv[1]))
    lines = open(p["loci"][0]).read().splitlines()
    k, b = p["n_shuffles"], p["batch_size"]
    pairs = len(lines) * k
    w = sys.stderr.write
    w("Loading Loci: 100%%|##########| %d/%d [00:00<00:00, 999.0it/s]\\n" % (len(lines), len(lines)))
    w("\\r  0%%|          | 0/%d [00:00<?, ?it/s]" % pairs); sys.stderr.flush()
    time.sleep(0.15)
    for n in range(b, pairs + 1, b):
        w("\\r%3d%%|#####     | %d/%d [00:00<00:00, 9.9it/s]" % (100 * n // pairs, n, pairs)); sys.stderr.flush()
        if n == 2 * b:
            w("RuntimeWarning: Convergence deltas too high: tensor([0.0011])\\n")
        time.sleep(0.02)
    w("\\r100%%|##########| %d/%d [00:00<00:00, 9.9it/s]\\n" % (pairs, pairs))
    rng = np.random.default_rng(0)  # 10 fixed rows; a run keeps the rows of its peaks
    ohe = np.eye(4, dtype=np.int8)[rng.integers(0, 4, (10, 8))].transpose(0, 2, 1)[: len(lines)].copy()
    attr = rng.normal(size=(10, 4, 8)).astype(np.float32)[: len(lines)]
    np.savez_compressed(p["ohe_filename"], ohe)
    np.savez_compressed(p["attr_filename"], attr)
    np.save(p["idx_filename"], np.ones(len(lines), dtype=bool))
    """
)

FAKE_FAST = textwrap.dedent(
    """
    import json, os, sys
    import numpy as np
    p = json.load(open(sys.argv[1]))
    lines = open(p["loci"][0]).read().splitlines()
    k = p["n_shuffles"]
    rng = np.random.default_rng(0)  # the fake stock's 10 rows: equal on the shared peaks
    ohe = np.eye(4, dtype=np.int8)[rng.integers(0, 4, (10, 8))].transpose(0, 2, 1)[: len(lines)].copy()
    attr = rng.normal(size=(10, 4, 8)).astype(np.float32)[: len(lines)]
    attr *= np.float32(1 + 1e-5)
    cache = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    if cache:  # a compile leaves files: the next run starts warm
        open(os.path.join(cache, "kernel.bin"), "w").write("x")
    np.savez_compressed(p["ohe_filename"], ohe)
    np.savez_compressed(p["attr_filename"], attr)
    np.save(p["idx_filename"], np.ones(len(lines), dtype=bool))
    meta = {
        "precision": "reference", "seqs_per_step": 8, "head_rows": 40, "compile": "default", "n_shuffles": k,
        "inputs": {"kept": len(lines)},
        "timings_s": {"imports": 2.0, "load": 0.1, "extract": 0.2, "self_check": 10.0, "run": 30.0, "write": 1.0,
                      "compile": 5.0, "steady_pairs_per_s": 50000.0, "total": 43.3},
        "memory": {"gpu_max_allocated_bytes": 6e9, "gpu_max_reserved_bytes": 6.5e9},
        "engine": {"steps": 2, "padded_peaks": 6, "host_syncs": 2, "oom": [],
                   "reference_stats": {"stalls": 0, "wait_s": 0.0}},
        "deltas": {"pairs": len(lines) * k, "above": 0, "max": 1e-4, "threshold": 1e-3},
        "self_check": {"pass": True, "max_abs": 1e-6, "seconds": 10.0},
        "gpu": {"name": "fake GPU", "driver": "0", "CUDA_VISIBLE_DEVICES": ""}, "code": {"git_commit": "x"},
    }
    json.dump(meta, open(p["attr_filename"][: -len(".npz")] + ".meta.json", "w"))
    """
)


def _prepare(tmp_path: Path) -> Path:
    peaks = tmp_path / "peaks.narrowPeak"
    peaks.write_text("".join(f"chr1\t{100 * i}\t{100 * i + 50}\tp{i}\n" for i in range(10)))
    base = {
        "model": "m.torch",
        "loci": [str(peaks)],
        "batch_size": 64,
        "n_shuffles": 20,
        "ohe_filename": "/elsewhere/attr.ohe.npz",
        "attr_filename": "/elsewhere/attr.attr.npz",
        "idx_filename": "/elsewhere/attr.idx.npy",
    }
    (tmp_path / "attribute.json").write_text(json.dumps(base))
    out = tmp_path / "c1"
    argv = [
        "prepare",
        "-p",
        str(tmp_path / "attribute.json"),
        "--out-dir",
        str(out),
        "--stock-peaks",
        "8",
        "--set",
        "batch_size=20",
        "--fast",
        '{"ref_workers": 3}',
        "--fast-env",
        f"TORCHINDUCTOR_CACHE_DIR={tmp_path / 'cache' / 'inductor'}",
        "--stock-env",
        "OMP_NUM_THREADS=4",
    ]
    assert dls_ab.main(argv) == 0
    return out


def test_prepare_writes_one_json_per_run(tmp_path):
    out = _prepare(tmp_path)
    plan = json.loads((out / "plan.json").read_text())
    assert [(r["tag"], r["tool"], r["rep"]) for r in plan["runs"]] == [
        ("1_stock", "stock", 1),
        ("2_fast", "fast", 1),
        ("3_stock", "stock", 2),
        ("4_fast", "fast", 2),
    ]
    assert plan["differ_only_in"] == ["attr_filename", "idx_filename", "loci", "ohe_filename"]
    params = {r["tag"]: json.loads(Path(r["json"]).read_text()) for r in plan["runs"]}
    assert all(p["batch_size"] == 20 and p["fast"] == {"ref_workers": 3} for p in params.values())
    stock_peaks = Path(params["1_stock"]["loci"][0])
    assert stock_peaks.name == "peaks_first8.narrowPeak" and len(stock_peaks.read_text().splitlines()) == 8
    assert params["2_fast"]["loci"] == [str((tmp_path / "peaks.narrowPeak").resolve())]
    assert params["3_stock"]["attr_filename"] == str(out / "3_stock" / "attr.attr.npz")
    stock_cmd, fast_cmd = plan["runs"][0]["command"], plan["runs"][1]["command"]
    assert Path(stock_cmd[0]).name == "cherimoya" and stock_cmd[1:3] == ["attribute", "-p"]
    assert fast_cmd[1].endswith("attribute_fast.py") and fast_cmd[2] == "-p"
    assert plan["runs"][1]["env"] == {"TORCHINDUCTOR_CACHE_DIR": str(tmp_path / "cache" / "inductor")}
    assert plan["runs"][0]["env"] == {"OMP_NUM_THREADS": "4"}


def test_run_and_report_end_to_end_with_fake_tools(tmp_path, monkeypatch):
    out = _prepare(tmp_path)
    plan = json.loads((out / "plan.json").read_text())
    for name, code in (("fake_stock.py", FAKE_STOCK), ("fake_fast.py", FAKE_FAST)):
        (tmp_path / name).write_text(code)
    for r in plan["runs"]:
        r["command"] = [sys.executable, str(tmp_path / f"fake_{r['tool']}.py"), r["json"]]
    (out / "plan.json").write_text(json.dumps(plan))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert dls_ab.main(["run", "--out-dir", str(out), "--poll-s", "30"]) == 0

    runs = json.loads((out / "runs.json").read_text())
    assert list(runs["runs"]) == ["1_stock", "2_fast", "3_stock", "4_fast"]
    s1, f2 = runs["runs"]["1_stock"], runs["runs"]["2_fast"]
    assert s1["returncode"] == 0 and s1["gpu_before"] == [] and runs["setups"][0]["gpu_uuid"] is None
    host = s1["steady"]["host_clock"]
    assert s1["steady"]["pairs_total"] == 160 and host["n_end"] == 160 and host["window_start_n"] == 100
    assert host["first_batch_s"] >= 0.1 and 100 < host["rate_ls"] < 2000  # 20 pairs per 0.02 s sleep
    assert s1["cache_files_before"] == {} and f2["cache_files_before"] == {"TORCHINDUCTOR_CACHE_DIR": 0}
    assert f2["meta"]["timings_s"]["steady_pairs_per_s"] == 50000.0
    assert set(s1["cpu"]) >= {"busy_s", "other_cpu_s", "run_delay_s", "tool_cpu_s"}
    log = (out / "1_stock.log").read_text()
    assert log.startswith("Loading Loci: 100%") and "RuntimeWarning" in log and "160/160" in log
    progress = (out / "1_stock.progress.tsv").read_text().splitlines()
    assert progress[0] == "t_s\tdesc\tn\ttotal\telapsed_s" and len(progress) == 1 + 1 + 1 + 8 + 1

    tsv, md = tmp_path / "c1.tsv", tmp_path / "c1.md"
    argv = ["report", "--out-dir", str(out), "--condition", "C1", "--tsv", str(tsv), "--md", str(md)]
    notes = tmp_path / "notes.md"
    notes.write_text("## Deviations\n\n- a note\n")
    assert dls_ab.main([*argv, "--notes", str(notes)]) == 0
    table = [line.split("\t") for line in tsv.read_text().splitlines()]
    assert tuple(table[0]) == dls_ab.COLUMNS and len(table) == 5
    row = dict(zip(table[0], table[2]))
    assert row["tool"] == "fast" and row["peaks"] == "10" and row["pairs"] == "200" and row["cache"] == "cold"
    assert row["steady_pairs_per_s"] == "50000" and row["S"] == "8" and row["gpu_idle_pct"] == "NA"
    assert dict(zip(table[0], table[4]))["cache"] == "warm"  # the first fast run left a file in its cache
    assert dict(zip(table[0], table[1]))["cache"] == "-"  # stock was given no cache directories
    report = json.loads(md.with_suffix(".json").read_text())
    assert report["g1_shared_pass"] and report["speedup"]["ratio_of_means"] > 5
    agree = {a["label"]: a for a in report["agreement"]}
    assert agree["stock rerun"]["rel_l2"]["max"] == 0.0 and agree["stock rerun"]["peaks"] == 8
    assert agree["ours vs stock"]["ohe_equal"] and agree["ours vs stock"]["peaks"] == 8
    assert agree["ours vs stock"]["rel_l2"]["max"] == pytest.approx(1e-5, rel=1e-2)
    text = md.read_text()
    assert "**shared GPU**" in text and "G1, shared-GPU clause" in text and "**PASS**" in text
    assert "## Projections (C1 rates, shared GPU)" in text and "## Deviations" in text
    runs_table = [line for line in text.splitlines() if line.startswith("| 2 | fast |")]
    assert len(runs_table) == 2 and runs_table[0].split(" | ")[9] == "10.0"  # self-check s, one decimal
    assert [line for line in text.splitlines() if line.startswith("| 1 | stock |")][0].split(" | ")[9] == "-"
