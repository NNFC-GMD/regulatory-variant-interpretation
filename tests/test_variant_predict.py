"""The variant-scoring predictors: windows in, VS's per-strand log counts out, the same way for both models.

The model-free parts (predict_common.py, vchecks.py, read_manifest.py, probe_throughput.py) need numpy only and run
here. The model code needs torch (the pixi `cherimoya` environment) or Keras/JAX (the chrombpnet fork's `cuda13`
environment), and neither of those has pytest, so the backend tests run tests/variant_predict_backends.py in that
environment's interpreter, on the CPU, with tiny random models:
  VS_CHERIMOYA_PYTHON    the cherimoya environment's python (else this interpreter, if it has torch and cherimoya)
  VS_CHROMBPNET_PYTHON   the fork's cuda13 python (else this interpreter, if it has chrombpnet and keras)
A backend test is skipped when neither is available, and fails when its variable names a missing interpreter.

VS is kundajelab/variant-scorer, main @0e1e341.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "workflows" / "variant_scoring"))

import predict_common as pc  # noqa: E402
import probe_throughput as pt  # noqa: E402
import read_manifest as rm  # noqa: E402
import vchecks as vc  # noqa: E402

BASES = "ACGTN"


def encode(seq: str) -> np.ndarray:
    return np.array([BASES.index(b) for b in seq], dtype=np.uint8)


def revcomp_string(seq: str) -> str:
    return seq.translate(str.maketrans("ACGTN", "TGCAN"))[::-1]


# ---------------------------------------------------------------------------------------------------- sequences
def test_reverse_complement_of_codes_is_the_string_reverse_complement():
    seq = "ACGTTNNGCAAAGTN"
    assert np.array_equal(pc.reverse_complement(encode(seq)[None])[0], encode(revcomp_string(seq)))
    t = np.random.default_rng(0).integers(0, 5, size=(6, 40), dtype=np.uint8)
    assert np.array_equal(pc.reverse_complement(pc.reverse_complement(t)), t)


def test_one_hot_layouts_and_reverse_complement_in_one_hot_space():
    t = np.random.default_rng(1).integers(0, 5, size=(4, pc.WINDOW), dtype=np.uint8)
    nlc, ncl = pc.one_hot(t, "NLC"), pc.one_hot(t, "NCL")
    assert nlc.shape == (4, pc.WINDOW, 4) and ncl.shape == (4, 4, pc.WINDOW) and nlc.dtype == np.float32
    assert np.array_equal(nlc[t == 4], np.zeros(((t == 4).sum(), 4), np.float32))  # N: all-zero column
    assert np.array_equal(nlc.argmax(-1)[t < 4], t[t < 4])
    # VS's reverse complement, seqs[:, ::-1, ::-1] (helpers.py:138-139), and Cherimoya's flip(dims=(1, 2))
    assert np.array_equal(pc.one_hot(pc.reverse_complement(t), "NLC"), nlc[:, ::-1, ::-1])
    assert np.array_equal(pc.one_hot(pc.reverse_complement(t), "NCL"), ncl[:, ::-1, ::-1])
    # both layouts carry the same base at the allele index 1057
    assert pc.ALLELE_INDEX == 1057
    assert np.array_equal(nlc[:, pc.ALLELE_INDEX, :], ncl[:, :, pc.ALLELE_INDEX])


def test_parse_window_spec():
    assert pc.parse_window_spec("variants=w/v.npz") == ("variants", Path("w/v.npz"))
    assert pc.parse_window_spec("w/peaks.dsA.npz") == ("peaks.dsA", Path("w/peaks.dsA.npz"))
    for bad in ("w/v.txt", "=w/v.npz"):
        with pytest.raises(ValueError):
            pc.parse_window_spec(bad)


def _write_windows(path: Path, n=6, a2=True, **over):
    rng = np.random.default_rng(3)
    arrays = {"a1": rng.integers(0, 5, size=(n, pc.WINDOW), dtype=np.uint8), "ids": np.array([f"v{i}" for i in range(n)]),
              "nonacgt": np.zeros(n, bool)}  # fmt: skip
    if a2:
        arrays["a2"] = arrays["a1"].copy()
        arrays["a2"][:, pc.ALLELE_INDEX] = (arrays["a1"][:, pc.ALLELE_INDEX] + 1) % 4
    arrays.update(over)
    np.savez(path, **arrays)
    return arrays


def test_load_windows_reads_and_checks(tmp_path):
    arrays = _write_windows(tmp_path / "variants.npz")
    (tmp_path / "variants.json").write_text(json.dumps({"fasta_md5": "f" * 32, "mode": "variants"}))
    ws = pc.load_windows(tmp_path / "variants.npz")
    assert ws.name == "variants" and len(ws) == 6 and list(ws.alleles) == ["a1", "a2"]
    assert np.array_equal(ws.alleles["a1"], arrays["a1"]) and ws.builder["fasta_md5"] == "f" * 32
    _write_windows(tmp_path / "peaks.x.npz", a2=False)
    assert list(pc.load_windows(tmp_path / "peaks.x.npz").alleles) == ["a1"]
    bad = {
        "dtype": {"a1": np.zeros((6, pc.WINDOW), np.int64)},
        "length": {"a1": np.zeros((6, 100), np.uint8)},
        "codes": {"a1": np.full((6, pc.WINDOW), 5, np.uint8)},
        "rows": {"ids": np.array(["a", "b"])},
    }
    for name, over in bad.items():
        _write_windows(tmp_path / f"{name}.npz", **over)
        with pytest.raises(ValueError):
            pc.load_windows(tmp_path / f"{name}.npz")
    np.savez(tmp_path / "noids.npz", a1=np.zeros((2, pc.WINDOW), np.uint8))
    with pytest.raises(ValueError):
        pc.load_windows(tmp_path / "noids.npz")


# ---------------------------------------------------------------------------------------------------- counts
def test_strand_average_is_vs_mean_of_counts():
    rng = np.random.default_rng(4)
    fwd, rev = rng.normal(4, 1, 1000).astype(np.float32), rng.normal(4, 1, 1000).astype(np.float32)
    avg = pc.strand_average(fwd, rev)
    assert avg.dtype == np.float32
    # VS: np.average([exp(fwd), exp(rev)], axis=0) (helpers.py:169-170, 195-197), then log2(c2 / c1) (:225)
    vs_counts = np.average([np.exp(fwd.astype(np.float64)), np.exp(rev.astype(np.float64))], axis=0)
    np.testing.assert_allclose(avg, np.log(vs_counts), rtol=0, atol=2e-6)
    assert np.max(np.abs(avg - (fwd + rev) / 2)) > 1e-3  # not the mean of the log counts
    fwd2, rev2 = fwd + 0.3, rev - 0.1
    ours = (pc.strand_average(fwd2, rev2).astype(np.float64) - avg) / pc.LN2
    vs2 = np.average([np.exp(fwd2.astype(np.float64)), np.exp(rev2.astype(np.float64))], axis=0)
    np.testing.assert_allclose(ours, np.log2(vs2 / vs_counts), atol=1e-5)


def test_derive_full_adds_counts_strand_by_strand_and_commutes_with_the_average():
    rng = np.random.default_rng(5)
    nb = {f"{a}_{s}": rng.normal(5, 1, 50).astype(np.float32) for a in pc.ALLELES for s in pc.STRANDS}
    b = {f"{a}_{s}": rng.normal(2, 1, 50).astype(np.float32) for a in pc.ALLELES for s in pc.STRANDS}
    nobias = pc.prediction_arrays({a: {s: nb[f"{a}_{s}"] for s in pc.STRANDS} for a in pc.ALLELES})
    bias = pc.prediction_arrays({a: {s: b[f"{a}_{s}"] for s in pc.STRANDS} for a in pc.ALLELES})
    full = pc.derive_full(nobias, bias)
    for a in pc.ALLELES:
        for s in pc.STRANDS:
            k = f"{a}_{s}"
            np.testing.assert_allclose(np.exp(full[k].astype(np.float64)),
                                       np.exp(nb[k].astype(np.float64)) + np.exp(b[k].astype(np.float64)),
                                       rtol=1e-6)  # fmt: skip
        # the strand-averaged full model is the sum of the strand-averaged parts, in counts
        np.testing.assert_allclose(full[f"{a}_logcounts"],
                                   np.logaddexp(nobias[f"{a}_logcounts"].astype(np.float64),
                                                bias[f"{a}_logcounts"]), atol=2e-6)  # fmt: skip
    peaks = pc.derive_full({k: v for k, v in nobias.items() if k.startswith("a1")},
                           {k: v for k, v in bias.items() if k.startswith("a1")})  # fmt: skip
    assert sorted(peaks) == ["a1_fwd", "a1_logcounts", "a1_rev"]


def test_rows_order_batches_and_padding():
    assert np.array_equal(pc.select_rows(10, None, 0), np.arange(10))
    assert np.array_equal(pc.select_rows(10, 20, 0), np.arange(10))
    rows = pc.select_rows(100, 7, 3)
    assert len(rows) == 7 and np.all(np.diff(rows) > 0) and np.array_equal(rows, pc.select_rows(100, 7, 3))
    order = pc.processing_order(9, 1)
    values = np.arange(9, dtype=np.float32) * 2
    assert pc.processing_order(9, None) is None and pc.restore_order(values, None) is values
    np.testing.assert_array_equal(pc.restore_order(values[order], order), values)
    assert list(pc.batch_slices(7, 3)) == [(0, 3), (3, 6), (6, 7)]
    chunk = np.ones((2, 5), np.uint8)
    padded = pc.pad_rows(chunk, 4)
    assert padded.shape == (4, 5) and (padded[2:] == pc.N_CODE).all() and (padded[:2] == 1).all()
    assert pc.pad_rows(chunk, 2) is chunk


# ---------------------------------------------------------------------------------------------------- driver
def _fake_predict(calls: list):
    """Two 'models' whose log counts are functions of each row alone, like a real model's."""

    def predict(tokens, batch):
        assert tokens.dtype == np.uint8 and tokens.flags.c_contiguous and tokens.shape[1] == pc.WINDOW
        calls.append((len(tokens), batch))
        rc = pc.reverse_complement(tokens)
        f = lambda t, w: (t == w).mean(1).astype(np.float32) * 3  # noqa: E731
        return {"m1": {"fwd": f(tokens, 0), "rev": f(rc, 0)}, "m2": {"fwd": f(tokens, 2) - 1, "rev": f(rc, 2) - 1}}

    return predict


def test_run_window_sets_writes_vs_arrays_and_keeps_rows(tmp_path):
    _write_windows(tmp_path / "variants.npz", n=9)
    _write_windows(tmp_path / "peaks.d.npz", n=5, a2=False)
    sets = [pc.load_windows(tmp_path / "variants.npz"), pc.load_windows(tmp_path / "peaks.d.npz")]
    calls: list = []
    out = tmp_path / "preds"
    meta = {"model_files": {"m1": {"path": "m1.bin"}}, "precision": "default"}
    pc.run_window_sets(sets, _fake_predict(calls), ["m1", "m2"], out, 4, meta=meta, derived={"mf": ("m1", "m2")})
    assert calls == [(9, 4), (9, 4), (5, 4)]  # one call per allele and set, with the batch size
    full = {m: pc.read_predictions(pc.output_path(out, m, "variants")) for m in ("m1", "m2", "mf")}
    a1 = sets[0].alleles["a1"]
    np.testing.assert_array_equal(full["m1"]["a1_fwd"], (a1 == 0).mean(1).astype(np.float32) * 3)
    np.testing.assert_array_equal(full["m1"]["ids"], sets[0].ids)
    for m in full:
        for a in pc.ALLELES:
            np.testing.assert_allclose(full[m][f"{a}_logcounts"],
                                       pc.strand_average(full[m][f"{a}_fwd"], full[m][f"{a}_rev"]))  # fmt: skip
    np.testing.assert_allclose(full["mf"]["a2_rev"], np.logaddexp(full["m1"]["a2_rev"].astype(np.float64),
                                                                  full["m2"]["a2_rev"]), atol=1e-6)  # fmt: skip
    peaks = pc.read_predictions(pc.output_path(out, "m2", "peaks.d"))
    assert "a2_fwd" not in peaks and "a2_logcounts" not in peaks and len(peaks["a1_fwd"]) == 5
    info = json.loads(pc.output_path(out, "m1", "variants").with_suffix(".json").read_text())
    assert info["model_file"] == {"path": "m1.bin"} and info["rows"] == 9 and info["precision"] == "default"
    assert info["strand_average"] == pc.STRAND_AVERAGE and "model_files" not in info
    assert json.loads(pc.output_path(out, "mf", "variants").with_suffix(".json").read_text())["derived_from"] == [
        "m1", "m2"]  # fmt: skip
    assert not list(out.glob("*.partial"))

    stamp = pc.output_path(out, "m1", "variants").stat().st_mtime_ns
    time.sleep(0.01)
    summary = pc.run_window_sets(sets, _fake_predict([]), ["m1", "m2"], out, 4, derived={"mf": ("m1", "m2")})
    assert all(s.get("skipped") for s in summary) and pc.output_path(out, "m1", "variants").stat().st_mtime_ns == stamp

    pc.run_window_sets(sets[:1], _fake_predict([]), ["m1", "m2"], tmp_path / "perm", 2, permute_seed=7)
    perm = pc.read_predictions(pc.output_path(tmp_path / "perm", "m1", "variants"))
    for k in ("ids", "a1_fwd", "a1_rev", "a2_logcounts"):
        np.testing.assert_array_equal(perm[k], full["m1"][k])
    pc.run_window_sets(sets[:1], _fake_predict([]), ["m1", "m2"], tmp_path / "sample", 2, sample=4, sample_seed=1)
    sample = pc.read_predictions(pc.output_path(tmp_path / "sample", "m2", "variants"))
    rows = pc.select_rows(9, 4, 1)
    np.testing.assert_array_equal(sample["ids"], full["m2"]["ids"][rows])
    np.testing.assert_array_equal(sample["a2_fwd"], full["m2"]["a2_fwd"][rows])


def test_probe_counts_passes_and_meets_at_barriers(tmp_path):
    results = pc.probe(_fake_predict([]), [2, 8], seconds=0.02, passes_per_row=4, rows_per_call=10,
                       barrier_dir=tmp_path / "barrier", barrier_n=1)  # fmt: skip
    assert [r["batch"] for r in results] == [2, 8]
    assert results[0]["rows_per_call"] == 10 and results[1]["rows_per_call"] == 16  # whole batches
    assert all(r["passes"] == r["calls"] * r["rows_per_call"] * 4 and r["passes_per_second"] > 0 for r in results)


def test_probe_records_a_failing_batch_size_and_goes_on():
    def predict(tokens, batch):
        if batch == 8:
            raise MemoryError("out of GPU memory, say")
        return {}

    seen = []
    results = pc.probe(predict, [2, 8, 4], seconds=0.01, passes_per_row=2, rows_per_call=8,
                       on_result=lambda r: seen.append(len(r)))  # fmt: skip
    assert [r["batch"] for r in results] == [2, 8, 4] and seen == [1, 2, 3]
    assert "error" in results[1] and "MemoryError" in results[1]["error"] and "passes_per_second" not in results[1]
    assert all("error" not in results[i] and results[i]["passes_per_second"] > 0 for i in (0, 2))


def test_barrier_waits_for_every_process(tmp_path):
    code = ("import sys; sys.path.insert(0, sys.argv[1]); import predict_common as pc; "
            "pc.barrier(__import__('pathlib').Path(sys.argv[2]), 'go', int(sys.argv[3]), timeout=float(sys.argv[4]))")  # fmt: skip
    vs_dir = str(HERE.parent / "workflows" / "variant_scoring")
    procs = [subprocess.Popen([sys.executable, "-c", code, vs_dir, str(tmp_path / "b"), "2", "60"]) for _ in range(2)]
    assert [p.wait(timeout=60) for p in procs] == [0, 0]
    lonely = subprocess.run([sys.executable, "-c", code, vs_dir, str(tmp_path / "c"), "2", "0.3"],
                            capture_output=True, text=True, timeout=60)  # fmt: skip
    assert lonely.returncode != 0 and "TimeoutError" in lonely.stderr


# ---------------------------------------------------------------------------------------------------- vchecks
def _preds(n=200, seed=0, a2=True, noise=0.0):
    rng = np.random.default_rng(seed)
    strands = {a: {s: rng.normal(4, 1, n).astype(np.float32) for s in pc.STRANDS} for a in pc.ALLELES[: 2 if a2 else 1]}
    arrays = pc.prediction_arrays(strands)
    if noise:
        arrays = {
            k: (v + np.random.default_rng(seed + 1).normal(0, noise, n)).astype(np.float32) for k, v in arrays.items()
        }
    return {"ids": np.array([f"v{i}" for i in range(n)]), **arrays}


def test_pair_statistics_and_tolerance():
    a = _preds()
    same = vc.compare_pair(a, a)
    assert all(q["max"] == 0 for q in same["quantities"].values()) and "logfc" in same["quantities"]
    assert vc.tolerance_pass(same["quantities"], 1e-9) is True and vc.tolerance_pass(same["quantities"], None) is None
    b = {k: (v + 5e-5).astype(np.float32) if k != "ids" else v for k, v in a.items()}
    q = vc.compare_pair(a, b)["quantities"]
    assert q["a1_fwd"]["max"] == pytest.approx(5e-5, rel=0.05) and q["logfc"]["max"] < 1e-5
    assert vc.tolerance_pass(q, 1e-4) is True and vc.tolerance_pass(q, 1e-5) is False
    with pytest.raises(ValueError):
        vc.compare_pair(a, {**a, "ids": a["ids"][::-1]})
    peaks = vc.compare_pair(_preds(a2=False), _preds(a2=False, noise=0.01))
    assert "logfc" not in peaks["quantities"] and peaks["quantities"]["a1_logcounts"]["pearson"] > 0.99


def test_pair_cli_exit_status(tmp_path):
    a, b = _preds(), _preds(noise=1e-3)
    np.savez(tmp_path / "a.npz", **a)
    np.savez(tmp_path / "b.npz", **b)
    common = ["pair", "--a", str(tmp_path / "a.npz"), "--b", str(tmp_path / "b.npz"), "--label", "x"]
    assert vc.main([*common, "--out", str(tmp_path / "ok.json")]) == 0
    assert vc.main([*common, "--tol", "1e-6", "--out", str(tmp_path / "fail.json")]) == vc.OUTSIDE_TOLERANCE == 3
    assert json.loads((tmp_path / "fail.json").read_text())["pass"] is False
    assert vc.main(["pair", "--a", str(tmp_path / "missing.npz"), "--b", str(tmp_path / "b.npz"), "--label", "x",
                    "--out", str(tmp_path / "err.json")]) == 1  # fmt: skip


def test_strands_v5(tmp_path):
    fold = tmp_path / "preds" / "dsA" / "fold_0"
    fold.mkdir(parents=True)
    p = _preds()
    p["a1_rev"] = p["a1_fwd"] + 0.01
    p["a2_rev"] = p["a2_fwd"] + 0.01
    p.update(pc.prediction_arrays({a: {"fwd": p[f"{a}_fwd"], "rev": p[f"{a}_rev"]} for a in pc.ALLELES}))
    np.savez(fold / "m1.variants.npz", **p)
    np.savez(fold / "m1.peaks.dsA.npz", **_preds(a2=False))
    assert vc.main(["strands", "--preds-root", str(tmp_path / "preds"), "--out", str(tmp_path / "v5.json")]) == 0
    rows = (tmp_path / "v5.tsv").read_text().splitlines()
    header = rows[0].split("\t")
    table = [dict(zip(header, r.split("\t"))) for r in rows[1:]]
    var = {r["quantity"]: r for r in table if r["set"] == "variants"}
    assert {"a1_fwd_vs_rev", "a2_fwd_vs_rev", "logfc_fwd_vs_rev", "logfc_fwd_vs_average"} <= set(var)
    assert float(var["a1_fwd_vs_rev"]["max"]) == pytest.approx(0.01, rel=1e-3)
    assert float(var["logfc_fwd_vs_rev"]["max"]) < 1e-5 and float(var["a1_fwd_vs_rev"]["pearson"]) > 0.999
    assert all(r["dataset"] == "dsA" and r["fold"] == "0" and r["model"] == "m1" for r in table)
    summary = json.loads((tmp_path / "v5.json").read_text())
    assert summary["files"] == 2 and "a1_fwd_vs_rev" in summary["by_model"]["m1"]


def test_strands_v5_ranks_a_seeded_sample(tmp_path):
    """V5 at scale: Spearman on a seeded sample of at most --rank-rows rows (n_spearman); the differences, Pearson and
    the summary over every row."""
    fold = tmp_path / "preds" / "dsA" / "fold_0"
    fold.mkdir(parents=True)
    n, k = 300, 100
    sample = vc.rank_sample(n, k)
    assert len(sample) == k and np.array_equal(sample, vc.rank_sample(n, k)) and (np.diff(sample) > 0).all()
    assert vc.rank_sample(n, n) is None and vc.rank_sample(n, 0) is None
    p = _preds(n=n)
    left_out = np.setdiff1d(np.arange(n), sample)
    outlier = int(left_out[np.argmin(p["a1_fwd"][left_out])])  # outside the sample, and far from its rank
    rev = p["a1_fwd"] + np.float32(0.01)
    rev[outlier] += np.float32(5.0)
    strands = {"a1": {"fwd": p["a1_fwd"], "rev": rev}, "a2": {"fwd": p["a2_fwd"], "rev": p["a2_rev"]}}
    p.update(pc.prediction_arrays(strands))
    np.savez(fold / "m1.variants.npz", **p)

    def run(rank_rows: int) -> dict:
        out = tmp_path / f"v5_{rank_rows}.json"
        assert vc.main(["strands", "--preds-root", str(tmp_path / "preds"), "--out", str(out),
                        "--rank-rows", str(rank_rows)]) == 0  # fmt: skip
        lines = out.with_suffix(".tsv").read_text().splitlines()
        rows = {r["quantity"]: r for r in (dict(zip(lines[0].split("\t"), ln.split("\t"))) for ln in lines[1:])}
        return {"rows": rows, "json": json.loads(out.read_text())}

    sampled, every = run(k), run(0)
    a1 = sampled["rows"]["a1_fwd_vs_rev"]
    assert int(a1["n"]) == n and int(a1["n_spearman"]) == k and int(every["rows"]["a1_fwd_vs_rev"]["n_spearman"]) == n
    assert float(a1["max"]) == pytest.approx(5.01, rel=1e-4)  # the outlier, which the sample leaves out
    assert float(a1["pearson"]) == pytest.approx(vc.pearson(p["a1_fwd"], p["a1_rev"]), rel=1e-5)
    assert float(a1["spearman"]) == pytest.approx(1.0, abs=1e-9)  # the sample is in order on both strands
    assert float(every["rows"]["a1_fwd_vs_rev"]["spearman"]) < 0.999  # all rows: the outlier moves
    assert sampled["rows"]["abs_logfc_average"]["n_spearman"] == ""
    m1 = sampled["json"]["by_model"]["m1"]["a1_fwd_vs_rev"]
    assert sampled["json"]["spearman_rows"] == k and m1 == every["json"]["by_model"]["m1"]["a1_fwd_vs_rev"]
    assert m1["pearson_min"] == pytest.approx(float(a1["pearson"]), rel=1e-5)


def test_vs_list_and_vs_compare(tmp_path):
    header = "variant_id\tchrom\tpos\tref\talt\tsource\tstratum\taf\tdist_to_summit\tfold_test\n"
    lines = [f"chr1:{1100 + i}:A:{'C' if i % 3 else 'CT'}\tchr1\t{1100 + i}\tA\t{'C' if i % 3 else 'CT'}\tgenome"
             f"\tsummit100\t0.01\t5\t0\n" for i in range(30)]  # fmt: skip
    (tmp_path / "list.tsv").write_text(header + "".join(lines))
    built = [ln.split("\t")[0] for ln in lines[:25]]  # the builder dropped 5 rows
    np.savez(tmp_path / "variants.npz", ids=np.array(built), a1=np.zeros((25, 4), np.uint8))
    (tmp_path / "g.fa.fai").write_text("chr1\t50000\t6\t60\t61\nchr2\t30000\t50860\t60\t61\n")
    info = vc.vs_list(tmp_path / "list.tsv", tmp_path / "variants.npz", tmp_path / "g.fa.fai", 10, 1, tmp_path / "v2")
    vs_rows = [r.split("\t") for r in (tmp_path / "v2" / "vs_variants.tsv").read_text().splitlines()]
    assert info["picked"] == 10 and len(vs_rows) == 10 and all(len(r) == 5 for r in vs_rows)
    assert all(r[4] in built and r[4] == f"{r[0]}:{r[1]}:{r[2]}:{r[3]}" for r in vs_rows)  # chr pos a1 a2 id
    assert (tmp_path / "v2" / "chrom.sizes").read_text() == "chr1\t50000\nchr2\t30000\n"

    ours = _preds(n=25)
    ours["ids"] = np.array(built)
    np.savez(tmp_path / "ours.npz", **ours)
    lfc = (ours["a2_logcounts"].astype(np.float64) - ours["a1_logcounts"]) / math.log(2)
    with open(tmp_path / "vs.tsv", "w") as fh:  # VS's variant_scores.tsv columns, minus one row VS dropped
        fh.write("chr\tpos\tallele1\tallele2\tvariant_id\tallele1_pred_counts\tallele2_pred_counts\tlogfc\tabs_logfc\n")
        for i in range(1, 25):
            c1, c2 = math.exp(float(ours["a1_logcounts"][i])), math.exp(float(ours["a2_logcounts"][i]))
            shifted, size = float(lfc[i]) + 1e-5, abs(float(lfc[i]))
            fh.write(f"chr1\t{1100 + i}\tA\tC\t{built[i]}\t{c1!r}\t{c2!r}\t{shifted!r}\t{size!r}\n")
    res = vc.vs_compare(tmp_path / "vs.tsv", tmp_path / "ours.npz", 1e-3)
    assert res["n_joined"] == 24 and res["n_vs"] == 24 and res["pass"] is True
    assert res["quantities"]["logfc"]["max"] == pytest.approx(1e-5, rel=1e-3)
    assert res["quantities"]["a1_logcounts"]["max"] < 1e-6
    assert vc.vs_compare(tmp_path / "vs.tsv", tmp_path / "ours.npz", 1e-6)["pass"] is False


def test_summary_flattens_every_check(tmp_path):
    a = _preds()
    pair = {"check": "V4", "label": "rerun", **vc.compare_pair(a, a), "tol": None, "pass": None}
    strands = {"check": "strands", "by_model": {"m1": {"a1_fwd_vs_rev": {"files": 2, "pearson_median": 0.9,
                                                                         "pearson_min": 0.8, "p99_median": 0.1,
                                                                         "p99_max": 0.2}}}}  # fmt: skip
    paths = []
    for i, payload in enumerate((pair, strands)):
        paths.append(tmp_path / f"{i}.json")
        paths[-1].write_text(json.dumps(payload))
    (tmp_path / "broken.json").write_text("{")
    assert vc.main(["summary", "--inputs", *map(str, paths), str(tmp_path / "broken.json"),
                    "--out", str(tmp_path / "s.tsv")]) == 0  # fmt: skip
    rows = (tmp_path / "s.tsv").read_text().splitlines()
    assert rows[0].split("\t") == list(vc.SUMMARY_COLUMNS)
    assert len(rows) == 1 + len(pair["quantities"]) + 1
    assert any(r.startswith("V5\tm1 (worst of 2 files)\ta1_fwd_vs_rev") for r in rows)


# ---------------------------------------------------------------------------------------------------- manifest
def _manifest(tmp_path: Path, missing: str | None = None) -> Path:
    files = {}
    for name in ("g.fa", "g.fa.fai", "peaks.narrowPeak", "c.torch", "nb.h5", "b.h5", "full.h5"):
        p = tmp_path / name
        if name != missing:
            p.write_text("x")
        files[name] = str(p)
    fold = {"chroms": {"train": ["chr2"], "valid": ["chr3"], "test": ["chr1", "chr4"]},
            "models": {"cherimoya": {"path": files["c.torch"]}, "cbp_nobias": {"path": files["nb.h5"]},
                       "cbp_bias": {"path": files["b.h5"]}, "cbp_full": {"path": files["full.h5"]}}}  # fmt: skip
    manifest = {"genome": {"path": files["g.fa"]}, "ok": True, "problems": [],
                "datasets": {"dsA": {"peaks": {"path": files["peaks.narrowPeak"]}, "folds": {"1": fold, "0": fold}},
                             "dsB": {"peaks": {"path": files["peaks.narrowPeak"]}, "folds": {"0": fold}}}}  # fmt: skip
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path


def test_read_manifest(tmp_path, capsys):
    path = _manifest(tmp_path)
    m = rm.load(path)
    rows = rm.rows(m)
    assert [(r["dataset"], r["fold"]) for r in rows] == [("dsA", "0"), ("dsA", "1"), ("dsB", "0")]
    assert rows[0]["cbp_bias"].endswith("b.h5") and rows[0]["test_chroms"] == "chr1,chr4"
    assert rm.check(m) == []
    assert rm.main(["tasks", str(path), "--datasets", "dsB"]) == 0
    line = capsys.readouterr().out.strip().split("\t")
    assert line[:2] == ["dsB", "0"] and len(line) == 7
    assert rm.main(["peaks", str(path)]) == 0 and len(capsys.readouterr().out.strip().splitlines()) == 2
    assert rm.main(["tasks", str(path), "--datasets", "nope"]) == 1
    broken = json.loads(path.read_text())
    del broken["datasets"]["dsA"]["folds"]["0"]["models"]["cbp_bias"]
    path.write_text(json.dumps(broken))
    with pytest.raises(rm.ManifestError, match="expected"):
        rm.rows(rm.load(path))
    other = tmp_path / "other"
    other.mkdir()
    assert any("nb.h5" in p for p in rm.check(rm.load(_manifest(other, missing="nb.h5"))))


# ---------------------------------------------------------------------------------------------------- probe driver
def test_probe_worker_env_caps_memory_for_both_backends():
    env = pt.worker_env({"XLA_PYTHON_CLIENT_MEM_FRACTION": "0.9", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
                        "3", 8, 0.45)  # fmt: skip
    assert env["CUDA_VISIBLE_DEVICES"] == "3" and env["OMP_NUM_THREADS"] == "8"
    assert "XLA_PYTHON_CLIENT_MEM_FRACTION" not in env and env["XLA_CLIENT_MEM_FRACTION"] == "0.450"
    assert env["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
    assert env["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True,per_process_memory_fraction:0.450"
    assert pt.merge_alloc_conf("per_process_memory_fraction:0.1", 0.2) == "per_process_memory_fraction:0.200"


def test_probe_collect_and_best(tmp_path):
    point = pt.Point("cherimoya", 2, [64, 512], gpu="0", exit_codes=[0, 0])
    for rank, rates in enumerate(([100.0, 900.0], [110.0, 800.0])):
        (tmp_path / f"{point.tag}.r{rank}.json").write_text(
            json.dumps(
                {
                    "results": [
                        {"batch": b, "passes_per_second": r, "warmup_seconds": 1.0} for b, r in zip([64, 512], rates)
                    ],
                    "peak_gpu_memory_gb": 1.5,
                    "device": "gpu",
                    "precision": "default",
                    "forward_path": "megakernel",
                }
            )
        )
    res = pt.collect(point, tmp_path)
    assert [b["total_passes_per_second"] for b in res["batches"]] == [210.0, 1700.0]
    lone = pt.collect(pt.Point("cbp_bias", 1, [64], gpu="1", exit_codes=[1]), tmp_path)
    assert lone["batches"][0]["total_passes_per_second"] is None and not lone["batches"][0]["complete"]
    assert pt.best_per_model([res, lone]) == {"cherimoya": {"batch": 512, "procs_per_gpu": 2,
                                                             "total_passes_per_second": 1700.0}}  # fmt: skip


# ---------------------------------------------------------------------------------------------------- backends
def _backend_python(var: str, modules: tuple[str, ...]) -> str:
    given = os.environ.get(var)
    if given:
        if not os.access(given, os.X_OK):
            pytest.fail(f"{var}={given} is not an executable interpreter")
        return given
    if all(importlib.util.find_spec(m) is not None for m in modules):
        return sys.executable
    pytest.skip(f"no {var} and no {', '.join(modules)} in this interpreter")


def _run_backend(python: str, backend: str, tmp_path: Path) -> dict:
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu", "PYTHONDONTWRITEBYTECODE": "1"}
    env.pop("PYTHONPATH", None)
    proc = subprocess.run([python, str(HERE / "variant_predict_backends.py"), backend, str(tmp_path / backend)],
                          capture_output=True, text=True, env=env, timeout=1800)  # fmt: skip
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")]
    report = json.loads(lines[-1]) if lines else {}
    assert proc.returncode == 0 and report.get("ok"), (
        f"{backend} backend checks failed (exit {proc.returncode}); passed: {report.get('checks')}\n"
        f"--- stderr ---\n{proc.stderr[-6000:]}\n--- stdout ---\n{proc.stdout[-3000:]}"
    )
    return report


def test_cherimoya_backend_with_a_tiny_model(tmp_path):
    report = _run_backend(_backend_python("VS_CHERIMOYA_PYTHON", ("torch", "cherimoya")), "cherimoya", tmp_path)
    assert len(report["checks"]) >= 9


def test_chrombpnet_backend_with_tiny_models(tmp_path):
    report = _run_backend(
        _backend_python("VS_CHROMBPNET_PYTHON", ("chrombpnet", "keras", "jax")), "chrombpnet", tmp_path
    )
    assert len(report["checks"]) >= 7
