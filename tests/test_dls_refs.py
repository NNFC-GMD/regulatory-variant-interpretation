"""dls_refs.py gives every (peak, shuffle) pair the reference `cherimoya attribute` would use, bit for bit.

tangermeme's deep_lift_shap draws the reference of pair (e, k) as
``dinucleotide_shuffle(X[e:e+1].float(), n=1, random_state=rs + k)[:, 0]`` (tangermeme/deep_lift_shap.py:706-708).
dls_refs calls that same public function in spawned worker processes and returns each reference as uint8 tokens.
These tests hold it to that contract (acceptance criterion B1) against tangermeme called in this process:

- Tokens re-expanded to a one-hot are torch.equal to tangermeme's float32 reference for every pair. This covers
  300 random sequences of length 2114 plus edge cases (homopolymers, a base absent, a base at most twice, so that
  it draws no random numbers, and alternating dinucleotides), K = 20, random_state 0 and 1234, and 1, 2 and 4
  workers.
- Tokens round-trip losslessly, N columns included, and anything that is not a one-hot is refused.
- Chunks arrive in order, with the prefetch window kept full.
- An exception that tangermeme raises in a worker reaches the caller with the same type and message, and the pool
  keeps working afterwards.
- Workers run one intra-op thread each and use the parent's node-local numba cache.

The tests also report shuffle throughput (pairs/s per worker) on the terminal.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
ersatz = pytest.importorskip("tangermeme.ersatz")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workflows" / "cherimoya"))

import dls_refs  # noqa: E402

L = 2114  # cherimoya attribute's default in_window
K = 20  # n_shuffles
RANDOM_STATES = (0, 1234)
N_RANDOM = 300
SEQS_PER_CHUNK = 7  # does not divide the number of sequences, so the last chunk is short


def _report(config: pytest.Config, line: str) -> None:
    """Write a measurement to the terminal, past pytest's output capture."""
    reporter = config.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None:
        reporter.write_line(f"[dls_refs] {line}")


def _onehot(tokens: np.ndarray) -> torch.Tensor:
    """(n, L) base indices -> (n, 4, L) int8 one-hot (extract_loci's dtype), built without dls_refs."""
    return torch.from_numpy(np.ascontiguousarray(np.eye(4, dtype=np.int8)[tokens].transpose(0, 2, 1)))


def _edge_cases(rng: np.random.RandomState) -> dict[str, np.ndarray]:
    """Sequences that take the shuffle kernel's corner paths. G (2) is the rare base.

    The kernel permutes each base's successor list with permutation(n - 1). A base with at most two occurrences
    in positions 0..L-2 therefore draws no random numbers.
    """

    def without_g() -> np.ndarray:
        return rng.choice([0, 1, 3], L)

    cases = {"homopolymer A": np.zeros(L, np.int64), "homopolymer T": np.full(L, 3, np.int64)}
    cases["no T"] = rng.randint(0, 3, L)
    cases["no A"] = rng.randint(1, 4, L)
    cases["A and T only"] = rng.choice([0, 3], L)
    seq = without_g()
    seq[rng.choice(np.arange(1, L - 1), 2, replace=False)] = 2
    cases["G twice, inside"] = seq
    seq = without_g()
    seq[-1] = 2
    cases["G once, last base"] = seq
    seq = without_g()
    seq[[0, -1]] = 2
    cases["G first and last base"] = seq
    cases["alternating AC"] = np.tile([0, 1], L // 2)
    cases["alternating GT"] = np.tile([2, 3], L // 2)
    return cases


@pytest.fixture(scope="module")
def sequences() -> tuple[list[str], np.ndarray, torch.Tensor]:
    """300 random sequences, then the edge cases: names, base indices (n, L) uint8 and the int8 one-hot (n, 4, L)."""
    rng = np.random.RandomState(20261008)
    cases = _edge_cases(rng)
    tokens = np.concatenate([rng.randint(0, 4, (N_RANDOM, L)), np.stack(list(cases.values()))])
    names = [f"random {i}" for i in range(N_RANDOM)] + list(cases)
    return names, tokens.astype(np.uint8), _onehot(tokens)


@pytest.fixture(scope="module")
def truth(sequences, pytestconfig) -> dict[int, torch.Tensor]:
    """tangermeme's reference for every pair, from the call deep_lift_shap makes: rs -> (n, K, 4, L) float32."""
    _, _, X = sequences
    out = {}
    start = time.perf_counter()
    for rs in RANDOM_STATES:
        out[rs] = torch.stack(
            [
                torch.cat(
                    [
                        ersatz.dinucleotide_shuffle(X[e : e + 1].float(), n=1, random_state=rs + k)[:, 0]
                        for k in range(K)
                    ]
                )
                for e in range(len(X))
            ]
        )
    pairs = len(RANDOM_STATES) * len(X) * K
    seconds = time.perf_counter() - start
    _report(
        pytestconfig,
        f"tangermeme called in this process ({torch.get_num_threads()} torch threads): "
        f"{pairs} pairs in {seconds:.1f} s = {pairs / seconds:.0f} pairs/s",
    )
    return out


def _mismatches(names: list[str], got: np.ndarray, expected: torch.Tensor, rows: list[int]) -> list[tuple[str, int]]:
    """Pairs whose tokens, re-expanded to a float32 one-hot, are not torch.equal to tangermeme's reference."""
    bad = []
    for i, e in enumerate(rows):
        for k in range(K):
            ref = dls_refs.tokens_to_onehot(got[i, k])
            if ref.dtype != torch.float32 or not torch.equal(ref, expected[e, k]):
                bad.append((names[e], k))
    return bad


def test_tokens_round_trip(sequences, truth):
    _, tokens, X = sequences
    got = dls_refs.onehot_to_tokens(X)
    assert got.dtype == np.uint8 and got.shape == tokens.shape
    np.testing.assert_array_equal(got, tokens)
    back = dls_refs.tokens_to_onehot(got, dtype=torch.int8)
    assert back.dtype == torch.int8 and torch.equal(back, X)

    refs = truth[RANDOM_STATES[0]]  # tangermeme's own float32 output, (n, K, 4, L)
    ref_tokens = dls_refs.onehot_to_tokens(refs)
    assert ref_tokens.shape == (len(X), K, L)
    again = dls_refs.tokens_to_onehot(ref_tokens)
    assert again.dtype == torch.float32 and torch.equal(again, refs)

    x = X[:2].clone()
    x[1, :, [0, 5, L - 1]] = 0  # N columns, as extract_loci writes them
    with_n = dls_refs.onehot_to_tokens(x)
    assert set(np.flatnonzero(with_n[1] == dls_refs.N_TOKEN)) == {0, 5, L - 1}
    assert torch.equal(dls_refs.tokens_to_onehot(with_n, dtype=torch.int8), x)


def test_non_onehot_inputs_and_bad_tokens_are_refused(sequences):
    _, tokens, X = sequences
    two_bases = X[:1].clone()
    two_bases[0, :, 9] = torch.tensor([1, 1, 0, 0], dtype=torch.int8)
    not_binary = X[:1].clone()
    not_binary[0, :, 9] = torch.tensor([0, 0, 2, 0], dtype=torch.int8)
    for bad in (two_bases, not_binary, X[:1].float() * 0.5, X[:1, :3]):
        with pytest.raises(ValueError):
            dls_refs.onehot_to_tokens(bad)

    out_of_range = tokens[:1].copy()
    out_of_range[0, 3] = dls_refs.N_TOKEN + 1
    with pytest.raises(ValueError):
        dls_refs.tokens_to_onehot(out_of_range)
    with pytest.raises(TypeError):
        dls_refs.tokens_to_onehot(tokens[:1].astype(np.int64))
    with pytest.raises(ValueError):
        dls_refs.shuffle_tokens(out_of_range, 1, 0)

    producer = dls_refs.RefProducer(workers=0)  # arguments are checked before any work starts
    for args, error in (
        ((tokens[:2], K, None, 1), TypeError),
        ((tokens[:2], 0, 0, 1), ValueError),
        ((tokens[:2], K, 0, 0), ValueError),
        ((tokens[:2, None], K, 0, 1), ValueError),
        ((tokens[:2].astype(np.int8), K, 0, 1), TypeError),
        ((out_of_range, K, 0, 1), ValueError),
    ):
        with pytest.raises(error):
            producer.chunks(*args)


def test_shuffle_tokens_in_process_matches_tangermeme(sequences, truth):
    """The worker body run in this process, alone and as RefProducer(workers=0): edge cases and 3 random sequences."""
    names, tokens, _ = sequences
    rows = list(range(N_RANDOM - 3, len(tokens)))
    for rs in RANDOM_STATES:
        got = dls_refs.shuffle_tokens(tokens[rows], K, rs)
        assert got.dtype == np.uint8 and got.shape == (len(rows), K, L)
        assert not _mismatches(names, got, truth[rs], rows)

    with dls_refs.RefProducer(workers=0) as producer:
        parts = list(producer.chunks(tokens[rows], K, RANDOM_STATES[1], SEQS_PER_CHUNK))
        stats = dict(producer.stats)
    assert [(a, b) for a, b, _ in parts] == [
        (a, min(a + SEQS_PER_CHUNK, len(rows))) for a in range(0, len(rows), SEQS_PER_CHUNK)
    ]
    # in process, every chunk is drawn when it is asked for: each one is a stall, the first one too
    assert stats["stalls"] == stats["chunks"] and 0.0 < stats["first_wait_s"] <= stats["wait_s"]
    got = np.concatenate([refs for _, _, refs in parts])
    assert not _mismatches(names, got, truth[RANDOM_STATES[1]], rows)


@pytest.mark.parametrize("workers", [1, 2, 4])
def test_pool_references_equal_tangermeme(sequences, truth, workers, pytestconfig):
    """B1: every pair, both seeds, through a spawned pool. The second seed runs on a warm pool and is timed."""
    names, _, X = sequences
    n = len(X)
    tokens = dls_refs.onehot_to_tokens(X)
    expected_bounds = [(a, min(a + SEQS_PER_CHUNK, n)) for a in range(0, n, SEQS_PER_CHUNK)]
    with dls_refs.RefProducer(workers=workers) as producer:
        assert producer.window >= max(3, workers)
        for run, rs in enumerate(RANDOM_STATES):
            got = np.full((n, K, L), 255, np.uint8)  # 255 is not a token, so a chunk never written would fail
            bounds = []
            start = time.perf_counter()
            for a, b, refs in producer.chunks(tokens, K, rs, SEQS_PER_CHUNK):
                bounds.append((a, b))
                got[a:b] = refs
            seconds = time.perf_counter() - start
            assert bounds == expected_bounds
            stats = dict(producer.stats)
            assert stats["max_in_flight"] == min(producer.window, len(expected_bounds))
            assert 0.0 <= stats["first_wait_s"] <= stats["wait_s"]
            assert (stats["first_wait_s"] > 0.0) <= (stats["stalls"] > 0)

            bad = _mismatches(names, got, truth[rs], list(range(n)))
            assert not bad, f"{len(bad)} of {n * K} pairs differ from tangermeme (random_state={rs}): {bad[:5]}"
            pairs = n * K
            _report(
                pytestconfig,
                f"{workers} worker(s), random_state={rs} ({'cold pool, includes spawn' if run == 0 else 'warm pool'}): "
                f"{pairs}/{pairs} pairs equal; {pairs / seconds:.0f} pairs/s = {pairs / seconds / workers:.0f} "
                f"per worker; consumer waited {stats['wait_s']:.2f} s on {stats['stalls']} of {stats['chunks']} chunks "
                f"({stats['first_wait_s']:.2f} s on the first)",
            )


def test_worker_exception_reaches_caller_with_same_type(sequences):
    _, _, X = sequences
    x = X[:5].clone()
    x[3, :, 1000] = 0  # an N, which tangermeme refuses
    with pytest.raises(Exception) as direct:
        ersatz.dinucleotide_shuffle(x[3:4].float(), n=1, random_state=0)
    tokens = dls_refs.onehot_to_tokens(x)

    with pytest.raises(direct.type) as inline:
        dls_refs.shuffle_tokens(tokens[3:4], 2, 0)
    assert type(inline.value) is direct.type and str(inline.value) == str(direct.value)

    with dls_refs.RefProducer(workers=2) as producer:
        delivered = []
        with pytest.raises(direct.type) as pooled:
            for a, _, _ in producer.chunks(tokens, 2, 0, 1):
                delivered.append(a)
        assert type(pooled.value) is direct.type and str(pooled.value) == str(direct.value)
        assert delivered == [0, 1, 2]  # the chunks before the bad one, in order

        again = np.concatenate([refs for _, _, refs in producer.chunks(tokens[:3], 2, 0, 2)])
        np.testing.assert_array_equal(again, dls_refs.shuffle_tokens(tokens[:3], 2, 0))


def test_workers_are_single_threaded_and_share_the_parent_numba_cache():
    with dls_refs.RefProducer(workers=1) as producer:
        cache_dir = os.environ["NUMBA_CACHE_DIR"]  # start() exports it when it is unset
        cache_path = Path(producer.numba_cache_path)
        assert cache_path.resolve().is_relative_to(Path(cache_dir).resolve())
        assert list(cache_path.glob("ersatz._fast_shuffle-*.nbi")), "the parent did not cache the kernel"
        assert producer._pool.submit(torch.get_num_threads).result() == 1
        assert producer._pool.submit(os.getenv, "NUMBA_CACHE_DIR").result() == cache_dir
