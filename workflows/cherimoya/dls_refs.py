"""DeepLIFT/DeepSHAP references for Cherimoya, bit for bit as `cherimoya attribute` draws them, off the GPU's path.

tangermeme's deep_lift_shap gives pair (e, k), sequence e with reference k, the reference
``dinucleotide_shuffle(X[e:e+1].float(), n=1, random_state=rs + k)[:, 0]`` (tangermeme/deep_lift_shap.py:693,
706-708). It makes one seeded call per pair, serially, between GPU batches. The fast engine needs exactly those
references. So this module calls that same public function on that same input, in spawned worker processes that run
ahead of the GPU:
- Processes, not threads: the numba kernel inside holds the GIL (tangermeme/ersatz.py:518).
- Spawn, not fork: the parent initialises CUDA.

References travel as uint8 tokens, one per position: 0-3 for the one-hot channel (A, C, G, T) and N_TOKEN (4) for
an all-zero column. A token array is a quarter of the int8 one-hot and a sixteenth of the float32 one. The encoding
is lossless. Anything that is not a one-hot with at most one 1 per position is refused, never rounded.

- shuffle_tokens() is the worker body. It re-expands each sequence to the float32 one-hot, calls tangermeme's
  dinucleotide_shuffle once per reference with seed rs + k, and returns the reference tokens.
- RefProducer is a spawn ProcessPoolExecutor whose workers run one intra-op thread each.
  - Before the pool starts, the parent exports a node-local NUMBA_CACHE_DIR and pre-warms tangermeme's numba kernel
    into it. The workers then load the compiled kernel instead of each compiling it, and nothing is written into
    the installed environment.
  - chunks() yields the references for consecutive chunks of sequences in order, with several chunks in flight.
  - An exception raised in a worker, by tangermeme or otherwise, reaches the caller with its own type.
"""

from __future__ import annotations

import collections
import getpass
import logging
import multiprocessing
import operator
import os
import sys
import tempfile
import time
from collections.abc import Iterator
from concurrent.futures import Future, ProcessPoolExecutor

import numpy as np
import torch

logger = logging.getLogger(__name__)

N_TOKEN = 4
"""Token of an all-zero one-hot column, an N. Tokens 0-3 are the one-hot channels."""

_BLOCK_ROWS = 4096  # rows per block when tokenising, which bounds the temporaries for a 290k-peak X


def onehot_to_tokens(X: torch.Tensor | np.ndarray) -> np.ndarray:
    """Encode a one-hot (..., 4, L) as uint8 tokens (..., L): the hot channel, or N_TOKEN for an all-zero column.

    The encoding is lossless: tokens_to_onehot(onehot_to_tokens(X), X.dtype) equals X. Raises ValueError for
    anything that is not a one-hot with at most one 1 per position.

    It works by comparisons and sums over the 4 channels. argmax over them would cost about 300 us per (4, 2114)
    sequence on CPU, half as much as a shuffle.
    """
    x = torch.as_tensor(X)
    if x.dim() < 2 or x.shape[-2] != 4:
        raise ValueError(f"expected a one-hot of shape (..., 4, L), got {tuple(x.shape)}")
    if x.dtype == torch.bool:
        x = x.to(torch.uint8)
    lead, length = tuple(x.shape[:-2]), x.shape[-1]
    flat = x.reshape(-1, 4, length)
    out = np.empty((flat.shape[0], length), np.uint8)
    for a in range(0, flat.shape[0], _BLOCK_ROWS):
        block = flat[a : a + _BLOCK_ROWS]
        if not bool(((block == 0) | (block == 1)).all()):
            raise ValueError("not a one-hot encoding: it holds values other than 0 and 1")
        hot = block[:, 0] + block[:, 1] + block[:, 2] + block[:, 3]
        if bool((hot > 1).any()):
            raise ValueError("not a one-hot encoding: a position has more than one base")
        code = block[:, 1] + 2 * block[:, 2] + 3 * block[:, 3]
        out[a : a + block.shape[0]] = torch.where(hot == 0, N_TOKEN, code).to(torch.uint8).cpu().numpy()
    return out.reshape(*lead, length)


def tokens_to_onehot(tokens: torch.Tensor | np.ndarray, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Expand uint8 tokens (..., L) to a one-hot (..., 4, L) of `dtype`. N_TOKEN becomes an all-zero column.

    This is the inverse of onehot_to_tokens. Token values are checked on CPU tensors only. A check on a GPU tensor
    would synchronise the device, and tokens reach a GPU only from host arrays that were already checked.
    """
    t = torch.as_tensor(tokens)
    if t.dtype != torch.uint8:
        raise TypeError(f"tokens must be uint8, got {t.dtype}")
    if t.dim() < 1:
        raise ValueError("tokens must have a length axis")
    if t.device.type == "cpu" and t.numel() and int(t.max()) > N_TOKEN:
        raise ValueError(f"tokens must lie in 0-{N_TOKEN}, got {int(t.max())}")
    codes = torch.arange(4, dtype=torch.uint8, device=t.device)
    return (t.unsqueeze(-2) == codes[:, None]).to(dtype)


def shuffle_tokens(tokens: np.ndarray, n_shuffles: int, random_state: int) -> np.ndarray:
    """Draw each sequence's references with tangermeme, exactly as deep_lift_shap does, and return their tokens.

    tokens: (n, L) uint8 sequence tokens. The result is (n, n_shuffles, L) uint8. Row [e, k] re-expands to
    ``dinucleotide_shuffle(x_e, n=1, random_state=random_state + k)[:, 0]``, where x_e is sequence e as the float32
    (1, 4, L) one-hot that deep_lift_shap passes.

    This is RefProducer's worker body, and it also runs in-process. Whatever tangermeme raises propagates unchanged,
    e.g. ValueError for an N column.
    """
    from tangermeme.ersatz import dinucleotide_shuffle

    tokens = _check_tokens(tokens)
    n_shuffles = _positive(n_shuffles, "n_shuffles")
    random_state = _seed(random_state)
    out = np.empty((tokens.shape[0], n_shuffles, tokens.shape[1]), np.uint8)
    for e in range(tokens.shape[0]):
        x = tokens_to_onehot(tokens[e : e + 1])
        refs = [dinucleotide_shuffle(x, n=1, random_state=random_state + k)[:, 0] for k in range(n_shuffles)]
        out[e] = onehot_to_tokens(torch.cat(refs))
    return out


def ensure_numba_cache_dir() -> str:
    """Make NUMBA_CACHE_DIR name a writable node-local directory, exporting a default if it is unset. Return it.

    Workers inherit the variable, so they load tangermeme's compiled kernel from wherever the parent cached it.
    - Unset: numba caches next to the installed tangermeme sources, inside the environment that running jobs share.
    - Unwritable: numba silently falls back to that same location. So an unwritable directory raises OSError.

    Call this before tangermeme is first imported. numba reads the variable when numba is imported, and it fixes a
    kernel's cache location when that kernel compiles, which for tangermeme's kernel is at import.
    """
    path = os.environ.get("NUMBA_CACHE_DIR")
    if not path:
        path = os.path.join(tempfile.gettempdir(), f"numba_{_user()}")
        os.environ["NUMBA_CACHE_DIR"] = path
        logger.info("NUMBA_CACHE_DIR was unset; exported %s", path)
        if "numba" in sys.modules:
            logger.warning(
                "numba was imported before NUMBA_CACHE_DIR was set; kernels it compiled already keep their earlier "
                "cache location"
            )
            from numba.core import config

            config.reload_config()
    os.makedirs(path, exist_ok=True)
    try:
        tempfile.TemporaryFile(dir=path).close()
    except OSError as exc:
        raise OSError(
            f"NUMBA_CACHE_DIR={path} is not writable, so numba would cache inside the installed environment; "
            "point it at a writable node-local directory"
        ) from exc
    return path


def default_workers() -> int:
    """The default pool size: min(8, usable CPUs - 2), and at least 1. The rest is left to the main process."""
    try:
        ncpu = len(os.sched_getaffinity(0))
    except AttributeError:
        ncpu = os.cpu_count() or 1
    return max(1, min(8, ncpu - 2))


class RefProducer:
    """Reference tokens for consecutive chunks of sequences, from a spawn process pool, in order and ahead of use.

    workers: the number of worker processes. The default is default_workers(). 0 shuffles in this process instead,
        for tests and tiny jobs.
    ahead: the minimum number of chunks kept in flight. The window is max(ahead, 2 * workers), so the pool keeps
        working while the consumer handles earlier chunks.

    Use it as a context manager, or call start() and close(). One producer serves any number of chunks() runs.
    `stats` describes the latest run: chunks, pairs, stalls (chunks not ready when asked for), wait_s (time the
    consumer spent blocked on them) and max_in_flight.
    """

    def __init__(self, workers: int | None = None, *, ahead: int = 3) -> None:
        self.workers = default_workers() if workers is None else operator.index(workers)
        if self.workers < 0:
            raise ValueError(f"workers must be 0 or more, got {self.workers}")
        self.ahead = _positive(ahead, "ahead")
        self.window = max(self.ahead, 2 * self.workers) if self.workers else 0
        self.numba_cache_path: str | None = None
        self.stats: dict[str, int | float] = {}
        self._pool: ProcessPoolExecutor | None = None
        self._warm = False

    def __enter__(self) -> RefProducer:
        return self.start()

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def start(self) -> RefProducer:
        """Export the node-local numba cache, pre-warm the kernel in this process, then start the pool.

        Calling it again does nothing.
        """
        if not self._warm:
            self.numba_cache_path = _prewarm_numba(ensure_numba_cache_dir())
            self._warm = True
        if self.workers and self._pool is None:
            self._pool = ProcessPoolExecutor(
                self.workers, mp_context=multiprocessing.get_context("spawn"), initializer=_init_worker
            )
            logger.info("reference pool: %d spawned workers, numba cache %s", self.workers, self.numba_cache_path)
        return self

    def close(self) -> None:
        """Stop the pool: cancel the chunks still queued and wait for the running ones."""
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None

    def chunks(
        self, tokens: np.ndarray, n_shuffles: int, random_state: int, seqs_per_chunk: int
    ) -> Iterator[tuple[int, int, np.ndarray]]:
        """Yield (a, b, refs) for sequences [a, b), in order. refs is (b - a, n_shuffles, L) uint8 from shuffle_tokens.

        tokens: (n, L) uint8 sequence tokens, as onehot_to_tokens returns them. random_state: the base seed rs, an
        integer. For `random_state: null`, draw one once and pass it.

        The arguments are checked, and the first window of chunks is submitted, before this returns. So the pool
        starts working while the caller does other things, such as compiling.

        A chunk whose task failed raises its exception, with its original type, when the iteration reaches that
        chunk. The chunks still queued are then cancelled.
        """
        tokens = _check_tokens(tokens)
        n_shuffles = _positive(n_shuffles, "n_shuffles")
        random_state = _seed(random_state)
        seqs_per_chunk = _positive(seqs_per_chunk, "seqs_per_chunk")
        bounds = collections.deque(
            (a, min(a + seqs_per_chunk, len(tokens))) for a in range(0, len(tokens), seqs_per_chunk)
        )
        self.start()
        stats = {
            "chunks": len(bounds),
            "pairs": len(tokens) * n_shuffles,
            "stalls": 0,
            "wait_s": 0.0,
            "max_in_flight": 0,
        }
        self.stats = stats
        task = (tokens, n_shuffles, random_state)
        if self._pool is None:
            return self._inline(task, bounds, stats)
        pending: collections.deque[tuple[int, int, Future]] = collections.deque()
        self._fill(self._pool, pending, bounds, task, stats)
        return self._drain(self._pool, pending, bounds, task, stats)

    def _fill(
        self, pool: ProcessPoolExecutor, pending: collections.deque, bounds: collections.deque, task: tuple, stats: dict
    ) -> None:
        tokens, n_shuffles, random_state = task
        while bounds and len(pending) < self.window:
            a, b = bounds.popleft()
            pending.append((a, b, pool.submit(shuffle_tokens, tokens[a:b], n_shuffles, random_state)))
        stats["max_in_flight"] = max(stats["max_in_flight"], len(pending))

    def _drain(
        self, pool: ProcessPoolExecutor, pending: collections.deque, bounds: collections.deque, task: tuple, stats: dict
    ) -> Iterator[tuple[int, int, np.ndarray]]:
        try:
            while pending:
                a, b, future = pending.popleft()
                if not future.done():
                    stats["stalls"] += 1
                    start = time.perf_counter()
                    future.exception()  # waits without raising, so the wait is counted either way
                    stats["wait_s"] += time.perf_counter() - start
                refs = future.result()
                self._fill(pool, pending, bounds, task, stats)
                yield a, b, refs
        finally:
            for _, _, future in pending:
                future.cancel()

    @staticmethod
    def _inline(task: tuple, bounds: collections.deque, stats: dict) -> Iterator[tuple[int, int, np.ndarray]]:
        tokens, n_shuffles, random_state = task
        for a, b in bounds:
            start = time.perf_counter()
            refs = shuffle_tokens(tokens[a:b], n_shuffles, random_state)
            stats["stalls"] += 1
            stats["wait_s"] += time.perf_counter() - start
            yield a, b, refs


def _init_worker() -> None:
    """Pool initializer: one intra-op thread per worker; tangermeme's numba kernel loads from the parent's cache."""
    torch.set_num_threads(1)
    import tangermeme.ersatz  # noqa: F401  (compiles nothing: the parent cached the kernel in NUMBA_CACHE_DIR)


def _prewarm_numba(cache_dir: str) -> str | None:
    """Compile or load tangermeme's numba shuffle kernel here, so it is cached before any worker starts.

    Returns the kernel's cache directory. A warning is logged when that directory is not under cache_dir.
    """
    from tangermeme import ersatz

    ersatz.dinucleotide_shuffle(tokens_to_onehot(np.array([[0, 1, 2, 3, 0, 2, 1, 3]], np.uint8)), n=1, random_state=0)
    kernel_cache = getattr(getattr(ersatz, "_fast_shuffle", None), "_cache", None)  # numba's private attributes
    cache_path = getattr(kernel_cache, "cache_path", None)
    root = os.path.realpath(cache_dir)
    if cache_path is None or os.path.commonpath([os.path.realpath(cache_path), root]) != root:
        logger.warning(
            "tangermeme's numba kernel is cached in %s, not under NUMBA_CACHE_DIR=%s, because tangermeme was imported "
            "before the variable was set; each worker compiles the kernel itself",
            cache_path,
            cache_dir,
        )
    return cache_path


def _check_tokens(tokens: np.ndarray | torch.Tensor) -> np.ndarray:
    """Return `tokens` as a C-contiguous (n, L) uint8 array with values in 0..N_TOKEN, or raise."""
    if isinstance(tokens, torch.Tensor):
        tokens = tokens.cpu().numpy()
    t = np.ascontiguousarray(tokens)
    if t.dtype != np.uint8:
        raise TypeError(f"tokens must be uint8, got {t.dtype}")
    if t.ndim != 2:
        raise ValueError(f"tokens must have shape (n_sequences, length), got {t.shape}")
    if t.size and int(t.max()) > N_TOKEN:
        raise ValueError(f"tokens must lie in 0-{N_TOKEN}, got {int(t.max())}")
    return t


def _seed(random_state: object) -> int:
    try:
        return operator.index(random_state)
    except TypeError:
        raise TypeError(
            f"random_state must be an integer; for random_state null, draw a base seed once. Got {random_state!r}"
        ) from None


def _positive(value: object, name: str) -> int:
    v = operator.index(value)
    if v < 1:
        raise ValueError(f"{name} must be at least 1, got {v}")
    return v


def _user() -> str:
    try:
        return getpass.getuser()
    except (KeyError, OSError):
        return str(os.getuid())
