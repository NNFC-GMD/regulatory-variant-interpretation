"""TF-MoDISco results, MEME motif files and tomtom-lite, for the comparison's levels 3 and 4.

Patterns are read from a modisco-lite results h5 (`pos_patterns`/`neg_patterns`, each `pattern_<i>` with `sequence`
(w, 4) PPM, `contrib_scores` (w, 4) CWM and `seqlets/{example_idx, start, end, is_revcomp}`; the root attribute
`window_size` is the central window TF-MoDISco cropped, so a seqlet's start and end are columns of
[summit - window/2, summit + window/2) of its example's peak, 0-based, end exclusive). A compiled compendium h5
(09.0's modisco_compiled.h5) holds one CWM per cluster under `<group>/<cluster id>/contrib_scores` and no seqlets.

Matching is tomtom-lite (memelite.tomtom: the TOMTOM p-value of the best alignment of each query against each
target, either strand), as modisco-lite's report runs it: queries are trimmed as its report trims them (the positions
whose summed |CWM| reaches 0.3 of the maximum, first to last) and given as PPMs; q-values are Benjamini-Hochberg over
each query's targets. tomtom-lite's p-values take the target columns as their background, so a small target set (a
few dozen patterns of one dataset) is always searched together with a motif database as decoys: the decoys make the
background and the multiple-testing burden those of a whole database, and a pattern's partner is its best target
among the patterns alone, kept when its q-value (over all targets, decoys included) is at most the threshold.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

TRIM_THRESHOLD = 0.3
"""modisco-lite's report: keep the positions whose sum of |CWM| is at least this share of the maximum."""
BIAS_PREFIXES = ("TN5_", "DNASE_")
"""chrombpnet's motifs.meme marks Tn5 and DNase I enzyme bias with these names."""
Q_MATCH = 0.05
GROUPS = (("pos", "pos_patterns"), ("neg", "neg_patterns"))
BASES = "ACGT"


@dataclass
class Pattern:
    """One TF-MoDISco pattern of one model's results."""

    model: str
    posneg: str
    name: str
    ppm: np.ndarray
    cwm: np.ndarray
    n_seqlets: int
    seqlets: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.posneg}.{self.name}"

    def trim_bounds(self) -> tuple[int, int]:
        score = np.abs(self.cwm).sum(axis=1)
        keep = np.flatnonzero(score >= TRIM_THRESHOLD * score.max()) if score.max() > 0 else np.arange(len(score))
        return int(keep.min()), int(keep.max()) + 1

    def query(self) -> np.ndarray:
        """The trimmed PPM, (4, w), as modisco-lite's report gives tomtom-lite."""
        a, b = self.trim_bounds()
        return np.ascontiguousarray(self.ppm[a:b].T, dtype=np.float64)

    def consensus(self) -> str:
        a, b = self.trim_bounds()
        out = []
        for row in self.ppm[a:b]:
            i = int(np.argmax(row))
            out.append(BASES[i] if row[i] >= 0.5 else BASES[i].lower() if row[i] >= 0.3 else "n")
        return "".join(out)


def pattern_order(name: str) -> int:
    try:
        return int(name.rsplit("_", 1)[-1])
    except ValueError:
        return 1 << 30


def read_modisco(path: Path, model: str, seqlets: bool = True) -> tuple[list[Pattern], int | None]:
    """The patterns of a modisco-lite results h5 (pos then neg, in pattern order) and its window_size."""
    import h5py

    out = []
    with h5py.File(path, "r") as fh:
        window = int(fh.attrs["window_size"]) if "window_size" in fh.attrs else None
        for posneg, group in GROUPS:
            if group not in fh:
                continue
            for name in sorted(fh[group], key=pattern_order):
                g = fh[group][name]
                sq = g["seqlets"]
                n = int(sq["n_seqlets"][0]) if "n_seqlets" in sq else int(sq["start"].shape[0])
                p = Pattern(model, posneg, name, np.asarray(g["sequence"][()], dtype=np.float64),
                            np.asarray(g["contrib_scores"][()], dtype=np.float64), n)  # fmt: skip
                if seqlets:
                    p.seqlets = {k: np.asarray(sq[k][()]) for k in ("example_idx", "start", "end", "is_revcomp")}
                out.append(p)
    return out, window


def read_compiled(path: Path) -> list[tuple[str, str, np.ndarray]]:
    """09.0's modisco_compiled.h5: (posneg, cluster id, CWM (w, 4)) per cluster."""
    import h5py

    out = []
    with h5py.File(path, "r") as fh:
        for posneg, group in GROUPS:
            if group not in fh:
                continue
            for cid in sorted(fh[group], key=lambda c: int(c) if c.isdigit() else 1 << 30):
                out.append((posneg, cid, np.asarray(fh[group][cid]["contrib_scores"][()], dtype=np.float64)))
    return out


def read_meme(path: Path) -> tuple[list[str], list[np.ndarray]]:
    """A MEME file's motifs: names (the first word of each MOTIF line) and matrices (4, w)."""
    from memelite.io import read_meme as _read

    motifs = _read(str(path))
    return [k.split()[0] for k in motifs], [np.ascontiguousarray(v, dtype=np.float64) for v in motifs.values()]


def is_bias(name: str) -> bool:
    return name.upper().startswith(BIAS_PREFIXES)


def tomtom_p(queries: list[np.ndarray], targets: list[np.ndarray]) -> np.ndarray:
    """(n_queries, n_targets) p-values of tomtom-lite (both strands), on NUMBA_NUM_THREADS threads."""
    from memelite import tomtom

    if not queries or not targets:
        return np.ones((len(queries), len(targets)))
    n_jobs = int(os.environ.get("NUMBA_NUM_THREADS", "1") or 1)
    p = tomtom(queries, targets, n_jobs=n_jobs)[0]
    return np.clip(np.nan_to_num(np.asarray(p, dtype=np.float64), nan=1.0), 0.0, 1.0)


def bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q-values along the last axis (each query's targets)."""
    p = np.asarray(p, dtype=np.float64)
    if p.size == 0:
        return p.copy()
    m = p.shape[-1]
    order = np.argsort(p, axis=-1)
    ranked = np.take_along_axis(p, order, axis=-1) * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[..., ::-1], axis=-1)[..., ::-1]
    q = np.empty_like(ranked)
    np.put_along_axis(q, order, np.minimum(ranked, 1.0), axis=-1)
    return q


@dataclass
class Database:
    """A motif database (names, matrices) to annotate against or to use as decoys."""

    names: list[str]
    mats: list[np.ndarray]

    @classmethod
    def read(cls, path: Path) -> Database:
        return cls(*read_meme(path))


def annotate(patterns: list[Pattern], db: Database, top: int = 3) -> list[dict]:
    """Per pattern: the best `top` database matches by p-value, with q-values."""
    p = tomtom_p([pt.query() for pt in patterns], db.mats)
    q = bh(p)
    out = []
    for i in range(len(patterns)):
        order = np.argsort(p[i])[:top]
        out.append({"matches": [db.names[j] for j in order], "p": [float(p[i, j]) for j in order],
                    "q": [float(q[i, j]) for j in order]})  # fmt: skip
    return out


def bias_flags(patterns: list[Pattern], db: Database, q_max: float = Q_MATCH) -> list[dict]:
    """Per pattern, against a database with bias entries (chrombpnet's motifs.meme): its best match overall, its best
    bias entry, and the flag: the best match overall is a bias entry, at q <= q_max."""
    p = tomtom_p([pt.query() for pt in patterns], db.mats)
    q = bh(p)
    bias_cols = np.array([is_bias(n) for n in db.names])
    out = []
    for i in range(len(patterns)):
        best = int(np.argmin(p[i]))
        entry = {"db_best": db.names[best], "db_best_q": float(q[i, best]), "bias_best": None, "bias_q": np.nan}
        if bias_cols.any():
            cols = np.flatnonzero(bias_cols)
            j = int(cols[np.argmin(p[i, cols])])
            entry.update(bias_best=db.names[j], bias_q=float(q[i, j]))
        entry["bias_flag"] = bool(bias_cols[best] and q[i, best] <= q_max)
        out.append(entry)
    return out


def cross_match(a: list[Pattern], b: list[Pattern], decoys: Database, q_max: float = Q_MATCH) -> dict:
    """Each pattern's best partner among the other set, both ways: tomtom-lite on trimmed PPMs against the other set
    plus the decoys, the partner being the best target among the other set's patterns, kept at q <= q_max (q over all
    targets, decoys included). `reciprocal` holds the pairs that are each other's partner.

    Returns {"a": [(j or None, p, q)], "b": [(i or None, p, q)], "reciprocal": [(i, j, p_ab, p_ba)]}, with the p- and
    q-value of the best pattern target also when it is not kept."""

    def one_way(qs: list[Pattern], ts: list[Pattern]) -> list[tuple]:
        if not qs:
            return []
        if not ts:
            return [(None, np.nan, np.nan) for _ in qs]
        p = tomtom_p([x.query() for x in qs], [t.query() for t in ts] + decoys.mats)
        q = bh(p)
        out = []
        for i in range(len(qs)):
            j = int(np.argmin(p[i, : len(ts)]))
            out.append((j if q[i, j] <= q_max else None, float(p[i, j]), float(q[i, j])))
        return out

    ab, ba = one_way(a, b), one_way(b, a)
    reciprocal = [(i, j, pa, ba[j][1]) for i, (j, pa, _) in enumerate(ab) if j is not None and ba[j][0] == i]
    return {"a": ab, "b": ba, "reciprocal": reciprocal}


# ---------------------------------------------------------------------------------------------------------------------
# seqlets in the genome


def seqlet_intervals(pattern: Pattern, bed, half: int) -> dict[str, np.ndarray]:
    """A pattern's seqlets as genomic intervals through its own model's bed (row example_idx; peak summit at
    start + column 10): [start + summit - half + seqlet start, ... + seqlet end), with each seqlet's peak name."""
    s = pattern.seqlets
    rows = s["example_idx"].astype(np.int64)
    if rows.size and (rows.min() < 0 or rows.max() >= len(bed)):
        raise ValueError(
            f"{pattern.model} {pattern.key}: example_idx {int(rows.max())} beyond the bed's {len(bed)} rows"
        )
    origin = bed["start"].to_numpy()[rows] + bed["summit"].to_numpy()[rows] - half
    return {"chrom": bed["chrom"].to_numpy()[rows], "start": origin + s["start"].astype(np.int64),
            "end": origin + s["end"].astype(np.int64), "peak": bed["name"].to_numpy()[rows]}  # fmt: skip


def reciprocal_overlap(a: dict, b: dict, min_frac: float = 0.5) -> np.ndarray:
    """For each interval of a, whether some interval of b on the same chromosome overlaps it by at least min_frac of
    both lengths."""
    n = len(a["start"])
    hit = np.zeros(n, dtype=bool)
    if n == 0 or len(b["start"]) == 0:
        return hit
    for chrom in np.unique(a["chrom"]):
        ia = np.flatnonzero(a["chrom"] == chrom)
        ib = np.flatnonzero(b["chrom"] == chrom)
        if ib.size == 0:
            continue
        order = np.argsort(b["start"][ib], kind="stable")
        bs, be = b["start"][ib][order], b["end"][ib][order]
        max_len = int((be - bs).max())
        as_, ae = a["start"][ia], a["end"][ia]
        lo = np.searchsorted(bs, as_ - max_len, side="left")
        hi = np.searchsorted(bs, ae, side="left")
        found = np.zeros(ia.size, dtype=bool)
        width = int((hi - lo).max()) if ia.size else 0
        for off in range(width):
            j = lo + off
            ok = j < hi
            jj = np.where(ok, j, 0)
            ov = np.minimum(ae, be[jj]) - np.maximum(as_, bs[jj])
            need = np.maximum(min_frac * (ae - as_), min_frac * (be[jj] - bs[jj]))
            found |= ok & (ov > 0) & (ov >= need)
        hit[ia] = found
    return hit
