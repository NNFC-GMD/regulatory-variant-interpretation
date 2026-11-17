"""QC on the signal and peaks that are about to be handed to ChromBPNet.

Runs after 01 (non-peaks) and before any GPU time, on the three artifacts that
actually go into training: the prepared bigwig of Tn5 insertion counts, the
filtered narrowPeak, and the GC-matched negatives. QC'ing those rather than the
upstream fragments means the numbers describe what the model will see, after
every filter.

Two halves, and the second is why this runs after 01 rather than before it:

- **individual** -- is the signal accessibility data, and does it agree with
  the peaks? (TSS enrichment, profiles, depth, signal per peak)
- **comparative** -- can signal tell a peak from its own GC-matched
  background? That is precisely the question training poses, so asking it here
  says whether the dataset is worth training on at all.

Deliberately not SnapATAC2. That is a single-cell toolkit: ``metrics.tsse``
wants an AnnData built by ``import_fragments`` and reports per-cell scores.
This pipeline trains on pseudobulk, so a per-cell distribution is not
actionable for "is this dataset worth training on", and building a cell x bin
matrix to reach the library aggregate is a lot of work for one number. Use
SnapATAC2 for per-cell QC upstream of pseudobulking; that is a different and
also worthwhile question.

Every metric here is computed from a bigwig, so it applies equally to a signal
that was supplied as a bigwig in the first place.
"""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

__all__ = [
    "aggregate_profile",
    "auroc",
    "peak_signal_distribution",
    "peak_vs_nonpeak_signal",
    "signal_summary",
    "tss_enrichment",
    "window_totals",
]

#: Profiles are averaged over at most this many regions. The aggregate shape
#: converges long before every region is used, and this keeps QC to seconds.
DEFAULT_MAX_REGIONS = 20_000


def _open(bigwig):
    import pyBigWig

    return pyBigWig.open(str(bigwig))


def _values(bw, chrom, start, end) -> np.ndarray | None:
    """Per-base values, NaN treated as zero coverage. None if out of bounds."""
    length = bw.chroms().get(chrom)
    if length is None or start < 0 or end > length:
        return None
    v = np.asarray(bw.values(chrom, start, end), dtype=np.float64)
    return np.nan_to_num(v, nan=0.0)


def _centres(regions, kind: str) -> list[tuple[str, int]]:
    """(chrom, centre) per region. narrowPeak centres on start+summit."""
    out = []
    for row in regions:
        if kind == "narrowpeak":
            out.append((row[0], int(row[1]) + int(row[9])))
        else:  # BED: use the midpoint
            out.append((row[0], (int(row[1]) + int(row[2])) // 2))
    return out


def aggregate_profile(
    bigwig,
    regions,
    flank: int = 1000,
    kind: str = "narrowpeak",
    max_regions: int = DEFAULT_MAX_REGIONS,
    seed: int = 0,
):
    """Mean insertion profile in +/-``flank`` around region centres.

    Returns ``(offsets, mean_profile, n_used)``. This is the cumulative /
    metaprofile plot: a sharp central enrichment is what open chromatin looks
    like, and a flat one means the signal and the regions disagree.
    """
    centres = _centres(regions, kind)
    rng = np.random.default_rng(seed)
    if len(centres) > max_regions:
        idx = rng.choice(len(centres), size=max_regions, replace=False)
        centres = [centres[i] for i in sorted(idx)]

    bw = _open(bigwig)
    try:
        total = np.zeros(2 * flank, dtype=np.float64)
        used = 0
        for chrom, centre in centres:
            v = _values(bw, chrom, centre - flank, centre + flank)
            if v is None:
                continue  # region's window runs off the chromosome
            total += v
            used += 1
    finally:
        bw.close()

    offsets = np.arange(-flank, flank)
    if used == 0:
        return offsets, np.zeros(2 * flank), 0
    return offsets, total / used, used


def tss_enrichment(bigwig, tss, flank: int = 2000, flank_width: int = 100, **kw):
    """TSS enrichment: central signal over the background in the far flanks.

    The ENCODE-style definition: build the mean profile around TSSs, take the
    background as the mean of the outermost ``flank_width`` bases at each end,
    and report the peak of the (background-normalised) profile.

    A value near 1 means no enrichment at all -- either the signal is not
    accessibility, or it does not match this genome's annotation.
    """
    offsets, profile, used = aggregate_profile(bigwig, tss, flank=flank, kind="bed", **kw)
    if used == 0:
        return {"tss_enrichment": None, "n_tss_used": 0}

    background = float(np.concatenate([profile[:flank_width], profile[-flank_width:]]).mean())
    if background <= 0:
        return {"tss_enrichment": None, "n_tss_used": used, "note": "zero flank signal"}

    normalised = profile / background
    return {
        "tss_enrichment": float(normalised.max()),
        "tss_enrichment_at_centre": float(normalised[flank]),
        "n_tss_used": used,
        "background_per_base": background,
        "profile_offsets": offsets.tolist(),
        "profile": normalised.tolist(),
    }


def peak_signal_distribution(bigwig, peaks, max_regions: int | None = None):
    """Total insertions per peak, and how unevenly they are spread.

    ``frac_peaks_zero_signal`` is the one to watch: peaks with no insertions
    under them are regions the model is asked to explain with no evidence, and
    a non-trivial fraction usually means the peaks and the signal came from
    different samples.
    """
    bw = _open(bigwig)
    try:
        totals = []
        for row in peaks if max_regions is None else peaks[:max_regions]:
            chrom, start, end = row[0], int(row[1]), int(row[2])
            v = _values(bw, chrom, start, end)
            totals.append(0.0 if v is None else float(v.sum()))
    finally:
        bw.close()

    t = np.asarray(totals, dtype=np.float64)
    if t.size == 0:
        return {"n_peaks": 0}
    q = np.percentile(t, [0, 25, 50, 75, 90, 99, 100])
    return {
        "n_peaks": int(t.size),
        "insertions_in_peaks": float(t.sum()),
        "signal_per_peak_mean": float(t.mean()),
        "signal_per_peak_min": float(q[0]),
        "signal_per_peak_q25": float(q[1]),
        "signal_per_peak_median": float(q[2]),
        "signal_per_peak_q75": float(q[3]),
        "signal_per_peak_q90": float(q[4]),
        "signal_per_peak_q99": float(q[5]),
        "signal_per_peak_max": float(q[6]),
        "frac_peaks_zero_signal": float((t == 0).mean()),
    }


def signal_summary(bigwig):
    """Genome-wide totals straight from the bigwig header."""
    bw = _open(bigwig)
    try:
        header = bw.header()
        chroms = bw.chroms()
    finally:
        bw.close()
    return {
        "total_insertions": float(header.get("sumData", 0.0)),
        "covered_bases": int(header.get("nBasesCovered", 0)),
        "max_per_base": float(header.get("maxVal", 0.0)),
        "mean_per_covered_base": float(header.get("sumData", 0.0) / header["nBasesCovered"])
        if header.get("nBasesCovered")
        else 0.0,
        "n_chromosomes": len(chroms),
        "genome_bases": int(sum(chroms.values())),
    }


# -- comparative: peaks against their own GC-matched background ---------------
# The negatives from 01 are the other half of what ChromBPNet trains on. They
# are matched to the peaks on GC and drawn from the same fold chromosomes, so
# the ONE thing that should separate them is accessibility. If it does not, the
# model has nothing to learn from this dataset and the GPU time is wasted --
# which is the whole reason this QC runs before training rather than after.


def _average_ranks(x: np.ndarray) -> np.ndarray:
    """Ranks with ties averaged, like ``scipy.stats.rankdata``.

    Written out rather than imported: ties are the common case here (peaks and
    background alike are full of zero-signal windows, which all tie at 0) and
    scipy is not installed in every environment this runs in.
    """
    order = np.argsort(x, kind="mergesort")
    sx = x[order]
    n = x.size
    is_new = np.empty(n, dtype=bool)
    is_new[0] = True
    np.not_equal(sx[1:], sx[:-1], out=is_new[1:])
    starts = np.flatnonzero(is_new)
    ends = np.append(starts[1:], n) - 1
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = ((starts + ends) / 2.0 + 1.0)[np.cumsum(is_new) - 1]
    return ranks


def auroc(positive, negative) -> float | None:
    """Probability a random positive outscores a random negative; ties count half.

    The Mann-Whitney U form, so it stays exact under the heavy tying that
    zero-signal windows produce.
    """
    positive = np.asarray(positive, dtype=np.float64)
    negative = np.asarray(negative, dtype=np.float64)
    if positive.size == 0 or negative.size == 0:
        return None
    ranks = _average_ranks(np.concatenate([positive, negative]))
    n_pos, n_neg = positive.size, negative.size
    u = ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def window_totals(
    bigwig,
    regions,
    window: int,
    kind: str = "narrowpeak",
    max_regions: int = DEFAULT_MAX_REGIONS,
    seed: int = 0,
    keep_chroms=None,
) -> np.ndarray:
    """Total insertions in a FIXED ``window`` centred on each region's summit.

    Fixed width is the point. :func:`peak_signal_distribution` sums each peak
    over its own called width, which describes the peaks well but cannot
    compare them to anything: the negatives are all ``inputlen`` wide, so a
    300bp peak would lose to a background window on width alone. Here both
    sides get the same number of bases, so what differs is signal density.

    Both peaks and chrombpnet's negatives are 10-column BED with the summit
    offset in column 10, so ``kind="narrowpeak"`` centres both correctly.
    """
    centres = _centres(regions, kind)
    if keep_chroms is not None:
        keep = set(keep_chroms)
        centres = [c for c in centres if c[0] in keep]
    rng = np.random.default_rng(seed)
    if len(centres) > max_regions:
        idx = rng.choice(len(centres), size=max_regions, replace=False)
        centres = [centres[i] for i in sorted(idx)]

    half = window // 2
    bw = _open(bigwig)
    try:
        totals = []
        for chrom, centre in centres:
            v = _values(bw, chrom, centre - half, centre + half)
            if v is not None:  # drop, rather than zero, a window off the end
                totals.append(float(v.sum()))
    finally:
        bw.close()
    return np.asarray(totals, dtype=np.float64)


def peak_vs_nonpeak_signal(bigwig, peaks, nonpeaks, window: int = 1000, **kw):
    """How well signal alone separates peaks from their GC-matched background.

    ``window`` defaults to 1000, ChromBPNet's OUTPUT window -- the span the
    counts head is asked to predict -- so the separation measured here is the
    same quantity the trained model gets scored on.

    Returns ``(metrics, peak_totals, nonpeak_totals)``. The headline is
    ``auroc_peaks_vs_nonpeaks``: 1.0 is perfect separation, 0.5 means signal
    cannot tell a peak from background and there is nothing to train on.
    """
    pos = window_totals(bigwig, peaks, window, **kw)
    neg = window_totals(bigwig, nonpeaks, window, **kw)
    if pos.size == 0 or neg.size == 0:
        return (
            {"n_peaks_compared": int(pos.size), "n_nonpeaks_compared": int(neg.size)},
            pos,
            neg,
        )

    pos_median, neg_median = float(np.median(pos)), float(np.median(neg))
    metrics = {
        "compare_window": int(window),
        "n_peaks_compared": int(pos.size),
        "n_nonpeaks_compared": int(neg.size),
        "peak_signal_mean": float(pos.mean()),
        "peak_signal_median": pos_median,
        "nonpeak_signal_mean": float(neg.mean()),
        "nonpeak_signal_median": neg_median,
        "signal_enrichment_peak_over_nonpeak": float(pos.mean() / neg.mean())
        if neg.mean() > 0
        else None,
        "auroc_peaks_vs_nonpeaks": auroc(pos, neg),
        # Background windows as accessible as a typical peak. A few percent is
        # expected -- there is open chromatin outside called peaks -- but a
        # large fraction means the peak calls left real signal behind.
        "frac_nonpeaks_above_peak_median": float((neg >= pos_median).mean()),
        "frac_nonpeaks_zero_signal": float((neg == 0).mean()),
        "frac_peaks_below_nonpeak_median": float((pos <= neg_median).mean()),
    }
    return metrics, pos, neg


def dropped_peaks_resampled(dropped_bed, nonpeaks, window: int):
    """How much of what 00.1 discarded came back as GC-matched background.

    Dropping a peak does not remove the region from the analysis. It removes it
    from the exclusion list `chrombpnet prep nonpeaks` builds, which makes the
    region ELIGIBLE to be sampled as background -- so the bias model can end up
    training on regions the pipeline just judged too weak to be peaks. That is
    the opposite of what the floor is for, and nothing else measures it.

    Reported both ways, because they answer different questions:
      frac_negatives_...  how contaminated the background is (small by
                          construction -- the sampler draws from the whole
                          genome)
      frac_dropped_...    how much of the discarded set came back (can be
                          large, and is the number that says whether the floor
                          actually removed those regions from training)
    """
    import bisect
    from pathlib import Path

    if not Path(dropped_bed).exists():
        # Not an error: the floor is optional, so there is usually nothing to
        # check. Say so rather than returning {} and looking like a clean
        # result -- a silently missing metric reads as "measured, found none".
        logging.getLogger(__name__).info(
            "no dropped-peaks file at %s; signal floor not in use, skipping the re-sampling check",
            dropped_bed,
        )
        return {}
    by = defaultdict(list)
    n_dropped = 0
    with open(dropped_bed) as fh:
        for line in fh:
            f = line.split("\t")
            if len(f) < 3:
                continue
            by[f[0]].append((int(f[1]), int(f[2])))
            n_dropped += 1
    for c in by:
        by[c].sort()
    starts = {c: [x[0] for x in v] for c, v in by.items()}

    n_neg = 0
    hit_neg = 0
    hit_regions = set()
    for row in nonpeaks:
        c = str(row[0])
        mid = int(row[1]) + (int(row[9]) if len(row) > 9 else (int(row[2]) - int(row[1])) // 2)
        n_neg += 1
        v = by.get(c)
        if not v:
            continue
        i = bisect.bisect_right(starts[c], mid)
        if i and v[i - 1][1] > mid:
            hit_neg += 1
            hit_regions.add((c, v[i - 1][0]))

    return {
        "n_peaks_dropped_by_floor": n_dropped,
        "n_negatives_in_dropped_peaks": hit_neg,
        "frac_negatives_in_dropped_peaks": round(hit_neg / n_neg, 6) if n_neg else None,
        "n_dropped_peaks_resampled": len(hit_regions),
        "frac_dropped_peaks_resampled": (
            round(len(hit_regions) / n_dropped, 6) if n_dropped else None
        ),
    }


def background_signal_quantile(
    bigwig,
    quantile: float,
    window: int,
    blacklist_intervals=None,
    n_sample: int = 50_000,
    seed: int = 0,
):
    """Signal level at `quantile` of this experiment's genome-wide windows.

    A floor for peak calling that is derived from the experiment itself rather
    than assumed. Because it is a QUANTILE of the same signal the peaks are
    measured in, it is depth-independent: a deeper library lifts the peaks and
    the background together.

    This is the experiment's distribution, not a background model -- peaks are
    NOT excluded, because the question is "what signal level does this
    experiment reach across the genome", and the peaks are part of it.
    Blacklist regions ARE excluded: artifact pileups are not experimental
    signal and they sit in the upper tail, exactly where a high quantile reads.

    Sampled, not exhaustive. Quantiles do not need every bin, and an
    exhaustive pass is ~200s per window size against ~2s here; 50k windows
    pins q01..q99 far tighter than a filtering decision needs. `seed` makes it
    reproducible, and the sample size is recorded so a run can be audited.

    Returns (threshold, diagnostics).
    """
    bw = _open(bigwig)
    try:
        chroms = bw.chroms()
        names = list(chroms)
        sizes = np.array([chroms[c] for c in names], dtype=np.int64)
        cum = np.cumsum(sizes)

        bl = {}
        for c, st, en in blacklist_intervals or []:
            if c in chroms:
                bl.setdefault(c, []).append((int(st), int(en)))
        for c in bl:
            a = np.array(sorted(bl[c]))
            bl[c] = (a[:, 0], a[:, 1])

        def hits_blacklist(c, s, e):
            if c not in bl:
                return False
            starts, ends = bl[c]
            i = int(np.searchsorted(starts, e))
            return i > 0 and ends[i - 1] > s

        rng = np.random.default_rng(seed)
        half = window // 2
        vals, tries, skipped_bl = [], 0, 0
        while len(vals) < n_sample and tries < n_sample * 6:
            tries += 1
            pos = int(rng.integers(0, cum[-1]))
            ci = int(np.searchsorted(cum, pos, side="right"))
            c = names[ci]
            off = pos - (cum[ci - 1] if ci else 0)
            s, e = off - half, off + half
            if s < 0 or e > chroms[c]:
                continue
            if hits_blacklist(c, s, e):
                skipped_bl += 1
                continue
            v = bw.stats(c, s, e, type="sum", exact=True)[0]
            vals.append(0.0 if v is None else float(v))
    finally:
        bw.close()

    g = np.asarray(vals, dtype=np.float64)
    if g.size == 0:
        return None, {}
    thr = float(np.quantile(g, quantile))
    diag = {
        "background_window": int(window),
        "background_quantile": float(quantile),
        "background_threshold": thr,
        "background_n_sampled": int(g.size),
        "background_n_blacklist_skipped": int(skipped_bl),
        "background_seed": int(seed),
        "background_frac_zero": round(float((g == 0).mean()), 6),
    }
    for q in (0.25, 0.50, 0.75, 0.90, 0.99):
        diag[f"background_q{int(q * 100):02d}"] = float(np.quantile(g, q))
    return thr, diag, g


def bias_suffix(factor: float) -> str:
    """The `_05` / `_08` suffix convention, extended past one decimal.

    The existing configs spell 0.5 as `_05` and 0.8 as `_08` -- the factor with
    its decimal point removed. Extending that rule gives `_055` for 0.55 and
    `_105` for 1.05, and it keeps every existing suffix unchanged. A whole
    number is written to one decimal first so 1.0 becomes `_10`, not `_1`.
    """
    text = f"{float(factor):g}"
    if "." not in text:
        text += ".0"
    return "_" + text.replace(".", "")


def bias_threshold_viability(
    bigwig,
    peaks,
    nonpeaks,
    factors=None,
    outputlen: int = 1000,
    outlier_threshold: float = 0.9999,
    max_regions: int = 10_000_000,
    fold_json=None,
):
    """Which bias_threshold_factor values 03.0 can actually train on.

    03.0's sweep is the most expensive thing in the pipeline -- folds x factors
    GPU jobs -- and some of those jobs cannot succeed, for a reason visible
    here with no GPU at all. `chrombpnet bias train` selects its background by

        counts_threshold = quantile(peak_counts, 0.01) * bias_threshold_factor
        kept  = nonpeak_counts[nonpeak_counts < counts_threshold]
        upper = quantile(kept, outlier_threshold)
        lower = quantile(kept, 1 - outlier_threshold)
        nonpeaks = nonpeaks[(counts < upper) & (counts > lower)]

    Counts are integers. When the cutoff is low, `kept` holds only a couple of
    distinct values, the two quantiles collapse onto adjacent integers, and the
    strict inequalities select nothing -- 0 non-peaks, counts_loss_weight nan,
    and a crash inside the one-hot encoder several frames later, AFTER the
    bigwig preprocessing has run. chrombpnet asserts `kept` is non-empty but
    never checks the post-outlier count, which is why the error surfaces so far
    from its cause.

    The same quantisation means neighbouring factors are often IDENTICAL: the
    training set only changes when q01 * factor crosses an integer.

    Everything needed is already in hand here -- 02.0 has the prepared bigwig,
    the filtered peaks and the GC-matched negatives -- so the answer costs a
    couple of minutes of CPU instead of a failed GPU job per bad factor.
    """
    # NOT subsampled by default: quantile(kept, 0.9999) is the whole point and
    # needs the real tail. chrombpnet tunes on train+valid only, so treat these
    # as indicative of the sweep rather than an exact replay of it.
    # A fine grid is free: the bigwig is read once, below, and the factor loop
    # is arithmetic. Sweeping past 1.0 matters because
    # docs/bias-factor-per-fold.md records winners piling up at the 0.8 ceiling
    # of the default grid, which cannot see an optimum outside its own range.
    if factors is None:
        factors = [round(f, 2) for f in np.arange(0.05, 2.001, 0.05)]

    # chrombpnet thresholds train+valid only -- the test chromosomes are held
    # out before any of this runs. Counting them here inflated every row by a
    # constant ~13% on d0 (the test split is 11.7% of the negatives), which is
    # invisible in the viability verdict and in the RANKING of factors, but
    # wrong in the absolute counts the sweep table reports.
    keep = None
    if fold_json:
        fold = json.loads(Path(fold_json).read_text())
        keep = set(fold.get("train", [])) | set(fold.get("valid", []))
    pk = window_totals(bigwig, peaks, window=outputlen, max_regions=max_regions, keep_chroms=keep)
    ng = window_totals(
        bigwig, nonpeaks, window=outputlen, max_regions=max_regions, keep_chroms=keep
    )
    if pk.size == 0 or ng.size == 0:
        return {}, [], np.array([]), np.array([])
    q01 = float(np.quantile(pk, 0.01))

    rows, viable, distinct = [], [], {}
    for f in factors:
        thr = q01 * f
        kept = ng[ng < thr]
        if kept.size == 0:
            rows.append(
                {
                    "factor": f,
                    "suffix": bias_suffix(f),
                    "counts_threshold": thr,
                    "n_after_cutoff": 0,
                    "n_nonpeaks": 0,
                    "distinct": False,
                    "verdict": "fail: cutoff admits no non-peaks",
                }
            )
            continue
        upper = np.quantile(kept, outlier_threshold)
        lower = np.quantile(kept, 1 - outlier_threshold)
        n = int(((ng < upper) & (ng > lower)).sum())
        verdict = (
            "fail: outlier quantiles collapse"
            if n == 0
            else "risky: very few non-peaks"
            if n < 1000
            else "ok"
        )
        is_distinct = verdict == "ok" and n not in distinct
        rows.append(
            {
                "factor": f,
                "suffix": bias_suffix(f),
                "counts_threshold": thr,
                "n_after_cutoff": int(kept.size),
                "n_nonpeaks": n,
                "distinct": is_distinct,
                "verdict": verdict,
            }
        )
        if verdict == "ok":
            viable.append(f)
            distinct.setdefault(n, f)

    summary = {
        "peak_signal_q01": q01,
        "bias_factors_viable": ",".join(str(f) for f in viable) or "NONE",
        "bias_factors_distinct": ",".join(str(f) for f in distinct.values()) or "NONE",
        "n_bias_factors_viable": len(viable),
        "n_bias_factors_distinct": len(distinct),
    }
    # ── the recommended factor ───────────────────────────────────────────
    # The background should not contain regions STRONGER than the weakest
    # peaks. chrombpnet admits non-peaks with count < q01*factor, so the
    # largest count it lets in is ceil(q01*factor)-1; once that exceeds q01 the
    # background holds regions above the weakest 1% of peaks, and the bias
    # model starts learning accessibility instead of Tn5 preference.
    #
    # On d0 that boundary is exactly where the trained models break: peaks
    # pearson r is -0.004, +0.003 while max_admitted <= q01, then jumps to
    # +0.251, +0.369 the moment it exceeds it. A step, not a curve -- so the
    # recommendation is the last factor before the step, not an optimum found
    # by search.
    #
    # Among the factors sharing that training set, take the SMALLEST: they are
    # identical, so parsimony costs nothing and keeps the cutoff furthest from
    # the boundary.
    recommended = None
    best_admitted = -1
    for r in rows:
        if r["verdict"] != "ok":
            continue
        admitted = math.ceil(r["counts_threshold"]) - 1
        if admitted <= q01 and admitted > best_admitted:
            best_admitted, recommended = admitted, r["factor"]
        elif admitted == best_admitted and r["factor"] < recommended:
            recommended = r["factor"]
    for r in rows:
        r["recommended"] = r["factor"] == recommended
    if recommended is not None:
        summary["bias_factor_recommended"] = recommended
        summary["bias_factor_recommended_suffix"] = bias_suffix(recommended)
        summary["bias_max_admitted_count"] = int(best_admitted)

    return summary, rows, pk, ng


def peak_width_summary(peaks, input_window: int = 2114, genome_bases: int | None = None):
    """Peak widths, disjointness, and how much sequence the model sees twice.

    Peak WIDTH does not reach the model: ChromBPNet extracts a fixed
    `input_window`bp window centred on (start + summit), so a 500bp and a
    4000bp peak produce the same sized training example. Width still matters
    for a different reason -- with a synthesised midpoint summit, the wider the
    peak the more arbitrary the window's placement within it, which is why
    `peak_width_max` is worth reading next to `summit_offset_max`.

    What does reach the model is WINDOW overlap. Peaks may be disjoint and
    their windows still overlap, because neighbours can sit closer together
    than the window is wide -- the same sequence then appears in several
    training examples. `window_redundancy` is summed window bp over unique bp
    covered: 1.0 means every example is disjoint sequence.
    """
    rows = [(str(r[0]), int(r[1]), int(r[2]), int(r[9]) if len(r) > 9 else None) for r in peaks]
    if not rows:
        return {}
    widths = np.array([e - s for _, s, e, _ in rows], dtype=np.int64)
    q = np.percentile(widths, [0, 50, 100])
    out = {
        "peak_width_min": int(q[0]),
        "peak_width_median": float(q[1]),
        "peak_width_max": int(q[2]),
        "peak_width_distinct": int(np.unique(widths).size),
        "peak_bases_total": int(widths.sum()),
    }

    def _n_overlapping(by_chrom):
        n = 0
        for spans in by_chrom.values():
            prev_end = None
            for a, b in sorted(spans):
                if prev_end is not None and a < prev_end:
                    n += 1
                prev_end = b if prev_end is None else max(prev_end, b)
        return n

    def _merged_bp(by_chrom):
        total = 0
        for spans in by_chrom.values():
            spans = sorted(spans)
            cs, ce = spans[0]
            for a, b in spans[1:]:
                if a <= ce:
                    ce = max(ce, b)
                else:
                    total += ce - cs
                    cs, ce = a, b
            total += ce - cs
        return total

    peaks_by = defaultdict(list)
    for c, s, e, _ in rows:
        peaks_by[c].append((s, e))
    out["n_peaks_overlapping"] = _n_overlapping(peaks_by)
    # Merged, so this stays correct if the peaks ever DO overlap --
    # peak_bases_total is a sum of widths and would double-count.
    out["peak_bases_merged"] = _merged_bp(peaks_by)

    if all(su is not None for _, _, _, su in rows):
        half = input_window // 2
        win_by = defaultdict(list)
        for c, s, _, su in rows:
            mid = s + su
            win_by[c].append((mid - half, mid + half))
        n_ov = _n_overlapping(win_by)
        summed = covered = 0
        for spans in win_by.values():
            spans = sorted(spans)
            summed += sum(b - a for a, b in spans)
            cs, ce = spans[0]
            for a, b in spans[1:]:
                if a <= ce:
                    ce = max(ce, b)
                else:
                    covered += ce - cs
                    cs, ce = a, b
            covered += ce - cs
        out["input_window"] = int(input_window)
        out["window_bases_merged"] = int(covered)
        out["n_windows_overlapping"] = int(n_ov)
        out["frac_windows_overlapping"] = round(n_ov / len(rows), 6)
        out["window_redundancy"] = round(summed / covered, 4) if covered else None

    # How much of the genome the peak set claims. Two numbers, because they
    # answer different questions:
    #   frac_genome_in_peaks    the called peaks themselves. Comparable to what
    #                           a peak caller reports; ~1-3% is typical for
    #                           ATAC, and a permissive candidate-region set
    #                           runs higher.
    #   frac_genome_in_windows  the ${input_window}bp windows the model
    #                           actually reads, merged. This is the number that
    #                           bounds the background: GC-matched negatives are
    #                           drawn from what is left, so as it grows the
    #                           background is sampled from an ever smaller and
    #                           less peak-like remainder.
    if genome_bases:
        out["genome_bases"] = int(genome_bases)
        out["frac_genome_in_peaks"] = round(out["peak_bases_merged"] / genome_bases, 6)
        if "window_bases_merged" in out:
            out["frac_genome_in_windows"] = round(out["window_bases_merged"] / genome_bases, 6)
    return out
