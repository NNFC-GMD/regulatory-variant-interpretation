"""Colourblind-safe colour and label maps for the pipeline's plots.

**Categorical series use Okabe-Ito; continuous scales use cividis.** Both are
safe for deuteranopia, protanopia and tritanopia, and both stay legible in
greyscale. Nothing here may use ``viridis`` (not safe in greyscale print for
all readers and not the house scale), and never ``jet`` or ``rainbow``.

Okabe-Ito, in the canonical order:

===============  =========  ===============  =========
black            ``#000000``  yellow           ``#F0E442``
orange           ``#E69F00``  blue             ``#0072B2``
sky blue         ``#56B4E9``  vermillion       ``#D55E00``
bluish green     ``#009E73``  reddish purple   ``#CC79A7``
===============  =========  ===============  =========

``cividis`` ships with matplotlib (``cmap="cividis"``), so this adds no
dependency. Use ``SEQUENTIAL_CMAP`` rather than the string, so a future change
happens in one place.

One thing colour alone cannot fix: the pass/warn/fail heatmap in
``select_bias_model.py`` encodes status by fill. Okabe-Ito keeps those three
distinguishable, but the cells are also labelled with their metric value, which
is the accessible fallback. Keep the labels.

The endothelial dataset still appears under two keys (``igvf17_endothelial`` in
``qc_datasets.py``, ``igvf_endothelial`` everywhere else). Both are present so
neither script changes behaviour; picking one is a rename that has to happen on
the cluster results tree too.
"""

from __future__ import annotations

# ── Okabe-Ito ─────────────────────────────────────────────────────────────────

OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
}

#: Cycle order for an arbitrary number of categorical series. Black is last: it
#: reads as "other/unspecified" next to the hues.
OKABE_ITO_CYCLE = [
    OKABE_ITO["blue"],
    OKABE_ITO["orange"],
    OKABE_ITO["bluish_green"],
    OKABE_ITO["vermillion"],
    OKABE_ITO["sky_blue"],
    OKABE_ITO["reddish_purple"],
    OKABE_ITO["yellow"],
    OKABE_ITO["black"],
]

#: Continuous/sequential scale. Use this, not "viridis".
SEQUENTIAL_CMAP = "cividis"

#: Neutral grey for a series with no assigned colour. Not from Okabe-Ito; it is
#: deliberately outside the palette so "unmapped" is visually obvious.
DATASET_COLOR_DEFAULT = "#999999"


def cycle_colors(n: int) -> list[str]:
    """First ``n`` Okabe-Ito colours, repeating only if ``n`` exceeds 8."""
    return [OKABE_ITO_CYCLE[i % len(OKABE_ITO_CYCLE)] for i in range(n)]


# ── Datasets ──────────────────────────────────────────────────────────────────

DATASET_LABELS = {
    "igvf11_h7_hesc": "igvf11_h7_hesc",
    "igvf3_cardiomyocyte": "igvf3_cardiomyocyte",
    "igvf6_definitive_endoderm": "igvf6_definitive_endoderm",
    "igvf_endothelial": "igvf_endothelial",
}

DATASET_LABELS_QC = {
    "igvf3_cardiomyocyte": "Cardiomyocyte\n(igvf3)",
    "igvf6_definitive_endoderm": "Def. Endoderm\n(igvf6)",
    "igvf11_h7_hesc": "hESC H7\n(igvf11)",
    "igvf17_endothelial": "Endothelial\n(igvf17)",
}

#: One Okabe-Ito hue per dataset, used by every dataset-coloured plot (04.1,
#: 04.2 combined, and the dataset QC). Previously there were two unrelated
#: hand-picked schemes; they are now one.
DATASET_COLORS_COMBINED = {
    "igvf11_h7_hesc": OKABE_ITO["blue"],
    "igvf3_cardiomyocyte": OKABE_ITO["vermillion"],
    "igvf6_definitive_endoderm": OKABE_ITO["bluish_green"],
    "igvf_endothelial": OKABE_ITO["reddish_purple"],
}

DATASET_COLORS_QC = {
    "igvf3_cardiomyocyte": OKABE_ITO["vermillion"],
    "igvf6_definitive_endoderm": OKABE_ITO["bluish_green"],
    "igvf11_h7_hesc": OKABE_ITO["blue"],
    "igvf17_endothelial": OKABE_ITO["reddish_purple"],
}

# ── Bias model selection (src/select_bias_model.py) ───────────────────────────

#: Keyed by bias-factor label with the leading underscore stripped ("_08" -> "08").
#: Ordered along the Okabe-Ito cycle so a sweep reads as a progression.
BIAS_COLORS = {
    "05": OKABE_ITO["blue"],
    "06": OKABE_ITO["sky_blue"],
    "07": OKABE_ITO["orange"],
    "08": OKABE_ITO["bluish_green"],
    "09": OKABE_ITO["vermillion"],
    "1": OKABE_ITO["reddish_purple"],
}

#: Traffic-light status. Bluish green / orange / vermillion is the Okabe-Ito
#: substitute for green/amber/red: the green-red pair that a deuteranope cannot
#: separate is replaced by a pair that differs in both hue and lightness.
STATUS_COLORS = {
    "pass": OKABE_ITO["bluish_green"],
    "warn": OKABE_ITO["orange"],
    "fail": OKABE_ITO["vermillion"],
}

#: Threshold guide lines in the Pearson-r scatter, matching STATUS_COLORS.
THRESHOLD_COLORS = {
    "warn": OKABE_ITO["orange"],
    "fail": OKABE_ITO["vermillion"],
}

#: Bias-threshold scan (02.0). The three verdicts plus the two distributions
#: the cutoff sits between. Green/orange/vermillion carry the same pass / warn
#: / fail meaning they do in STATUS_COLORS.
BIAS_SCAN_COLORS = {
    "ok": OKABE_ITO["bluish_green"],
    "risky": OKABE_ITO["orange"],
    "fail": OKABE_ITO["vermillion"],
    "nonpeaks": OKABE_ITO["sky_blue"],
    "peaks": OKABE_ITO["reddish_purple"],
    "cutoff": OKABE_ITO["black"],
}

# ── Fragment-size classes (src/qc_datasets.py) ────────────────────────────────

#: NFR / mono / di / tri nucleosome bins. An ordered series, so it walks the
#: cycle rather than using four arbitrary blues and greys.
FRAGMENT_SIZE_COLORS = [
    OKABE_ITO["blue"],
    OKABE_ITO["sky_blue"],
    OKABE_ITO["orange"],
    OKABE_ITO["black"],
]
