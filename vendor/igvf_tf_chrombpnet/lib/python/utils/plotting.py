"""Shared matplotlib setup for the pipeline's QC plots.

Replaces the rcParams block that was copy-pasted into ``select_bias_model.py``,
``qc_full_model.py`` and ``qc_datasets.py``. The three copies were identical
except for the font size, which is why ``apply_style`` takes one.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

logger = logging.getLogger(__name__)


def apply_style(font_size: int = 10, backend: str | None = "Agg") -> None:
    """Set the pipeline's plotting defaults.

    Call this at import time, before pyplot draws anything. ``backend="Agg"``
    is the default because every caller runs headless under SLURM; pass
    ``backend=None`` to leave the backend alone when working interactively.
    """
    if backend is not None:
        matplotlib.use(backend)
    matplotlib.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": font_size,
            "axes.labelsize": font_size,
            "axes.titlesize": font_size,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
            "legend.fontsize": font_size,
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.transparent": True,
        }
    )


def save_fig(fig, stem, formats=("pdf", "png"), quiet: bool = False) -> None:
    """Write ``fig`` to ``<stem>.<ext>`` for each format, creating parent dirs.

    ``stem`` is a path without an extension. The pipeline saves both a PDF (for
    figures) and a PNG (for quick review in a browser) everywhere.
    """
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        fig.savefig(f"{stem}.{ext}")
    if not quiet:
        logger.info("Saved: %s.%s", stem, "/.".join(formats))
