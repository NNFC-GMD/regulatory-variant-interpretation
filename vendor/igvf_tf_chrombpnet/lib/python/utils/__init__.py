"""Shared helpers for the IGVF TF ChromBPNet pipeline.

Importable without installation: every script in ``src/`` puts ``lib/python`` on
``sys.path`` with a three-line shim before importing this package, so it works
identically under the cluster conda envs, under pixi, and under a bare
``python``.

``utils`` is a deliberately plain name, but it is a *top-level* one, and the
shim puts it at the front of ``sys.path``. If some other installed package ever
ships a top-level ``utils``, this one wins inside ``src/`` scripts. Nothing in
the four cluster environments does today (chrombpnet's own helpers live under
``chrombpnet.training.utils``, which is namespaced and unaffected). If that ever
changes, the symptom is an unexpected ``ImportError`` from a *third-party*
import, and the fix is to rename this package rather than to reorder the shim.

Module rules:

- Nothing here may import ``chrombpnet``, ``tensorflow``, ``finemo`` or
  ``MotifCompendium`` — those live in different, mutually incompatible
  environments, and these helpers are shared across all of them.
- ``intervals`` is the only module allowed to import ``pyranges1``, which needs
  Python >= 3.12 and so is only present in the ``preprocess`` environment. That
  keeps ``folds``, ``palettes``, ``plotting`` and ``regions`` importable from the
  chrombpnet environment, where the QC scripts run.
"""

__all__ = [
    "compression",
    "folds",
    "intervals",
    "palettes",
    "plotting",
    "references",
    "regions",
]
