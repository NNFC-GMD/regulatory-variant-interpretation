"""Shared pytest settings: the ``gpu`` marker, GPU tests skipped unless asked for, and the attribution engine's
tests skipped outside the environment it was validated in.

GPU tests need a CUDA device, so they run only inside a SLURM box and only on request. A test marked
``@pytest.mark.gpu`` is skipped unless the run's marker expression names it (``-m gpu``). A run with
``-m 'not gpu'`` deselects GPU tests through that same expression.

The engine (workflows/attribution/cheri_dls.py, attribute_fast.py) reproduces Cherimoya 0.3.0's and tangermeme
1.5.0's arithmetic and refuses other versions. Its tests (test_dls_*.py) run in the cherimoya-030 environment
(``pixi run -e cherimoya-030 test-engine``) and are skipped, with that reason, where other versions are installed
(the cherimoya environment's 0.3.1).

This file must not import torch. The default pixi environment has no torch, and ``pixi run test`` must stay
green there.
"""

from __future__ import annotations

import re
from importlib import metadata

import pytest

ENGINE_VERSIONS = {"cherimoya": "0.3.0", "tangermeme": "1.5.0"}
"""cheri_dls.EXPECTED_VERSIONS, read here without importing the engine (it imports torch)."""


def _engine_mismatch() -> str | None:
    found = {}
    for name in ENGINE_VERSIONS:
        try:
            found[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            return None  # not installed: the tests' own importorskip skips them
    if found != ENGINE_VERSIONS:
        have = ", ".join(f"{k} {v}" for k, v in found.items())
        return f"the attribution engine's tests need cherimoya 0.3.0 and tangermeme 1.5.0 (here {have}): pixi run -e cherimoya-030 test-engine"
    return None


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "gpu: needs a CUDA GPU; skipped unless selected with -m gpu")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    mismatch = _engine_mismatch()
    if mismatch is not None:
        skip_engine = pytest.mark.skip(reason=mismatch)
        for item in items:
            if item.path.name.startswith("test_dls_"):
                item.add_marker(skip_engine)
    if re.search(r"\bgpu\b", config.getoption("markexpr") or ""):
        return
    skip = pytest.mark.skip(reason="GPU test: select it with -m gpu, inside a box with a GPU")
    for item in items:
        if item.get_closest_marker("gpu") is not None:
            item.add_marker(skip)
