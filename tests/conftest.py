"""Shared pytest settings: the ``gpu`` marker, and GPU tests skipped unless asked for.

GPU tests need a CUDA device, so they run only inside a SLURM box and only on request. A test marked
``@pytest.mark.gpu`` is skipped unless the run's marker expression names it (``-m gpu``). A run with
``-m 'not gpu'`` deselects GPU tests through that same expression.

This file must not import torch. The default pixi environment has no torch, and ``pixi run test`` must stay
green there.
"""

from __future__ import annotations

import re

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "gpu: needs a CUDA GPU; skipped unless selected with -m gpu")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if re.search(r"\bgpu\b", config.getoption("markexpr") or ""):
        return
    skip = pytest.mark.skip(reason="GPU test: select it with -m gpu, inside a box with a GPU")
    for item in items:
        if item.get_closest_marker("gpu") is not None:
            item.add_marker(skip)
