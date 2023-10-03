from __future__ import annotations

import importlib.metadata

import tiled_uproot as m


def test_version():
    assert importlib.metadata.version("tiled_uproot") == m.__version__
