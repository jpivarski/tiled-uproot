from __future__ import annotations

import importlib.metadata

import tiled_uproot


def test_version():
    assert importlib.metadata.version("tiled_uproot") == tiled_uproot.__version__
