"""
Copyright (c) 2023 Jim Pivarski. All rights reserved.

tiled-uproot: Stores ROOT metadata in Tiled for quicker sliced-array access.
"""


from __future__ import annotations

import tiled_uproot.extract
import tiled_uproot.populate  # noqa: F401

from ._version import version as __version__

__all__ = ("__version__", "populate", "extract")
