"""Compatibility module for Twin query routes.

The implementation lives in :mod:`twindb_lightrag_memgraph.server.query.router`.
The module object is aliased so monkeypatches against the historical import
path still affect the functions used by ``build_twin_query_router``.
"""

from __future__ import annotations

import sys

from .query import router as _router_module

sys.modules[__name__] = _router_module
