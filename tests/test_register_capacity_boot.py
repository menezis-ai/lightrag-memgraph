"""``register()`` refuses to boot on a malformed ``TWIN_VECTOR_INDEX_CAPACITY``.

The variable is applied at index creation, which can be minutes after boot
(first ingestion, or a query against a missing index). Validating it in
``register()`` — before any LightRAG patching — turns that late, confusing
failure into an immediate configuration error (audit 2026-08-25).

OFFLINE — the error is raised before ``register()`` touches LightRAG, so no
fake ``lightrag.kg`` module is needed; the registration flag is restored.
"""

from __future__ import annotations

import pytest

import twindb_lightrag_memgraph
from twindb_lightrag_memgraph.patches import registry


@pytest.fixture
def _fresh_registration():
    before = (twindb_lightrag_memgraph._registered, registry._registered)
    twindb_lightrag_memgraph._registered = False
    registry._registered = False
    try:
        yield
    finally:
        twindb_lightrag_memgraph._registered, registry._registered = before


@pytest.mark.parametrize("raw", ["0", "-1", "abc"])
def test_register_refuses_to_boot_on_malformed_capacity(
    monkeypatch, _fresh_registration, raw
):
    monkeypatch.setenv("TWIN_VECTOR_INDEX_CAPACITY", raw)

    with pytest.raises(ValueError, match="TWIN_VECTOR_INDEX_CAPACITY"):
        registry.register()

    # Nothing was registered: the next boot with a fixed env starts clean.
    assert registry._registered is False
