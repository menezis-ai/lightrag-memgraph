from __future__ import annotations

import sys

from twindb_lightrag_memgraph.patches import registry
from twindb_lightrag_memgraph.server.settings import (
    TWIN_ENTITY_TYPES,
    TWIN_ENTITY_TYPES_GUIDANCE,
)


def test_native_taxonomy_overlay_preserves_language_and_adds_technology_guidance():
    original = {"addon_params": {"language": "French"}}

    configured = registry._with_twin_entity_taxonomy(original)

    assert configured is not original
    assert configured["addon_params"]["language"] == "French"
    assert configured["addon_params"]["entity_types"] == list(TWIN_ENTITY_TYPES)
    assert (
        configured["addon_params"]["entity_types_guidance"]
        == TWIN_ENTITY_TYPES_GUIDANCE
    )
    assert "Latin" in configured["addon_params"]["entity_types_guidance"]
    assert original == {"addon_params": {"language": "French"}}


# The 1.4.x stock list (lightrag.constants.DEFAULT_ENTITY_TYPES) — inlined
# because 1.5.x moved it out of lightrag.constants; the helper import is
# degradable and these tests pin the LOGIC against a synthetic stock list.
_STOCK_14X = [
    "Person",
    "Creature",
    "Organization",
    "Location",
    "Event",
    "Concept",
    "Method",
    "Content",
    "Data",
    "Artifact",
    "NaturalObject",
]


def test_native_taxonomy_replaces_the_stock_default_list(monkeypatch):
    """QA GRA-tech root cause: LightRAG 1.4.9.11's server ALWAYS passes
    ``entity_types`` (``DEFAULT_ENTITY_TYPES`` when no ``ENTITY_TYPES`` env is
    set — a list with no "Technology"), so the previous ``setdefault`` never
    applied on the native path and the category stayed structurally empty.
    A stock-default incoming list must be replaced by Twin's taxonomy."""
    monkeypatch.setattr(
        registry, "_stock_default_entity_types", lambda: list(_STOCK_14X)
    )
    original = {
        "addon_params": {
            "language": "French",
            "entity_types": list(_STOCK_14X),
        }
    }

    configured = registry._with_twin_entity_taxonomy(original)

    assert configured["addon_params"]["entity_types"] == list(TWIN_ENTITY_TYPES)
    assert "Technology" in configured["addon_params"]["entity_types"]


def test_native_taxonomy_conservative_when_stock_list_unknown(monkeypatch):
    """When the installed LightRAG exposes no stock list (1.5.x moved it),
    an incoming list cannot be attributed → it is preserved (the 1.5.x
    server passes no entity_types at all, so Twin's list still lands via
    the None branch)."""
    monkeypatch.setattr(registry, "_stock_default_entity_types", lambda: None)
    original = {"addon_params": {"entity_types": list(_STOCK_14X)}}

    configured = registry._with_twin_entity_taxonomy(original)

    assert configured["addon_params"]["entity_types"] == list(_STOCK_14X)


def test_native_taxonomy_preserves_an_operator_customized_list():
    """An explicit ``ENTITY_TYPES`` env (any list that is NOT the stock
    default) is an operator decision — it must survive the overlay."""
    custom = ["Person", "Organization", "Malware", "Indicator"]
    original = {"addon_params": {"entity_types": list(custom)}}

    configured = registry._with_twin_entity_taxonomy(original)

    assert configured["addon_params"]["entity_types"] == custom
    # The guidance stays additive (inert on 1.4.x, consumed on 1.5.x).
    assert (
        configured["addon_params"]["entity_types_guidance"]
        == TWIN_ENTITY_TYPES_GUIDANCE
    )


def test_native_server_constructor_receives_twin_taxonomy(monkeypatch):
    saved_argv = sys.argv
    sys.argv = ["lightrag"]
    try:
        import lightrag.api.lightrag_server as server
    finally:
        sys.argv = saved_argv

    received: dict = {}

    def fake_lightrag(*_args, **kwargs):
        received.update(kwargs)
        return object()

    monkeypatch.setattr(server, "LightRAG", fake_lightrag)
    monkeypatch.setattr(server, "_twindb_entity_taxonomy_patched", False, raising=False)

    registry._patch_native_entity_taxonomy()
    # The REAL 1.4.9.11 call shape: the native server always fills
    # entity_types itself (stock default when ENTITY_TYPES is unset). The
    # earlier version of this test omitted the key — modelling the 1.5.x
    # caller — and green-lit a patch that was inert on that pin.
    monkeypatch.setattr(
        registry, "_stock_default_entity_types", lambda: list(_STOCK_14X)
    )
    server.LightRAG(
        addon_params={
            "language": "English",
            "entity_types": list(_STOCK_14X),
        }
    )

    assert received["addon_params"]["language"] == "English"
    assert received["addon_params"]["entity_types"] == list(TWIN_ENTITY_TYPES)
    assert (
        received["addon_params"]["entity_types_guidance"] == TWIN_ENTITY_TYPES_GUIDANCE
    )
