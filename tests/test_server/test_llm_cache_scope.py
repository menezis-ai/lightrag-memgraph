"""Query-cache wiring: the LightRAG LLM query cache must be off by default.

Finding (audit 2026-06-27): LightRAG's ``enable_llm_cache`` defaults True and
its query-cache key (``compute_args_hash`` in ``lightrag.operate``) is computed
from the query + keywords only -- it does NOT include the retrieved context
(``sys_prompt``), the conversation history, the active folder, or the
doc/tag/min_score filters. Twin keeps a single physical workspace and layers
folders as ``MEMBER_OF`` membership, so an enabled query cache returns folder
A's generated answer for the same question asked in folder B (false-grounded
answer + cross-folder text leak).

The non-fork doctrine forbids patching ``operate.py`` to widen the hash, and
``QueryParam`` exposes no per-query cache toggle, so the only doctrine-compliant
lever is the constructor flag ``enable_llm_cache``. These tests lock the
Twin-safe default and the opt-in override.
"""

from __future__ import annotations

import inspect

from lightrag import LightRAG

from twindb_lightrag_memgraph.server.app import _build_rag_kwargs
from twindb_lightrag_memgraph.server.settings import (
    LightRAGServerSettings,
    TWIN_ENTITY_TYPES,
    TWIN_ENTITY_TYPES_GUIDANCE,
)


def _settings(**overrides: object) -> LightRAGServerSettings:
    base: dict[str, object] = {
        "memgraph_uri": "bolt://localhost:7687",
        "api_key": "t",
        "llm_binding_api_key": "t",
        "embedding_binding_api_key": "t",
    }
    base.update(overrides)
    return LightRAGServerSettings(**base)


def test_query_cache_disabled_by_default() -> None:
    """Default settings must construct LightRAG with the query cache off."""
    settings = _settings()
    assert settings.enable_llm_cache is False

    kwargs = _build_rag_kwargs(settings, embedding_func=object(), llm_func=object())
    assert kwargs["enable_llm_cache"] is False


def test_query_cache_opt_in_passthrough() -> None:
    """Operators can re-enable the cache; the flag must reach the constructor."""
    settings = _settings(enable_llm_cache=True)
    assert settings.enable_llm_cache is True

    kwargs = _build_rag_kwargs(settings, embedding_func=object(), llm_func=object())
    assert kwargs["enable_llm_cache"] is True


def test_rag_kwargs_carry_static_wiring() -> None:
    """Extraction of the helper must not drop the storage/chunk wiring."""
    settings = _settings(workspace="acme")
    kwargs = _build_rag_kwargs(settings, embedding_func="ef", llm_func="lf")

    assert kwargs["kv_storage"] == settings.kv_storage
    assert kwargs["vector_storage"] == settings.vector_storage
    assert kwargs["graph_storage"] == settings.graph_storage
    assert kwargs["doc_status_storage"] == settings.doc_status_storage
    assert kwargs["chunk_token_size"] == settings.chunk_token_size
    assert kwargs["embedding_func"] == "ef"
    assert kwargs["llm_model_func"] == "lf"
    assert kwargs["workspace"] == "acme"


def test_rag_kwargs_carry_twin_entity_extraction_profile() -> None:
    """Technology must be explicit in LightRAG's extraction prompt profile."""
    settings = _settings(max_gleaning=3)
    kwargs = _build_rag_kwargs(settings, embedding_func=object(), llm_func=object())
    supported = inspect.signature(LightRAG).parameters

    if "entity_extract_max_gleaning" in supported:
        assert kwargs["entity_extract_max_gleaning"] == 3
    else:
        assert "entity_extract_max_gleaning" not in kwargs
    if "entity_extract_max_records" in supported:
        assert (
            kwargs["entity_extract_max_records"] == settings.entity_extract_max_records
        )
    else:
        assert "entity_extract_max_records" not in kwargs
    if "entity_extract_max_entities" in supported:
        assert (
            kwargs["entity_extract_max_entities"]
            == settings.entity_extract_max_entities
        )
    else:
        assert "entity_extract_max_entities" not in kwargs
    if "addon_params" in supported:
        assert kwargs["addon_params"]["entity_types"] == list(TWIN_ENTITY_TYPES)
        assert "Technology" in kwargs["addon_params"]["entity_types"]
        assert (
            kwargs["addon_params"]["entity_types_guidance"]
            == TWIN_ENTITY_TYPES_GUIDANCE
        )
        assert "- Technology:" in kwargs["addon_params"]["entity_types_guidance"]
        for expected in ("UNIX", "grep", "sed", "ssh", "KnowRob", "Accelerate"):
            assert expected in kwargs["addon_params"]["entity_types_guidance"]
        assert "Latin" in kwargs["addon_params"]["entity_types_guidance"]
    else:
        assert "addon_params" not in kwargs


def test_rag_kwargs_omit_workspace_when_unset() -> None:
    """Empty workspace must not be forwarded (LightRAG default workspace)."""
    settings = _settings(workspace="")
    kwargs = _build_rag_kwargs(settings, embedding_func=object(), llm_func=object())
    assert "workspace" not in kwargs
