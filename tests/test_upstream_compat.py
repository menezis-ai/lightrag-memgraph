"""
Upstream LightRAG compatibility guard tests (offline, no Memgraph required).

This package bridges LightRAG's internal storage registry without modifying
LightRAG's source. That contract is fragile across LightRAG releases: the
version pin (``lightrag-hku>=1.4.9,<2.0.0``) permits minor releases that add
abstract methods to the storage base classes or rename ``QueryParam`` fields.
When that happens the failure is severe and silent until runtime:

  * A new abstract method on ``DocStatusStorage`` makes
    ``MemgraphDocStatusStorage`` impossible to instantiate — the whole
    doc-status backend breaks (regression observed against LightRAG 1.5.x,
    which added ``get_doc_by_content_hash`` / ``get_doc_by_file_basename``).
  * A removed/renamed ``QueryParam`` field (e.g. ``history_turns`` dropped in
    1.5) makes every Twin query 500 with ``TypeError``.

These tests assert the compatibility contract against whatever LightRAG is
actually installed, so the class of "upstream changed its interface" breakage
is caught at test time instead of in production.
"""

import dataclasses
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from twindb_lightrag_memgraph.docstatus_impl import MemgraphDocStatusStorage
from twindb_lightrag_memgraph.kv_impl import MemgraphKVStorage
from twindb_lightrag_memgraph.vector_impl import MemgraphVectorDBStorage

STORAGE_CLASSES = [
    MemgraphKVStorage,
    MemgraphVectorDBStorage,
    MemgraphDocStatusStorage,
]


@pytest.mark.parametrize(
    "storage_cls", STORAGE_CLASSES, ids=lambda c: c.__name__
)
def test_storage_class_is_concrete(storage_cls):
    """Every Memgraph storage backend must implement all abstract methods
    declared by the installed LightRAG base class.

    A non-empty ``__abstractmethods__`` means LightRAG added an abstract
    method that this package does not override — the backend cannot be
    instantiated and the storage slot is broken for that LightRAG release.
    """
    unimplemented = sorted(getattr(storage_cls, "__abstractmethods__", set()))
    assert unimplemented == [], (
        f"{storage_cls.__name__} does not implement abstract method(s) "
        f"{unimplemented} required by the installed LightRAG. "
        "Implement them (or retighten the lightrag-hku version pin)."
    )


def test_docstatus_storage_instantiates():
    """The DocStatus backend must be constructible against installed LightRAG.

    This guards the exact regression seen on 1.5.x where the class was abstract
    and ``__new__`` / ``__init__`` raised ``TypeError``.
    """
    store = MemgraphDocStatusStorage(
        namespace="docstatus",
        global_config={},
        embedding_func=MagicMock(),
    )
    assert store is not None


def test_docstatus_serializes_content_hash():
    """LightRAG 1.5 duplicate detection writes ``content_hash`` through
    DocProcessingStatus; the Memgraph row must preserve it as a top-level
    property so ``get_doc_by_content_hash`` can use an indexed lookup.
    """
    from lightrag.base import DocProcessingStatus, DocStatus

    kwargs = {}
    if "content_hash" in getattr(DocProcessingStatus, "__dataclass_fields__", {}):
        kwargs["content_hash"] = "abc123"
    props = MemgraphDocStatusStorage._serialize_status(
        "doc-1",
        DocProcessingStatus(
            content_summary="hello",
            content_length=5,
            file_path="report.pdf",
            status=DocStatus.PENDING,
            created_at="2026-06-17T00:00:00Z",
            updated_at="2026-06-17T00:00:00Z",
            **kwargs,
        ),
    )

    if kwargs:
        assert props["content_hash"] == "abc123"
    else:
        assert "content_hash" not in props


def _docstatus_read_session(record: dict | None):
    result = AsyncMock()
    result.single = AsyncMock(return_value=record)
    result.consume = AsyncMock()

    session = AsyncMock()
    session.run = AsyncMock(return_value=result)
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    return session


@pytest.mark.asyncio
async def test_docstatus_get_doc_by_file_basename_queries_canonical_file_path():
    storage = MemgraphDocStatusStorage(
        namespace="docstatus",
        global_config={},
        embedding_func=MagicMock(),
    )
    session = _docstatus_read_session(
        {
            "id": "doc-1",
            "props": {
                "file_path": "report.pdf",
                "metadata": '{"source": "upload"}',
                "chunks_list": '["chunk-1"]',
            },
        }
    )

    with patch(
        "twindb_lightrag_memgraph._pool.get_read_session",
        return_value=session,
    ):
        result = await storage.get_doc_by_file_basename("report.pdf")

    assert result == (
        "doc-1",
        {
            "file_path": "report.pdf",
            "metadata": {"source": "upload"},
            "chunks_list": ["chunk-1"],
        },
    )
    assert session.run.call_args.kwargs["basename"] == "report.pdf"


@pytest.mark.asyncio
async def test_docstatus_get_doc_by_content_hash_queries_top_level_hash():
    storage = MemgraphDocStatusStorage(
        namespace="docstatus",
        global_config={},
        embedding_func=MagicMock(),
    )
    session = _docstatus_read_session(
        {
            "id": "doc-2",
            "props": {
                "file_path": "other.pdf",
                "content_hash": "abc123",
            },
        }
    )

    with patch(
        "twindb_lightrag_memgraph._pool.get_read_session",
        return_value=session,
    ):
        result = await storage.get_doc_by_content_hash("abc123")

    assert result == (
        "doc-2",
        {
            "file_path": "other.pdf",
            "content_hash": "abc123",
        },
    )
    assert session.run.call_args.kwargs["content_hash"] == "abc123"


def _query_param_cls():
    from lightrag.base import QueryParam

    return QueryParam


def test_query_param_builder_survives_unknown_fields():
    """``_make_query_param`` must not crash on kwargs the installed
    ``QueryParam`` constructor does not accept.

    ``tag_filter`` is a Twin-only extension never present upstream, and
    ``history_turns`` was removed from ``QueryParam`` in LightRAG 1.5. Both
    must round-trip as runtime attributes rather than 500-ing the endpoint.
    """
    from twindb_lightrag_memgraph.server.twin_query_routes import _make_query_param

    qp_cls = _query_param_cls()
    param = _make_query_param(
        qp_cls,
        {
            "mode": "mix",
            "top_k": 5,
            "history_turns": 3,
            "tag_filter": {"all": ["rman"], "any": []},
        },
    )

    assert param.top_k == 5
    assert getattr(param, "history_turns") == 3
    assert getattr(param, "tag_filter") == {"all": ["rman"], "any": []}


def test_query_param_builder_passes_known_fields_through_constructor():
    """Known fields must reach the dataclass constructor (not just setattr),
    so dataclass defaults/validation still apply."""
    from twindb_lightrag_memgraph.server.twin_query_routes import _make_query_param

    qp_cls = _query_param_cls()
    ctor_fields = {f.name for f in dataclasses.fields(qp_cls)}
    assert "top_k" in ctor_fields  # sanity: field name is still current

    param = _make_query_param(qp_cls, {"top_k": 11})
    assert param.top_k == 11
