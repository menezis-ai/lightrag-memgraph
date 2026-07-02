"""The Twin query routes bind ``doc_filter`` / ``tag_filter`` / ``min_score``
into ``storage_filter_context`` at the grounding call.

This is the route-wiring half of the fix for the audit "faux grounding" gap:
these filters used to be attached to ``QueryParam`` and read by nothing in the
retrieval path — the LLM grounded on the *unfiltered* context and only the
Sources panel was trimmed afterwards. They are now enforced at the Memgraph
storage layer, which reads them from a ContextVar.

Three layers prove the guarantee "the LLM never grounds on an out-of-filter
chunk":

1. **here** — the request's filters are *bound* into
   ``get_active_retrieval_filters()`` at the moment ``aquery_llm`` /
   ``aquery_data`` / ``aquery`` fires (so the storage layer is engaged with the
   right filters);
2. ``tests/test_retrieval_filters_scoping.py`` — those filters become the
   correct Cypher exclusion predicates;
3. the ``@pytest.mark.integration`` cases there — the predicates actually drop
   out-of-filter rows on a real Memgraph.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from twindb_lightrag_memgraph._constants import (
    RetrievalFilters,
    get_active_retrieval_filters,
)
from twindb_lightrag_memgraph.server.twin_query_routes import build_twin_query_router


def _envelope(answer: str = "grounded answer") -> dict[str, Any]:
    return {
        "status": "success",
        "message": "ok",
        "data": {
            "entities": [],
            "relationships": [],
            "chunks": [],
            "references": [{"reference_id": "1", "file_path": "/kb/a.pdf"}],
        },
        "metadata": {},
        "llm_response": {
            "content": answer,
            "response_iterator": None,
            "is_streaming": False,
        },
    }


class CapturingRag:
    """Records the retrieval filters visible (via the ContextVar) in each API."""

    def __init__(self) -> None:
        self.filters_in_llm: RetrievalFilters | None = "<unset>"  # type: ignore[assignment]
        self.filters_in_query: RetrievalFilters | None = "<unset>"  # type: ignore[assignment]
        self.filters_in_data: RetrievalFilters | None = "<unset>"  # type: ignore[assignment]

    async def aquery_llm(self, query: str, *, param):
        self.filters_in_llm = get_active_retrieval_filters()
        return _envelope()

    async def aquery(self, query: str, *, param):
        self.filters_in_query = get_active_retrieval_filters()
        return "context body"

    async def aquery_data(self, query: str, *, param):
        self.filters_in_data = get_active_retrieval_filters()
        return {"status": "success", "message": "ok", "data": {}, "metadata": {}}


@pytest.fixture()
async def client(monkeypatch):
    monkeypatch.setenv("TWIN_DEFAULT_FOLDER", "default")
    monkeypatch.setenv(
        "TWIN_FOLDERS_JSON",
        json.dumps([{"id": "default", "label": "Default", "kind": "primary"}]),
    )
    rag = CapturingRag()
    app = FastAPI()
    app.include_router(build_twin_query_router(lambda: rag))
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        c._rag = rag
        yield c


class TestQueryBindsFilterContext:
    async def test_doc_filter_bound_into_aquery_llm(self, client):
        # "/query with doc_filter must never call the LLM with a chunk outside
        # the doc" — proven here by the filter being active when aquery_llm runs;
        # the actual chunk exclusion is the storage Cypher (unit + integration).
        r = await client.post(
            "/query",
            json={"query": "q", "mode": "mix", "doc_filter": {"any": ["docA"]}},
        )
        assert r.status_code == 200
        bound = client._rag.filters_in_llm
        assert isinstance(bound, RetrievalFilters)
        assert bound.doc_any == frozenset({"docA"})
        assert not bound.has_tag

    async def test_tag_filter_bound_and_lowercased(self, client):
        # "/query with tag_filter must never call the LLM with a chunk outside
        # the tag." Tags are normalised to lower-case (case-insensitive ids).
        r = await client.post(
            "/query",
            json={
                "query": "q",
                "mode": "mix",
                "tag_filter": {"all": ["Oracle"], "any": ["RMAN"]},
            },
        )
        assert r.status_code == 200
        bound = client._rag.filters_in_llm
        assert isinstance(bound, RetrievalFilters)
        assert bound.tag_all == frozenset({"oracle"})
        assert bound.tag_any == frozenset({"rman"})

    async def test_query_data_respects_doc_tag_and_min_score(self, client):
        # #2 fix: /query/data used to ignore doc_filter + min_score entirely.
        r = await client.post(
            "/query/data",
            json={
                "query": "q",
                "mode": "mix",
                "doc_filter": {"all": ["docA", "docB"]},
                "tag_filter": {"any": ["oracle"]},
                "min_score": 0.42,
            },
        )
        assert r.status_code == 200
        bound = client._rag.filters_in_data
        assert isinstance(bound, RetrievalFilters)
        assert bound.doc_all == frozenset({"docA", "docB"})
        assert bound.tag_any == frozenset({"oracle"})
        assert bound.min_score == pytest.approx(0.42)

    async def test_min_score_bound_into_aquery_llm(self, client):
        r = await client.post(
            "/query",
            json={"query": "q", "mode": "mix", "min_score": 0.5},
        )
        assert r.status_code == 200
        bound = client._rag.filters_in_llm
        assert isinstance(bound, RetrievalFilters)
        assert bound.min_score == pytest.approx(0.5)

    async def test_only_need_context_binds_filters_into_aquery(self, client):
        r = await client.post(
            "/query",
            json={
                "query": "q",
                "mode": "mix",
                "only_need_context": True,
                "doc_filter": {"any": ["docA"]},
            },
        )
        assert r.status_code == 200
        bound = client._rag.filters_in_query
        assert isinstance(bound, RetrievalFilters)
        assert bound.doc_any == frozenset({"docA"})

    async def test_stream_binds_filters_inside_generator(self, client):
        r = await client.post(
            "/query/stream",
            json={"query": "q", "mode": "mix", "doc_filter": {"any": ["docA"]}},
        )
        assert r.status_code == 200
        bound = client._rag.filters_in_llm
        assert isinstance(bound, RetrievalFilters)
        assert bound.doc_any == frozenset({"docA"})

    async def test_absent_filters_leave_context_unset(self, client):
        # Strict compat: no filters in the body ⇒ no retrieval-filter context, so
        # the storage layer takes the byte-for-byte legacy/folder path.
        r = await client.post("/query", json={"query": "q", "mode": "mix"})
        assert r.status_code == 200
        assert client._rag.filters_in_llm is None
