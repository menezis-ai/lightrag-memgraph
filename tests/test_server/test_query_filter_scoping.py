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


class TestTagFilterGroupsBinding:
    """OR-of-groups ``tag_filter`` (``{"groups": [...]}``): OR between groups,
    flat semantics inside a group, both wire forms bound at the same storage
    seam."""

    async def test_grouped_filter_binds_tag_groups_lowercased(self, client):
        r = await client.post(
            "/query",
            json={
                "query": "q",
                "mode": "mix",
                "tag_filter": {
                    "groups": [
                        {"all": ["CFT_VM", "Client"]},
                        {"any": ["Transverse"]},
                    ]
                },
            },
        )
        assert r.status_code == 200
        bound = client._rag.filters_in_llm
        assert isinstance(bound, RetrievalFilters)
        assert bound.tag_groups == (
            (frozenset({"cft_vm", "client"}), frozenset()),
            (frozenset(), frozenset({"transverse"})),
        )
        assert bound.tag_all == frozenset()
        assert bound.tag_any == frozenset()
        assert bound.has_tag

    async def test_single_group_binds_exactly_like_flat(self, client):
        # Strict compat inside the new form: one group ≡ the flat filter, so it
        # binds the flat sets and produces the identical storage Cypher.
        r = await client.post(
            "/query",
            json={
                "query": "q",
                "mode": "mix",
                "tag_filter": {"groups": [{"all": ["Oracle"], "any": ["RMAN"]}]},
            },
        )
        assert r.status_code == 200
        bound = client._rag.filters_in_llm
        assert isinstance(bound, RetrievalFilters)
        assert bound.tag_all == frozenset({"oracle"})
        assert bound.tag_any == frozenset({"rman"})
        assert bound.tag_groups == ()

    async def test_grouped_filter_bound_into_query_data(self, client):
        # /query/data is the surface the (all…) OR (any…) need was reported on.
        r = await client.post(
            "/query/data",
            json={
                "query": "q",
                "mode": "mix",
                "tag_filter": {"groups": [{"all": ["a", "b"]}, {"any": ["c"]}]},
            },
        )
        assert r.status_code == 200
        bound = client._rag.filters_in_data
        assert isinstance(bound, RetrievalFilters)
        assert len(bound.tag_groups) == 2

    @pytest.mark.parametrize(
        "payload",
        [
            {"groups": [{"all": ["a"]}], "any": ["b"]},  # mixed forms
            {"groups": []},  # empty list
            {"groups": "not-a-list"},  # wrong container type
            {"groups": [{"all": ["a"]}] * 6},  # over the 5-group cap
            {"groups": [["a"]]},  # non-object group
            {"groups": [{"bogus": ["a"]}]},  # unknown group key
            {"groups": [{"all": ["   "]}]},  # blank-only group
            {"groups": [{"all": "a"}]},  # non-list group value
        ],
    )
    async def test_invalid_grouped_filters_are_422(self, client, payload):
        r = await client.post(
            "/query", json={"query": "q", "mode": "mix", "tag_filter": payload}
        )
        assert r.status_code == 422

    async def test_doc_filter_rejects_groups_loudly(self, client):
        # doc_filter stays flat-only. 'groups' must 422 — never be silently
        # ignored — so API callers can feature-detect safely.
        r = await client.post(
            "/query",
            json={
                "query": "q",
                "mode": "mix",
                "doc_filter": {"groups": [{"any": ["docA"]}]},
            },
        )
        assert r.status_code == 422

    def test_grouped_filter_arms_the_mix_fallback(self):
        # A grouped-only filter is an advanced filter: hybrid /query/data with
        # zero graph rows must retry as mix instead of reading as a broken
        # filter (the exact shape of the reported no_results).
        from twindb_lightrag_memgraph.server.query.models import TwinQueryBody
        from twindb_lightrag_memgraph.server.query.request_scope import (
            _query_data_fallback_mode,
        )

        body = TwinQueryBody(
            query="q",
            mode="hybrid",
            tag_filter={
                "groups": [{"all": ["cft_vm", "client"]}, {"any": ["transverse"]}]
            },
        )
        assert _query_data_fallback_mode(body) == "mix"


class TestTagFilterGroupsSemantics:
    """Pure verdicts of the grouped filter at the post-filter guard-rail layer
    (``source_filters``) — the mirror of the storage Cypher pins in
    ``tests/test_retrieval_filters_scoping.py``."""

    def test_or_between_groups_matches_either_group(self):
        from twindb_lightrag_memgraph.server.query.source_filters import (
            _doc_tags_match_filter,
        )

        f = {"groups": [{"all": ["cft_vm", "client"]}, {"any": ["transverse"]}]}
        assert _doc_tags_match_filter({"cft_vm", "client"}, f)
        assert _doc_tags_match_filter({"transverse"}, f)
        assert _doc_tags_match_filter({"cft_vm", "client", "transverse"}, f)
        assert not _doc_tags_match_filter({"cft_vm"}, f)  # half of group 1
        assert not _doc_tags_match_filter({"client"}, f)
        assert not _doc_tags_match_filter(set(), f)

    def test_group_inner_all_and_any_stay_conjunctive(self):
        from twindb_lightrag_memgraph.server.query.source_filters import (
            _doc_tags_match_filter,
        )

        f = {"groups": [{"all": ["a"], "any": ["b", "c"]}]}
        assert _doc_tags_match_filter({"a", "c"}, f)
        assert not _doc_tags_match_filter({"a"}, f)
        assert not _doc_tags_match_filter({"b", "c"}, f)

    def test_flat_form_verdicts_unchanged(self):
        from twindb_lightrag_memgraph.server.query.source_filters import (
            _doc_tags_match_filter,
        )

        f = {"all": ["a"], "any": ["b", "c"]}
        assert _doc_tags_match_filter({"a", "b"}, f)
        assert not _doc_tags_match_filter({"a"}, f)
        assert not _doc_tags_match_filter({"b"}, f)

    def test_normaliser_lowercases_and_drops_blank_groups(self):
        from twindb_lightrag_memgraph.server.query.source_filters import (
            _tag_filter_groups,
        )

        groups = _tag_filter_groups(
            {"groups": [{"all": ["  X  "]}, {"any": ["", "   "]}]}
        )
        assert groups == ((frozenset({"x"}), frozenset()),)

    def test_normaliser_flat_form_is_one_group(self):
        from twindb_lightrag_memgraph.server.query.source_filters import (
            _tag_filter_groups,
        )

        assert _tag_filter_groups({"all": ["A"], "any": ["b"]}) == (
            (frozenset({"a"}), frozenset({"b"})),
        )

    def test_terms_extractor_refuses_grouped_form(self):
        # Reading a grouped payload through the flat extractor would silently
        # disable filtering — it must refuse loudly instead.
        from twindb_lightrag_memgraph.server.query.source_filters import (
            _tag_filter_terms,
        )

        with pytest.raises(ValueError):
            _tag_filter_terms({"groups": [{"all": ["a"]}]})

    def test_activity_check_covers_both_forms(self):
        from twindb_lightrag_memgraph.server.query.source_filters import (
            _tag_filter_active,
        )

        assert _tag_filter_active({"groups": [{"any": ["a"]}]})
        assert _tag_filter_active({"all": ["a"]})
        assert not _tag_filter_active(None)
        assert not _tag_filter_active({})
        assert not _tag_filter_active({"all": []})
        assert not _tag_filter_active({"groups": []})


class TestTagFilterOpenApiSchema:
    """The generated schema must expose the typed contract — both forms,
    ``extra="forbid"``, the 5-group cap and string lists — not an arbitrary
    object (PR #426 review, blocking issue 2)."""

    def test_grouped_form_is_fully_typed_in_the_schema(self):
        from twindb_lightrag_memgraph.server.query.models import TwinQueryBody

        schema = TwinQueryBody.model_json_schema()
        defs = schema["$defs"]

        grouped = defs["TagFilterGrouped"]
        assert grouped["additionalProperties"] is False
        groups = grouped["properties"]["groups"]
        assert groups["minItems"] == 1
        assert groups["maxItems"] == 5

        group = defs["TagFilterGroup"]
        assert group["additionalProperties"] is False
        assert group["properties"]["all"]["items"] == {"type": "string"}
        assert group["properties"]["any"]["items"] == {"type": "string"}

        flat = defs["TagFilterFlat"]
        assert flat["additionalProperties"] is False
        assert flat["properties"]["all"]["items"] == {"type": "string"}
        assert flat["properties"]["any"]["items"] == {"type": "string"}

        refs = {
            option.get("$ref") for option in schema["properties"]["tag_filter"]["anyOf"]
        }
        assert "#/$defs/TagFilterFlat" in refs
        assert "#/$defs/TagFilterGrouped" in refs

    def test_model_kept_payload_derived_serialisation_clean(self):
        # PR #426 review round 2: the validated union model stays in
        # body.tag_filter (so model_dump()/model_dump_json() are warning-free)
        # and the downstream chain reads the wire dict via tag_filter_payload.
        import warnings

        from twindb_lightrag_memgraph.server.query.models import (
            TagFilterFlat,
            TagFilterGrouped,
            TwinQueryBody,
        )

        grouped = TwinQueryBody(
            query="q",
            tag_filter={"groups": [{"all": ["A"]}, {"any": ["b"]}]},
        )
        assert isinstance(grouped.tag_filter, TagFilterGrouped)
        assert grouped.tag_filter_payload == {
            "groups": [{"all": ["A"]}, {"any": ["b"]}]
        }

        flat = TwinQueryBody(query="q", tag_filter={"any": ["x"]})
        assert isinstance(flat.tag_filter, TagFilterFlat)
        assert flat.tag_filter_payload == {"any": ["x"]}

        assert TwinQueryBody(query="q").tag_filter_payload is None

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            grouped.model_dump()
            grouped.model_dump_json()
            flat.model_dump()
            flat.model_dump_json()
