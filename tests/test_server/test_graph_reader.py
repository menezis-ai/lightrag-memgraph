"""Unit tests for ``server.graph_reader`` — type mapping + layout +
shape projection from Memgraph rows to WebUI GraphEntity/Relation.

Integration coverage (real Memgraph) lives in
``tests/test_server/test_webui_router_graph.py`` (marked ``integration``).
"""

from __future__ import annotations

import pytest

from twindb_lightrag_memgraph.server.graph_reader import (
    _edge_record_to_relation,
    _build_native_relations,
    _entity_id_to_node_id,
    _project_relation_rows,
    _node_record_to_entity,
    _RELATION_ENDPOINT_CACHE,
    layout_position,
    map_entity_type,
)


class TestMapEntityType:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("organization", "ORG"),
            ("ORG", "ORG"),
            ("Company", "ORG"),
            ("Person", "PERSON"),
            ("PEOPLE", "PERSON"),
            ("geo", "LOCATION"),
            ("Datacenter", "LOCATION"),
            ("product", "PRODUCT"),
            ("Database", "PRODUCT"),
            ("technology", "TECHNOLOGY"),
            ("PROTOCOL", "TECHNOLOGY"),
            ("concept", "CONCEPT"),
            ("event", "CONCEPT"),
        ],
    )
    def test_maps_common_lightrag_types(self, raw, expected):
        assert map_entity_type(raw) == expected

    def test_falls_back_to_concept_on_unknown(self):
        assert map_entity_type("random-bullshit-bingo") == "CONCEPT"

    def test_falls_back_to_concept_on_empty_or_none(self):
        assert map_entity_type("") == "CONCEPT"
        assert map_entity_type(None) == "CONCEPT"
        assert map_entity_type("   ") == "CONCEPT"


class TestLayoutPosition:
    def test_same_id_same_position(self):
        # Determinism is the whole point — page reloads must not shuffle.
        assert layout_position("Oracle Database", "PRODUCT") == layout_position(
            "Oracle Database", "PRODUCT"
        )

    def test_different_ids_get_different_positions(self):
        # Probabilistic but the hash space is huge — for 4 distinct ids
        # we shouldn't see a collision.
        positions = {
            layout_position(eid, "PRODUCT")
            for eid in ("a", "b", "c", "d")
        }
        assert len(positions) == 4

    def test_different_types_use_different_centroids(self):
        # Same id, different type → different cluster.
        p1 = layout_position("same-id", "PRODUCT")
        p2 = layout_position("same-id", "ORG")
        assert p1 != p2

    def test_positions_stay_in_canvas(self):
        for eid in ("foo", "bar", "baz", "very-long-entity-id-that-may-stress-layout"):
            for t in ("PRODUCT", "TECHNOLOGY", "CONCEPT", "ORG", "PERSON", "LOCATION"):
                x, y = layout_position(eid, t)
                # canvas 960x620 with 40 margin → coords ∈ [40, 920]×[40, 580]
                assert 40 <= x <= 920
                assert 40 <= y <= 580


class TestNodeRecordProjection:
    def test_full_record_projects_to_webui_entity(self):
        row = {
            "entity_id": "Oracle Database",
            "entity_type": "Database",
            "description": "Relational engine.",
            "source_id": "chunk-1,chunk-2,chunk-3",
        }
        out = _node_record_to_entity(row)
        assert out["id"] == "kg_Oracle Database"
        assert out["name"] == "Oracle Database"
        assert out["type"] == "PRODUCT"
        assert out["summary"] == "Relational engine."
        assert out["mentions"] == 3
        assert out["sources"] == 3  # chunk count proxy
        # Position deterministic + clamped
        assert 40 <= out["x"] <= 920
        assert 40 <= out["y"] <= 580

    def test_handles_missing_fields_gracefully(self):
        row = {"entity_id": "lone"}
        out = _node_record_to_entity(row)
        assert out["name"] == "lone"
        assert out["type"] == "CONCEPT"  # fallback
        assert out["summary"] == ""
        assert out["mentions"] == 0
        assert out["sources"] == 0

    def test_source_id_with_sep_separator(self):
        # LightRAG sometimes joins source ids with `<SEP>` instead of comma.
        row = {
            "entity_id": "x",
            "entity_type": "concept",
            "source_id": "chunk-a<SEP>chunk-b",
        }
        out = _node_record_to_entity(row)
        assert out["mentions"] == 2
        # "sources" tracks distinct chunks until we wire DocStatus join
        assert out["sources"] == 2

    def test_source_id_deduplicates_repeated_chunks(self):
        row = {
            "entity_id": "x",
            "entity_type": "concept",
            "source_id": "chunk-a,chunk-a,chunk-b",
        }
        out = _node_record_to_entity(row)
        # de-dup: same chunk listed twice doesn't double-count
        assert out["mentions"] == 2

    def test_sources_collapses_chunks_to_distinct_docs(self):
        # 5 chunks → 2 parent docs → sources should be 2, mentions stays 5.
        row = {
            "entity_id": "Speaker 1",
            "entity_type": "person",
            "source_id": "c1,c2,c3,c4,c5",
        }
        chunk_to_doc = {
            "c1": "doc-A",
            "c2": "doc-A",
            "c3": "doc-A",
            "c4": "doc-B",
            "c5": "doc-B",
        }
        out = _node_record_to_entity(row, chunk_to_doc)
        assert out["mentions"] == 5
        assert out["sources"] == 2
        assert out["source_docs"] == ["doc-A", "doc-B"]

    def test_sources_falls_back_to_mentions_for_orphan_chunks(self):
        # Chunks not present in the index (e.g. DocStatus row missing
        # chunks_list) should keep the legacy behaviour rather than 0.
        row = {
            "entity_id": "x",
            "entity_type": "concept",
            "source_id": "orphan-1,orphan-2",
        }
        out = _node_record_to_entity(row, {"unrelated": "doc-Z"})
        assert out["mentions"] == 2
        assert out["sources"] == 2
        assert out["source_docs"] == []

    def test_description_sep_marker_replaced(self):
        # LightRAG joins per-chunk descriptions with <SEP> — must not
        # leak into the WebUI inspector summary.
        row = {
            "entity_id": "Ubuntu",
            "entity_type": "org",
            "description": "Ubuntu is Linux-based.<SEP>Ubuntu is popular.",
        }
        out = _node_record_to_entity(row)
        assert "<SEP>" not in out["summary"]
        assert out["summary"] == "Ubuntu is Linux-based. · Ubuntu is popular."

    def test_summary_truncated_at_600_chars(self):
        row = {
            "entity_id": "wordy",
            "entity_type": "concept",
            "description": "x" * 1000,
        }
        out = _node_record_to_entity(row)
        assert len(out["summary"]) == 600

    def test_twin_overlay_fields_round_trip_from_node_properties(self):
        row = {
            "entity_id": "Oracle Database",
            "display_name": "Oracle DB",
            "entity_type": "Database",
            "description": "Relational engine.",
            "source_id": "chunk-1",
            "twin_tags_json": '["critical", "db"]',
            "twin_props_json": '{"owner": "dba", "tier": "gold"}',
        }
        out = _node_record_to_entity(row)
        assert out["name"] == "Oracle DB"
        assert out["tags"] == ["critical", "db"]
        assert out["properties"] == {"owner": "dba", "tier": "gold"}


class TestEdgeRecordProjection:
    def test_full_edge_projects(self):
        row = {
            "source_id": "Oracle Database",
            "target_id": "RHEL 9",
            "keywords": "runs on",
            "weight": 0.88,
            "twin_props_json": '{"since": "2024"}',
        }
        out = _edge_record_to_relation(row, 0)
        # id is endpoint-derived, stable, opaque
        assert out["id"].startswith("kr_")
        assert len(out["id"]) == 3 + 12
        assert out["source"] == "kg_Oracle Database"
        assert out["target"] == "kg_RHEL 9"
        assert out["label"] == "RUNS_ON"
        assert out["strength"] == 0.88
        assert out["properties"] == {"since": "2024"}

    def test_edge_id_stable_across_calls(self):
        row = {
            "source_id": "a",
            "target_id": "b",
            "keywords": "uses",
            "weight": 0.5,
        }
        out1 = _edge_record_to_relation(row, 0)
        out2 = _edge_record_to_relation(row, 99)  # different index — same id
        assert out1["id"] == out2["id"]

    def test_edge_id_distinguishes_direction(self):
        # a→b ≠ b→a (LightRAG :DIRECTED edges are direction-bearing on read)
        a_to_b = _edge_record_to_relation(
            {"source_id": "a", "target_id": "b", "keywords": "", "weight": 0.5}, 0
        )
        b_to_a = _edge_record_to_relation(
            {"source_id": "b", "target_id": "a", "keywords": "", "weight": 0.5}, 0
        )
        assert a_to_b["id"] != b_to_a["id"]

    def test_weight_above_one_is_normalised(self):
        # LightRAG sometimes outputs weights on a 0..10 scale.
        row = {
            "source_id": "a",
            "target_id": "b",
            "keywords": "uses",
            "weight": 9,
        }
        out = _edge_record_to_relation(row, 0)
        assert out["strength"] == 0.9

    def test_missing_weight_defaults_to_05(self):
        row = {
            "source_id": "a",
            "target_id": "b",
            "keywords": "uses",
            "weight": None,
        }
        out = _edge_record_to_relation(row, 0)
        assert out["strength"] == 0.5

    def test_empty_keywords_default_label(self):
        row = {
            "source_id": "a",
            "target_id": "b",
            "keywords": "",
            "weight": 0.5,
        }
        out = _edge_record_to_relation(row, 0)
        assert out["label"] == "RELATED_TO"

    def test_node_id_namespacing_is_consistent(self):
        # Ensure the same prefix is applied on both sides so endpoints
        # match the entities returned by `_node_record_to_entity`.
        assert _entity_id_to_node_id("x") == "kg_x"


class TestPatchTranslators:
    def test_entity_patch_summary_maps_to_description(self):
        from twindb_lightrag_memgraph.server.graph_reader import (
            _entity_patch_to_props,
        )

        props = _entity_patch_to_props({"summary": "New summary"})
        assert props == {"description": "New summary"}

    def test_entity_patch_type_maps_to_entity_type(self):
        from twindb_lightrag_memgraph.server.graph_reader import (
            _entity_patch_to_props,
        )

        props = _entity_patch_to_props({"type": "TECHNOLOGY"})
        assert props == {"entity_type": "TECHNOLOGY"}

    def test_entity_patch_name_maps_to_display_name_not_pk(self):
        from twindb_lightrag_memgraph.server.graph_reader import (
            _entity_patch_to_props,
        )

        # The immutable entity_id (Memgraph PK) must NOT be touched —
        # only an auxiliary display_name property.
        props = _entity_patch_to_props({"name": "Renamed"})
        assert props == {"display_name": "Renamed"}
        assert "entity_id" not in props

    def test_entity_patch_tags_serialized_as_json(self):
        from twindb_lightrag_memgraph.server.graph_reader import (
            _entity_patch_to_props,
        )

        props = _entity_patch_to_props({"tags": ["a", "b", "c"]})
        assert props == {"twin_tags_json": '["a", "b", "c"]'}

    def test_entity_patch_properties_serialized_as_json(self):
        from twindb_lightrag_memgraph.server.graph_reader import (
            _entity_patch_to_props,
        )

        props = _entity_patch_to_props({"properties": {"k": "v"}})
        assert props == {"twin_props_json": '{"k": "v"}'}

    def test_entity_patch_empty_returns_empty(self):
        from twindb_lightrag_memgraph.server.graph_reader import (
            _entity_patch_to_props,
        )

        assert _entity_patch_to_props({}) == {}

    def test_entity_patch_ignores_none_fields(self):
        from twindb_lightrag_memgraph.server.graph_reader import (
            _entity_patch_to_props,
        )

        # Pydantic's exclude_unset is the caller's responsibility, but
        # if a None slips through it shouldn't write a property.
        props = _entity_patch_to_props({"summary": None, "type": "ORG"})
        assert props == {"entity_type": "ORG"}

    def test_relation_patch_label_maps_to_keywords(self):
        from twindb_lightrag_memgraph.server.graph_reader import (
            _relation_patch_to_props,
        )

        props = _relation_patch_to_props({"label": "USES"})
        assert props == {"keywords": "USES"}

    def test_relation_patch_strength_maps_to_weight_float(self):
        from twindb_lightrag_memgraph.server.graph_reader import (
            _relation_patch_to_props,
        )

        props = _relation_patch_to_props({"strength": 0.42})
        assert props == {"weight": 0.42}
        assert isinstance(props["weight"], float)

    def test_relation_patch_properties_serialized(self):
        from twindb_lightrag_memgraph.server.graph_reader import (
            _relation_patch_to_props,
        )

        props = _relation_patch_to_props({"properties": {"note": "x"}})
        assert props == {"twin_props_json": '{"note": "x"}'}


class TestRelationEndpointCache:
    def test_lookup_returns_none_for_unknown_id(self):
        from twindb_lightrag_memgraph.server.graph_reader import (
            lookup_relation_endpoints,
        )

        assert lookup_relation_endpoints("kr_nonexistent") is None

    def test_lookup_returns_workspace_and_endpoints(self):
        from twindb_lightrag_memgraph.server.graph_reader import (
            _remember_relation,
            lookup_relation_endpoints,
        )

        _remember_relation("cib", "kr_test123", "A", "B")
        got = lookup_relation_endpoints("kr_test123")
        assert got == ("cib", "A", "B")

    def test_project_relation_rows_remembers_stripped_endpoints(self):
        from twindb_lightrag_memgraph.server.graph_reader import (
            lookup_relation_endpoints,
        )

        _RELATION_ENDPOINT_CACHE.clear()
        rows = [
            {
                "source_id": "A",
                "target_id": "B",
                "keywords": "depends_on",
                "weight": 0.8,
            },
        ]
        relations = _project_relation_rows(
            workspace="cib",
            rows=rows,
            valid_ids={"kg_A", "kg_B"},
            chunk_to_doc=None,
            member_docs=None,
            folder=None,
            rel_overrides={},
        )
        assert len(relations) == 1
        rel_id = relations[0]["id"]
        assert relations[0]["source"] == "kg_A"
        assert relations[0]["target"] == "kg_B"
        assert lookup_relation_endpoints(rel_id) == ("cib", "A", "B")

    def test_build_native_relations_remembers_stripped_endpoints(self):
        from twindb_lightrag_memgraph.server.graph_reader import (
            lookup_relation_endpoints,
        )

        _RELATION_ENDPOINT_CACHE.clear()

        class _Edge:
            def __init__(self, source, target):
                self.source = source
                self.target = target
                self.properties = {}

        class _Graph:
            edges = [_Edge("A", "B")]

        relations = _build_native_relations(
            _Graph(),
            workspace="cib",
            valid_ids={"kg_A", "kg_B"},
            chunk_to_doc=None,
            member_docs=None,
            active_folder=None,
            overrides={},
        )
        assert len(relations) == 1
        rel_id = relations[0]["id"]
        assert relations[0]["source"] == "kg_A"
        assert relations[0]["target"] == "kg_B"
        assert lookup_relation_endpoints(rel_id) == ("cib", "A", "B")


# ----------------------------------------------------------------------
# Contract guards for create_graph_entity (TR-KG-01)
#
# These exist to prevent a future ``except Exception: return None`` from
# reintroducing the faux-409 contract. The function-level API is
# "returns dict on success, raises typed exception on failure" — never
# ``None``.
# ----------------------------------------------------------------------


class _FakeResult:
    """Minimal async-iterable result with ``consume()``."""

    def __init__(self, rows: list[Any]):
        self._rows = rows

    def __aiter__(self):
        async def _gen():
            for row in self._rows:
                yield row

        return _gen()

    async def consume(self) -> None:  # pragma: no cover - trivial
        return None


class _FakeSession:
    def __init__(self, rows: list[Any] | None = None, *, raise_on_run: bool = False):
        self._rows = rows or []
        self._raise = raise_on_run

    async def run(self, *_args, **_kwargs):
        if self._raise:
            raise RuntimeError("memgraph unavailable")
        return _FakeResult(self._rows)


def _fake_session_cm(session: _FakeSession):
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _cm():
        yield session

    return _cm


def _fake_write_slot():
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _cm():
        yield

    return _cm


class TestCreateGraphEntityContract:
    """The function-level API contract for ``create_graph_entity``.

    Route-level coverage lives in
    ``tests/test_server/test_webui_router_graph.py``. These tests
    pin the function shape so a future regression to the
    ``return None`` sentinel is caught even if the route mapping is
    rewritten.
    """

    async def test_raises_entity_exists_error_on_duplicate(self, monkeypatch):
        from twindb_lightrag_memgraph.server import graph_reader as gr

        async def fake_exists(workspace, entity_id):
            return True

        monkeypatch.setattr(gr, "entity_exists", fake_exists)

        with pytest.raises(gr.EntityExistsError):
            await gr.create_graph_entity(
                "cib", {"name": "Existing", "type": "PRODUCT"}
            )

    async def test_raises_backend_error_on_empty_name(self):
        from twindb_lightrag_memgraph.server import graph_reader as gr

        # Direct callers that bypass Pydantic must still see a typed
        # failure rather than a silent ``None``.
        with pytest.raises(gr.EntityCreateBackendError):
            await gr.create_graph_entity("cib", {"name": "   ", "type": "PRODUCT"})

    async def test_raises_backend_error_when_session_run_fails(self, monkeypatch):
        from twindb_lightrag_memgraph.server import graph_reader as gr

        async def fake_exists(workspace, entity_id):
            return False

        monkeypatch.setattr(gr, "entity_exists", fake_exists)
        monkeypatch.setattr(
            gr, "acquire_write_slot", _fake_write_slot()
        )
        monkeypatch.setattr(
            gr,
            "get_session",
            _fake_session_cm(_FakeSession(raise_on_run=True)),
        )

        with pytest.raises(gr.EntityCreateBackendError):
            await gr.create_graph_entity(
                "cib", {"name": "FreshOne", "type": "PRODUCT"}
            )

    async def test_raises_projection_error_when_reread_fails(self, monkeypatch):
        from twindb_lightrag_memgraph.server import graph_reader as gr

        async def fake_exists(workspace, entity_id):
            return False

        async def fake_reread_fails(workspace, entity_id):
            raise RuntimeError("read session timeout")

        monkeypatch.setattr(gr, "entity_exists", fake_exists)
        monkeypatch.setattr(
            gr, "acquire_write_slot", _fake_write_slot()
        )
        monkeypatch.setattr(
            gr,
            "get_session",
            _fake_session_cm(_FakeSession(rows=[{"entity_id": "FreshOne"}])),
        )
        monkeypatch.setattr(gr, "_read_one_entity", fake_reread_fails)

        with pytest.raises(gr.EntityProjectionError):
            await gr.create_graph_entity(
                "cib", {"name": "FreshOne", "type": "PRODUCT"}
            )

    async def test_returns_dict_on_success_never_none(self, monkeypatch):
        """The success path must return the projected dict — not ``None``.
        This is the explicit guard Codex asked for against a future
        ``except Exception: return None`` regression that would resurrect
        the faux-409 contract."""
        from twindb_lightrag_memgraph.server import graph_reader as gr

        async def fake_exists(workspace, entity_id):
            return False

        async def fake_reread_ok(workspace, entity_id):
            return {
                "id": "kg_FreshOne",
                "name": "FreshOne",
                "type": "PRODUCT",
                "x": 100,
                "y": 100,
                "mentions": 0,
                "sources": 0,
                "summary": "",
            }

        monkeypatch.setattr(gr, "entity_exists", fake_exists)
        monkeypatch.setattr(
            gr, "acquire_write_slot", _fake_write_slot()
        )
        monkeypatch.setattr(
            gr,
            "get_session",
            _fake_session_cm(_FakeSession(rows=[{"entity_id": "FreshOne"}])),
        )
        monkeypatch.setattr(gr, "_read_one_entity", fake_reread_ok)

        out = await gr.create_graph_entity(
            "cib", {"name": "FreshOne", "type": "PRODUCT"}
        )
        assert out is not None
        assert isinstance(out, dict)
        assert out["id"] == "kg_FreshOne"
