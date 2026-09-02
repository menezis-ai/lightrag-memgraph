"""T0.4 — registry parity with the code, folder scoping, schema guards."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from twindb_lightrag_memgraph._folders import load_folder_catalog
from twindb_lightrag_memgraph.portability.stores import (
    STORES,
    SchemaViolation,
    Scope,
    StoreSchema,
    StoreSpec,
    declared_label_prefixes,
    exportable_stores,
    folder_scoped_labels,
    never_label_prefixes,
    portable_store,
    project_record,
    store_by_file,
    store_by_name,
)

SRC = Path(__file__).resolve().parents[2] / "src" / "twindb_lightrag_memgraph"
# Every f-string label the runtime builds: ``f"Prefix_{...}"`` — the plan's §9.5 list.
_LABEL_RE = re.compile(
    r'f"(KV|Vec|DocStatus|Folder|Webui[A-Za-z]+|TwinSourceLink|Graph(?:Rel)?Override)_\{'
)


def _label_prefixes_in_code() -> set[str]:
    found: set[str] = set()
    for path in SRC.rglob("*.py"):
        if "portability" in path.parts:
            continue
        found.update(
            m.group(1) for m in _LABEL_RE.finditer(path.read_text(encoding="utf-8"))
        )
    return found


def test_every_label_prefix_in_the_code_is_declared():
    in_code = _label_prefixes_in_code()
    assert in_code, "the grep found no label prefix — the regex drifted"
    undeclared = in_code - declared_label_prefixes()
    assert (
        not undeclared
    ), f"label prefixes written by the code but absent from the registry: {sorted(undeclared)}"
    # and the plan's closed list is fully present
    for prefix in (
        "KV",
        "Vec",
        "DocStatus",
        "Folder",
        "WebuiTag",
        "WebuiTagCategory",
        "WebuiActivity",
        "WebuiNotification",
        "WebuiApiKey",
        "WebuiSettings",
        "TwinSourceLink",
        "GraphOverride",
        "GraphRelOverride",
    ):
        assert prefix in declared_label_prefixes(), prefix


def test_every_store_declares_scoping_and_portability():
    for spec in STORES:
        assert spec.scoping in ("workspace", "folder", "global"), spec.name
        assert spec.portability in ("always", "optional", "never"), spec.name
        assert spec.plane in ("memgraph", "overlay"), spec.name
        if spec.file:
            assert store_by_file(spec.file) is spec
    assert store_by_name("kv.llm_response_cache").portability == "never"  # Q1
    assert store_by_name("activity").portability == "optional"  # Q6
    assert store_by_name("procedures").portability == "optional"  # Q6


def test_exportable_never_contains_a_never_store_and_optionals_are_opt_in():
    names = {s.name for s in exportable_stores()}
    assert not any(store_by_name(n).portability == "never" for n in names)
    assert (
        "kv.llm_response_cache" not in names
        and "api_keys" not in names
        and "notifications" not in names
    )
    assert "activity" not in names and "procedures" not in names
    both = {
        s.name
        for s in exportable_stores(include_activity=True, include_procedures=True)
    }
    assert {"activity", "procedures"} <= both
    assert never_label_prefixes() == {"WebuiApiKey", "WebuiNotification"}


def test_two_folders_give_two_tag_labels(monkeypatch):
    monkeypatch.setenv("TWIN_DEFAULT_FOLDER", "f1")
    monkeypatch.setenv(
        "TWIN_FOLDERS_JSON",
        json.dumps(
            [
                {"id": "f1", "label": "One", "kind": "primary"},
                {"id": "f2", "label": "Two", "kind": "custom"},
            ]
        ),
    )
    catalog = load_folder_catalog()
    scope = Scope(workspace="base", folder_ids=tuple(f.id for f in catalog.folders))
    labels = folder_scoped_labels(store_by_name("tags"), scope)
    assert labels == [("f1", "WebuiTag_f1"), ("f2", "WebuiTag_f2")]
    with pytest.raises(ValueError, match="not folder-scoped"):
        folder_scoped_labels(store_by_name("docstatus"), scope)
    assert store_by_name("kv.full_docs").label("base") == "KV_base_full_docs"
    assert store_by_name("docstatus").label("base") == "DocStatus_base"


def test_schema_guard_refuses_secret_fields_but_not_tokens_counter():
    with pytest.raises(ValueError, match="password"):
        StoreSchema(key=("id",), fields=frozenset({"id", "password"}))
    with pytest.raises(ValueError, match="api_key"):
        StoreSchema(key=("id",), fields=frozenset({"id", "vision_api_key"}))
    with pytest.raises(ValueError, match="token"):
        StoreSchema(key=("id",), fields=frozenset({"id", "access_token"}))
    StoreSchema(
        key=("id",), fields=frozenset({"id", "tokens", "content_hash"})
    )  # legit
    with pytest.raises(ValueError, match="key"):
        StoreSchema(key=("id",), fields=frozenset({"x"}))
    with pytest.raises(ValueError, match="both"):
        StoreSchema(key=("id",), fields=frozenset({"id"}), transient=frozenset({"id"}))


def test_project_record_drops_transient_and_refuses_unknown():
    spec = store_by_name("docstatus")
    out = project_record(
        spec,
        {
            "id": "d",
            "status": "processed",
            "__membership_epoch": 3,
            "__delete_claim": None,
        },
    )
    assert out == {"id": "d", "status": "processed"}
    with pytest.raises(SchemaViolation, match="secret_marker"):
        project_record(spec, {"id": "d", "secret_marker": "x"})


def test_chunk_schema_keeps_persisted_block_provenance():
    spec = store_by_name("vec.chunks")
    sidecar = '{"type":"block","id":"b1","refs":[{"type":"block","id":"b1"}]}'
    boundaries = '[{"block_id":"b1","start":0,"end":11}]'

    assert project_record(
        spec,
        {
            "id": "chunk-1",
            "sidecar": sidecar,
            "twin_block_boundaries": boundaries,
        },
    ) == {
        "id": "chunk-1",
        "sidecar": sidecar,
        "twin_block_boundaries": boundaries,
    }


def test_scope_validates_identifiers_and_maps_folders():
    scope = Scope(workspace="ws", folder_ids=("a", "b"), folder_map={"a": "z"})
    assert scope.mapped_folder("a") == "z" and scope.mapped_folder("b") == "b"
    with pytest.raises(ValueError):
        Scope(workspace="bad-ws")
    with pytest.raises(ValueError):
        Scope(workspace="ws", folder_ids=("a b",))
    with pytest.raises(ValueError):
        Scope(workspace="ws", batch_size=0)


def test_spec_invariants():
    with pytest.raises(ValueError, match="never store has no bundle file"):
        StoreSpec(
            "x",
            "overlay",
            "workspace",
            "never",
            StoreSchema(("id",), frozenset({"id"})),
            ("X",),
            "overlay/x.jsonl",
        )
    with pytest.raises(ValueError, match="needs a bundle file"):
        StoreSpec(
            "x",
            "overlay",
            "workspace",
            "always",
            StoreSchema(("id",), frozenset({"id"})),
            ("X",),
        )


def test_every_exportable_store_has_an_implementation():
    for spec in exportable_stores(include_activity=True, include_procedures=True):
        implementation = portable_store(spec)
        assert implementation.spec is spec
    with pytest.raises(Exception, match="never portable"):
        portable_store(store_by_name("api_keys"))


async def test_graph_member_of_rejects_unknown_relationship_properties(monkeypatch):
    from twindb_lightrag_memgraph.portability import stores_graph
    from twindb_lightrag_memgraph.portability.stores_graph import GraphMemberOfStore

    async def rows(query: str, **_params):
        assert "properties(m) AS p" in query
        return [
            {
                "entity_id": "Alice",
                "folder_id": "f1",
                "p": {"future_property": "must-not-disappear"},
            }
        ]

    monkeypatch.setattr(stores_graph, "read_rows", rows)
    with pytest.raises(SchemaViolation, match="future_property"):
        _ = [
            record
            async for record in GraphMemberOfStore().export_records(
                Scope(workspace="base", folder_ids=("f1",))
            )
        ]
