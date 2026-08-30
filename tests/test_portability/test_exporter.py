"""T1.4 orchestration without a live Memgraph or embedding endpoint."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from twindb_lightrag_memgraph.portability import exporter
from twindb_lightrag_memgraph.portability.bundle import inspect_bundle
from twindb_lightrag_memgraph.portability.stores import (
    Scope,
    exportable_stores,
)


class FakeEmbedding:
    embedding_dim = 4
    model = "fake-bge"

    async def func(self, texts: list[str]) -> list[list[float]]:
        return [
            [float(index + 1), float(len(text)), 0.25, -0.5]
            for index, text in enumerate(texts)
        ]


class FakeStore:
    def __init__(self, spec, records: list[dict], *, mutate: bool = False):
        self.spec = spec
        self.records = records
        self.mutate = mutate
        self.fingerprint_calls = 0

    async def export_records(self, scope: Scope) -> AsyncIterator[dict]:
        del scope
        for record in self.records:
            yield record

    async def import_records(self, records, scope: Scope) -> int:  # pragma: no cover
        del records, scope
        return 0

    async def fingerprint(self, scope: Scope) -> dict:
        del scope
        self.fingerprint_calls += 1
        return {
            "count": len(self.records) + int(self.mutate and self.fingerprint_calls > 1)
        }

    async def count(self, scope: Scope) -> int:
        del scope
        return len(self.records)


@pytest.fixture
def fake_runtime(monkeypatch):
    specs = exportable_stores()
    data = {spec.name: [] for spec in specs}
    data.update(
        {
            "docstatus": [
                {"id": "d1", "metadata": {"classification": {"class_id": "C1"}}},
                {"id": "d2", "metadata": {"classification": {"class_id": "C2"}}},
            ],
            "vec.chunks": [
                {"id": "c1", "props": {"content": "one"}, "embedding": [1.0] * 4},
                {"id": "c2", "props": {"content": "two"}, "embedding": [2.0] * 4},
                {"id": "c3", "props": {"content": "three"}, "embedding": [3.0] * 4},
            ],
            "folders": [{"id": "f1"}, {"id": "f2"}],
            "graph.nodes": [{"entity_id": "Alice", "labels": ["Person"], "props": {}}],
            "graph.edges": [
                {"src": "Alice", "tgt": "Acme", "props": {"keywords": "works"}}
            ],
            "tags": [{"folder_id": "f1", "id": "ops", "value": {"tag": "ops"}}],
        }
    )

    monkeypatch.setattr(exporter, "exportable_stores", lambda **_kwargs: specs)
    monkeypatch.setattr(
        exporter,
        "portable_store",
        lambda spec, **_kwargs: FakeStore(spec, data[spec.name]),
    )
    monkeypatch.setattr(
        exporter,
        "_folder_manifest",
        lambda: [
            {"id": "f1", "label": "One", "kind": "primary"},
            {"id": "f2", "label": "Two", "kind": "team"},
        ],
    )

    async def idle(_workspace: str) -> bool:
        return True

    async def source() -> dict[str, str]:
        return {
            "database": "memgraph",
            "version": "3.12.0",
            "mage": "present",
            "lightrag_version": "1.5.6",
        }

    monkeypatch.setattr(exporter, "_pipeline_is_idle", idle)
    monkeypatch.setattr(exporter, "_memgraph_source", source)
    return specs, data


async def test_two_exports_of_same_state_have_same_state_hash(fake_runtime, tmp_path):
    first = await exporter.export_kb(
        tmp_path / "first",
        workspace="base",
        embedding_func=FakeEmbedding(),
        actor="admin",
    )
    second = await exporter.export_kb(
        tmp_path / "second",
        workspace="base",
        embedding_func=FakeEmbedding(),
        actor="admin",
    )
    assert first.bundle_id != second.bundle_id
    assert first.state_hash == second.state_hash
    assert first.consistency.status == second.consistency.status == "verified"
    assert first.classification.max_detected == "C2"
    assert first.classification.unknown_present is False
    assert first.counts == {
        "documents": 2,
        "chunks": 3,
        "entities": 1,
        "relations": 1,
        "folders": 2,
        "tags": 1,
    }
    assert inspect_bundle(tmp_path / "first").ok
    assert [entry.path for entry in first.files] == sorted(
        entry.path for entry in first.files
    )


async def test_busy_pipeline_refuses_or_force_marks_unverified(
    fake_runtime, monkeypatch, tmp_path
):
    async def busy(_workspace: str) -> bool:
        return False

    monkeypatch.setattr(exporter, "_pipeline_is_idle", busy)
    with pytest.raises(exporter.ExportRefused, match="pipeline is busy"):
        await exporter.export_kb(
            tmp_path / "refused",
            workspace="base",
            embedding_func=FakeEmbedding(),
        )
    assert not (tmp_path / "refused").exists()

    manifest = await exporter.export_kb(
        tmp_path / "forced",
        workspace="base",
        embedding_func=FakeEmbedding(),
        force=True,
    )
    assert manifest.consistency.status == "unverified"
    assert manifest.consistency.pipeline_idle is False


async def test_mutation_between_fingerprints_marks_unverified(
    fake_runtime, monkeypatch, tmp_path
):
    specs, data = fake_runtime
    stores = {
        spec.name: FakeStore(spec, data[spec.name], mutate=spec.name == "docstatus")
        for spec in specs
    }
    monkeypatch.setattr(
        exporter, "portable_store", lambda spec, **_kwargs: stores[spec.name]
    )
    manifest = await exporter.export_kb(
        tmp_path / "mutated",
        workspace="base",
        embedding_func=FakeEmbedding(),
    )
    assert manifest.consistency.status == "unverified"
    assert (
        manifest.consistency.fingerprints_before["docstatus"]
        != manifest.consistency.fingerprints_after["docstatus"]
    )


async def test_unknown_classification_is_fail_closed_in_manifest(
    fake_runtime, monkeypatch, tmp_path
):
    specs, data = fake_runtime
    data["docstatus"].append(
        {"id": "d3", "metadata": {"classification": {"class_id": "UNKNOWN"}}}
    )
    monkeypatch.setattr(
        exporter,
        "portable_store",
        lambda spec, **_kwargs: FakeStore(spec, data[spec.name]),
    )
    manifest = await exporter.export_kb(
        tmp_path / "unknown",
        workspace="base",
        embedding_func=FakeEmbedding(),
    )
    assert manifest.classification.max_detected == "C2"
    assert manifest.classification.unknown_present is True


async def test_bad_probe_cleans_new_target(fake_runtime, tmp_path):
    class BadEmbedding(FakeEmbedding):
        async def func(self, texts: list[str]) -> list[list[float]]:
            return [[1.0] * 4 for _ in texts[:2]]

    with pytest.raises(exporter.ExportRefused, match="exactly three"):
        await exporter.export_kb(
            tmp_path / "bad-probe",
            workspace="base",
            embedding_func=BadEmbedding(),
        )
    assert not (tmp_path / "bad-probe").exists()


def test_native_embedding_environment_uses_upstream_resolver(monkeypatch):
    monkeypatch.setenv("EMBEDDING_BINDING", "openai")
    monkeypatch.setenv("EMBEDDING_MODEL", "actual-model")
    monkeypatch.setenv("EMBEDDING_DIM", "384")
    monkeypatch.setenv("EMBEDDING_TOKEN_LIMIT", "777")
    monkeypatch.setenv("EMBEDDING_ASYMMETRIC", "true")
    monkeypatch.setenv("EMBEDDING_DOCUMENT_PREFIX", "doc: ")
    monkeypatch.setenv("EMBEDDING_QUERY_PREFIX", "query: ")
    # The obsolete private resolver used these legacy names and would select
    # exactly the wrong values reproduced by the review.
    monkeypatch.setenv("LIGHTRAG_EMBEDDING_MODEL", "bge-m3")
    monkeypatch.setenv("LIGHTRAG_EMBEDDING_DIM", "1024")

    embedding_func, model, dim = exporter._default_embedding_func()

    assert model == "actual-model"
    assert dim == embedding_func.embedding_dim == 384
    assert embedding_func.model_name == "actual-model"
    assert embedding_func.max_token_size == 777
    assert embedding_func.supports_asymmetric is True


async def test_memgraph_source_preserves_unknown_mage_state(monkeypatch):
    from twindb_lightrag_memgraph import _capabilities
    from twindb_lightrag_memgraph.patches import canary
    from twindb_lightrag_memgraph.portability import _io

    async def snapshot():
        return _capabilities.MageCapabilitySnapshot(available=None, procedures=None)

    async def rows(_query: str):
        return [{"version": "3.12.0"}]

    monkeypatch.setattr(_capabilities, "get_mage_capability_snapshot", snapshot)
    monkeypatch.setattr(_io, "read_rows", rows)
    monkeypatch.setattr(canary, "installed_lightrag_version", lambda: "1.5.6")

    assert (await exporter._memgraph_source())["mage"] == "unknown"
