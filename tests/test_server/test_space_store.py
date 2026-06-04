"""Unit tests for the runtime space store — env-free, no FastAPI."""

from __future__ import annotations

import json

import pytest

from twindb_lightrag_memgraph.server import space_store


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    """Each test starts from an empty in-memory store with no file."""
    monkeypatch.delenv("TWIN_SPACES_RUNTIME_FILE", raising=False)
    space_store.reset_runtime_store()
    yield
    space_store.reset_runtime_store()


class TestInMemoryStore:
    def test_add_and_list(self):
        s = space_store.add_runtime_space(
            space_id="sandbox", label="Sandbox", kind="sandbox"
        )
        assert s.id == "sandbox"
        spaces = space_store.list_runtime_spaces()
        assert [sp.id for sp in spaces] == ["sandbox"]

    def test_get(self):
        space_store.add_runtime_space(space_id="x", label="X")
        got = space_store.get_runtime_space("x")
        assert got is not None
        assert got.label == "X"
        assert space_store.get_runtime_space("ghost") is None

    def test_add_invalid_id_raises_value_error(self):
        with pytest.raises(ValueError):
            space_store.add_runtime_space(space_id="bad space!", label="x")

    def test_add_duplicate_raises_key_error(self):
        space_store.add_runtime_space(space_id="x", label="X")
        with pytest.raises(KeyError):
            space_store.add_runtime_space(space_id="x", label="X-bis")

    def test_update_changes_fields(self):
        space_store.add_runtime_space(space_id="x", label="X", description="old")
        updated = space_store.update_runtime_space(
            "x", label="X-new", description="new"
        )
        assert updated is not None
        assert updated.label == "X-new"
        assert updated.description == "new"

    def test_update_unknown_returns_none(self):
        assert space_store.update_runtime_space("ghost", label="x") is None

    def test_delete_returns_true_then_false(self):
        space_store.add_runtime_space(space_id="x", label="X")
        assert space_store.delete_runtime_space("x") is True
        assert space_store.delete_runtime_space("x") is False


class TestFilePersistence:
    def test_add_writes_to_disk(self, monkeypatch, tmp_path):
        path = tmp_path / "twin-spaces.json"
        monkeypatch.setenv("TWIN_SPACES_RUNTIME_FILE", str(path))
        space_store.reset_runtime_store()
        space_store.add_runtime_space(space_id="x", label="X")
        data = json.loads(path.read_text())
        assert any(item["id"] == "x" for item in data)

    def test_load_from_disk_on_first_access(self, monkeypatch, tmp_path):
        path = tmp_path / "twin-spaces.json"
        path.write_text(
            json.dumps(
                [
                    {
                        "id": "preloaded",
                        "label": "Preloaded",
                        "kind": "custom",
                        "description": "From disk",
                        "sources": 0,
                    }
                ]
            )
        )
        monkeypatch.setenv("TWIN_SPACES_RUNTIME_FILE", str(path))
        space_store.reset_runtime_store()

        loaded = space_store.list_runtime_spaces()
        assert [s.id for s in loaded] == ["preloaded"]

    def test_corrupt_file_falls_back_to_empty(self, monkeypatch, tmp_path):
        path = tmp_path / "twin-spaces.json"
        path.write_text("not even json {{")
        monkeypatch.setenv("TWIN_SPACES_RUNTIME_FILE", str(path))
        space_store.reset_runtime_store()

        # Falls back to empty; no exception raised.
        assert space_store.list_runtime_spaces() == []

    def test_invalid_ids_skipped_on_load(self, monkeypatch, tmp_path):
        path = tmp_path / "twin-spaces.json"
        path.write_text(
            json.dumps(
                [
                    {"id": "good", "label": "Good"},
                    {"id": "bad name!", "label": "Bad"},
                    {"id": "", "label": "Empty"},
                ]
            )
        )
        monkeypatch.setenv("TWIN_SPACES_RUNTIME_FILE", str(path))
        space_store.reset_runtime_store()

        loaded = space_store.list_runtime_spaces()
        assert [s.id for s in loaded] == ["good"]

    def test_delete_rewrites_file(self, monkeypatch, tmp_path):
        path = tmp_path / "twin-spaces.json"
        monkeypatch.setenv("TWIN_SPACES_RUNTIME_FILE", str(path))
        space_store.reset_runtime_store()
        space_store.add_runtime_space(space_id="x", label="X")
        space_store.add_runtime_space(space_id="y", label="Y")
        space_store.delete_runtime_space("x")
        data = json.loads(path.read_text())
        assert [item["id"] for item in data] == ["y"]
