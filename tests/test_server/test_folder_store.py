"""Unit tests for the runtime folder store — env-free, no FastAPI."""

from __future__ import annotations

import json

import pytest

from twindb_lightrag_memgraph.server import folder_store


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    """Each test starts from an empty in-memory store with no file."""
    monkeypatch.delenv("TWIN_FOLDERS_RUNTIME_FILE", raising=False)
    folder_store.reset_runtime_store()
    yield
    folder_store.reset_runtime_store()


class TestInMemoryStore:
    def test_add_and_list(self):
        s = folder_store.add_runtime_folder(
            folder_id="sandbox", label="Sandbox", kind="sandbox"
        )
        assert s.id == "sandbox"
        folders = folder_store.list_runtime_folders()
        assert [sp.id for sp in folders] == ["sandbox"]

    def test_get(self):
        folder_store.add_runtime_folder(folder_id="x", label="X")
        got = folder_store.get_runtime_folder("x")
        assert got is not None
        assert got.label == "X"
        assert folder_store.get_runtime_folder("ghost") is None

    def test_add_invalid_id_raises_value_error(self):
        with pytest.raises(ValueError):
            folder_store.add_runtime_folder(folder_id="bad folder!", label="x")

    def test_add_duplicate_raises_key_error(self):
        folder_store.add_runtime_folder(folder_id="x", label="X")
        with pytest.raises(KeyError):
            folder_store.add_runtime_folder(folder_id="x", label="X-bis")

    def test_update_changes_fields(self):
        folder_store.add_runtime_folder(folder_id="x", label="X", description="old")
        updated = folder_store.update_runtime_folder(
            "x", label="X-new", description="new"
        )
        assert updated is not None
        assert updated.label == "X-new"
        assert updated.description == "new"

    def test_update_unknown_returns_none(self):
        assert folder_store.update_runtime_folder("ghost", label="x") is None

    def test_delete_returns_true_then_false(self):
        folder_store.add_runtime_folder(folder_id="x", label="X")
        assert folder_store.delete_runtime_folder("x") is True
        assert folder_store.delete_runtime_folder("x") is False


class TestFilePersistence:
    def test_add_writes_to_disk(self, monkeypatch, tmp_path):
        path = tmp_path / "twin-folders.json"
        monkeypatch.setenv("TWIN_FOLDERS_RUNTIME_FILE", str(path))
        folder_store.reset_runtime_store()
        folder_store.add_runtime_folder(folder_id="x", label="X")
        data = json.loads(path.read_text())
        assert any(item["id"] == "x" for item in data)

    def test_load_from_disk_on_first_access(self, monkeypatch, tmp_path):
        path = tmp_path / "twin-folders.json"
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
        monkeypatch.setenv("TWIN_FOLDERS_RUNTIME_FILE", str(path))
        folder_store.reset_runtime_store()

        loaded = folder_store.list_runtime_folders()
        assert [s.id for s in loaded] == ["preloaded"]

    def test_corrupt_file_falls_back_to_empty(self, monkeypatch, tmp_path):
        path = tmp_path / "twin-folders.json"
        path.write_text("not even json {{")
        monkeypatch.setenv("TWIN_FOLDERS_RUNTIME_FILE", str(path))
        folder_store.reset_runtime_store()

        # Falls back to empty; no exception raised.
        assert folder_store.list_runtime_folders() == []

    def test_invalid_ids_skipped_on_load(self, monkeypatch, tmp_path):
        path = tmp_path / "twin-folders.json"
        path.write_text(
            json.dumps(
                [
                    {"id": "good", "label": "Good"},
                    {"id": "bad name!", "label": "Bad"},
                    {"id": "", "label": "Empty"},
                ]
            )
        )
        monkeypatch.setenv("TWIN_FOLDERS_RUNTIME_FILE", str(path))
        folder_store.reset_runtime_store()

        loaded = folder_store.list_runtime_folders()
        assert [s.id for s in loaded] == ["good"]

    def test_delete_rewrites_file(self, monkeypatch, tmp_path):
        path = tmp_path / "twin-folders.json"
        monkeypatch.setenv("TWIN_FOLDERS_RUNTIME_FILE", str(path))
        folder_store.reset_runtime_store()
        folder_store.add_runtime_folder(folder_id="x", label="X")
        folder_store.add_runtime_folder(folder_id="y", label="Y")
        folder_store.delete_runtime_folder("x")
        data = json.loads(path.read_text())
        assert [item["id"] for item in data] == ["y"]
