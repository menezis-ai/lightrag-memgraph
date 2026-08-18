"""Approval-workflow routes for procedure bundles (PR 2).

Route-level tests on ``build_procedure_router`` with a fake rag — no
Memgraph, no vision. Covers: folder-bound list projection (no path leak),
admin detail, approve (context rebind: primary folder + strictest operator
classification; duplicate-request memberships; MIP-gate refusal surfaced),
reject/retry/reroute state machine (atomic transitions -> 409 on races),
degraded-store 503 + explicit recovery, and the seam event sink bridge.
"""

from __future__ import annotations

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from twindb_lightrag_memgraph import _procedure, _procedure_store
from twindb_lightrag_memgraph.server.auth import configure_auth
from twindb_lightrag_memgraph.server.idp_jwt import configure_idp
from twindb_lightrag_memgraph.server.procedure_routes import (
    _approved_doc_id,
    build_procedure_router,
    install_procedure_event_sink,
)

ROOT_KEY = "test-infra-root"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path):
    monkeypatch.setenv(
        "TWIN_PROCEDURE_STORE_FILE", str(tmp_path / "bundles" / "store.json")
    )
    monkeypatch.setenv("TWIN_DEFAULT_FOLDER", "f1")
    monkeypatch.setenv(
        "TWIN_FOLDERS_JSON",
        json.dumps(
            [
                {"id": "f1", "label": "Folder 1", "kind": "kb"},
                {"id": "f2", "label": "Folder 2", "kind": "kb"},
            ]
        ),
    )
    configure_idp(None)
    _procedure.reset_caches()
    yield
    configure_idp(None)
    configure_auth(api_key=None, jwt_secret=None)
    _procedure.reset_caches()


class _FakeDocStatus:
    def __init__(self):
        self.memberships = []
        self.rows: dict[str, dict] = {}

    async def add_to_folder(self, doc_id, folder):
        self.memberships.append((doc_id, folder))
        return True

    async def get_by_id(self, doc_id):
        return self.rows.get(doc_id)


class _FakeRag:
    def __init__(self):
        self.enqueues = []
        self.doc_status = _FakeDocStatus()

    async def apipeline_enqueue_documents(
        self, content, ids=None, file_paths=None, track_id=None, **kwargs
    ):
        from twindb_lightrag_memgraph._constants import (
            get_active_operator_classification,
            get_active_storage_folder,
        )

        self.enqueues.append(
            {
                "content": content,
                "file_paths": file_paths,
                "track_id": track_id,
                "folder": get_active_storage_folder(),
                "operator_classification": get_active_operator_classification(),
            }
        )
        return track_id or "trk"


@pytest.fixture
def rag():
    return _FakeRag()


@pytest.fixture
def client(rag, monkeypatch):
    async def enabled_settings():
        return {"procedure_enabled": True}

    _procedure.set_settings_provider(enabled_settings)
    monkeypatch.setattr(_procedure, "is_available", lambda: True)
    app = FastAPI()
    app.include_router(build_procedure_router(lambda: rag), prefix="/twin/api")
    configure_auth(api_key=ROOT_KEY)
    return TestClient(
        app,
        raise_server_exceptions=False,
        headers={"Authorization": f"Bearer {ROOT_KEY}", "X-Twin-Folder": "f1"},
    )


_INFORMED = {
    "title": "Qualify the incident",
    "description": "The L1 support qualifies the ticket.",
    "tasks": [
        {
            "id": "T4.1",
            "title": "Categorize and Enrich",
            "responsible": "Incident Manager",
            "actors": "L1 Support",
            "inputs": "Incident ticket",
            "outputs": "Updated CI",
            "conditions": "",
            "links": "CONF",
        }
    ],
}


def _park(
    state="pending",
    *,
    folder="f1",
    file_name="itg0162.pdf",
    original_path="/inputs/itg0162.pdf",
    with_schematic=True,
    operator_classification=None,
    content_hash="hash-1",
):
    schematics = (
        [
            {
                "page": 3,
                "png_base64": "cG5n",
                "blind": {"title": "b", "description": "blind", "tasks": []},
                "informed": _INFORMED,
                "divergence": {"coherent": True, "divergences": [], "summary": "ok"},
                "error": None,
            }
        ]
        if with_schematic
        else []
    )
    return _procedure_store.create_bundle(
        file_name=file_name,
        original_path=original_path,
        track_id="trk-up",
        state=state,
        reason="ok",
        source="detected",
        folder=folder,
        content_hash=content_hash,
        full_text="ITG0162 procedure body text",
        schematics=schematics,
        schematics_total=len(schematics),
        classification={"class_id": None, "class_name": None, "reason": "no label"},
        operator_classification=operator_classification,
    )


# ---------------------------------------------------------------------------
# List / detail
# ---------------------------------------------------------------------------


def test_list_is_folder_bound_and_leaks_no_paths(client):
    visible = _park(folder="f1", content_hash="h1")
    _park(folder="f2", file_name="other.pdf", content_hash="h2")
    requested = _park(
        folder=None, file_name="scan.pdf", content_hash="h3"
    )  # folderless + f1 request
    _procedure_store.record_request(
        requested,
        path="/inputs/scan.pdf",
        folder="f1",
        track_id="t",
        operator_classification=None,
        file_name="scan.pdf",
    )
    # Unassigned (scan-created, no operator request): reachable from EVERY
    # folder list — otherwise it would be reviewable from nowhere.
    unassigned = _park(folder=None, file_name="orphan.pdf", content_hash="h4")

    resp = client.get("/twin/api/procedures")
    assert resp.status_code == 200
    items = resp.json()
    ids = {item["id"] for item in items}
    assert ids == {visible, requested, unassigned}
    for item in items:
        assert "original_path" not in item
        assert "full_text" not in item
        assert "schematics" not in item
        assert "duplicate_requests" not in item
        # track_id IS projected: the WebUI reconciles parked uploads on it.
        assert item["track_id"] == "trk-up"


def test_detail_returns_full_bundle(client):
    bundle_id = _park()
    resp = client.get(f"/twin/api/procedures/{bundle_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["full_text"]
    assert body["schematics"][0]["informed"]["title"] == _INFORMED["title"]
    assert client.get("/twin/api/procedures/ghost").status_code == 404


# ---------------------------------------------------------------------------
# Approve
# ---------------------------------------------------------------------------


def test_approve_enqueues_with_rebound_contexts(client, rag):
    bundle_id = _park(operator_classification="C1")
    _procedure_store.record_request(
        bundle_id,
        path="/inputs/copy.pdf",
        folder="f2",
        track_id="t2",
        operator_classification="C2",
        file_name="copy.pdf",
    )

    resp = client.post(f"/twin/api/procedures/{bundle_id}/approve")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["state"] == "approved"

    assert len(rag.enqueues) == 1
    call = rag.enqueues[0]
    # Composed markdown: full text + informed description, ORIGINAL name.
    assert "ITG0162 procedure body text" in call["content"]
    assert _INFORMED["description"] in call["content"]
    assert "T4.1" in call["content"]
    assert call["file_paths"] == "itg0162.pdf"
    # Contexts rebound: primary folder + STRICTEST operator class (C2>C1).
    assert call["folder"] == "f1"
    assert call["operator_classification"] == "C2"
    # Duplicate-request folder membership applied on the derived doc id.
    doc_id = _approved_doc_id(call["content"], {"file_name": "itg0162.pdf"})
    assert rag.doc_status.memberships == [(doc_id, "f2")]
    assert body["approved_doc_id"] == doc_id


def test_approve_conflicts_and_missing(client):
    assert client.post("/twin/api/procedures/ghost/approve").status_code == 404
    failed = _park(state="failed")
    assert client.post(f"/twin/api/procedures/{failed}/approve").status_code == 409


def test_approve_folderless_requires_folder(client, rag):
    bundle_id = _park(folder=None)
    resp = client.post(f"/twin/api/procedures/{bundle_id}/approve")
    assert resp.status_code == 422
    resp = client.post(
        f"/twin/api/procedures/{bundle_id}/approve", json={"folder": "f2"}
    )
    assert resp.status_code == 200
    assert rag.enqueues[0]["folder"] == "f2"


def test_approve_surfaces_mip_gate_refusal(client, rag):
    bundle_id = _park()
    bundle = _procedure_store.get_bundle(bundle_id)
    markdown = _procedure.compose_approved_markdown(bundle)
    doc_id = _approved_doc_id(markdown, bundle)
    rag.doc_status.rows[doc_id] = {
        "status": "failed",
        "error_msg": "classification C3 above ceiling",
    }

    resp = client.post(f"/twin/api/procedures/{bundle_id}/approve")
    assert resp.status_code == 200
    body = resp.json()
    assert body["state"] == "failed"
    assert "mip-rejected-at-enqueue" in body["reason"]


# ---------------------------------------------------------------------------
# Reject / retry / reroute
# ---------------------------------------------------------------------------


def test_reject_transitions_and_conflicts(client):
    bundle_id = _park()
    resp = client.post(
        f"/twin/api/procedures/{bundle_id}/reject", json={"comment": "bad scan"}
    )
    assert resp.status_code == 200
    assert resp.json()["state"] == "rejected"
    assert "bad scan" in resp.json()["reason"]
    # Terminal: rejecting again conflicts.
    assert client.post(f"/twin/api/procedures/{bundle_id}/reject").status_code == 409


def test_retry_reprocesses_failed_bundle(client, monkeypatch, tmp_path):
    original = tmp_path / "itg0162.pdf"
    original.write_bytes(b"%PDF-1.4")
    bundle_id = _park(state="failed", original_path=str(original))

    async def fake_profile(_path):
        return {
            "state": "pending",
            "reason": "ok",
            "full_text": "text",
            "schematics": [],
            "schematics_total": 0,
            "classification": None,
        }

    monkeypatch.setattr(_procedure, "_run_profile", fake_profile)

    resp = client.post(f"/twin/api/procedures/{bundle_id}/retry")
    assert resp.status_code == 200
    assert resp.json()["state"] == "pending"

    # pending is not retryable — the ONLY relaunch path stays scoped.
    assert client.post(f"/twin/api/procedures/{bundle_id}/retry").status_code == 409


def test_retry_with_missing_original_fails_visibly(client, tmp_path):
    bundle_id = _park(state="failed", original_path=str(tmp_path / "gone.pdf"))
    resp = client.post(f"/twin/api/procedures/{bundle_id}/retry")
    assert resp.status_code == 200
    assert resp.json()["state"] == "failed"
    assert "original-missing" in resp.json()["reason"]


def test_retry_is_blocked_while_admin_toggle_is_off(client, tmp_path):
    bundle_id = _park(
        state="failed",
        original_path=str(tmp_path / "still-parked.pdf"),
    )

    async def disabled_settings():
        return {"procedure_enabled": False}

    _procedure.set_settings_provider(disabled_settings)
    resp = client.post(f"/twin/api/procedures/{bundle_id}/retry")

    assert resp.status_code == 409
    assert "Settings > Vision" in resp.json()["detail"]
    assert _procedure_store.get_bundle(bundle_id)["state"] == "failed"


def test_retry_is_blocked_when_prerequisites_are_unavailable(
    client, monkeypatch, tmp_path
):
    bundle_id = _park(
        state="failed",
        original_path=str(tmp_path / "still-parked.pdf"),
    )
    monkeypatch.setattr(_procedure, "is_available", lambda: False)

    resp = client.post(f"/twin/api/procedures/{bundle_id}/retry")

    assert resp.status_code == 409
    assert "prerequisites are unavailable" in resp.json()["detail"]
    assert _procedure_store.get_bundle(bundle_id)["state"] == "failed"


def test_reroute_standard_uses_standard_context(client, rag, monkeypatch, tmp_path):
    import sys

    monkeypatch.setattr(sys, "argv", ["pytest"])  # document_routes parses argv
    original = tmp_path / "report.pdf"
    original.write_bytes(b"%PDF-1.4")
    bundle_id = _park(state="failed", original_path=str(original))
    seen = {}

    async def fake_enqueue_file(rag_arg, file_path, *args, **kwargs):
        from twindb_lightrag_memgraph._constants import (
            get_active_doc_type,
            get_active_storage_folder,
        )

        seen["doc_type"] = get_active_doc_type()
        seen["folder"] = get_active_storage_folder()
        seen["path"] = file_path
        return True, "trk-std"

    import lightrag.api.routers.document_routes as dr

    monkeypatch.setattr(dr, "pipeline_enqueue_file", fake_enqueue_file)

    resp = client.post(f"/twin/api/procedures/{bundle_id}/reroute-standard")
    assert resp.status_code == 200, resp.text
    assert resp.json()["state"] == "rerouted"
    # Explicit standard override + folder rebound: the seam honors the
    # operator decision instead of re-claiming the file.
    assert seen["doc_type"] == "standard"
    assert seen["folder"] == "f1"
    assert str(seen["path"]) == str(original)


def test_reroute_missing_original_conflicts(client, tmp_path):
    bundle_id = _park(state="failed", original_path=str(tmp_path / "gone.pdf"))
    resp = client.post(f"/twin/api/procedures/{bundle_id}/reroute-standard")
    assert resp.status_code == 409


# ---------------------------------------------------------------------------
# Degraded store: honest 503 + explicit recovery
# ---------------------------------------------------------------------------


def test_degraded_store_returns_503_then_recovers(client, tmp_path):
    _park()
    store_file = _procedure_store.store_path()
    store_file.write_text("{not json", encoding="utf-8")

    resp = client.get("/twin/api/procedures")
    assert resp.status_code == 503
    assert "recover" in resp.json()["detail"]

    health = client.get("/twin/api/procedures/store/health").json()
    assert health["degraded"] is True
    assert len(health["quarantine_files"]) == 1

    recover = client.post("/twin/api/procedures/store/recover")
    assert recover.status_code == 200
    assert recover.json()["degraded"] is False
    assert client.get("/twin/api/procedures").status_code == 200


# ---------------------------------------------------------------------------
# Auth gating + seam event sink
# ---------------------------------------------------------------------------


def test_anonymous_is_rejected_when_auth_configured(rag):
    app = FastAPI()
    app.include_router(build_procedure_router(lambda: rag), prefix="/twin/api")
    configure_auth(api_key=ROOT_KEY)
    anonymous = TestClient(app, raise_server_exceptions=False)
    assert anonymous.get("/twin/api/procedures").status_code == 401
    assert anonymous.post("/twin/api/procedures/x/approve").status_code == 401


async def test_event_sink_bridges_without_breaking_ingestion():
    install_procedure_event_sink()
    assert _procedure._event_sink is not None
    # No webui store configured: the bridge must swallow the failure.
    _procedure._emit(
        "procedure-parked",
        {"bundle_id": "b1", "file_name": "doc.pdf", "state": "pending"},
    )
    import asyncio

    await asyncio.sleep(0)  # let the scheduled task run (and fail silently)
    _procedure.set_event_sink(None)
