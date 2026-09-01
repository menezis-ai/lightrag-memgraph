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
# Malformed-bundle resilience: ONE bad record must not 500 the review queue
#
# The store is a JSON file that predates several schema revisions and is also
# an import target (restore_bundle), so type drift is reachable in production.
# A parked procedure is invisible in /documents — an unlistable bundle is a
# document stuck forever, which is why the list degrades instead of failing.
# ---------------------------------------------------------------------------


def _poison(bundle_id: str, **fields):
    """Write raw (schema-violating) values straight into the store file."""
    store_file = _procedure_store.store_path()
    store_payload = json.loads(store_file.read_text(encoding="utf-8"))
    store_payload["bundles"][bundle_id].update(fields)
    store_file.write_text(json.dumps(store_payload), encoding="utf-8")


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        # int/float/bool in a str|None slot: pydantic v2 does NOT coerce.
        ("track_id", 42, "42"),
        ("created_at", 1735689600.0, "1735689600.0"),
        ("updated_at", True, "True"),
        # non-scalar in a scalar slot: dropped, never stringified into junk.
        ("track_id", {"id": "t"}, None),
        ("created_at", ["2026-01-01"], None),
        ("operator_classification", {"level": "C1"}, None),
    ],
)
def test_malformed_scalar_field_is_coerced_not_fatal(client, field, value, expected):
    bundle_id = _park()
    _poison(bundle_id, **{field: value})

    resp = client.get("/twin/api/procedures")

    assert resp.status_code == 200, resp.text
    (item,) = resp.json()
    assert item["id"] == bundle_id
    assert item[field] == expected


@pytest.mark.parametrize(
    "value",
    [{"a": 1}, ["x"], "not-a-number", None, float("inf"), float("-inf")],
)
def test_malformed_schematics_total_projects_as_zero(client, value):
    bundle_id = _park(with_schematic=False)
    _poison(bundle_id, schematics_total=value)

    resp = client.get("/twin/api/procedures")

    assert resp.status_code == 200, resp.text
    assert resp.json()[0]["schematics_total"] == 0
    assert resp.json()[0]["reason"] == "ok"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("file_name", {"path": "/sensitive/input.pdf"}),
        ("state", ["pending", "secret-state"]),
        ("reason", {"full_text": "sensitive procedure body"}),
        ("source", ["detected", "secret-source"]),
    ],
)
def test_required_scalar_container_is_dropped_not_stringified(client, field, value):
    bundle_id = _park()
    _poison(bundle_id, **{field: value})

    resp = client.get("/twin/api/procedures")

    assert resp.status_code == 200, resp.text
    (item,) = resp.json()
    assert item[field] == ""
    assert "sensitive" not in resp.text


def test_invalid_integer_log_omits_the_imported_value(client, caplog):
    bundle_id = _park(with_schematic=False)
    sensitive_value = "sensitive-imported-value-" + ("x" * 2048)
    _poison(bundle_id, schematics_total=sensitive_value)

    with caplog.at_level("WARNING", logger="twindb_lightrag_memgraph"):
        resp = client.get("/twin/api/procedures")

    assert resp.status_code == 200, resp.text
    assert resp.json()[0]["schematics_total"] == 0
    assert sensitive_value not in caplog.text
    assert "value of type str" in caplog.text


@pytest.mark.parametrize("value", [["c1"], "C1", 7])
def test_malformed_classification_is_dropped_not_fatal(client, value):
    bundle_id = _park()
    _poison(bundle_id, classification=value)

    resp = client.get("/twin/api/procedures")

    assert resp.status_code == 200, resp.text
    assert resp.json()[0]["classification"] is None


def test_one_malformed_bundle_does_not_hide_the_healthy_ones(client):
    """The actual production symptom: parked procedures exist, list 500s."""
    healthy_a = _park(file_name="a.pdf", content_hash="ha")
    poisoned = _park(file_name="b.pdf", content_hash="hb")
    healthy_b = _park(file_name="c.pdf", content_hash="hc")
    _poison(poisoned, track_id={"nested": "object"}, schematics_total=["bad"])

    resp = client.get("/twin/api/procedures")

    assert resp.status_code == 200, resp.text
    ids = {item["id"] for item in resp.json()}
    assert ids == {healthy_a, poisoned, healthy_b}


def test_duplicate_request_drift_does_not_break_the_fold(client):
    """strictest_operator_classification folds caller-supplied requests."""
    bundle_id = _park(operator_classification="C1")
    _poison(bundle_id, duplicate_requests=[{"operator_classification": {"bad": 1}}])

    resp = client.get("/twin/api/procedures")

    assert resp.status_code == 200, resp.text
    assert resp.json()[0]["id"] == bundle_id


def test_projection_crash_serves_a_degraded_row_not_a_500(client, monkeypatch):
    """Belt beyond coercion: the row stays reachable, flagged, never dropped."""
    from twindb_lightrag_memgraph.server import procedure_routes

    bundle_id = _park()

    def _boom(bundle):
        raise RuntimeError("projection exploded")

    monkeypatch.setattr(procedure_routes, "_summary", _boom)

    resp = client.get("/twin/api/procedures")

    assert resp.status_code == 200, resp.text
    (item,) = resp.json()
    assert item["id"] == bundle_id
    assert "could not be read" in item["reason"]


def test_degraded_summary_drops_required_scalar_containers(client, monkeypatch):
    """The last-resort row must honor the same no-content list contract."""
    from twindb_lightrag_memgraph.server import procedure_routes

    bundle_id = _park()
    _poison(
        bundle_id,
        file_name={"path": "/sensitive/input.pdf"},
        state=["pending", "sensitive-state"],
        source={"full_text": "sensitive procedure body"},
    )

    def _boom(bundle):
        raise RuntimeError("projection exploded")

    monkeypatch.setattr(procedure_routes, "_summary", _boom)

    resp = client.get("/twin/api/procedures")

    assert resp.status_code == 200, resp.text
    (item,) = resp.json()
    assert item["file_name"] == ""
    assert item["state"] == ""
    assert item["source"] == ""
    assert "sensitive" not in resp.text


@pytest.mark.parametrize(
    ("label", "fields"),
    [
        # bundle_folders() collapses each of these to [] WITHOUT raising, so
        # they used to inherit the "unassigned -> visible everywhere"
        # exception meant for scan-created records.
        ("primary folder is a list", {"folder": []}),
        ("primary folder is an int", {"folder": 0}),
        ("primary folder is a bool", {"folder": True}),
        ("primary folder is empty", {"folder": ""}),
        ("primary folder identifier is invalid", {"folder": "bad-folder"}),
        (
            "duplicate_requests is an object",
            {"folder": None, "duplicate_requests": {"f2": {"folder": "f2"}}},
        ),
        (
            "duplicate request entry is not a dict",
            {"folder": None, "duplicate_requests": ["f2"]},
        ),
        (
            "duplicate request folder is a list",
            {"folder": None, "duplicate_requests": [{"folder": ["f2"]}]},
        ),
        (
            "duplicate request folder is empty",
            {"folder": None, "duplicate_requests": [{"folder": ""}]},
        ),
        (
            "duplicate request folder identifier is invalid",
            {"folder": None, "duplicate_requests": [{"folder": "bad-folder"}]},
        ),
    ],
)
def test_malformed_folder_metadata_never_leaks_into_a_folder(client, label, fields):
    """Persisted-data regression: malformed-but-FALSY folder metadata must not
    be mistaken for a legitimately unassigned bundle and shown everywhere."""
    legit = _park(folder="f1", file_name="mine.pdf", content_hash="h-mine")
    leaky = _park(folder="f2", file_name="foreign.pdf", content_hash="h-foreign")
    _poison(leaky, **fields)

    resp = client.get("/twin/api/procedures")

    assert resp.status_code == 200, resp.text
    ids = {item["id"] for item in resp.json()}
    assert ids == {legit}, f"{label}: {leaky} leaked into folder f1"


def test_malformed_folder_metadata_excludes_even_from_its_own_folder(client):
    """Fail-closed is deliberate: an unenumerable claim set excludes the bundle
    from its OWN folder too. It stays reachable via the admin detail route,
    which is not folder-bound — the boundary wins over the convenience."""
    bundle_id = _park(folder="f1")
    _poison(bundle_id, duplicate_requests=["not-a-dict"])

    listed = client.get("/twin/api/procedures")
    assert listed.status_code == 200
    assert listed.json() == []

    detail = client.get(f"/twin/api/procedures/{bundle_id}")
    assert detail.status_code == 200
    assert detail.json()["id"] == bundle_id


def test_restored_bundle_with_malformed_folder_does_not_leak(client):
    """The reviewer's reproduction path: restore_bundle() persists a record
    without validating its folder shape."""
    _park(folder="f1", file_name="mine.pdf", content_hash="h-mine")
    _procedure_store.restore_bundle(
        {
            "id": "restored-leak",
            "state": "pending",
            "folder": [],
            "file_name": "foreign.pdf",
            "reason": "",
            "source": "detected",
        }
    )

    resp = client.get("/twin/api/procedures")

    assert resp.status_code == 200, resp.text
    assert "restored-leak" not in {item["id"] for item in resp.json()}


def test_legitimate_folder_shapes_are_unchanged(client):
    """The fix must not narrow valid data: a scan-created bundle stays visible
    everywhere, a foreign bundle stays invisible, a claimed one stays claimed."""
    mine = _park(folder="f1", file_name="mine.pdf", content_hash="h1")
    foreign = _park(folder="f2", file_name="foreign.pdf", content_hash="h2")
    unassigned = _park(folder=None, file_name="scan.pdf", content_hash="h3")
    claimed = _park(folder=None, file_name="claimed.pdf", content_hash="h4")
    _procedure_store.record_request(
        claimed,
        path="/inputs/claimed.pdf",
        folder="f2",
        track_id="t",
        operator_classification=None,
        file_name="claimed.pdf",
    )

    ids = {item["id"] for item in client.get("/twin/api/procedures").json()}
    assert ids == {mine, unassigned}
    assert foreign not in ids
    assert claimed not in ids


def test_undecidable_visibility_fails_closed(client, monkeypatch):
    """Outer belt: an UNEXPECTED raise in folder resolution still excludes.

    The structural cases above are the real ones (they never raise); this
    guards the residual path where something else blows up entirely.
    """
    from twindb_lightrag_memgraph.server import procedure_routes

    _park(folder="f1")

    def _boom(bundle, bundle_id):
        raise RuntimeError("folders unreadable")

    monkeypatch.setattr(procedure_routes, "_validated_bundle_folders", _boom)

    resp = client.get("/twin/api/procedures")

    assert resp.status_code == 200, resp.text
    assert resp.json() == []


def test_store_access_failure_stays_a_500_and_is_logged_as_such(client, caplog):
    """A filesystem/permission failure is NOT a malformed bundle: the log must
    say so, because the two have identical HTTP signatures and opposite fixes.
    """
    store_file = _procedure_store.store_path()
    store_file.parent.mkdir(parents=True, exist_ok=True)
    store_file.write_text("{}", encoding="utf-8")

    def _explode(path):
        raise PermissionError(13, "Permission denied")

    with caplog.at_level("ERROR", logger="twindb_lightrag_memgraph"):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(_procedure_store, "_load", _explode)
            resp = client.get("/twin/api/procedures")

    assert resp.status_code == 500
    assert "STORE ACCESS failed" in caplog.text


def test_business_error_is_not_mislabelled_as_store_access(client, caplog):
    """``_store_call`` also runs mutations: an invalid-state ValueError is a
    business error and must not send the reader hunting for a permissions bug.
    """
    store_file = _procedure_store.store_path()
    store_file.parent.mkdir(parents=True, exist_ok=True)
    store_file.write_text("{}", encoding="utf-8")

    def _explode(*args, **kwargs):
        raise ValueError("invalid bundle state: 'nope'")

    with caplog.at_level("ERROR", logger="twindb_lightrag_memgraph"):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(_procedure_store, "list_bundles", _explode)
            resp = client.get("/twin/api/procedures")

    assert resp.status_code == 500
    assert "STORE ACCESS failed" not in caplog.text


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
