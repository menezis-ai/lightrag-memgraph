"""Admin portability routes and persistent job-state contract (PR-P3)."""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from httpx import ASGITransport, AsyncClient

from twindb_lightrag_memgraph.server import portability_jobs, portability_routes
from twindb_lightrag_memgraph.server.app import create_app
from twindb_lightrag_memgraph.server.auth import _create_jwt, configure_auth
from twindb_lightrag_memgraph.server.idp_jwt import configure_idp

REPORT_HASH = "a" * 64


async def _wait_for_status(
    client: AsyncClient,
    job_id: str,
    expected: str,
    *,
    timeout: float = 2.0,
) -> dict:
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        response = await client.get(f"/twin/api/admin/portability/imports/{job_id}")
        assert response.status_code == 200, response.text
        payload = response.json()
        if payload["status"] == expected:
            return payload
        if payload["status"] == "failed":
            raise AssertionError(payload["error"])
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError(
                f"timeout waiting for {expected}; current={payload['status']}"
            )
        await asyncio.sleep(0.01)


@pytest.fixture(autouse=True)
def _reset(monkeypatch, tmp_path):
    monkeypatch.setenv("TWIN_PORTABILITY_DIR", str(tmp_path / "portability"))
    monkeypatch.setenv("WORKSPACE", "target")
    monkeypatch.setenv("MEMGRAPH_WORKSPACE", "target")
    monkeypatch.setenv("LIGHTRAG_API_KEY", "test-infra-root")
    configure_idp(None)
    portability_jobs.reset_portability_jobs_for_tests()
    yield
    portability_jobs.reset_portability_jobs_for_tests()
    configure_idp(None)
    configure_auth()


@pytest.fixture()
async def admin_client():
    app = create_app()
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
        headers={"Authorization": "Bearer test-infra-root"},
    ) as client:
        yield client


@pytest.fixture()
def fake_import_primitives(monkeypatch):
    calls: list[str] = []

    async def fake_dry_run(*_args, **kwargs):
        calls.append("dry-run")
        return {
            "format": "twin-kb-import-report",
            "format_version": "1.0",
            "report_hash": REPORT_HASH,
            "blocking": [],
            "compat": [
                {
                    "dimension": "embedding",
                    "ok": True,
                    "source": {"dim": 1024},
                    "target": {"dim": 1024, "min_cosine": 1.0},
                    "reason": "all three probe cosines must be >= 0.999",
                }
            ],
            "classification": {
                "source_max": "C2",
                "target_ceiling": "C2",
                "unknown_present": False,
            },
            "target": {"workspace": kwargs["workspace"]},
            "folders": {
                "requested_mapping": kwargs["folder_map"],
                "effective_mapping": {"staging": "production"},
            },
            "stats": {"counts": {"documents": 2}},
        }

    async def fake_apply(*_args, **kwargs):
        assert kwargs["approved_report_hash"] == REPORT_HASH
        calls.append("apply")
        return {
            "ok": True,
            "status": "applied",
            "bundle_id": "bundle-1",
            "state_hash": "b" * 64,
            "workspace": "target",
            "resumed": False,
            "imported": {"docstatus": 2},
            "warnings": [],
        }

    async def fake_validate(*_args, **_kwargs):
        calls.append("validate")
        return {
            "ok": True,
            "bundle_id": "bundle-1",
            "workspace": "target",
            "problems": [],
        }

    monkeypatch.setattr(portability_jobs, "create_dry_run", fake_dry_run)
    monkeypatch.setattr(portability_jobs, "apply_import", fake_apply)
    monkeypatch.setattr(portability_jobs, "validate_import", fake_validate)
    return calls


async def test_non_admin_legacy_user_is_forbidden(monkeypatch):
    monkeypatch.delenv("LIGHTRAG_API_KEY", raising=False)
    monkeypatch.setenv("LIGHTRAG_JWT_SECRET", "legacy-secret-long-enough-for-sha256")
    monkeypatch.setenv("LIGHTRAG_JWT_PASSWORD", "non-default-password")
    app = create_app()
    token = _create_jwt({"sub": "reader"})
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
        headers={"Authorization": f"Bearer {token}"},
    ) as client:
        response = await client.post("/twin/api/admin/portability/exports", json={})
    assert response.status_code == 403


async def test_admin_surface_refuses_cross_workspace_requests(admin_client):
    exported = await admin_client.post(
        "/twin/api/admin/portability/exports", json={"workspace": "other"}
    )
    assert exported.status_code == 422
    assert "bound to the runtime workspace" in exported.json()["detail"]

    imported = await admin_client.post(
        "/twin/api/admin/portability/imports",
        data={"workspace": "other"},
        files={"bundle": ("kb.tar.gz", b"bundle", "application/gzip")},
    )
    assert imported.status_code == 422
    assert "bound to the runtime workspace" in imported.json()["detail"]


async def test_import_cycle_requires_report_approval_and_validates(
    admin_client, fake_import_primitives, monkeypatch
):
    monkeypatch.setattr(
        portability_routes,
        "_request_actor",
        lambda request: request.headers.get("x-test-actor", "api_key"),
    )
    created = await admin_client.post(
        "/twin/api/admin/portability/imports",
        data={
            "workspace": "target",
            "folder_map": '{"staging":"production"}',
        },
        files={"bundle": ("kb.tar.gz", b"canonical-placeholder", "application/gzip")},
    )
    assert created.status_code == 202, created.text
    job_id = created.json()["id"]

    dry_run = await _wait_for_status(admin_client, job_id, "awaiting-approval")
    assert dry_run["report"]["report_hash"] == REPORT_HASH
    assert dry_run["report"]["blocking"] == []
    assert "upload_path" not in dry_run
    assert "report_path" not in dry_run

    wrong = await admin_client.post(
        f"/twin/api/admin/portability/imports/{job_id}/approve",
        json={"report_hash": "0" * 64},
    )
    assert wrong.status_code == 409

    approved = await admin_client.post(
        f"/twin/api/admin/portability/imports/{job_id}/approve",
        json={"report_hash": REPORT_HASH},
        headers={"X-Test-Actor": "bob.approver"},
    )
    assert approved.status_code == 200
    assert approved.json()["status"] == "approved"
    assert approved.json()["approved_report_hash"] == REPORT_HASH
    assert approved.json()["approved_by"] == "bob.approver"

    applying = await admin_client.post(
        f"/twin/api/admin/portability/imports/{job_id}/apply",
        headers={"X-Test-Actor": "carol.applier"},
    )
    assert applying.status_code == 202
    applied = await _wait_for_status(admin_client, job_id, "applied")
    assert applied["result"]["bundle_id"] == "bundle-1"
    assert applied["applied_by"] == "carol.applier"

    deadline = asyncio.get_running_loop().time() + 2
    while True:
        activity = (
            await admin_client.get(
                "/twin/api/activity",
                params={"kind": "kb-imported", "resource.id": "target"},
            )
        ).json()["items"]
        notifications = (await admin_client.get("/twin/api/notifications")).json()
        imported_notifications = [
            item for item in notifications if item.get("kind") == "kb-imported"
        ]
        if activity and imported_notifications:
            break
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError("import event/notification was not emitted")
        await asyncio.sleep(0.01)
    assert len(activity) == 1
    assert activity[0]["target"]["id"] == "target"
    assert activity[0]["meta"]["bundle_id"] == "bundle-1"
    assert activity[0]["actor"]["user"] == "carol.applier"
    assert len(imported_notifications) == 1

    validating = await admin_client.post(
        f"/twin/api/admin/portability/imports/{job_id}/validate",
        headers={"X-Test-Actor": "dana.validator"},
    )
    assert validating.status_code == 202
    validated = await _wait_for_status(admin_client, job_id, "validated")
    assert validated["validation"]["ok"] is True
    assert validated["validated_by"] == "dana.validator"
    assert fake_import_primitives == ["dry-run", "apply", "validate"]


async def test_workspace_allows_only_one_non_terminal_job(
    admin_client, fake_import_primitives
):
    first = await admin_client.post(
        "/twin/api/admin/portability/imports",
        data={"workspace": "target"},
        files={"bundle": ("first.tar.gz", b"one", "application/gzip")},
    )
    assert first.status_code == 202
    job_id = first.json()["id"]
    await _wait_for_status(admin_client, job_id, "awaiting-approval")

    second = await admin_client.post(
        "/twin/api/admin/portability/imports",
        data={"workspace": "target"},
        files={"bundle": ("second.tar.gz", b"two", "application/gzip")},
    )
    assert second.status_code == 409
    assert job_id in second.json()["detail"]

    cancelled = await admin_client.post(
        f"/twin/api/admin/portability/imports/{job_id}/cancel"
    )
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"
    assert cancelled.json()["cancelled_by"] == "api_key"


async def test_browser_upload_is_stream_capped(
    admin_client, monkeypatch, fake_import_primitives
):
    monkeypatch.setattr(portability_routes, "API_UPLOAD_MAX_BYTES", 3)
    response = await admin_client.post(
        "/twin/api/admin/portability/imports",
        files={"bundle": ("too-large.tar.gz", b"1234", "application/gzip")},
    )
    assert response.status_code == 413
    manager = portability_jobs.get_portability_jobs()
    assert list(manager.uploads_dir.rglob("bundle.tar.gz")) == []


async def test_disconnected_upload_fails_job_and_releases_workspace(
    admin_client, monkeypatch
):
    async def disconnect(*_args, **_kwargs):
        raise asyncio.CancelledError

    monkeypatch.setattr(portability_routes, "_stream_upload", disconnect)
    # Starlette's BaseHTTPMiddleware converts a cancelled response path into
    # "No response returned"; the route still observes CancelledError and
    # must complete its shielded cleanup before either signal escapes.
    with pytest.raises((asyncio.CancelledError, RuntimeError)):
        await admin_client.post(
            "/twin/api/admin/portability/imports",
            files={"bundle": ("disconnected.tar.gz", b"partial", "application/gzip")},
        )

    manager = portability_jobs.get_portability_jobs()
    job = next(iter(manager._jobs.values()))  # noqa: SLF001 - disconnect contract.
    assert job["status"] == "failed"
    assert job["owner_pid"] is None
    assert not manager._lock_path("target").exists()  # noqa: SLF001


async def test_archive_compression_does_not_block_event_loop(monkeypatch, tmp_path):
    manager = portability_jobs.PortabilityJobManager(tmp_path / "portability")
    archive_started = threading.Event()
    release_archive = threading.Event()

    async def fake_export(target, **_kwargs):
        Path(target).mkdir(parents=True)
        return SimpleNamespace(
            bundle_id="bundle-export",
            state_hash="c" * 64,
            consistency=SimpleNamespace(status="verified"),
            classification=SimpleNamespace(max_detected="C2", unknown_present=False),
            counts={},
        )

    def slow_archive(_source, target):
        archive_started.set()
        release_archive.wait(timeout=2)
        target.write_bytes(b"archive")

    async def no_event(_job):
        return None

    monkeypatch.setattr(portability_jobs, "export_kb", fake_export)
    monkeypatch.setattr(portability_jobs, "archive_bundle", slow_archive)
    monkeypatch.setattr(portability_jobs, "_emit_export_event", no_event)

    job = await manager.create_export(
        workspace="target",
        actor="admin",
        include_activity=False,
        include_procedures=False,
        force=False,
    )
    started_at = asyncio.get_running_loop().time()
    assert await asyncio.to_thread(archive_started.wait, 2)
    assert asyncio.get_running_loop().time() - started_at < 0.75
    release_archive.set()
    await manager._tasks[job["id"]]  # noqa: SLF001
    assert (await manager.get(job["id"], kind="export"))["status"] == "completed"


async def test_late_activity_cancellation_keeps_completed_export(monkeypatch, tmp_path):
    manager = portability_jobs.PortabilityJobManager(tmp_path / "portability")
    event_started = asyncio.Event()

    async def fake_export(target, **_kwargs):
        Path(target).mkdir(parents=True)
        return SimpleNamespace(
            bundle_id="bundle-export",
            state_hash="c" * 64,
            consistency=SimpleNamespace(status="verified"),
            classification=SimpleNamespace(max_detected="C2", unknown_present=False),
            counts={},
        )

    def fake_archive(_source, target):
        target.write_bytes(b"archive")

    async def blocked_event(_job):
        event_started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(portability_jobs, "export_kb", fake_export)
    monkeypatch.setattr(portability_jobs, "archive_bundle", fake_archive)
    monkeypatch.setattr(portability_jobs, "_emit_export_event", blocked_event)
    job = await manager.create_export(
        workspace="target",
        actor="admin",
        include_activity=False,
        include_procedures=False,
        force=False,
    )
    await asyncio.wait_for(event_started.wait(), 1)
    task = manager._tasks[job["id"]]  # noqa: SLF001
    task.cancel()
    await task

    persisted = await manager.get(job["id"], kind="export")
    assert persisted["status"] == "completed"
    assert persisted["download_available"] is True


@pytest.mark.parametrize("status", ["completed", "applied"])
async def test_durable_success_cannot_be_downgraded_by_late_failure(status, tmp_path):
    manager = portability_jobs.PortabilityJobManager(tmp_path / status)
    kind = "export" if status == "completed" else "import"
    job_id = ("exp_" if kind == "export" else "imp_") + "9" * 24
    job = {
        "id": job_id,
        "kind": kind,
        "workspace": "target",
        "status": status,
        "created_at": "2026-08-26T12:00:00Z",
        "updated_at": "2026-08-26T12:00:00Z",
        "actor": "admin",
        "options": {},
        "result": {},
        "report": None,
        "validation": None,
        "error": None,
    }
    manager._jobs[job_id] = job  # noqa: SLF001
    manager._write(job)  # noqa: SLF001

    await manager._fail(job_id, RuntimeError("late Activity failure"))  # noqa: SLF001
    assert (await manager.get(job_id, kind=kind))["status"] == status


@pytest.mark.parametrize("interruption", ["cancelled", "failed"])
async def test_interrupted_validation_returns_to_applied_and_can_retry(
    interruption, monkeypatch, tmp_path
):
    manager = portability_jobs.PortabilityJobManager(tmp_path / "portability")
    job_id = "imp_" + "8" * 24
    validation_started = asyncio.Event()
    block_validation = asyncio.Event()
    job = {
        "id": job_id,
        "kind": "import",
        "workspace": "target",
        "status": "applied",
        "created_at": "2026-08-26T12:00:00Z",
        "updated_at": "2026-08-26T12:00:00Z",
        "actor": "uploader",
        "validated_by": None,
        "options": {"folder_map": {"staging": "production"}},
        "result": {"ok": True},
        "report": None,
        "validation": None,
        "error": None,
        "upload_path": str(tmp_path / "bundle.tar.gz"),
    }
    manager._jobs[job_id] = job  # noqa: SLF001 - state-machine contract.
    manager._write(job)  # noqa: SLF001

    async def interrupted_validation(*_args, **_kwargs):
        validation_started.set()
        if interruption == "failed":
            raise RuntimeError("validation backend unavailable")
        await block_validation.wait()

    monkeypatch.setattr(portability_jobs, "validate_import", interrupted_validation)
    await manager.start_validate(job_id, actor="validator-one")
    await asyncio.wait_for(validation_started.wait(), 1)
    task = manager._tasks[job_id]  # noqa: SLF001
    if interruption == "cancelled":
        task.cancel()
    await task

    interrupted = manager._jobs[job_id]  # noqa: SLF001
    assert interrupted["status"] == "applied"
    assert interrupted["owner_pid"] is None
    assert interrupted["owner_process_identity"] is None
    assert (
        "ready to validate again" in interrupted["error"]
        or "retry" in interrupted["error"]
    )

    async def successful_validation(*_args, **_kwargs):
        return {"ok": True, "problems": []}

    monkeypatch.setattr(portability_jobs, "validate_import", successful_validation)
    restarted = await manager.start_validate(job_id, actor="validator-two")
    assert restarted["error"] is None
    await manager._tasks[job_id]  # noqa: SLF001
    retried = await manager.get(job_id, kind="import")
    assert retried["status"] == "validated"
    assert retried["validated_by"] == "validator-two"
    assert retried["error"] is None


async def test_export_can_be_polled_and_downloaded(admin_client, monkeypatch):
    events: list[dict] = []

    async def fake_export(target, **_kwargs):
        Path(target).mkdir(parents=True)
        return SimpleNamespace(
            bundle_id="bundle-export",
            state_hash="c" * 64,
            consistency=SimpleNamespace(status="verified"),
            classification=SimpleNamespace(max_detected="C2", unknown_present=False),
            counts={"documents": 3},
        )

    def fake_archive(_source, out_path):
        out_path.write_bytes(b"archive")
        out_path.chmod(0o600)
        return out_path

    async def capture(job):
        events.append(job)

    monkeypatch.setattr(portability_jobs, "export_kb", fake_export)
    monkeypatch.setattr(portability_jobs, "archive_bundle", fake_archive)
    monkeypatch.setattr(portability_jobs, "_emit_export_event", capture)

    response = await admin_client.post("/twin/api/admin/portability/exports", json={})
    assert response.status_code == 202
    job_id = response.json()["id"]

    deadline = asyncio.get_running_loop().time() + 2
    while True:
        polled = await admin_client.get(f"/twin/api/admin/portability/exports/{job_id}")
        if polled.json()["status"] == "completed":
            break
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("export did not complete")
        await asyncio.sleep(0.01)
    assert polled.json()["download_available"] is True
    assert polled.json()["result"]["counts"] == {"documents": 3}

    downloaded = await admin_client.get(
        f"/twin/api/admin/portability/exports/{job_id}?download=true"
    )
    assert downloaded.status_code == 200
    assert downloaded.content == b"archive"
    assert downloaded.headers["content-type"] == "application/gzip"
    assert len(events) == 1


def test_job_manager_fails_interrupted_jobs_closed(tmp_path):
    root = tmp_path / "portability"
    manager = portability_jobs.PortabilityJobManager(root)
    job_id = "imp_" + "1" * 24
    job = {
        "id": job_id,
        "kind": "import",
        "workspace": "target",
        "status": "applying",
        "created_at": "2026-08-26T12:00:00Z",
        "updated_at": "2026-08-26T12:00:00Z",
        "actor": "admin",
        "options": {},
        "result": None,
        "report": None,
        "validation": None,
        "error": None,
    }
    manager._write(job)  # noqa: SLF001 - persisted restart contract.
    reloaded = portability_jobs.PortabilityJobManager(root)
    assert reloaded._jobs[job_id]["status"] == "failed"  # noqa: SLF001
    assert "restarted" in reloaded._jobs[job_id]["error"]  # noqa: SLF001


def test_job_manager_preserves_live_other_worker_and_refreshes(tmp_path, monkeypatch):
    root = tmp_path / "portability"
    writer = portability_jobs.PortabilityJobManager(root)
    job_id = "imp_" + "2" * 24
    job = {
        "id": job_id,
        "kind": "import",
        "workspace": "target",
        "status": "applying",
        "created_at": "2026-08-26T12:00:00Z",
        "updated_at": "2026-08-26T12:00:00Z",
        "actor": "admin",
        "owner_pid": 4242,
        "owner_process_identity": "linux:boot:123",
        "options": {},
        "result": None,
        "report": None,
        "validation": None,
        "error": None,
    }
    writer._write(job)  # noqa: SLF001 - cross-worker persistence contract.
    monkeypatch.setattr(portability_jobs, "_pid_is_alive", lambda pid: pid == 4242)
    monkeypatch.setattr(
        portability_jobs,
        "_process_identity",
        lambda pid: "linux:boot:123" if pid == 4242 else None,
    )

    reader = portability_jobs.PortabilityJobManager(root)
    assert reader._jobs[job_id]["status"] == "applying"  # noqa: SLF001

    job.update(
        status="applied",
        owner_pid=None,
        owner_process_identity=None,
        updated_at="2026-08-26T12:00:01Z",
    )
    writer._write(job)  # noqa: SLF001
    assert reader._require(job_id, "import")["status"] == "applied"  # noqa: SLF001


def test_reused_pid_without_matching_process_identity_releases_job_lock(
    tmp_path, monkeypatch
):
    root = tmp_path / "portability"
    writer = portability_jobs.PortabilityJobManager(root)
    job_id = "imp_" + "3" * 24
    job = {
        "id": job_id,
        "kind": "import",
        "workspace": "target",
        "status": "applying",
        "created_at": "2026-08-26T12:00:00Z",
        "updated_at": "2026-08-26T12:00:00Z",
        "actor": "admin",
        "owner_pid": 1,
        "owner_process_identity": "linux:boot:old-start",
        "options": {},
        "result": None,
        "report": None,
        "validation": None,
        "error": None,
    }
    writer._write(job)  # noqa: SLF001 - persisted restart contract.
    lock = writer._lock_path("target")  # noqa: SLF001
    lock.write_text(job_id + "\n", encoding="utf-8")
    monkeypatch.setattr(portability_jobs, "_pid_is_alive", lambda pid: pid == 1)
    monkeypatch.setattr(
        portability_jobs,
        "_process_identity",
        lambda pid: "linux:boot:new-start" if pid == 1 else None,
    )

    reloaded = portability_jobs.PortabilityJobManager(root)
    assert reloaded._jobs[job_id]["status"] == "failed"  # noqa: SLF001
    assert "restarted" in reloaded._jobs[job_id]["error"]  # noqa: SLF001
    assert not lock.exists()


def test_restart_during_validation_restores_resumable_applied_state(
    tmp_path, monkeypatch
):
    root = tmp_path / "portability"
    writer = portability_jobs.PortabilityJobManager(root)
    job_id = "imp_" + "4" * 24
    job = {
        "id": job_id,
        "kind": "import",
        "workspace": "target",
        "status": "validating",
        "created_at": "2026-08-26T12:00:00Z",
        "updated_at": "2026-08-26T12:00:00Z",
        "actor": "admin",
        "owner_pid": 1,
        "owner_process_identity": "linux:boot:old-start",
        "options": {},
        "result": {"ok": True},
        "report": None,
        "validation": None,
        "error": None,
    }
    writer._write(job)  # noqa: SLF001 - persisted restart contract.
    monkeypatch.setattr(portability_jobs, "_pid_is_alive", lambda pid: pid == 1)
    monkeypatch.setattr(
        portability_jobs,
        "_process_identity",
        lambda pid: "linux:boot:new-start" if pid == 1 else None,
    )

    reloaded = portability_jobs.PortabilityJobManager(root)
    recovered = reloaded._jobs[job_id]  # noqa: SLF001
    assert recovered["status"] == "applied"
    assert recovered["owner_pid"] is None
    assert recovered["owner_process_identity"] is None
    assert "ready to validate again" in recovered["error"]


def test_fresh_unbound_workspace_lock_is_not_stolen(tmp_path):
    manager = portability_jobs.PortabilityJobManager(tmp_path / "portability")
    lock = manager._lock_path("target")  # noqa: SLF001
    lock.write_text("imp_pending\n", encoding="utf-8")

    with pytest.raises(portability_jobs.JobConflictError, match="being claimed"):
        manager._claim_workspace("target", "imp_competing")  # noqa: SLF001
