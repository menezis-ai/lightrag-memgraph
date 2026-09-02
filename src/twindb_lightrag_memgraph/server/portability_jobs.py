"""Persistent admin jobs for KB portability (ADR 010, decision 5).

Design record: ``docs/adr/010-kb-portability-contract.md``.

The CLI remains the primitive for large/offline transfers.  This module adds a
bounded operator surface without weakening the import contract: uploads are
sealed to one private job directory, dry-run reports are persisted before an
approval can be recorded, and apply always consumes that exact report.

Only one non-terminal job may own a workspace at a time.  The lock is enforced
from the persisted job ledger as well as in-process, so a server restart cannot
silently admit a competing import.  Interrupted running jobs are failed closed
on reload; their bundle/report/checkpoint stay available to an administrator
for diagnosis or CLI resume.
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime as dt
import fcntl
import json
import logging
import os
import secrets
import time
from pathlib import Path
from typing import Any, Iterator, Literal

from .._constants import (
    resolve_portability_batch_size,
    resolve_portability_dir,
    validate_identifier,
)
from ..portability.bundle import archive_bundle
from ..portability.exporter import export_kb
from ..portability.importer import apply_import
from ..portability.plan import create_dry_run, write_report
from ..portability.validate import validate_import

logger = logging.getLogger(__name__)

JobKind = Literal["export", "import"]

_RUNNING_STATES = frozenset(
    {"queued", "uploading", "running", "dry-running", "applying", "validating"}
)
_TERMINAL_STATES = frozenset(
    {"completed", "failed", "cancelled", "validated", "validation-failed"}
)
_DURABLE_SUCCESS_STATES = frozenset(
    {"completed", "applied", "validated", "validation-failed"}
)
_PRIVATE_KEYS = frozenset(
    {
        "archive_path",
        "bundle_path",
        "checkpoint_path",
        "owner_pid",
        "owner_process_identity",
        "report_path",
        "upload_path",
    }
)
_FRESH_LOCK_SECONDS = 60
_PROCESS_INSTANCE_TOKEN = secrets.token_hex(16)


class JobConflictError(RuntimeError):
    """A workspace already has a non-terminal portability job."""


class JobNotFoundError(LookupError):
    """The requested job does not exist or has the wrong operation kind."""


class JobStateError(RuntimeError):
    """The requested transition is not legal from the current state."""


def _utc_now() -> str:
    return (
        dt.datetime.now(dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _job_id(kind: JobKind) -> str:
    return f"{'exp' if kind == 'export' else 'imp'}_{secrets.token_hex(12)}"


def _json_clone(value: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(value))


def _pid_is_alive(pid: object) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _process_identity(pid: object) -> str | None:
    """Return an identity that changes when a PID is reused.

    Linux exposes the process start tick in ``/proc/<pid>/stat``. Pairing it
    with the kernel boot id distinguishes a new container process that reused
    PID 1 from the worker that originally owned the job. Non-Linux runtimes
    can still identify this process through a module-lifetime token; ownership
    of other processes then fails closed.
    """
    if not isinstance(pid, int) or pid <= 0:
        return None
    try:
        stat_line = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        _, separator, fields = stat_line.rpartition(") ")
        values = fields.split()
        if not separator or len(values) <= 19:
            return None
        start_ticks = values[19]
        boot_id = (
            Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()
        )
        if not boot_id or not start_ticks.isdigit():
            return None
        return f"linux:{boot_id}:{start_ticks}"
    except OSError:
        if pid == os.getpid() and _pid_is_alive(pid):
            return f"local:{_PROCESS_INSTANCE_TOKEN}"
        return None


def _owner_fields() -> dict[str, Any]:
    pid = os.getpid()
    identity = _process_identity(pid)
    if identity is None:  # pragma: no cover - current-process fallback is total.
        raise RuntimeError("cannot establish portability worker identity")
    return {"owner_pid": pid, "owner_process_identity": identity}


class PortabilityJobManager:
    """Small persistent state machine around the portability primitives."""

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root or resolve_portability_dir())
        self.jobs_dir = self.root / "jobs"
        self.locks_dir = self.jobs_dir / "locks"
        self.uploads_dir = self.root / "uploads"
        self.exports_dir = self.root / "exports"
        for directory in (
            self.root,
            self.jobs_dir,
            self.locks_dir,
            self.uploads_dir,
            self.exports_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            directory.chmod(0o700)
        self._jobs: dict[str, dict[str, Any]] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._lock = asyncio.Lock()
        self._load()

    def _load(self) -> None:
        for path in sorted(self.jobs_dir.glob("*.json")):
            try:
                job = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(job, dict) or job.get("id") != path.stem:
                    raise ValueError("job id/file name mismatch")
                self._recover_interrupted(job)
                self._jobs[str(job["id"])] = job
                if job.get("status") in _TERMINAL_STATES:
                    self._release_workspace(job)
                else:
                    self._claim_workspace(str(job["workspace"]), str(job["id"]))
            except Exception:  # noqa: BLE001 - one corrupt job must not hide the rest.
                logger.exception("portability: cannot load job ledger %s", path)

    def _recover_interrupted(self, job: dict[str, Any]) -> None:
        if job.get("status") not in _RUNNING_STATES:
            return
        owner_pid = job.get("owner_pid")
        owner_identity = job.get("owner_process_identity")
        observed_identity = (
            _process_identity(owner_pid) if _pid_is_alive(owner_pid) else None
        )
        if (
            isinstance(owner_identity, str)
            and owner_identity
            and owner_identity == observed_identity
        ):
            return
        interrupted_validation = job.get("status") == "validating"
        job.update(
            status="applied" if interrupted_validation else "failed",
            owner_pid=None,
            owner_process_identity=None,
            error=(
                "Validation was interrupted by a server restart; the applied import "
                "is ready to validate again."
                if interrupted_validation
                else "The server restarted while this job was running. "
                "Its private report/checkpoint was retained for operator recovery."
            ),
            updated_at=_utc_now(),
        )
        self._write(job)

    def _refresh(self, job_id: str) -> dict[str, Any] | None:
        path = self.jobs_dir / f"{job_id}.json"
        try:
            job = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return self._jobs.get(job_id)
        if not isinstance(job, dict) or job.get("id") != job_id:
            raise ValueError(f"invalid persisted portability job {job_id!r}")
        self._recover_interrupted(job)
        self._jobs[job_id] = job
        if job.get("status") in _TERMINAL_STATES:
            self._release_workspace(job)
        return job

    def _refresh_all(self) -> None:
        for path in self.jobs_dir.glob("*.json"):
            self._refresh(path.stem)

    def _write(self, job: dict[str, Any]) -> None:
        target = self.jobs_dir / f"{job['id']}.json"
        temporary = target.with_suffix(".json.part")
        try:
            temporary.write_text(
                json.dumps(job, ensure_ascii=False, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            temporary.chmod(0o600)
            os.replace(temporary, target)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    def _job_dir(self, job_id: str, *, kind: JobKind) -> Path:
        parent = self.exports_dir if kind == "export" else self.uploads_dir
        target = parent / job_id
        target.mkdir(parents=True, exist_ok=True, mode=0o700)
        target.chmod(0o700)
        return target

    def _assert_workspace_available(self, workspace: str) -> None:
        self._refresh_all()
        busy = next(
            (
                job
                for job in self._jobs.values()
                if job.get("workspace") == workspace
                and job.get("status") not in _TERMINAL_STATES
            ),
            None,
        )
        if busy is not None:
            raise JobConflictError(
                f"workspace {workspace!r} already has active job {busy['id']} "
                f"({busy['status']})"
            )

    def _lock_path(self, workspace: str) -> Path:
        return self.locks_dir / f"{validate_identifier(workspace, 'workspace')}.lock"

    def _claim_workspace(self, workspace: str, job_id: str) -> None:
        """Claim a workspace atomically across ASGI worker processes."""
        path = self._lock_path(workspace)
        for _ in range(2):
            try:
                descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError:
                try:
                    owner = path.read_text(encoding="utf-8").strip()
                except OSError:
                    owner = ""
                if owner == job_id:
                    return
                owner_job = self._jobs.get(owner)
                if owner_job is None and owner:
                    owner_path = self.jobs_dir / f"{owner}.json"
                    try:
                        loaded = json.loads(owner_path.read_text(encoding="utf-8"))
                        owner_job = loaded if isinstance(loaded, dict) else None
                    except (OSError, ValueError):
                        owner_job = None
                if (
                    owner_job is not None
                    and owner_job.get("status") not in _TERMINAL_STATES
                ):
                    raise JobConflictError(
                        f"workspace {workspace!r} already has active job {owner} "
                        f"({owner_job.get('status')})"
                    )
                if owner_job is None:
                    try:
                        lock_age = time.time() - path.stat().st_mtime
                    except FileNotFoundError:
                        continue
                    if lock_age < _FRESH_LOCK_SECONDS:
                        raise JobConflictError(
                            f"workspace {workspace!r} is being claimed by another job"
                        )
                path.unlink(missing_ok=True)
                continue
            try:
                os.write(descriptor, (job_id + "\n").encode("utf-8"))
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            return
        raise JobConflictError(f"workspace {workspace!r} lock could not be acquired")

    def _release_workspace(self, job: dict[str, Any]) -> None:
        path = self._lock_path(str(job["workspace"]))
        try:
            if path.read_text(encoding="utf-8").strip() == str(job["id"]):
                path.unlink(missing_ok=True)
        except FileNotFoundError:
            return

    @contextlib.contextmanager
    def _transition(self, job_id: str) -> Iterator[None]:
        """Serialize a job transition across ASGI worker processes."""
        path = self.locks_dir / f"{job_id}.transition"
        with path.open("a+", encoding="utf-8") as handle:
            os.fchmod(handle.fileno(), 0o600)
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _require(self, job_id: str, kind: JobKind) -> dict[str, Any]:
        job = self._refresh(job_id)
        if job is None or job.get("kind") != kind:
            raise JobNotFoundError(f"{kind} job {job_id!r} not found")
        return job

    def public(self, job: dict[str, Any]) -> dict[str, Any]:
        payload = {key: value for key, value in job.items() if key not in _PRIVATE_KEYS}
        payload["download_available"] = bool(
            job.get("kind") == "export"
            and job.get("status") == "completed"
            and Path(str(job.get("archive_path") or "")).is_file()
        )
        return _json_clone(payload)

    async def get(self, job_id: str, *, kind: JobKind) -> dict[str, Any]:
        async with self._lock:
            return self.public(self._require(job_id, kind))

    async def archive_path(self, job_id: str) -> Path:
        async with self._lock:
            job = self._require(job_id, "export")
            if job.get("status") != "completed":
                raise JobStateError("export is not complete")
            path = Path(str(job.get("archive_path") or ""))
            if not path.is_file():
                raise JobStateError("export artifact is unavailable")
            return path

    async def create_export(
        self,
        *,
        workspace: str,
        actor: str,
        include_activity: bool,
        include_procedures: bool,
        force: bool,
    ) -> dict[str, Any]:
        workspace = validate_identifier(workspace, "workspace")
        async with self._lock:
            self._assert_workspace_available(workspace)
            job_id = _job_id("export")
            self._claim_workspace(workspace, job_id)
            try:
                job_dir = self._job_dir(job_id, kind="export")
                job = {
                    "id": job_id,
                    "kind": "export",
                    "workspace": workspace,
                    "status": "queued",
                    "created_at": _utc_now(),
                    "updated_at": _utc_now(),
                    "actor": actor,
                    **_owner_fields(),
                    "options": {
                        "include_activity": include_activity,
                        "include_procedures": include_procedures,
                        "force": force,
                    },
                    "result": None,
                    "report": None,
                    "validation": None,
                    "error": None,
                    "bundle_path": str(job_dir / "bundle"),
                    "archive_path": str(job_dir / "bundle.tar.gz"),
                }
                self._jobs[job_id] = job
                self._write(job)
            except BaseException:
                self._release_workspace({"workspace": workspace, "id": job_id})
                raise
            self._tasks[job_id] = asyncio.create_task(
                self._run_export(job_id), name=f"portability-{job_id}"
            )
            return self.public(job)

    async def reserve_import(
        self,
        *,
        workspace: str,
        actor: str,
        folder_map: dict[str, str],
        allow_unverified: bool,
        upload_name: str,
    ) -> tuple[dict[str, Any], Path]:
        workspace = validate_identifier(workspace, "workspace")
        mapping = {
            validate_identifier(str(source), "source folder"): validate_identifier(
                str(destination), "target folder"
            )
            for source, destination in folder_map.items()
        }
        async with self._lock:
            self._assert_workspace_available(workspace)
            job_id = _job_id("import")
            self._claim_workspace(workspace, job_id)
            try:
                job_dir = self._job_dir(job_id, kind="import")
                upload_path = job_dir / "bundle.tar.gz"
                job = {
                    "id": job_id,
                    "kind": "import",
                    "workspace": workspace,
                    "status": "uploading",
                    "created_at": _utc_now(),
                    "updated_at": _utc_now(),
                    "actor": actor,
                    "approved_by": None,
                    "approved_report_hash": None,
                    "applied_by": None,
                    "validated_by": None,
                    "cancelled_by": None,
                    **_owner_fields(),
                    "options": {
                        "folder_map": mapping,
                        "allow_unverified": allow_unverified,
                        "upload_name": Path(upload_name).name,
                    },
                    "result": None,
                    "report": None,
                    "validation": None,
                    "error": None,
                    "upload_path": str(upload_path),
                    "report_path": str(job_dir / "report.json"),
                    "checkpoint_path": str(job_dir / "checkpoint.json"),
                }
                self._jobs[job_id] = job
                self._write(job)
            except BaseException:
                self._release_workspace({"workspace": workspace, "id": job_id})
                raise
            return self.public(job), upload_path

    async def fail_upload(self, job_id: str, message: str) -> None:
        async with self._lock:
            with self._transition(job_id):
                job = self._require(job_id, "import")
                job.update(
                    status="failed",
                    owner_pid=None,
                    owner_process_identity=None,
                    error=message,
                    updated_at=_utc_now(),
                )
                self._write(job)
                self._release_workspace(job)

    async def start_dry_run(self, job_id: str) -> dict[str, Any]:
        async with self._lock:
            with self._transition(job_id):
                job = self._require(job_id, "import")
                if job.get("status") != "uploading":
                    raise JobStateError("import upload is not awaiting dry-run")
                job.update(
                    status="dry-running",
                    **_owner_fields(),
                    updated_at=_utc_now(),
                )
                self._write(job)
            self._tasks[job_id] = asyncio.create_task(
                self._run_dry_run(job_id), name=f"portability-{job_id}"
            )
            return self.public(job)

    async def approve(
        self, job_id: str, *, report_hash: str, actor: str
    ) -> dict[str, Any]:
        async with self._lock:
            with self._transition(job_id):
                job = self._require(job_id, "import")
                if job.get("status") != "awaiting-approval":
                    raise JobStateError("import is not awaiting approval")
                report = job.get("report") or {}
                if not report_hash or report_hash != report.get("report_hash"):
                    raise JobStateError(
                        "approved report_hash does not match the dry-run"
                    )
                if report.get("blocking"):
                    raise JobStateError("a blocking dry-run cannot be approved")
                job.update(
                    status="approved",
                    approved_report_hash=report_hash,
                    approved_by=actor,
                    updated_at=_utc_now(),
                )
                self._write(job)
                return self.public(job)

    async def start_apply(self, job_id: str, *, actor: str) -> dict[str, Any]:
        async with self._lock:
            with self._transition(job_id):
                job = self._require(job_id, "import")
                if job.get("status") != "approved":
                    raise JobStateError("import must be approved before apply")
                if not job.get("approved_report_hash"):
                    raise JobStateError("import has no persisted report approval")
                job.update(
                    status="applying",
                    applied_by=actor,
                    **_owner_fields(),
                    updated_at=_utc_now(),
                )
                self._write(job)
            self._tasks[job_id] = asyncio.create_task(
                self._run_apply(job_id), name=f"portability-{job_id}"
            )
            return self.public(job)

    async def start_validate(self, job_id: str, *, actor: str) -> dict[str, Any]:
        async with self._lock:
            with self._transition(job_id):
                job = self._require(job_id, "import")
                if job.get("status") != "applied":
                    raise JobStateError("import must be applied before validation")
                job.update(
                    status="validating",
                    validated_by=actor,
                    error=None,
                    **_owner_fields(),
                    updated_at=_utc_now(),
                )
                self._write(job)
            self._tasks[job_id] = asyncio.create_task(
                self._run_validate(job_id), name=f"portability-{job_id}"
            )
            return self.public(job)

    async def cancel(self, job_id: str, *, kind: JobKind, actor: str) -> dict[str, Any]:
        async with self._lock:
            with self._transition(job_id):
                job = self._require(job_id, kind)
                if job.get("status") in _TERMINAL_STATES:
                    raise JobStateError(f"{kind} job is already terminal")
                if job.get("status") in {"applying", "applied", "validating"}:
                    raise JobStateError(
                        "import cannot be cancelled after apply has started"
                    )
                task = self._tasks.get(job_id)
                if task is not None and not task.done():
                    task.cancel()
                job.update(
                    status="cancelled",
                    cancelled_by=actor,
                    owner_pid=None,
                    owner_process_identity=None,
                    updated_at=_utc_now(),
                    error=None,
                )
                self._write(job)
                self._release_workspace(job)
                return self.public(job)

    async def _update(self, job_id: str, **changes: Any) -> dict[str, Any]:
        async with self._lock:
            with self._transition(job_id):
                kind: JobKind = (
                    "export" if self._jobs[job_id]["kind"] == "export" else "import"
                )
                job = self._require(job_id, kind)
                # A user cancellation wins races with a finishing coroutine.
                if job.get("status") == "cancelled":
                    return job
                status = changes.get("status")
                # Publishing an export or applying an import is irreversible.
                # A late cancellation while the Activity event is emitted must
                # never hide the already durable result.
                if job.get("status") in _DURABLE_SUCCESS_STATES and status in {
                    "cancelled",
                    "failed",
                }:
                    return job
                if status not in _RUNNING_STATES:
                    changes.setdefault("owner_pid", None)
                    changes.setdefault("owner_process_identity", None)
                job.update(changes, updated_at=_utc_now())
                self._write(job)
                if job.get("status") in _TERMINAL_STATES:
                    self._release_workspace(job)
                return job

    async def _fail(self, job_id: str, exc: BaseException) -> None:
        if isinstance(exc, asyncio.CancelledError):
            await self._update(job_id, status="cancelled", error=None)
            return
        updated = await self._update(
            job_id, status="failed", error=str(exc) or type(exc).__name__
        )
        if updated.get("status") == "failed":
            logger.error(
                "portability: job %s failed",
                job_id,
                exc_info=(type(exc), exc, exc.__traceback__),
            )
        else:
            logger.warning(
                "portability: ignored late side-effect failure for durable job %s (%s)",
                job_id,
                updated.get("status"),
            )

    async def _restore_applied_after_validation(
        self, job_id: str, exc: BaseException
    ) -> None:
        """Keep an applied import resumable when validation is interrupted."""
        cancelled = isinstance(exc, asyncio.CancelledError)
        detail = str(exc) or type(exc).__name__
        message = (
            "Validation was cancelled; the applied import is ready to validate again."
            if cancelled
            else f"Validation was interrupted ({detail}); retry validation."
        )
        updated = await self._update(job_id, status="applied", error=message)
        if not cancelled and updated.get("status") == "applied":
            logger.error(
                "portability: validation for job %s was interrupted",
                job_id,
                exc_info=(type(exc), exc, exc.__traceback__),
            )

    async def _run_export(self, job_id: str) -> None:
        job = self._jobs[job_id]
        try:
            await self._update(job_id, status="running")
            options = job["options"]
            manifest = await export_kb(
                job["bundle_path"],
                workspace=job["workspace"],
                include_activity=bool(options["include_activity"]),
                include_procedures=bool(options["include_procedures"]),
                batch=resolve_portability_batch_size(),
                force=bool(options["force"]),
                actor=job["actor"],
            )
            await asyncio.to_thread(
                archive_bundle,
                Path(job["bundle_path"]),
                Path(job["archive_path"]),
            )
            result = {
                "bundle_id": manifest.bundle_id,
                "state_hash": manifest.state_hash,
                "consistency": manifest.consistency.status,
                "classification": {
                    "max_detected": manifest.classification.max_detected,
                    "unknown_present": manifest.classification.unknown_present,
                },
                "counts": manifest.counts,
            }
            updated = await self._update(job_id, status="completed", result=result)
            if updated.get("status") == "completed":
                await _emit_export_event(updated)
        except (asyncio.CancelledError, Exception) as exc:
            await self._fail(job_id, exc)

    async def _run_dry_run(self, job_id: str) -> None:
        job = self._jobs[job_id]
        try:
            options = job["options"]
            report = await create_dry_run(
                job["upload_path"],
                workspace=job["workspace"],
                folder_map=options["folder_map"],
                allow_unverified=bool(options["allow_unverified"]),
            )
            await asyncio.to_thread(write_report, job["report_path"], report)
            await self._update(
                job_id,
                status="awaiting-approval",
                report=report,
                error=None,
            )
        except (asyncio.CancelledError, Exception) as exc:
            await self._fail(job_id, exc)

    async def _run_apply(self, job_id: str) -> None:
        job = self._jobs[job_id]
        try:
            result = await apply_import(
                job["upload_path"],
                report_path=job["report_path"],
                checkpoint_path=job["checkpoint_path"],
                approved_report_hash=job["approved_report_hash"],
                batch=resolve_portability_batch_size(),
            )
            updated = await self._update(job_id, status="applied", result=result)
            if updated.get("status") == "applied":
                await _emit_import_event(updated)
        except (asyncio.CancelledError, Exception) as exc:
            await self._fail(job_id, exc)

    async def _run_validate(self, job_id: str) -> None:
        job = self._jobs[job_id]
        try:
            options = job["options"]
            validation = await validate_import(
                job["upload_path"],
                workspace=job["workspace"],
                folder_map=options["folder_map"],
                batch=resolve_portability_batch_size(),
            )
            await self._update(
                job_id,
                status="validated" if validation.get("ok") else "validation-failed",
                validation=validation,
                error=None,
            )
        except (asyncio.CancelledError, Exception) as exc:
            await self._restore_applied_after_validation(job_id, exc)


async def _emit_export_event(job: dict[str, Any]) -> None:
    from .activity_events import emit_activity_event

    result = job.get("result") or {}
    await emit_activity_event(
        kind="kb-exported",
        sev="info",
        actor=str(job.get("actor") or "operator"),
        target_type="workspace",
        target_label=str(job["workspace"]),
        target_id=str(job["workspace"]),
        summary=f"KB workspace {job['workspace']} exported",
        meta={
            "operation": "export",
            "job_id": job["id"],
            "bundle_id": result.get("bundle_id"),
            "state_hash": result.get("state_hash"),
            "consistency": result.get("consistency"),
        },
    )


async def _emit_import_event(job: dict[str, Any]) -> None:
    from .activity_events import emit_activity_event
    from .webui.events import _make_notification
    from .webui.store import get_store

    result = job.get("result") or {}
    actor = str(job.get("applied_by") or job.get("actor") or "operator")
    await emit_activity_event(
        kind="kb-imported",
        sev="info",
        actor=actor,
        target_type="workspace",
        target_label=str(job["workspace"]),
        target_id=str(job["workspace"]),
        summary=f"KB bundle imported into workspace {job['workspace']}",
        meta={
            "operation": "import",
            "job_id": job["id"],
            "bundle_id": result.get("bundle_id"),
            "state_hash": result.get("state_hash"),
            "resumed": bool(result.get("resumed")),
        },
    )
    try:
        await get_store().push_notification(
            _make_notification(
                kind="kb-imported",
                title="KB import completed",
                tagname=None,
                suffix=None,
                sub=f"Workspace {job['workspace']} is ready for validation.",
            )
        )
    except Exception:  # noqa: BLE001 - notification cannot invalidate the import.
        logger.exception("portability: target notification emission failed")


_manager: PortabilityJobManager | None = None


def get_portability_jobs() -> PortabilityJobManager:
    global _manager
    resolved = Path(resolve_portability_dir())
    if _manager is None or _manager.root != resolved:
        _manager = PortabilityJobManager(resolved)
    return _manager


def reset_portability_jobs_for_tests() -> None:
    global _manager
    for task in (
        _manager._tasks.values() if _manager is not None else ()
    ):  # noqa: SLF001
        if not task.done():
            task.cancel()
    _manager = None


__all__ = [
    "JobConflictError",
    "JobNotFoundError",
    "JobStateError",
    "PortabilityJobManager",
    "get_portability_jobs",
    "reset_portability_jobs_for_tests",
]
