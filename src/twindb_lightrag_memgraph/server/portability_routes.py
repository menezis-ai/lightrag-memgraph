"""Admin HTTP surface for canonical KB portability (plan PR-P3 / T3.1).

These routes orchestrate the exact CLI primitives; they do not introduce a
second import implementation.  The browser surface is intentionally bounded
to 100 MiB compressed uploads.  Larger bundles remain a maintenance-window CLI
operation, both to avoid tying up an ASGI worker and to preserve the streamed
body-limit posture established after security finding R-02.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Annotated, Any

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Path as PathParameter,
    Query,
    Request,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse

from .._constants import (
    TWIN_PORTABILITY_INCLUDE_ACTIVITY_ENV,
    TWIN_PORTABILITY_INCLUDE_PROCEDURES_ENV,
    portability_flag_enabled,
    resolve_workspace,
)
from .idp_jwt import require_admin_user
from .portability_jobs import (
    JobConflictError,
    JobNotFoundError,
    JobStateError,
    get_portability_jobs,
)
from .webui.events import _request_actor
from .webui_models import (
    PortabilityApproval,
    PortabilityExportCreate,
    PortabilityJobResponse,
)

API_UPLOAD_MAX_BYTES = 100 * 1024 * 1024
_UPLOAD_CHUNK_BYTES = 1024 * 1024
JobId = Annotated[
    str,
    PathParameter(description="Opaque portability job id returned at creation."),
]

_ERROR_RESPONSES = {
    401: {"description": "Authentication credentials are missing or invalid."},
    403: {"description": "Administrator scope is required."},
    404: {"description": "The portability job does not exist."},
    409: {"description": "The workspace is busy or the transition is invalid."},
    422: {"description": "The request payload is invalid."},
}

router = APIRouter(
    prefix="/admin/portability",
    tags=["admin-portability"],
    dependencies=[Depends(require_admin_user)],
)


def _raise_job_error(exc: Exception) -> None:
    if isinstance(exc, JobNotFoundError):
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if isinstance(exc, (JobConflictError, JobStateError)):
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if isinstance(exc, ValueError):
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    raise exc


def _folder_map(raw: str) -> dict[str, str]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=422, detail="folder_map must be a JSON object"
        ) from exc
    if not isinstance(value, dict) or not all(
        isinstance(key, str) and isinstance(item, str) for key, item in value.items()
    ):
        raise HTTPException(
            status_code=422,
            detail="folder_map must map source folder ids to target folder ids",
        )
    return value


def _bound_workspace(requested: str | None) -> str:
    """Keep the admin surface inside the workspace bound at server boot."""
    runtime = resolve_workspace()
    if requested is not None and requested != runtime:
        raise HTTPException(
            status_code=422,
            detail=(
                "Admin portability is bound to the runtime workspace; "
                "use the target instance or CLI for another workspace."
            ),
        )
    return runtime


async def _stream_upload(upload: UploadFile, target: Path) -> int:
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    total = 0
    handle = target.open("xb")
    try:
        target.chmod(0o600)
        while chunk := await upload.read(_UPLOAD_CHUNK_BYTES):
            total += len(chunk)
            if total > API_UPLOAD_MAX_BYTES:
                raise HTTPException(
                    status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                    detail=(
                        "Browser portability uploads are limited to 100 MiB; "
                        "use the CLI for larger bundles."
                    ),
                )
            await asyncio.to_thread(handle.write, chunk)
        await asyncio.to_thread(handle.flush)
        await asyncio.to_thread(os.fsync, handle.fileno())
    except BaseException:
        handle.close()
        target.unlink(missing_ok=True)
        raise
    finally:
        if not handle.closed:
            handle.close()
        await upload.close()
    return total


@router.post(
    "/exports",
    response_model=PortabilityJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start a canonical KB export (admin)",
    description=(
        "Start a workspace-wide canonical export in a private background job. "
        "Poll the returned id; once completed, GET it with download=true."
    ),
    responses={
        202: {"description": "Export job accepted."},
        **_ERROR_RESPONSES,
    },
)
async def create_export_job(
    body: PortabilityExportCreate,
    request: Request,
) -> dict[str, Any]:
    try:
        return await get_portability_jobs().create_export(
            workspace=_bound_workspace(body.workspace),
            actor=_request_actor(request),
            include_activity=body.include_activity
            or portability_flag_enabled(TWIN_PORTABILITY_INCLUDE_ACTIVITY_ENV),
            include_procedures=body.include_procedures
            or portability_flag_enabled(TWIN_PORTABILITY_INCLUDE_PROCEDURES_ENV),
            force=body.force,
        )
    except Exception as exc:  # mapped to stable HTTP errors below.
        _raise_job_error(exc)
        raise AssertionError("unreachable")


@router.get(
    "/exports/{job_id}",
    response_model=PortabilityJobResponse,
    summary="Inspect or download a KB export job (admin)",
    description=(
        "Return the persisted export state. Set download=true after completion "
        "to receive the canonical tar.gz artifact."
    ),
    responses={
        200: {
            "description": "Export state or completed gzip archive.",
            "content": {"application/gzip": {}},
        },
        **_ERROR_RESPONSES,
    },
)
async def get_export_job(
    job_id: JobId,
    download: Annotated[
        bool,
        Query(description="Return the completed tar.gz artifact instead of JSON."),
    ] = False,
) -> Any:
    manager = get_portability_jobs()
    try:
        if download:
            path = await manager.archive_path(job_id)
            job = await manager.get(job_id, kind="export")
            return FileResponse(
                path,
                media_type="application/gzip",
                filename=f"twin-kb-{job['workspace']}-{job_id}.tar.gz",
            )
        return await manager.get(job_id, kind="export")
    except Exception as exc:
        _raise_job_error(exc)
        raise AssertionError("unreachable")


@router.post(
    "/imports",
    response_model=PortabilityJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload and dry-run a canonical KB import (admin)",
    description=(
        "Stream a canonical tar.gz bundle into a private job directory, then "
        "run target compatibility checks. The compressed upload is capped at "
        "100 MiB; larger bundles must use the CLI."
    ),
    responses={
        202: {"description": "Upload persisted and dry-run job accepted."},
        413: {"description": "Compressed upload exceeds the browser limit."},
        **_ERROR_RESPONSES,
    },
)
async def create_import_job(
    request: Request,
    bundle: Annotated[
        UploadFile,
        File(description="Canonical twin-kb-bundle tar.gz archive (max 100 MiB)."),
    ],
    workspace: Annotated[
        str | None,
        Form(description="Target workspace; omit to use runtime WORKSPACE."),
    ] = None,
    folder_map: Annotated[
        str,
        Form(description="JSON object mapping source folder ids to target ids."),
    ] = "{}",
    allow_unverified: Annotated[
        bool,
        Form(description="Allow an explicitly unverified bundle in this dry-run."),
    ] = False,
) -> dict[str, Any]:
    manager = get_portability_jobs()
    job: dict[str, Any] | None = None
    try:
        job, target = await manager.reserve_import(
            workspace=_bound_workspace(workspace),
            actor=_request_actor(request),
            folder_map=_folder_map(folder_map),
            allow_unverified=allow_unverified,
            upload_name=bundle.filename or "bundle.tar.gz",
        )
        await _stream_upload(bundle, target)
        return await manager.start_dry_run(str(job["id"]))
    except asyncio.CancelledError:
        if job is not None:
            await asyncio.shield(
                manager.fail_upload(
                    str(job["id"]), "Upload cancelled before the bundle was complete"
                )
            )
        raise
    except HTTPException as exc:
        if job is not None:
            await manager.fail_upload(str(job["id"]), str(exc.detail))
        raise
    except Exception as exc:
        if job is not None:
            await manager.fail_upload(str(job["id"]), str(exc) or type(exc).__name__)
        _raise_job_error(exc)
        raise AssertionError("unreachable")


@router.get(
    "/imports/{job_id}",
    response_model=PortabilityJobResponse,
    summary="Inspect a KB import job (admin)",
    description="Return the dry-run, approval, apply and validation state.",
    responses={200: {"description": "Current import job state."}, **_ERROR_RESPONSES},
)
async def get_import_job(job_id: JobId) -> dict[str, Any]:
    try:
        return await get_portability_jobs().get(job_id, kind="import")
    except Exception as exc:
        _raise_job_error(exc)
        raise AssertionError("unreachable")


@router.post(
    "/imports/{job_id}/approve",
    response_model=PortabilityJobResponse,
    summary="Approve an import dry-run (admin)",
    description=(
        "Approve the exact displayed report_hash. Reports with blocking findings "
        "cannot be approved."
    ),
    responses={200: {"description": "Dry-run report approved."}, **_ERROR_RESPONSES},
)
async def approve_import_job(
    job_id: JobId, body: PortabilityApproval, request: Request
) -> dict[str, Any]:
    try:
        return await get_portability_jobs().approve(
            job_id,
            report_hash=body.report_hash,
            actor=_request_actor(request),
        )
    except Exception as exc:
        _raise_job_error(exc)
        raise AssertionError("unreachable")


@router.post(
    "/imports/{job_id}/apply",
    response_model=PortabilityJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Apply an approved KB import (admin)",
    description="Start checkpointed apply using the exact approved dry-run report.",
    responses={202: {"description": "Apply job accepted."}, **_ERROR_RESPONSES},
)
async def apply_import_job(job_id: JobId, request: Request) -> dict[str, Any]:
    try:
        return await get_portability_jobs().start_apply(
            job_id, actor=_request_actor(request)
        )
    except Exception as exc:
        _raise_job_error(exc)
        raise AssertionError("unreachable")


@router.post(
    "/imports/{job_id}/validate",
    response_model=PortabilityJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Validate an applied KB import (admin)",
    description="Start store, folder, index, graph and normalized state validation.",
    responses={202: {"description": "Validation job accepted."}, **_ERROR_RESPONSES},
)
async def validate_import_job(job_id: JobId, request: Request) -> dict[str, Any]:
    try:
        return await get_portability_jobs().start_validate(
            job_id, actor=_request_actor(request)
        )
    except Exception as exc:
        _raise_job_error(exc)
        raise AssertionError("unreachable")


@router.post(
    "/imports/{job_id}/cancel",
    response_model=PortabilityJobResponse,
    summary="Cancel a KB import job (admin)",
    description=(
        "Cancel an upload, dry-run or approved import without deleting its "
        "private bundle/report. Cancellation is refused once apply has started."
    ),
    responses={200: {"description": "Import job cancelled."}, **_ERROR_RESPONSES},
)
async def cancel_import_job(job_id: JobId, request: Request) -> dict[str, Any]:
    try:
        return await get_portability_jobs().cancel(
            job_id, kind="import", actor=_request_actor(request)
        )
    except Exception as exc:
        _raise_job_error(exc)
        raise AssertionError("unreachable")


__all__ = ["API_UPLOAD_MAX_BYTES", "router"]
