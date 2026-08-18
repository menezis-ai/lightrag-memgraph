"""Approval-bundle store for the procedure ingestion profile (PR 1).

A procedure document is NOT enqueued at upload time: the profile parks an
approval bundle (rendered schematic PNGs + blind/informed vision outputs +
divergence reports + the full document text) and a human decides. This module
is that parking lot.

Design constraints, in order:

- **FastAPI-free** (family of ``_folders.py``): the registry enqueue seam
  writes bundles without the ``[server]`` extra; the approval routes (PR 2)
  read the same store through this interface.
- **File-backed JSON, multi-worker safe**: every mutation runs a
  read-modify-write under an *inter-process* ``fcntl`` lock (sidecar
  ``.lock`` file) plus the in-process thread lock, and lands via a
  uniquely-named tempfile + atomic ``os.replace`` — two gunicorn workers can
  never clobber each other's snapshot or share a temp path. On platforms
  without ``fcntl`` (Windows dev) the store degrades to thread-level locking
  with a one-time warning; the BNP runtime is Linux.
- **Corrupt input is quarantined, never overwritten**: a store file that no
  longer parses is renamed to ``<name>.corrupt-<suffix>`` (bytes preserved
  for forensics) and every subsequent read/mutation raises
  :class:`StoreDegradedError` *under the same lock* — no second empty store
  can be written next to the quarantined truth. Recovery is explicit
  (operator inspects/merges then deletes the ``.corrupt-*`` files).
- **Exception contract**: ``StoreDegradedError`` (degraded store),
  ``LookupError`` (bundle vanished), ``OSError``/``ValueError`` — all
  handled fail-closed by the ingestion profile (explicit error-document,
  never a silent standard enqueue).

Bundle states: ``processing`` (reserved, vision in flight), ``pending``
(awaiting review), ``failed`` (render/vision error — visible, retryable),
``approved`` (enqueued by PR 2), ``rejected`` (terminal until the PR 2
``retry`` action — a rescan never re-runs a rejected document).

Idempotence is enforced by :func:`reserve_bundle` — an atomic get-or-create
keyed on ``content_hash`` inside ONE store transaction, so two workers
racing on the same document cannot both spend vision calls: exactly one
creates the ``processing`` reservation, the other reuses it. A same-content
upload under a different path or folder is recorded as a structured
``duplicate_requests`` entry (path, folder, track id, operator
classification) on the existing bundle instead of a new run — PR 2 applies
membership for the bundle's folder AND each request's folder at approve
time, and keeps the strictest operator classification for the gate.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from ._constants import TWIN_PROCEDURE_STORE_FILE_ENV

try:  # POSIX only; the BNP runtime is Linux, macOS dev has it too.
    import fcntl
except ImportError:  # pragma: no cover - Windows dev fallback
    fcntl = None

logger = logging.getLogger("twindb_lightrag_memgraph")

_CORRUPT_SUFFIX_GLOB = ".corrupt-*"

BUNDLE_STATES = frozenset(
    {"processing", "pending", "failed", "approved", "rejected", "rerouted"}
)


class StoreDegradedError(RuntimeError):
    """The claim index was lost (quarantine marker present).

    Raised by every read/mutation while a ``.corrupt-*`` sibling exists, so
    the degraded state is enforced atomically under the store lock — a
    mutator can never write a fresh empty store next to the quarantined
    truth. The ingestion profile turns this into a fail-closed refusal.
    """


_thread_lock = threading.Lock()
_flock_missing_warned = False


def store_path() -> Path:
    """Resolve the bundle-store file path (env, then LightRAG working dir)."""
    raw = os.environ.get(TWIN_PROCEDURE_STORE_FILE_ENV, "").strip()
    if raw:
        return Path(raw)
    base = os.environ.get("WORKING_DIR", "").strip() or "."
    return Path(base) / "twin_procedure_bundles.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def _store_lock(path: Path):
    """Thread lock + inter-process ``flock`` around a store transaction."""
    global _flock_missing_warned
    with _thread_lock:
        if fcntl is None:
            if not _flock_missing_warned:
                _flock_missing_warned = True
                logger.warning(
                    "twindb procedure: fcntl unavailable on this platform — "
                    "bundle store is thread-safe but NOT multi-process safe"
                )
            yield
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = path.with_name(path.name + ".lock")
        with open(lock_path, "a") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)


def _quarantine(path: Path) -> None:
    """Move a corrupt store aside (bytes preserved), best-effort.

    The quarantine file doubles as the persistent DEGRADED marker read by
    :func:`is_degraded`: losing the claim index means the profile can no
    longer tell which files are parked, so ingestion must refuse enqueues
    until an operator explicitly recovers (inspect/merge then delete the
    ``.corrupt-*`` files).
    """
    target = path.with_name(f"{path.name}.corrupt-{uuid.uuid4().hex[:8]}")
    try:
        os.replace(path, target)
        logger.error(
            "twindb procedure: bundle store %s is not valid JSON — "
            "quarantined as %s. The claim index is LOST: the procedure "
            "profile now refuses every enqueue until the .corrupt-* files "
            "are explicitly recovered and removed",
            path,
            target.name,
        )
    except OSError as exc:
        logger.warning(
            "twindb procedure: bundle store %s is corrupt and could not be "
            "quarantined (%s) — refusing to overwrite it",
            path,
            exc,
        )
        raise


def _degraded_marker_exists(path: Path) -> bool:
    try:
        return any(path.parent.glob(path.name + _CORRUPT_SUFFIX_GLOB))
    except FileNotFoundError:
        return False
    except OSError:
        return True  # cannot even inspect the store directory — fail closed


def is_degraded() -> bool:
    """Whether the claim index was lost (quarantine marker present).

    While degraded, the ingestion profile fails CLOSED for every file (it
    cannot know which ones were claimed). Recovery is explicit: an operator
    inspects and removes the ``.corrupt-*`` files next to the store.
    """
    return _degraded_marker_exists(store_path())


def _load(path: Path) -> dict[str, dict]:
    """Read the store file under the caller's lock.

    Raises :class:`StoreDegradedError` when a quarantine marker exists (or
    was just created by this read finding corrupt content) — every
    transaction shares this check under the same lock, so degradation is
    enforced atomically for reads AND mutations.
    """
    if _degraded_marker_exists(path):
        raise StoreDegradedError(
            f"bundle store {path} is degraded (quarantine marker present)"
        )
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError as exc:
        logger.warning("twindb procedure: cannot read bundle store %s (%s)", path, exc)
        raise
    try:
        data = json.loads(raw)
    except ValueError:
        _quarantine(path)
        raise StoreDegradedError(f"bundle store {path} was corrupt — quarantined")
    bundles = data.get("bundles") if isinstance(data, dict) else None
    if not isinstance(bundles, dict):
        _quarantine(path)
        raise StoreDegradedError(f"bundle store {path} was corrupt — quarantined")
    return {k: v for k, v in bundles.items() if isinstance(v, dict)}


def _write(path: Path, bundles: dict[str, dict]) -> None:
    """Unique tempfile + atomic replace: no half-writes, no tmp collisions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp:
            json.dump({"version": 1, "bundles": bundles}, tmp, ensure_ascii=False)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def create_bundle(
    *,
    file_name: str,
    original_path: str,
    track_id: str | None,
    state: str,
    reason: str,
    source: str,
    folder: str | None,
    content_hash: str | None,
    full_text: str,
    schematics: list[dict],
    classification: dict | None,
    schematics_total: int = 0,
    operator_classification: str | None = None,
) -> str:
    """Park a new bundle; returns its id. ``source`` is ``forced|detected``."""
    if state not in BUNDLE_STATES:
        raise ValueError(f"invalid bundle state: {state!r}")
    bundle_id = uuid.uuid4().hex
    now = _now_iso()
    bundle = {
        "id": bundle_id,
        "file_name": file_name,
        "original_path": original_path,
        "track_id": track_id,
        "state": state,
        "reason": reason,
        "source": source,
        "folder": folder,
        "content_hash": content_hash,
        "full_text": full_text,
        "schematics": schematics,
        "schematics_total": schematics_total,
        "classification": classification,
        "operator_classification": operator_classification,
        "created_at": now,
        "updated_at": now,
    }
    path = store_path()
    with _store_lock(path):
        bundles = _load(path)
        bundles[bundle_id] = bundle
        _write(path, bundles)
    return bundle_id


def get_bundle(bundle_id: str) -> dict | None:
    path = store_path()
    with _store_lock(path):
        return _load(path).get(bundle_id)


def list_bundles(state: str | None = None) -> list[dict]:
    """Bundles (newest first), optionally filtered by state."""
    path = store_path()
    with _store_lock(path):
        bundles = list(_load(path).values())
    if state is not None:
        bundles = [b for b in bundles if b.get("state") == state]
    bundles.sort(key=lambda b: str(b.get("created_at") or ""), reverse=True)
    return bundles


def _bundle_paths(bundle: dict) -> set[str]:
    paths = {str(bundle.get("original_path") or "")}
    requests = bundle.get("duplicate_requests")
    if isinstance(requests, list):
        paths.update(str(r.get("path") or "") for r in requests if isinstance(r, dict))
    paths.discard("")
    return paths


def _known_request_keys(bundle: dict) -> set[tuple[str, str | None]]:
    known = {(str(bundle.get("original_path") or ""), bundle.get("folder"))}
    requests = bundle.get("duplicate_requests")
    if isinstance(requests, list):
        known.update(
            (str(r.get("path") or ""), r.get("folder"))
            for r in requests
            if isinstance(r, dict)
        )
    return known


def _stricter_classification(new: str | None, current: str | None) -> str | None:
    """The stricter of two operator classes (ladder from ``classification``)."""
    if new is None:
        return current
    if current is None:
        return new
    try:
        from .classification import is_above

        return new if is_above(new, current) else current
    except ValueError:
        return current  # current not in the ladder: keep it (fail-closed)


def _append_request(
    bundle: dict,
    *,
    path: str,
    folder: str | None,
    track_id: str | None,
    operator_classification: str | None,
    file_name: str,
) -> bool:
    """Record a duplicate ingestion request on ``bundle`` (in-place).

    A same-content upload from another folder or path is NOT a new vision
    run, but its context must not be lost either: PR 2 needs the folder to
    apply membership at approve time, the operator classification to keep
    the strictest gate, and the track id for traceability. Folder-bound
    views must project a bundle through its own folder AND each request's
    folder — never leak another folder's paths.

    A (path, folder) key that is already known — the primary request
    included — is NOT re-appended, but its operator classification is
    raised to the stricter of the two (a C2 re-upload behind a C1 must not
    silently keep C1). Returns True when the bundle changed.
    """
    key = (path, folder)
    primary = (str(bundle.get("original_path") or ""), bundle.get("folder"))
    if key == primary:
        merged = _stricter_classification(
            operator_classification, bundle.get("operator_classification")
        )
        if merged != bundle.get("operator_classification"):
            bundle["operator_classification"] = merged
            return True
        return False
    for request in bundle.get("duplicate_requests") or []:
        if not isinstance(request, dict):
            continue
        if (str(request.get("path") or ""), request.get("folder")) == key:
            merged = _stricter_classification(
                operator_classification, request.get("operator_classification")
            )
            if merged != request.get("operator_classification"):
                request["operator_classification"] = merged
                return True
            return False
    bundle.setdefault("duplicate_requests", []).append(
        {
            "path": path,
            "folder": folder,
            "track_id": track_id,
            "operator_classification": operator_classification,
            "file_name": file_name,
            "requested_at": _now_iso(),
        }
    )
    return True


def find_bundles_by_path(original_path: str) -> list[dict]:
    """Bundles (newest first, ANY state, ANY folder) that claimed a path.

    The pre-selection rescan guard: a file whose bundle already exists must
    never fall through to the standard enqueue — regardless of whether the
    new scan carries the forcing header or the folder of the original
    upload. Duplicate-request paths are matched too.
    """
    if not original_path:
        return []
    path = store_path()
    with _store_lock(path):
        bundles = list(_load(path).values())
    matches = [b for b in bundles if original_path in _bundle_paths(b)]
    matches.sort(key=lambda b: str(b.get("created_at") or ""), reverse=True)
    return matches


_paths_cache_lock = threading.Lock()
_paths_cache: tuple[tuple[str, int, int], frozenset[str]] | None = None


def claimed_paths() -> frozenset[str]:
    """Cheap membership set of every path any bundle has claimed.

    The seam guard consults this for EVERY ingested file when the tier is
    active — re-parsing the (base64-PNG-laden) store JSON per file would be
    quadratic pain on a scan. The set is cached against the store file's
    ``(path, mtime_ns, size)`` and only reloaded after a write. Raises on a
    stat/read error (the guard fails closed on that).
    """
    global _paths_cache
    path = store_path()
    with _store_lock(path):
        # The quarantine marker is authoritative even when the primary store
        # no longer exists or a pre-quarantine cache entry is available.  A
        # lost claim index must never be represented as "no paths claimed".
        if _degraded_marker_exists(path):
            raise StoreDegradedError(
                f"bundle store {path} is degraded (quarantine marker present)"
            )
        try:
            stat = path.stat()
        except FileNotFoundError:
            return frozenset()
        key = (str(path), stat.st_mtime_ns, stat.st_size)
        with _paths_cache_lock:
            if _paths_cache is not None and _paths_cache[0] == key:
                return _paths_cache[1]
        bundles = _load(path)
    paths = frozenset(p for b in bundles.values() for p in _bundle_paths(b))
    with _paths_cache_lock:
        _paths_cache = (key, paths)
    return paths


def record_request(
    bundle_id: str,
    *,
    path: str,
    folder: str | None,
    track_id: str | None,
    operator_classification: str | None,
    file_name: str,
) -> bool:
    """Persist a duplicate request on an existing bundle (see
    :func:`_append_request`).

    Returns False when the (path, folder, classification) tuple brought
    nothing new. Raises ``LookupError`` when the bundle vanished — the
    caller must fail closed, not pretend the request was recorded.
    """
    store_file = store_path()
    with _store_lock(store_file):
        bundles = _load(store_file)
        bundle = bundles.get(bundle_id)
        if bundle is None:
            raise LookupError(f"bundle {bundle_id} no longer exists")
        changed = _append_request(
            bundle,
            path=path,
            folder=folder,
            track_id=track_id,
            operator_classification=operator_classification,
            file_name=file_name,
        )
        if changed:
            bundle["updated_at"] = _now_iso()
            _write(store_file, bundles)
        return changed


def reserve_bundle(
    *,
    content_hash: str,
    file_name: str,
    original_path: str,
    track_id: str | None,
    source: str,
    folder: str | None,
    operator_classification: str | None,
    via_scan: bool = False,
) -> tuple[dict, bool]:
    """Atomic get-or-create keyed on ``content_hash``; returns (bundle, created).

    In ONE store transaction: an existing bundle with the same content hash
    (ANY state, ANY folder — ``rejected`` included, it is terminal until the
    PR 2 retry) is returned as-is, with the new request's full context
    (path, folder, track id, operator classification) recorded as a
    ``duplicate_requests`` entry. Otherwise a ``processing`` reservation is
    created — the caller owns it and must settle it to ``pending``/``failed``
    via :func:`update_bundle`. Two racing workers can therefore never both
    spend vision calls on the same content.

    ``via_scan`` marks a request coming from the global /documents/scan
    surface: the path still enters the claim index (rescan guard), but the
    request carries NO folder — a scan is not an operator ingestion request
    and must never silently grant a future membership.
    """
    if not content_hash:
        raise ValueError("reserve_bundle requires a content_hash")
    request_folder = None if via_scan else folder
    path = store_path()
    with _store_lock(path):
        bundles = _load(path)
        matches = [b for b in bundles.values() if b.get("content_hash") == content_hash]
        if matches:
            matches.sort(key=lambda b: str(b.get("created_at") or ""), reverse=True)
            existing = matches[0]
            if _append_request(
                existing,
                path=original_path,
                folder=request_folder,
                track_id=track_id,
                operator_classification=operator_classification,
                file_name=file_name,
            ):
                existing["updated_at"] = _now_iso()
                _write(path, bundles)
            return existing, False
        bundle_id = uuid.uuid4().hex
        now = _now_iso()
        bundle = {
            "id": bundle_id,
            "file_name": file_name,
            "original_path": original_path,
            "track_id": track_id,
            "state": "processing",
            "reason": "processing",
            "source": source,
            "folder": request_folder,
            "content_hash": content_hash,
            "full_text": "",
            "schematics": [],
            "schematics_total": 0,
            "classification": None,
            "operator_classification": operator_classification,
            "created_at": now,
            "updated_at": now,
        }
        bundles[bundle_id] = bundle
        _write(path, bundles)
        return bundle, True


def update_bundle(bundle_id: str, **fields) -> dict | None:
    """Patch a bundle (state transitions, retry results); None if unknown."""
    new_state = fields.get("state")
    if new_state is not None and new_state not in BUNDLE_STATES:
        raise ValueError(f"invalid bundle state: {new_state!r}")
    path = store_path()
    with _store_lock(path):
        bundles = _load(path)
        bundle = bundles.get(bundle_id)
        if bundle is None:
            return None
        bundle.update(fields)
        bundle["updated_at"] = _now_iso()
        _write(path, bundles)
        return bundle


def transition_bundle(
    bundle_id: str, from_states: tuple[str, ...], **fields
) -> dict | None:
    """Conditional update in ONE transaction: apply ``fields`` only when the
    bundle currently sits in one of ``from_states``.

    The approval workflow's optimistic lock: two admins racing on the same
    approve/reject/retry cannot both win — the loser gets ``None`` (bundle
    missing OR not in an accepted state) and must surface a conflict, never
    pretend the action happened.
    """
    new_state = fields.get("state")
    if new_state is not None and new_state not in BUNDLE_STATES:
        raise ValueError(f"invalid bundle state: {new_state!r}")
    path = store_path()
    with _store_lock(path):
        bundles = _load(path)
        bundle = bundles.get(bundle_id)
        if bundle is None or bundle.get("state") not in from_states:
            return None
        bundle.update(fields)
        bundle["updated_at"] = _now_iso()
        _write(path, bundles)
        return bundle


def quarantine_files() -> list[str]:
    """Names of the ``.corrupt-*`` quarantine files next to the store."""
    path = store_path()
    try:
        return sorted(
            p.name for p in path.parent.glob(path.name + _CORRUPT_SUFFIX_GLOB)
        )
    except FileNotFoundError:
        return []


def recover_store() -> list[str]:
    """Explicit degraded-store recovery: remove the quarantine markers.

    This is the documented, deliberate operator action (PR 2 admin route) —
    the quarantined bytes are DELETED, so the caller is expected to have
    inspected/merged them first. Returns the removed file names.
    """
    path = store_path()
    removed: list[str] = []
    with _store_lock(path):
        for marker in sorted(path.parent.glob(path.name + _CORRUPT_SUFFIX_GLOB)):
            marker.unlink()
            removed.append(marker.name)
    if removed:
        logger.warning(
            "twindb procedure: degraded-store recovery — removed quarantine "
            "file(s) %s; the profile resumes normal operation",
            ", ".join(removed),
        )
    return removed


def delete_bundle(bundle_id: str) -> bool:
    path = store_path()
    with _store_lock(path):
        bundles = _load(path)
        if bundle_id not in bundles:
            return False
        del bundles[bundle_id]
        _write(path, bundles)
        return True
