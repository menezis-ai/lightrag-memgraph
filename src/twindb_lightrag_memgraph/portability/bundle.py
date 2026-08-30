"""Bundle container and integrity — KB-PORTABILITY-PLAN §3.1 / §8.3 / T0.3.

A bundle is a directory (``manifest.json`` + ``memgraph/`` + ``overlay/`` +
``files/``) or a ``.tar.gz`` of that directory whose **first member is
``manifest.json``** — so a reader can refuse an archive before extracting a
single data member. Extraction is hostile-by-default: ``tarfile``'s ``data``
filter (Python ≥ 3.12), regular files only, every member must be listed in
``manifest.files[]``, a decompressed-size ceiling checked member by member
(``TWIN_PORTABILITY_MAX_BYTES``), everything into a private temporary
directory that is removed on the first problem — never a partial bundle left
on disk. Directory bundles get the same listing/hash/stray checks in place.
"""

from __future__ import annotations

import asyncio
import shutil
import stat
import tarfile
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .._constants import resolve_portability_max_bytes
from .jsonl import IntegrityError, JsonlWriter, iter_jsonl, sha256_of_file
from .manifest import FileEntry, Manifest, ManifestError, validate_bundle_path

MANIFEST_NAME = "manifest.json"
PLANES = ("memgraph", "overlay", "files")
_DIR_MODE = 0o700
_FILE_MODE = 0o600


class BundleError(RuntimeError):
    """The bundle cannot be written or read; nothing partial is left behind."""


# ---------------------------------------------------------------- writer


class BundleWriter:
    """Build a bundle directory, manifest last, then optionally archive it."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        if self.root.exists() and any(self.root.iterdir()):
            raise BundleError(f"bundle directory is not empty: {self.root}")
        self.root.mkdir(parents=True, exist_ok=True, mode=_DIR_MODE)
        self.root.chmod(_DIR_MODE)
        self._entries: list[FileEntry] = []
        self._paths: set[str] = set()
        self._finalized = False

    def _reserve(self, bundle_path: str) -> Path:
        validate_bundle_path(bundle_path)
        if bundle_path in self._paths:
            raise BundleError(f"duplicate bundle path {bundle_path!r}")
        self._paths.add(bundle_path)
        target = self.root / bundle_path
        target.parent.mkdir(parents=True, exist_ok=True, mode=_DIR_MODE)
        return target

    def open_jsonl(self, bundle_path: str, *, store: str) -> JsonlWriter:
        """A canonical JSONL writer; ``close()`` registers its FileEntry."""
        target = self._reserve(bundle_path)
        return _RegisteringJsonlWriter(
            self, target, store=store, bundle_path=bundle_path
        )

    def add_file(self, bundle_path: str, data: bytes, *, store: str) -> FileEntry:
        """A binary member of the ``files/`` plane (records = 0)."""
        if not bundle_path.startswith("files/"):
            raise BundleError("binary members belong to the files/ plane")
        target = self._reserve(bundle_path)
        target.write_bytes(data)
        target.chmod(_FILE_MODE)
        digest, size = sha256_of_file(target)
        entry = FileEntry(
            path=bundle_path, store=store, records=0, sha256=digest, bytes=size
        )
        self._entries.append(entry)
        return entry

    def _register(self, entry: FileEntry) -> None:
        (self.root / entry.path).chmod(_FILE_MODE)
        self._entries.append(entry)

    @property
    def entries(self) -> list[FileEntry]:
        return sorted(self._entries, key=lambda e: e.path)

    def finalize(self, manifest: Manifest) -> Manifest:
        """Write ``manifest.json`` (sealed) — the last thing written."""
        if self._finalized:
            raise BundleError("bundle already finalized")
        listed = {e.path for e in manifest.files}
        if listed != self._paths:
            raise BundleError(
                "manifest.files[] must list exactly the written members "
                f"(missing {sorted(self._paths - listed)}, extra {sorted(listed - self._paths)})"
            )
        sealed = manifest.sealed()
        target = self.root / MANIFEST_NAME
        target.write_text(sealed.to_json(), encoding="utf-8")
        target.chmod(_FILE_MODE)
        self._finalized = True
        return sealed

    def archive(self, out_path: Path) -> Path:
        """``.tar.gz`` with ``manifest.json`` as the first member, the rest sorted."""
        if not self._finalized:
            raise BundleError("finalize() the bundle before archiving it")
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_name(out_path.name + ".part")
        try:
            with tarfile.open(tmp, "w:gz") as tar:
                tar.add(
                    self.root / MANIFEST_NAME, arcname=MANIFEST_NAME, recursive=False
                )
                for entry in self.entries:
                    tar.add(self.root / entry.path, arcname=entry.path, recursive=False)
            tmp.chmod(_FILE_MODE)
            tmp.replace(out_path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        return out_path


class _RegisteringJsonlWriter(JsonlWriter):
    def __init__(
        self, owner: BundleWriter, path: Path, *, store: str, bundle_path: str
    ) -> None:
        super().__init__(path, store=store, bundle_path=bundle_path)
        self._owner = owner
        self._registered = False

    def close(self) -> FileEntry:
        entry = super().close()
        if not self._registered:
            self._registered = True
            self._owner._register(entry)
        return entry


# ---------------------------------------------------------------- reader


@dataclass
class Inspection:
    manifest: Manifest
    ok: bool
    problems: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "problems": list(self.problems),
            "bundle_id": self.manifest.bundle_id,
            "state_hash": self.manifest.state_hash,
            "format_version": self.manifest.format_version,
            "workspace": self.manifest.source["workspace"],
            "consistency": self.manifest.consistency.status,
            "classification": {
                "max_detected": self.manifest.classification.max_detected,
                "unknown_present": self.manifest.classification.unknown_present,
            },
            "counts": dict(self.manifest.counts),
            "files": len(self.manifest.files),
        }


class BundleReader:
    """Open a bundle directory or archive; verify it; expose its members.

    Use as a context manager: an archive is extracted into a private temporary
    directory that is removed on exit (and immediately on any refusal).
    """

    def __init__(self, source: Path, *, max_bytes: int | None = None) -> None:
        self.source = Path(source)
        self.max_bytes = (
            max_bytes if max_bytes is not None else resolve_portability_max_bytes()
        )
        self.root: Path | None = None
        self._tmp: Path | None = None
        self.manifest: Manifest | None = None

    # -- lifecycle -----------------------------------------------------
    def __enter__(self) -> BundleReader:
        try:
            if self.source.is_dir():
                if self.source.is_symlink():
                    raise BundleError("bundle directory itself must not be a symlink")
                self.root = self.source
                self.manifest = _read_manifest(
                    _confined_regular_file(
                        self.root, MANIFEST_NAME, validate_member=False
                    )
                )
            elif self.source.is_file():
                self._extract()
            else:
                raise BundleError(f"no such bundle: {self.source}")
        except BaseException:
            self.close()
            raise
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self._tmp is not None:
            shutil.rmtree(self._tmp, ignore_errors=True)
            self._tmp = None

    # -- archive extraction -------------------------------------------
    def _extract(self) -> None:
        parent = self.source.resolve().parent
        self._tmp = Path(tempfile.mkdtemp(prefix=".kb-bundle-", dir=parent))
        self._tmp.chmod(_DIR_MODE)
        try:
            tar = tarfile.open(self.source, "r:gz")
        except (tarfile.TarError, OSError) as exc:
            raise BundleError(f"not a readable tar.gz archive: {exc}") from exc
        with tar:
            first = tar.next()
            if first is None or first.name != MANIFEST_NAME or not first.isreg():
                raise BundleError("the first archive member must be manifest.json")
            if first.size > self.max_bytes:
                raise BundleError("manifest.json exceeds TWIN_PORTABILITY_MAX_BYTES")
            self._extract_member(tar, first)
            self.manifest = _read_manifest(self._tmp / MANIFEST_NAME)
            listed = {entry.path: entry for entry in self.manifest.files}
            total = first.size
            for member in iter(tar.next, None):
                if not member.isreg():
                    raise BundleError(
                        f"archive member {member.name!r} is not a regular file"
                    )
                if member.name not in listed:
                    raise BundleError(
                        f"archive member {member.name!r} is not listed in the manifest"
                    )
                expected = listed[member.name]
                if member.size != expected.bytes:
                    raise BundleError(
                        f"archive member {member.name!r}: size differs from the manifest"
                    )
                total += member.size
                if total > self.max_bytes:
                    raise BundleError(
                        "decompressed bundle exceeds TWIN_PORTABILITY_MAX_BYTES"
                    )
                self._extract_member(tar, member)
        self.root = self._tmp

    def _extract_member(self, tar: tarfile.TarFile, member: tarfile.TarInfo) -> None:
        assert self._tmp is not None
        try:
            tar.extract(member, path=self._tmp, filter="data")
        except (tarfile.FilterError, tarfile.TarError, OSError) as exc:
            raise BundleError(
                f"refusing archive member {member.name!r}: {exc}"
            ) from exc
        target = self._tmp / member.name
        if target.is_symlink() or not stat.S_ISREG(target.lstat().st_mode):
            raise BundleError(
                f"refusing archive member {member.name!r}: not a regular file"
            )
        target.chmod(_FILE_MODE)

    # -- verification ---------------------------------------------------
    def inspect(self, *, parse_records: bool = True) -> Inspection:
        """Integrity of every listed member, no stray members, state_hash."""
        assert self.root is not None and self.manifest is not None
        problems: list[str] = []
        listed = {entry.path for entry in self.manifest.files}
        present: set[str] = set()
        for candidate in self.root.rglob("*"):
            relative = candidate.relative_to(self.root).as_posix()
            if candidate.is_symlink():
                problems.append(f"symlink path component: {relative}")
                continue
            try:
                is_regular = stat.S_ISREG(candidate.lstat().st_mode)
            except OSError:
                is_regular = False
            if is_regular and candidate.name != MANIFEST_NAME:
                present.add(relative)
        for stray in sorted(present - listed):
            problems.append(f"file not listed in manifest: {stray}")
        for entry in self.manifest.files:
            try:
                path = _confined_regular_file(self.root, entry.path)
            except BundleError as exc:
                problems.append(str(exc))
                continue
            digest, size = sha256_of_file(path)
            if digest != entry.sha256 or size != entry.bytes:
                problems.append(f"hash/size mismatch: {entry.path}")
                continue
            if parse_records and entry.path.endswith(".jsonl"):
                try:
                    count = sum(1 for _ in iter_jsonl(path, entry.sha256))
                except IntegrityError as exc:
                    problems.append(str(exc))
                    continue
                if count != entry.records:
                    problems.append(
                        f"{entry.path}: {count} records on disk, manifest says {entry.records}"
                    )
        return Inspection(manifest=self.manifest, ok=not problems, problems=problems)

    def path_of(self, bundle_path: str) -> Path:
        assert self.root is not None
        return _confined_regular_file(self.root, bundle_path)


def _confined_regular_file(
    root: Path, bundle_path: str, *, validate_member: bool = True
) -> Path:
    """Return one regular member with no symlink from *root* to the file."""
    if validate_member:
        validate_bundle_path(bundle_path)
    elif bundle_path != MANIFEST_NAME:
        raise BundleError(f"invalid bundle control path {bundle_path!r}")

    try:
        root_mode = root.lstat().st_mode
    except OSError as exc:
        raise BundleError(f"cannot inspect bundle root: {exc}") from exc
    if stat.S_ISLNK(root_mode) or not stat.S_ISDIR(root_mode):
        raise BundleError("bundle root must be a real directory")
    try:
        resolved_root = root.resolve(strict=True)
    except OSError as exc:
        raise BundleError(f"cannot resolve bundle root: {exc}") from exc

    current = root
    segments = bundle_path.split("/")
    for index, segment in enumerate(segments):
        current = current / segment
        try:
            mode = current.lstat().st_mode
        except OSError as exc:
            raise BundleError(f"missing file: {bundle_path} ({exc})") from exc
        if stat.S_ISLNK(mode):
            relative = "/".join(segments[: index + 1])
            raise BundleError(f"symlink path component: {relative}")
        if index < len(segments) - 1 and not stat.S_ISDIR(mode):
            raise BundleError(f"non-directory path component: {bundle_path}")
    if not stat.S_ISREG(current.lstat().st_mode):
        raise BundleError(f"missing file: {bundle_path}")
    try:
        current.resolve(strict=True).relative_to(resolved_root)
    except (OSError, ValueError) as exc:
        raise BundleError(f"bundle member escapes root: {bundle_path}") from exc
    return current


def _read_manifest(path: Path) -> Manifest:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise BundleError(f"cannot read {MANIFEST_NAME}: {exc}") from exc
    try:
        return Manifest.from_json(text)
    except ManifestError as exc:
        raise BundleError(f"invalid {MANIFEST_NAME}: {exc}") from exc


async def run_reader_io(operation: Any, /, *args: Any, **kwargs: Any) -> Any:
    """Finish one reader thread before propagating task cancellation.

    ``asyncio.to_thread()`` cannot stop its worker once extraction or cleanup
    has begun. Shielding the worker and delaying ``CancelledError`` until it
    finishes prevents an archive extraction from continuing after its owning
    coroutine has skipped cleanup.
    """
    worker = asyncio.create_task(asyncio.to_thread(operation, *args, **kwargs))
    cancelled = False
    while not worker.done():
        try:
            await asyncio.shield(worker)
        except asyncio.CancelledError:
            cancelled = True
        except BaseException:
            # The worker exception is re-raised below via ``result()``.
            break
    if cancelled:
        if not worker.cancelled():
            # Retrieve a possible worker exception before preserving the
            # caller's cancellation as the authoritative outcome.
            worker.exception()
        raise asyncio.CancelledError
    return worker.result()


async def open_bundle_reader(reader: BundleReader) -> BundleReader:
    """Open *reader* off-loop without abandoning extraction on cancellation."""
    await run_reader_io(reader.__enter__)
    return reader


async def close_bundle_reader(reader: BundleReader) -> None:
    """Close *reader* off-loop before propagating cancellation."""
    await run_reader_io(reader.close)


def inspect_bundle(source: Path, *, max_bytes: int | None = None) -> Inspection:
    """Convenience: open, verify, close."""
    with BundleReader(source, max_bytes=max_bytes) as reader:
        return reader.inspect()


def archive_bundle(source: Path, out_path: Path) -> Path:
    """Archive an already-finalized directory bundle after full inspection.

    ``manifest.json`` is always the first member; the remaining members follow
    the manifest's deterministic path order.  The archive lands atomically via
    a sibling ``.part`` file, matching :meth:`BundleWriter.archive`.
    """
    source = Path(source)
    out_path = Path(out_path)
    with BundleReader(source) as reader:
        inspection = reader.inspect()
        if not inspection.ok:
            raise BundleError(
                "cannot archive an invalid bundle: " + "; ".join(inspection.problems)
            )
        assert reader.root is not None and reader.manifest is not None
        root = reader.root
        entries = sorted(reader.manifest.files, key=lambda entry: entry.path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_name(out_path.name + ".part")
        try:
            with tarfile.open(tmp, "w:gz") as tar:
                tar.add(root / MANIFEST_NAME, arcname=MANIFEST_NAME, recursive=False)
                for entry in entries:
                    tar.add(
                        reader.path_of(entry.path),
                        arcname=entry.path,
                        recursive=False,
                    )
            tmp.chmod(_FILE_MODE)
            tmp.replace(out_path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    return out_path
