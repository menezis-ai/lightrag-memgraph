"""Canonical JSONL files of a bundle — ADR 010, decision 2.

Design record: ``docs/adr/010-kb-portability-contract.md``.

One record per line, keys sorted recursively (JCS, ``canonical.py``), NFC,
UTF-8, ``\\n`` terminated. The writer hashes and counts as it streams, so a
:class:`~.manifest.FileEntry` is available at ``close()`` without a second
pass; the reader re-hashes what it yields and raises :class:`IntegrityError`
at end of stream when the digest differs — which is why an ``apply`` must be
preceded by an ``inspect``: the reader can only *report* corruption after the
last line, never prevent the first line from being consumed.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .canonical import jcs_dumps, jcs_loads
from .manifest import FileEntry


class IntegrityError(ValueError):
    """A JSONL file does not match its manifest entry."""


class JsonlWriter:
    """Streaming canonical JSONL writer with an incremental sha256."""

    def __init__(self, path: Path, *, store: str, bundle_path: str) -> None:
        self._path = Path(path)
        self._store = store
        self._bundle_path = bundle_path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self._path.open("wb")
        self._digest = hashlib.sha256()
        self._records = 0
        self._bytes = 0
        self._closed = False

    def write(self, record: dict[str, Any]) -> None:
        if self._closed:
            raise ValueError("writer is closed")
        if not isinstance(record, dict):
            raise TypeError("a JSONL record must be a dict")
        line = (jcs_dumps(record) + "\n").encode("utf-8")
        self._fh.write(line)
        self._digest.update(line)
        self._records += 1
        self._bytes += len(line)

    def close(self) -> FileEntry:
        if not self._closed:
            self._fh.close()
            self._closed = True
        return FileEntry(
            path=self._bundle_path,
            store=self._store,
            records=self._records,
            sha256=self._digest.hexdigest(),
            bytes=self._bytes,
        )

    def __enter__(self) -> JsonlWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def iter_jsonl(
    path: Path, expected_sha256: str | None = None
) -> Iterator[dict[str, Any]]:
    """Stream the records of *path*; verify the digest once the file is exhausted.

    Every line must be in canonical form (re-serialising it yields the same
    bytes) — a bundle edited by hand is refused rather than silently accepted.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for line_no, raw in enumerate(fh, start=1):
            digest.update(raw)
            where = f"{Path(path).name}:{line_no}"
            if not raw.endswith(b"\n"):
                raise IntegrityError(f"{where}: missing final newline")
            try:
                text = raw.decode("utf-8")
                record = jcs_loads(text)
            except ValueError as exc:
                raise IntegrityError(f"{where}: invalid JSON ({exc})") from exc
            if not isinstance(record, dict):
                raise IntegrityError(f"{where}: record is not an object")
            if jcs_dumps(record) + "\n" != text:
                raise IntegrityError(f"{where}: not in canonical form")
            yield record
    if expected_sha256 is not None and digest.hexdigest() != expected_sha256:
        raise IntegrityError(f"{Path(path).name}: sha256 mismatch")


def sha256_of_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size
