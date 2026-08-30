#!/usr/bin/env python3
"""Fail-closed hygiene scan for a reconstructed delivery tree.

This tool deliberately scans the directory passed on the command line, rather
than the source checkout.  It is therefore safe to use after the export
allow-list has been applied and before a public push.

The patterns target identifiers that make synthetic demo material look like a
private deployment: former personas, private DNS suffixes, internal-looking
paths, and known infrastructure markers.  They do not ban public technical
vocabulary.  ``ALLOWED_PUBLIC_TECHNICAL_TERMS`` documents examples that remain
valid in rich demos.
"""

from __future__ import annotations

import argparse
import os
import re
import stat
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

# These public technical terms are intentionally *not* forbidden markers.  A
# demo may use them without suggesting a particular customer's topology.
ALLOWED_PUBLIC_TECHNICAL_TERMS = (
    "Oracle",
    "RMAN",
    "CFT",
    "Memgraph",
    "MAGE",
    "SWIFT",
    "ISO 20022",
)


class ExportHygieneError(RuntimeError):
    """The requested tree could not be inspected completely."""


@dataclass(frozen=True)
class Finding:
    """A forbidden marker found in one regular file."""

    marker: str
    path: Path
    line: int


def _expression(*parts: str) -> bytes:
    """Build an ASCII regular-expression fragment without storing a marker.

    The scanner is shipped in the tree it checks.  Splitting the sensitive
    literals keeps the scanner from matching its own pattern catalogue.
    """

    return "".join(parts).encode("ascii")


def _word(*parts: str) -> bytes:
    return rb"(?i)\b" + _expression(*parts) + rb"\b"


def _compile(marker: str, expression: bytes) -> tuple[str, re.Pattern[bytes]]:
    return marker, re.compile(expression)


FORBIDDEN_PATTERNS: tuple[tuple[str, re.Pattern[bytes]], ...] = (
    _compile("legacy fixture persona", _word("claire", r"\.", "benoit")),
    _compile("legacy fixture persona", _word("marc", r"\.", "berthier")),
    _compile("legacy fixture persona", _word("yann", r"\.", "dubois")),
    _compile("legacy fixture persona", _word("philippe", r"\.", "marchand")),
    _compile("legacy fixture persona", _word("manu", r"\.", "dev")),
    _compile("former internal workspace marker", _word("c", "ib")),
    _compile("known stakeholder", _word("al", "berto")),
    _compile("known stakeholder", _word("fab", "rice")),
    _compile("known stakeholder", _word("vi", "hn")),
    _compile("known stakeholder", _word("hor", "vat")),
    _compile("known stakeholder", _word("sa", "lah")),
    _compile("known stakeholder", _word("an", "as")),
    _compile("known stakeholder", _word("geof", "frey")),
    _compile("known stakeholder", _word("cassan", "dre")),
    _compile("known stakeholder", _word("cha", "ki")),
    _compile("known stakeholder", _word("ya", "zid")),
    _compile("known stakeholder", _word("cha", "fi")),
    _compile("known stakeholder", _word("fay", "cal")),
    _compile("known stakeholder", _word("timo", "thee")),
    _compile(
        "private DNS suffix",
        rb"(?i)\b(?:[a-z0-9-]+\.)+" + _expression("inter", "nal") + rb"\b",
    ),
    _compile(
        "private DNS suffix",
        rb"(?i)\b(?:[a-z0-9-]+\.)+" + _expression("lo", "cal") + rb"\b",
    ),
    _compile(
        "private DNS suffix",
        rb"(?i)\b(?:[a-z0-9-]+\.)+" + _expression("co", "rp") + rb"\b",
    ),
    _compile(
        "former internal path convention",
        rb"(?i)/(?:[a-z0-9_.-]+/)*" + _expression("c", "ib") + rb"(?:/|$)",
    ),
    _compile("former corporate group convention", _word("corp", r"\.", "c", "ib")),
    _compile("known infrastructure host", _expression("sig", "ilum", r"\.fr")),
    _compile("known infrastructure host", _expression("maquette", r"\.sig")),
    _compile("known infrastructure host", _word("twin", "-real")),
    _compile("known infrastructure host", _word("ovh", "-twin")),
    _compile("known infrastructure host", _expression("37", r"\.59\.104\.111")),
    _compile(
        "known infrastructure host", _expression("192", r"\.168\.1\.(?:49|61|212)")
    ),
    _compile("local workstation path", _expression("/Users/", "julien")),
    _compile("known personal address", _expression("julien", r"\.dabert")),
    _compile("known personal address", _expression("jdabert", "@")),
)


def _iter_regular_files(root: Path) -> Iterable[Path]:
    if root.is_symlink() or not root.is_dir():
        raise ExportHygieneError(f"export root is not a real directory: {root}")

    def _raise_walk_error(exc: OSError) -> None:
        failed_path = exc.filename or root
        raise ExportHygieneError(
            f"could not traverse export entry {failed_path}: {exc}"
        ) from exc

    for directory, child_directories, child_files in os.walk(
        root,
        followlinks=False,
        onerror=_raise_walk_error,
    ):
        current = Path(directory)

        # The canonical export is reconstructed inside a linked Git worktree,
        # whose root ``.git`` control file points back to private worktree
        # metadata.  It is not delivery content.  Exclude exactly that root
        # entry; a nested path named ``.git`` remains part of the scan.
        if current == root:
            child_directories[:] = [
                name for name in child_directories if name != ".git"
            ]
            child_files = [name for name in child_files if name != ".git"]

        child_directories.sort()
        child_files.sort()

        for name in child_directories:
            candidate = current / name
            if candidate.is_symlink():
                raise ExportHygieneError(
                    f"refusing to scan symlinked directory: {candidate}"
                )

        for name in child_files:
            candidate = current / name
            try:
                mode = candidate.lstat().st_mode
            except OSError as exc:
                raise ExportHygieneError(f"could not stat {candidate}: {exc}") from exc
            if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
                raise ExportHygieneError(
                    f"refusing non-regular export entry: {candidate}"
                )
            yield candidate


def scan_export_tree(root: Path | str) -> list[Finding]:
    """Return every forbidden marker found in ``root``.

    Reading bytes keeps the check independent of file encoding and ensures a
    marker in a generated asset cannot hide behind a decoding error.
    """

    root_path = Path(root)
    findings: list[Finding] = []
    for path in _iter_regular_files(root_path):
        try:
            contents = path.read_bytes()
        except OSError as exc:
            raise ExportHygieneError(f"could not read {path}: {exc}") from exc
        relative_path = path.relative_to(root_path)
        for marker, pattern in FORBIDDEN_PATTERNS:
            for match in pattern.finditer(contents):
                findings.append(
                    Finding(
                        marker=marker,
                        path=relative_path,
                        line=contents.count(b"\n", 0, match.start()) + 1,
                    )
                )
    return findings


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fail closed when a reconstructed export contains private-looking markers."
    )
    parser.add_argument(
        "export_tree", type=Path, help="already reconstructed export directory"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        findings = scan_export_tree(args.export_tree)
    except ExportHygieneError as exc:
        print(f"ERROR: export hygiene scan could not complete: {exc}", file=sys.stderr)
        return 2

    if not findings:
        return 0

    print("ERROR: forbidden private-looking marker in export tree:", file=sys.stderr)
    for finding in findings:
        print(
            f"  {finding.path}:{finding.line}: {finding.marker}",
            file=sys.stderr,
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
