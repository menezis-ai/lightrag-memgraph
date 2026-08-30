"""Contract tests for the reconstructed-export hygiene scan."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest

_SCANNER_PATH = Path(__file__).with_name("export_hygiene.py")
_SCANNER_SPEC = importlib.util.spec_from_file_location("export_hygiene", _SCANNER_PATH)
assert _SCANNER_SPEC is not None and _SCANNER_SPEC.loader is not None
_SCANNER = importlib.util.module_from_spec(_SCANNER_SPEC)
sys.modules[_SCANNER_SPEC.name] = _SCANNER
_SCANNER_SPEC.loader.exec_module(_SCANNER)

ALLOWED_PUBLIC_TECHNICAL_TERMS = _SCANNER.ALLOWED_PUBLIC_TECHNICAL_TERMS
ExportHygieneError = _SCANNER.ExportHygieneError
scan_export_tree = _SCANNER.scan_export_tree


def _legacy_persona() -> str:
    return "".join(("claire", ".", "benoit"))


def _private_host() -> str:
    return ".".join(("idp", "inter" + "nal"))


def _former_path() -> str:
    return "/" + "c" + "ib" + "/runbooks/example.md"


def _linked_worktree_control_file() -> str:
    return "gitdir: " + "".join(("/Users/", "julien", "/repo/.git/worktrees/export"))


def test_scan_accepts_public_technical_terms(tmp_path: Path) -> None:
    export_tree = tmp_path / "export"
    export_tree.mkdir()
    (export_tree / "fixture.txt").write_text(" ".join(ALLOWED_PUBLIC_TECHNICAL_TERMS))

    assert scan_export_tree(export_tree) == []


def test_scan_checks_the_reconstructed_tree_not_the_source_parent(
    tmp_path: Path,
) -> None:
    source_tree = tmp_path / "source"
    export_tree = tmp_path / "export"
    source_tree.mkdir()
    export_tree.mkdir()
    (source_tree / "private-fixture.txt").write_text(_legacy_persona())
    (export_tree / "public-fixture.txt").write_text("synthetic fixture")

    assert scan_export_tree(export_tree) == []


def test_scan_excludes_only_the_root_git_worktree_control_entry(
    tmp_path: Path,
) -> None:
    export_tree = tmp_path / "export"
    nested = export_tree / "nested"
    nested.mkdir(parents=True)
    (export_tree / ".git").write_text(_linked_worktree_control_file())
    (nested / ".git").write_text(_private_host())

    findings = scan_export_tree(export_tree)

    assert {finding.path for finding in findings} == {Path("nested/.git")}
    assert {finding.marker for finding in findings} == {"private DNS suffix"}


def test_scan_rejects_legacy_persona_path_and_private_host(tmp_path: Path) -> None:
    export_tree = tmp_path / "export"
    export_tree.mkdir()
    (export_tree / "fixture.txt").write_text(
        "\n".join((_legacy_persona(), _former_path(), _private_host()))
    )

    findings = scan_export_tree(export_tree)

    assert {finding.marker for finding in findings} == {
        "former internal workspace marker",
        "legacy fixture persona",
        "former internal path convention",
        "private DNS suffix",
    }
    assert {finding.line for finding in findings} == {1, 2, 3}


def test_scan_fails_closed_when_the_tree_is_missing(tmp_path: Path) -> None:
    missing_tree = tmp_path / "missing"

    try:
        scan_export_tree(missing_tree)
    except ExportHygieneError as exc:
        assert "not a real directory" in str(exc)
    else:  # pragma: no cover - assertion explains the intended fail-closed path.
        raise AssertionError("a missing export tree must fail closed")


def test_scan_fails_closed_when_walk_cannot_enter_a_subdirectory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    export_tree = tmp_path / "export"
    locked = export_tree / "locked"
    locked.mkdir(parents=True)

    def walk_with_permission_error(*args, onerror=None, **kwargs):
        assert onerror is not None
        onerror(PermissionError(13, "Permission denied", str(locked)))
        return ()

    monkeypatch.setattr(_SCANNER.os, "walk", walk_with_permission_error)

    with pytest.raises(ExportHygieneError, match="could not traverse export entry"):
        scan_export_tree(export_tree)


def test_export_procedure_runs_the_scan_on_its_current_tree() -> None:
    procedure = Path("EXPORT_PROCEDURE.md").read_text()

    assert "python tests/export_hygiene.py ." in procedure
    assert "root `.git` control entry" in procedure
