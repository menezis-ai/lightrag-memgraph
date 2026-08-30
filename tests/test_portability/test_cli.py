"""T1.5 command-line exit codes and JSON reports."""

from __future__ import annotations

import json

import pytest

from twindb_lightrag_memgraph.portability import __main__ as cli

from .test_bundle import _build


def test_inspect_valid_and_tampered_bundle(tmp_path, capsys):
    _build(tmp_path / "bundle")
    assert cli.main(["inspect", str(tmp_path / "bundle")]) == cli.EXIT_OK
    valid = json.loads(capsys.readouterr().out)
    assert valid["ok"] is True and valid["workspace"] == "base"

    target = tmp_path / "bundle" / "memgraph" / "docstatus.jsonl"
    target.write_bytes(target.read_bytes().replace(b"doc-2", b"doc-9"))
    assert cli.main(["inspect", str(tmp_path / "bundle")]) == cli.EXIT_INTEGRITY
    invalid = json.loads(capsys.readouterr().out)
    assert invalid["ok"] is False
    assert any("hash/size mismatch" in problem for problem in invalid["problems"])


def test_inspect_missing_path_is_integrity_error(tmp_path, capsys):
    assert cli.main(["inspect", str(tmp_path / "absent")]) == cli.EXIT_INTEGRITY
    error = json.loads(capsys.readouterr().err)
    assert error["ok"] is False and "no such bundle" in error["error"]


def test_export_requires_explicit_memgraph_uri(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("MEMGRAPH_URI", raising=False)
    assert (
        cli.main(["export", "--workspace", "base", "--out", str(tmp_path)])
        == cli.EXIT_REFUSED
    )
    error = json.loads(capsys.readouterr().err)
    assert error["ok"] is False and "MEMGRAPH_URI is required" in error["error"]


@pytest.mark.parametrize(
    "arguments",
    [
        ["dry-run", "bundle", "--report", "report.json"],
        ["apply", "bundle", "--report", "report.json"],
        ["validate", "--bundle", "bundle"],
    ],
)
def test_import_commands_require_explicit_memgraph_uri(monkeypatch, arguments, capsys):
    monkeypatch.delenv("MEMGRAPH_URI", raising=False)
    assert cli.main(arguments) == cli.EXIT_REFUSED
    error = json.loads(capsys.readouterr().err)
    assert error["ok"] is False and "MEMGRAPH_URI is required" in error["error"]


def test_validate_command_reports_semantic_mismatch(monkeypatch, tmp_path, capsys):
    async def fake_validate(*_args, **_kwargs):
        return {"ok": False, "problems": ["state hash mismatch"]}

    async def fake_close():
        return None

    monkeypatch.setenv("MEMGRAPH_URI", "bolt://target:7687")
    monkeypatch.setattr(cli, "validate_import", fake_validate)
    monkeypatch.setattr(cli._pool, "close_driver", fake_close)
    output = tmp_path / "validation.json"

    assert (
        cli.main(
            [
                "validate",
                "--bundle",
                "bundle",
                "--workspace",
                "target",
                "--map-folder",
                "staging=production",
                "--out",
                str(output),
            ]
        )
        == cli.EXIT_REFUSED
    )
    assert json.loads(capsys.readouterr().out)["ok"] is False
    assert json.loads(output.read_text(encoding="utf-8"))["problems"] == [
        "state hash mismatch"
    ]
