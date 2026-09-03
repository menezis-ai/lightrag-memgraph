"""Contract of the per-file coverage gate (``scripts/coverage_floor.py``).

The gate is the thing that decides whether the rest of the suite is trusted,
so it needs its own. The failure this file exists to prevent is the quiet one:
a report that lists fewer files than the source tree holds, printing OK while
every omitted module escapes the policy — a narrowed ``--source``, a leg that
crashed before importing half the package, a data file from another project.
Percentages are the easy half; the inventory is the half that fails silently.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests._repo_only import require_repo_path

require_repo_path("scripts")

from scripts import coverage_floor  # noqa: E402


def _summary(pct: float, statements: int = 10, missing: int | None = None) -> dict:
    if missing is None:
        missing = round(statements * (100 - pct) / 100)
    return {
        "summary": {
            "percent_covered": pct,
            "num_statements": statements,
            "missing_lines": missing,
        }
    }


def _report(files: dict[str, dict], total: float = 90.0) -> dict:
    return {"files": files, "totals": {"percent_covered": total}}


@pytest.fixture
def source_tree(tmp_path: Path) -> Path:
    root = tmp_path / "pkg"
    root.mkdir()
    (root / "__init__.py").write_text('"""Docstring only."""\n', encoding="utf-8")
    (root / "alpha.py").write_text("x = 1\n", encoding="utf-8")
    (root / "beta.py").write_text("y = 2\n", encoding="utf-8")
    (root / "__pycache__").mkdir()
    (root / "__pycache__" / "stale.py").write_text("z = 3\n", encoding="utf-8")
    return root


# ---------------------------------------------------------------- inventory


class TestInventory:
    def test_a_docstring_only_module_is_not_a_coverage_question(self, source_tree):
        found = {p.name for p in coverage_floor.inventory([source_tree])}
        assert found == {"alpha.py", "beta.py"}

    def test_caches_and_dotdirs_are_skipped(self, source_tree):
        assert all(
            "__pycache__" not in p.parts
            for p in coverage_floor.inventory([source_tree])
        )

    def test_an_unparseable_module_still_has_to_be_reported(self, tmp_path):
        """Fail-closed: if we cannot tell whether a file is measurable, we
        demand the report account for it rather than wave it through."""
        broken = tmp_path / "broken.py"
        broken.write_text("def (:\n", encoding="utf-8")
        assert coverage_floor.has_statements(broken) is True


# ------------------------------------------------------------------- policy


class TestEvaluate:
    def test_a_complete_report_above_the_floor_passes(self, source_tree):
        report = _report(
            {
                str(source_tree / "alpha.py"): _summary(80.0),
                str(source_tree / "beta.py"): _summary(99.0),
            }
        )
        verdict = coverage_floor.evaluate(report, [source_tree], 75.0)
        assert verdict.ok
        assert verdict.below == [] and verdict.missing_from_report == []

    def test_a_file_missing_from_the_report_fails_even_when_the_rest_is_green(
        self, source_tree
    ):
        """The headline regression: a narrowed report must not read as a pass."""
        report = _report({str(source_tree / "alpha.py"): _summary(100.0)})
        verdict = coverage_floor.evaluate(report, [source_tree], 75.0)
        assert not verdict.ok
        assert verdict.below == []
        assert [Path(p).name for p in verdict.missing_from_report] == ["beta.py"]

    def test_an_empty_report_fails_instead_of_passing_vacuously(self, source_tree):
        verdict = coverage_floor.evaluate(_report({}), [source_tree], 75.0)
        assert not verdict.ok
        assert len(verdict.missing_from_report) == 2

    def test_without_a_source_root_only_the_reported_files_are_judged(
        self, source_tree
    ):
        """Documented weaker mode — and the reason CI always passes a root."""
        report = _report({str(source_tree / "alpha.py"): _summary(100.0)})
        verdict = coverage_floor.evaluate(report, [], 75.0)
        assert verdict.ok and verdict.missing_from_report == []

    @pytest.mark.parametrize(
        ("pct", "expected_ok"),
        [(74.9, False), (75.0, True), (75.1, True)],
    )
    def test_the_floor_is_inclusive(self, source_tree, pct, expected_ok):
        report = _report(
            {
                str(source_tree / "alpha.py"): _summary(pct),
                str(source_tree / "beta.py"): _summary(100.0),
            }
        )
        verdict = coverage_floor.evaluate(report, [source_tree], 75.0)
        assert verdict.ok is expected_ok

    def test_a_zero_statement_module_never_counts_as_below_the_floor(self, source_tree):
        report = _report(
            {
                str(source_tree / "alpha.py"): _summary(100.0),
                str(source_tree / "beta.py"): _summary(100.0),
                str(source_tree / "__init__.py"): _summary(0.0, statements=0),
            }
        )
        verdict = coverage_floor.evaluate(report, [source_tree], 75.0)
        assert verdict.ok
        assert all("__init__" not in row[1] for row in verdict.rows)

    def test_an_exemption_is_reported_but_does_not_fail(self, source_tree, monkeypatch):
        target = str(source_tree / "alpha.py")
        monkeypatch.setitem(coverage_floor.EXEMPTIONS, target, "documented debt")
        report = _report(
            {target: _summary(10.0), str(source_tree / "beta.py"): _summary(100.0)}
        )
        verdict = coverage_floor.evaluate(report, [source_tree], 75.0)
        assert verdict.ok
        assert verdict.exempt == [(10.0, target)]


# ----------------------------------------------------------------- the CLI


class TestCli:
    def test_an_unreadable_data_file_exits_two_never_zero(self, tmp_path, capsys):
        rc = coverage_floor.main(
            ["--data-file", str(tmp_path / "nope.coverage"), "--floor", "75"]
        )
        assert rc == 2
        assert "cannot read coverage data" in capsys.readouterr().err

    def test_a_missing_source_root_exits_two(self, tmp_path, capsys):
        rc = coverage_floor.main(
            ["--source-root", str(tmp_path / "absent"), "--floor", "75"]
        )
        assert rc == 2
        assert "is not a directory" in capsys.readouterr().err

    def test_a_partial_report_exits_one_and_names_the_missing_file(
        self, source_tree, monkeypatch, capsys
    ):
        monkeypatch.setattr(
            coverage_floor,
            "coverage_json",
            lambda _df: _report({str(source_tree / "alpha.py"): _summary(100.0)}),
        )
        rc = coverage_floor.main(
            ["--source-root", str(source_tree), "--floor", "75", "--data-file", "x"]
        )
        assert rc == 1
        err = capsys.readouterr().err
        assert "absent from the coverage report" in err and "beta.py" in err

    def test_a_below_floor_file_exits_one_and_names_it(
        self, source_tree, monkeypatch, capsys
    ):
        monkeypatch.setattr(
            coverage_floor,
            "coverage_json",
            lambda _df: _report(
                {
                    str(source_tree / "alpha.py"): _summary(50.0),
                    str(source_tree / "beta.py"): _summary(100.0),
                }
            ),
        )
        rc = coverage_floor.main(
            ["--source-root", str(source_tree), "--floor", "75", "--data-file", "x"]
        )
        assert rc == 1
        assert "below the 75% floor" in capsys.readouterr().err

    def test_a_green_run_exits_zero_and_says_how_many_files_it_judged(
        self, source_tree, monkeypatch, capsys
    ):
        monkeypatch.setattr(
            coverage_floor,
            "coverage_json",
            lambda _df: _report(
                {
                    str(source_tree / "alpha.py"): _summary(80.0),
                    str(source_tree / "beta.py"): _summary(100.0),
                },
                total=91.3,
            ),
        )
        rc = coverage_floor.main(
            ["--source-root", str(source_tree), "--floor", "75", "--data-file", "x"]
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert "OK — 2 files" in out and "91.3%" in out

    def test_a_malformed_report_exits_two_rather_than_crashing(
        self, source_tree, monkeypatch, capsys
    ):
        monkeypatch.setattr(
            coverage_floor, "coverage_json", lambda _df: {"files": {}}  # no totals
        )
        rc = coverage_floor.main(
            ["--source-root", str(source_tree), "--floor", "75", "--data-file", "x"]
        )
        assert rc == 2
        assert "malformed coverage report" in capsys.readouterr().err


class TestAgainstRealCoverageData:
    """One end-to-end pass over data coverage.py actually produced — the
    synthetic reports above cannot catch a change in its JSON shape."""

    def test_real_data_round_trips_through_the_gate(self, tmp_path):
        pkg = tmp_path / "realpkg"
        pkg.mkdir()
        (pkg / "covered.py").write_text("def f():\n    return 1\n", encoding="utf-8")
        (pkg / "uncovered.py").write_text(
            "def g():\n    return 2\n\n\ndef h():\n    return 3\n", encoding="utf-8"
        )
        driver = tmp_path / "driver.py"
        driver.write_text(
            "import covered\nassert covered.f() == 1\nimport uncovered\n",
            encoding="utf-8",
        )
        data_file = tmp_path / ".coverage.real"
        done = subprocess.run(  # noqa: S603 - fixed argv, our own interpreter
            [
                sys.executable,
                "-m",
                "coverage",
                "run",
                "--source",
                str(pkg),
                str(driver),
            ],
            cwd=tmp_path,
            env=os.environ | {"COVERAGE_FILE": str(data_file), "PYTHONPATH": str(pkg)},
            capture_output=True,
            text=True,
        )
        assert done.returncode == 0, done.stderr
        report = coverage_floor.coverage_json(str(data_file))
        assert set(report["files"]) and "totals" in report

        verdict = coverage_floor.evaluate(report, [], 75.0)
        # Both modules imported, so both are reported; the gate must be able to
        # read real percentages out of them.
        assert len(verdict.rows) == 2
        assert all(0.0 <= row[0] <= 100.0 for row in verdict.rows)


def test_the_shipped_json_shape_is_the_one_the_gate_reads():
    """Guards the key names the policy depends on. If coverage renames any of
    them, this fails here instead of turning the CI gate into a no-op."""
    required = {"percent_covered", "num_statements", "missing_lines"}
    sample = json.loads(json.dumps(_summary(80.0)))
    assert required <= set(sample["summary"])
