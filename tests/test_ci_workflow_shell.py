"""Every shell payload in the Forgejo workflow must actually be shell.

The failure this file exists to prevent is invisible on review. Jobs pass
their script to a container as ``sh -lc '<multi-line payload>'``, a
single-quoted string. One apostrophe anywhere inside it — in a *comment*, in
an English possessive — closes the quote early, and every line after it is
executed by the OUTER shell instead of the container. Nothing warns: the job
runs, the container does part of the work, and the remainder dies on the host
with something unrelated-looking.

That is not hypothetical. CI task 36013 (2026-09-02) failed with
``python: command not found`` because a comment reading ``the gate's own
tests`` had been added inside such a payload; the pip installs had already run
in the container, so the log looked like a PATH problem rather than a quoting
one.

These checks are static and cost nothing, and they cover the workflow the CI
gate itself lives in — the one file whose breakage silently disables a gate.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

WORKFLOW = Path(__file__).resolve().parents[1] / ".forgejo" / "workflows" / "ci.yml"

#: ``docker run ... sh -lc '`` … the payload … ``'`` — non-greedy up to the
#: first line that is only whitespace plus a closing quote, which is how every
#: job in this workflow terminates its payload.
PAYLOAD = re.compile(r"sh -lc '\n(.*?)\n\s*'\n", re.S)


def _workflow() -> dict:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def _run_blocks() -> list[tuple[str, str]]:
    blocks = []
    for name, job in _workflow()["jobs"].items():
        for step in job.get("steps", []):
            if isinstance(step, dict) and isinstance(step.get("run"), str):
                blocks.append((name, step["run"]))
    return blocks


def test_the_workflow_declares_at_least_one_containerised_payload():
    """Guards the guard: if the extraction regex stops matching, every test
    below would pass vacuously on an empty list."""
    total = sum(len(PAYLOAD.findall(run)) for _, run in _run_blocks())
    assert total >= 3, f"only {total} sh -lc payloads found — regex drifted?"


@pytest.mark.parametrize("job,run", _run_blocks(), ids=[j for j, _ in _run_blocks()])
def test_no_apostrophe_closes_a_container_payload_early(job: str, run: str):
    """An apostrophe inside ``sh -lc '...'`` silently splits the script."""
    for payload in PAYLOAD.findall(run):
        assert "'" not in payload, (
            f"job {job}: an apostrophe inside the sh -lc payload closes it early, "
            "spilling the rest into the runner's host shell. Rewrite the wording "
            "(no possessives, no contractions) — this is the CI task 36013 bug.\n"
            + "\n".join(line for line in payload.splitlines() if "'" in line)
        )


@pytest.mark.parametrize("job,run", _run_blocks(), ids=[j for j, _ in _run_blocks()])
def test_every_shell_block_parses(job: str, run: str, tmp_path: Path):
    """``sh -n`` the outer block and each container payload.

    Catches unbalanced quotes, stray ``fi``/``done`` and the like before a
    runner spends a Memgraph and a full install discovering them.
    """
    sh = shutil.which("sh")
    assert sh, "no POSIX shell available to syntax-check with"

    scripts = [("outer", run)] + [
        (f"payload[{i}]", payload) for i, payload in enumerate(PAYLOAD.findall(run))
    ]
    for label, script in scripts:
        target = tmp_path / f"{label.replace('[', '_').replace(']', '')}.sh"
        target.write_text(script, encoding="utf-8")
        result = subprocess.run(  # noqa: S603 - fixed argv
            [sh, "-n", str(target)], capture_output=True, text=True
        )
        assert (
            result.returncode == 0
        ), f"job {job} {label} is not valid shell:\n{result.stderr}"


def test_the_coverage_floor_job_gates_on_a_source_root():
    """Without ``--source-root`` the gate judges only what the report happens
    to contain, which is the no-op this whole job exists to prevent."""
    jobs = _workflow()["jobs"]
    for name in ("coverage-floor", "catalog-tests"):
        run = jobs[name]["steps"][-1]["run"]
        assert "coverage_floor.py" in run, f"{name} no longer runs the gate"
        assert "--source-root" in run, f"{name} runs the gate without an inventory"


def test_the_coverage_floor_job_blocks_ci_success():
    """A gate outside ``ci-success`` is documentation, not a gate."""
    ci_success = _workflow()["jobs"]["ci-success"]
    assert "coverage-floor" in ci_success["needs"]
    assert (
        'needs.coverage-floor.result }}" = "success"' in ci_success["steps"][0]["run"]
    )
