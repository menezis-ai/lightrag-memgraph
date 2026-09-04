"""Packaging contract for the dependency-free operational counters."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import tomllib
from pathlib import Path


def test_prometheus_client_is_not_a_package_dependency():
    project_root = Path(__file__).parents[1]
    pyproject = tomllib.loads((project_root / "pyproject.toml").read_text("utf-8"))
    project = pyproject["project"]
    requirements = list(project["dependencies"])
    for extra_requirements in project["optional-dependencies"].values():
        requirements.extend(extra_requirements)

    assert not any(
        requirement.startswith("prometheus-client") for requirement in requirements
    )
    for constraint_name in ("constraints-dev.txt", "constraints-prod.txt"):
        constraints = (project_root / "requirements" / constraint_name).read_text(
            "utf-8"
        )
        assert "prometheus-client" not in constraints.casefold()


def test_metrics_router_imports_when_prometheus_client_is_unavailable():
    project_root = Path(__file__).parents[1]
    script = textwrap.dedent("""
        import builtins

        real_import = builtins.__import__

        def import_without_prometheus(name, *args, **kwargs):
            if name == "prometheus_client" or name.startswith("prometheus_client."):
                raise ModuleNotFoundError("prometheus_client is unavailable", name=name)
            return real_import(name, *args, **kwargs)

        builtins.__import__ = import_without_prometheus

        from twindb_lightrag_memgraph.server.metrics_routes import build_metrics_router

        paths = {route.path for route in build_metrics_router().routes}
        assert paths == {"/ops/metrics"}
        """)
    env = os.environ.copy()
    source_root = str(project_root / "src")
    env["PYTHONPATH"] = os.pathsep.join(
        item for item in (source_root, env.get("PYTHONPATH")) if item
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=project_root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
