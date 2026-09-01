"""Packaging contract for the Prometheus dependency used by tracing fixtures."""

from __future__ import annotations

import tomllib
from pathlib import Path


def test_prometheus_client_is_declared_on_all_metric_import_surfaces():
    pyproject = tomllib.loads(
        (Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    )
    project = pyproject["project"]
    extras = project["optional-dependencies"]

    assert all(
        any(requirement.startswith("prometheus-client") for requirement in extras[name])
        for name in ("server", "test", "tracing")
    )
    assert not any(
        requirement.startswith("prometheus-client")
        for requirement in project["dependencies"]
    ), "storage-only installs must not acquire the server metrics dependency"
