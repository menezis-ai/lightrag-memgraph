"""The ``python -m twindb_lightrag_memgraph.server`` entrypoint wires the
configured host/port into uvicorn. We run the module under ``runpy`` with
uvicorn stubbed so no socket is opened."""

from __future__ import annotations

import runpy
from unittest.mock import patch


def test_module_entrypoint_invokes_uvicorn_with_settings(monkeypatch):
    monkeypatch.setenv("LIGHTRAG_HOST", "0.0.0.0")
    monkeypatch.setenv("LIGHTRAG_PORT", "8123")

    with patch("uvicorn.run") as run:
        runpy.run_module(
            "twindb_lightrag_memgraph.server", run_name="__main__"
        )

    assert run.call_count == 1
    args, kwargs = run.call_args
    assert args[0] == "twindb_lightrag_memgraph.server.app:create_app"
    assert kwargs["factory"] is True
    assert kwargs["port"] == 8123
    assert kwargs["host"] == "0.0.0.0"
