"""Unit tests for the workspace resolution chain.

The fallback chain MEMGRAPH_WORKSPACE → WORKSPACE → DEFAULT_WORKSPACE lets
deploys ship a single env var instead of dual-writing both.
"""

from __future__ import annotations

import pytest

from lightrag_memgraph._constants import (
    DEFAULT_WORKSPACE,
    MEMGRAPH_WORKSPACE_ENV,
    WORKSPACE_ENV,
    resolve_workspace,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Each test starts with all workspace env vars cleared."""
    for key in (MEMGRAPH_WORKSPACE_ENV, WORKSPACE_ENV):
        monkeypatch.delenv(key, raising=False)
    yield


class TestResolveWorkspace:
    def test_explicit_workspace_has_priority_over_global_config_and_env(
        self, monkeypatch
    ):
        monkeypatch.setenv(MEMGRAPH_WORKSPACE_ENV, "environment")
        assert resolve_workspace("alpha", {"workspace": "configured"}) == "alpha"

    def test_global_config_used_before_environment(self, monkeypatch):
        monkeypatch.setenv(MEMGRAPH_WORKSPACE_ENV, "environment")
        assert (
            resolve_workspace(global_config={"workspace": "configured"}) == "configured"
        )

    @pytest.mark.parametrize("value", ["unsafe`name", 123, True])
    def test_invalid_explicit_workspace_is_not_silently_replaced(self, value):
        with pytest.raises(ValueError):
            resolve_workspace(value)

    def test_memgraph_workspace_wins_when_set(self, monkeypatch):
        monkeypatch.setenv(MEMGRAPH_WORKSPACE_ENV, "memgraph_one")
        monkeypatch.setenv(WORKSPACE_ENV, "core_two")
        assert resolve_workspace() == "memgraph_one"

    def test_falls_back_to_workspace_env(self, monkeypatch):
        monkeypatch.setenv(WORKSPACE_ENV, "core_two")
        assert resolve_workspace() == "core_two"

    def test_falls_back_to_default_workspace_when_nothing_set(self):
        assert resolve_workspace() == DEFAULT_WORKSPACE

    def test_whitespace_only_value_is_treated_as_unset(self, monkeypatch):
        monkeypatch.setenv(MEMGRAPH_WORKSPACE_ENV, "   ")
        monkeypatch.setenv(WORKSPACE_ENV, "real_value")
        assert resolve_workspace() == "real_value"

    def test_raises_on_unsafe_identifier(self, monkeypatch):
        monkeypatch.setenv(MEMGRAPH_WORKSPACE_ENV, "invalid folder!")
        with pytest.raises(ValueError):
            resolve_workspace()
