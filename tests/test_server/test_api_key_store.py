"""Integration tests for the ApiKey Memgraph store.

Auto-skips when MEMGRAPH_URI is unset (see tests/conftest.py).
"""

from __future__ import annotations

import secrets

import pytest

from twindb_lightrag_memgraph.server import api_key_store


@pytest.fixture
def _ws():
    """Unique workspace per test run."""
    return f"apikey_{secrets.token_hex(4)}"


@pytest.mark.integration
class TestApiKeyStore:
    async def test_create_returns_full_value_only_once(self, _ws):
        try:
            await api_key_store.initialize(_ws)
            entry = await api_key_store.create_key(
                _ws, name="my agent", created_by="alice@local"
            )
            assert entry["name"] == "my agent"
            assert entry["created_by"] == "alice@local"
            assert entry["full_value"].startswith(api_key_store.KEY_PREFIX)
            assert entry["prefix"].startswith(api_key_store.KEY_PREFIX)
            assert entry["prefix"].endswith("…")
            assert "hash" not in entry  # never exposed
            assert entry["revoked_at"] is None
            assert entry["last_used_at"] is None

            # Subsequent reads must NOT contain the full value.
            listed = await api_key_store.list_keys(_ws)
            assert len(listed) == 1
            row = listed[0]
            assert "full_value" not in row
            assert "hash" not in row
            assert row["id"] == entry["id"]
            assert row["prefix"] == entry["prefix"]
        finally:
            await api_key_store.reset_workspace(_ws)

    async def test_validate_bearer_matches_then_misses_after_revoke(self, _ws):
        try:
            await api_key_store.initialize(_ws)
            entry = await api_key_store.create_key(
                _ws, name="agent-1", created_by="bob"
            )
            full = entry["full_value"]

            # Positive match
            hit = await api_key_store.validate_bearer(_ws, full)
            assert hit is not None
            assert hit["id"] == entry["id"]
            assert "hash" not in hit

            # Wrong token → no match
            miss = await api_key_store.validate_bearer(
                _ws, f"{api_key_store.KEY_PREFIX}not_a_real_one"
            )
            assert miss is None

            # Empty / non-prefix tokens skip the lookup entirely
            assert await api_key_store.validate_bearer(_ws, "") is None
            assert await api_key_store.validate_bearer(_ws, "raw_value") is None

            # Revoke → match fails afterwards
            revoked = await api_key_store.revoke_key(_ws, entry["id"])
            assert revoked is not None
            assert revoked["revoked_at"] is not None

            assert await api_key_store.validate_bearer(_ws, full) is None
        finally:
            await api_key_store.reset_workspace(_ws)

    async def test_revoke_unknown_returns_none(self, _ws):
        try:
            await api_key_store.initialize(_ws)
            assert (await api_key_store.revoke_key(_ws, "no-such-id")) is None
        finally:
            await api_key_store.reset_workspace(_ws)

    async def test_double_revoke_is_idempotent(self, _ws):
        try:
            await api_key_store.initialize(_ws)
            entry = await api_key_store.create_key(
                _ws, name="agent-2", created_by="carol"
            )
            first = await api_key_store.revoke_key(_ws, entry["id"])
            assert first is not None
            first_stamp = first["revoked_at"]
            second = await api_key_store.revoke_key(_ws, entry["id"])
            assert second is not None
            assert second["revoked_at"] == first_stamp  # not bumped
        finally:
            await api_key_store.reset_workspace(_ws)

    async def test_mark_used_updates_last_used_at(self, _ws):
        try:
            await api_key_store.initialize(_ws)
            entry = await api_key_store.create_key(
                _ws, name="agent-3", created_by="dan"
            )
            await api_key_store.mark_used(_ws, entry["id"])
            reloaded = await api_key_store.get_key(_ws, entry["id"])
            assert reloaded is not None
            assert isinstance(reloaded["last_used_at"], int)
            assert reloaded["last_used_at"] > 0
        finally:
            await api_key_store.reset_workspace(_ws)

    async def test_two_keys_are_independent(self, _ws):
        try:
            await api_key_store.initialize(_ws)
            a = await api_key_store.create_key(_ws, name="a", created_by="x")
            b = await api_key_store.create_key(_ws, name="b", created_by="y")
            assert a["full_value"] != b["full_value"]
            assert a["id"] != b["id"]
            assert a["prefix"] != b["prefix"]

            await api_key_store.revoke_key(_ws, a["id"])
            # 'a' is revoked but 'b' still matches
            assert (await api_key_store.validate_bearer(_ws, a["full_value"])) is None
            assert (
                await api_key_store.validate_bearer(_ws, b["full_value"])
            ) is not None
        finally:
            await api_key_store.reset_workspace(_ws)

    async def test_long_names_truncated(self, _ws):
        try:
            await api_key_store.initialize(_ws)
            long_name = "x" * 500
            entry = await api_key_store.create_key(_ws, name=long_name, created_by="z")
            assert len(entry["name"]) == 120
        finally:
            await api_key_store.reset_workspace(_ws)

    async def test_blank_name_falls_back(self, _ws):
        try:
            await api_key_store.initialize(_ws)
            entry = await api_key_store.create_key(_ws, name="   ", created_by="z")
            assert entry["name"] == "Unnamed key"
        finally:
            await api_key_store.reset_workspace(_ws)
