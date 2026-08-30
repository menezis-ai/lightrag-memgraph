"""
Tests for USE DATABASE behavior across pool and graph wrapper.

Enterprise: USE DATABASE is sent and succeeds.
Community:  USE DATABASE fails with "enterprise feature" error → the session
            is REFUSED (``MemgraphDatabaseUnavailableError``), on every
            attempt — never cached, never a silent fallback to the default
            database (fail-closed since 2026-08-25). ``database="memgraph"``
            never issues the command at all.

OFFLINE — no Memgraph needed. All driver calls are mocked.
"""

import twindb_lightrag_memgraph._pool as _pool_module
from neo4j.exceptions import ClientError as Neo4jClientError
from unittest.mock import AsyncMock, patch

import pytest


@pytest.fixture(autouse=True)
def _reset_enterprise_flag():
    """Reset the pool's Enterprise detection flag between tests."""
    _pool_module._enterprise_supported = None
    yield
    _pool_module._enterprise_supported = None


def _make_mock_session(*, enterprise=True):
    """Build a mock async session that records run() calls.

    Args:
        enterprise: If False, session.run("USE DATABASE ...") raises
            the Memgraph Community "enterprise feature" error.
    """
    session = AsyncMock()

    if enterprise:
        session.run = AsyncMock(return_value=AsyncMock())
    else:

        async def _run_side_effect(query, *args, **kwargs):
            if query.startswith("USE DATABASE"):
                raise Neo4jClientError(
                    "Trying to use enterprise feature without a valid license."
                )
            return AsyncMock()

        session.run = AsyncMock(side_effect=_run_side_effect)

    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    return session


def _make_mock_driver(mock_session):
    """Build a mock async driver returning the given session."""
    driver = AsyncMock()
    driver.session = lambda **kwargs: mock_session
    return driver


# ── _pool.get_session() tests ────────────────────────────────────────


class TestPoolGetSessionEnterprise:
    """Tests for _pool.get_session() on Enterprise (USE DATABASE succeeds)."""

    async def test_use_database_skipped_for_default_memgraph(self):
        """USE DATABASE is skipped when database is 'memgraph' (Community default)."""
        mock_session = _make_mock_session(enterprise=True)
        mock_driver = _make_mock_driver(mock_session)

        with patch(
            "twindb_lightrag_memgraph._pool.get_driver",
            return_value=(mock_driver, "memgraph"),
        ):
            from twindb_lightrag_memgraph._pool import get_session

            async with get_session() as session:
                pass

        mock_session.run.assert_not_called()

    async def test_use_database_sent_for_custom_name(self):
        """USE DATABASE custom_db must be sent for non-default names."""
        mock_session = _make_mock_session(enterprise=True)
        mock_driver = _make_mock_driver(mock_session)

        with patch(
            "twindb_lightrag_memgraph._pool.get_driver",
            return_value=(mock_driver, "custom_db"),
        ):
            from twindb_lightrag_memgraph._pool import get_session

            async with get_session() as session:
                pass

        mock_session.run.assert_any_call("USE DATABASE custom_db")

    async def test_use_database_skipped_when_empty(self):
        """USE DATABASE must NOT be sent when database is empty."""
        mock_session = _make_mock_session(enterprise=True)
        mock_driver = _make_mock_driver(mock_session)

        with patch(
            "twindb_lightrag_memgraph._pool.get_driver",
            return_value=(mock_driver, ""),
        ):
            from twindb_lightrag_memgraph._pool import get_session

            async with get_session() as session:
                pass

        mock_session.run.assert_not_called()


class TestPoolGetSessionCommunity:
    """Tests for _pool.get_session() on Community (USE DATABASE fails)."""

    async def test_community_skipped_for_default_memgraph(self):
        """database='memgraph' skips USE DATABASE entirely — no Community probe."""
        mock_session = _make_mock_session(enterprise=False)
        mock_driver = _make_mock_driver(mock_session)

        with patch(
            "twindb_lightrag_memgraph._pool.get_driver",
            return_value=(mock_driver, "memgraph"),
        ):
            from twindb_lightrag_memgraph._pool import get_session

            async with get_session() as session:
                pass

        # No USE DATABASE attempted — _enterprise_supported stays None
        mock_session.run.assert_not_called()
        assert _pool_module._enterprise_supported is None

    async def test_custom_database_on_community_refuses_the_session(self):
        """A refused USE DATABASE on a non-default database is fail-closed.

        Before 2026-08-25 the pool flagged "Community" and silently kept
        working on the default ``memgraph`` database — in a one-database-per-
        KB topology that merges several KBs without any error (audit P0).
        """
        mock_session = _make_mock_session(enterprise=False)
        mock_driver = _make_mock_driver(mock_session)

        with patch(
            "twindb_lightrag_memgraph._pool.get_driver",
            return_value=(mock_driver, "custom_db"),
        ):
            from twindb_lightrag_memgraph._pool import (
                MemgraphDatabaseUnavailableError,
                get_session,
            )

            with pytest.raises(MemgraphDatabaseUnavailableError) as excinfo:
                async with get_session():
                    pass  # pragma: no cover - the session must never open

        # The message names the variable, the database and the remedy.
        message = str(excinfo.value)
        assert "MEMGRAPH_DATABASE='custom_db'" in message
        assert "unset MEMGRAPH_DATABASE" in message
        assert "Enterprise licence" in message
        # It is a ClientError chained, and it is NOT cached as "Community".
        assert isinstance(excinfo.value.__cause__, Neo4jClientError)
        assert _pool_module._enterprise_supported is None

    async def test_refusal_is_not_cached_so_a_restored_licence_recovers(self):
        """Every session re-issues USE DATABASE; a licence installed at
        runtime recovers without a restart, a broken one never goes silent."""
        state = {"licensed": False}

        async def _run_side_effect(query, *args, **kwargs):
            if query.startswith("USE DATABASE") and not state["licensed"]:
                raise Neo4jClientError(
                    "Trying to use enterprise feature without a valid license."
                )
            return AsyncMock()

        mock_session = AsyncMock()
        mock_session.run = AsyncMock(side_effect=_run_side_effect)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_driver = _make_mock_driver(mock_session)

        with patch(
            "twindb_lightrag_memgraph._pool.get_driver",
            return_value=(mock_driver, "custom_db"),
        ):
            from twindb_lightrag_memgraph._pool import (
                MemgraphDatabaseUnavailableError,
                get_session,
            )

            # First session — refused.
            with pytest.raises(MemgraphDatabaseUnavailableError):
                async with get_session():
                    pass  # pragma: no cover

            # Second session, still unlicensed — refused AGAIN (not skipped).
            with pytest.raises(MemgraphDatabaseUnavailableError):
                async with get_session():
                    pass  # pragma: no cover
            use_calls = [
                c
                for c in mock_session.run.call_args_list
                if str(c.args[0]).startswith("USE DATABASE")
            ]
            assert len(use_calls) == 2

            # Licence installed at runtime — the next session just works.
            state["licensed"] = True
            async with get_session():
                pass
            assert _pool_module._enterprise_supported is True

    async def test_read_session_is_fail_closed_too(self):
        """The read pool applies the same contract as the write pool."""
        mock_session = _make_mock_session(enterprise=False)
        mock_driver = _make_mock_driver(mock_session)

        with patch(
            "twindb_lightrag_memgraph._pool._get_read_driver",
            return_value=(mock_driver, "custom_db"),
        ):
            from twindb_lightrag_memgraph._pool import (
                MemgraphDatabaseUnavailableError,
                get_read_session,
            )

            with pytest.raises(MemgraphDatabaseUnavailableError):
                async with get_read_session():
                    pass  # pragma: no cover

    async def test_non_enterprise_client_error_still_raises(self):
        """Non-enterprise ClientErrors must propagate, not be swallowed."""
        session = AsyncMock()

        async def _run_side_effect(query, *args, **kwargs):
            if query.startswith("USE DATABASE"):
                raise Neo4jClientError("some other error")
            return AsyncMock()

        session.run = AsyncMock(side_effect=_run_side_effect)
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=False)
        mock_driver = _make_mock_driver(session)

        with patch(
            "twindb_lightrag_memgraph._pool.get_driver",
            return_value=(mock_driver, "custom_db"),
        ):
            from twindb_lightrag_memgraph._pool import get_session

            with pytest.raises(Neo4jClientError, match="some other error"):
                async with get_session():
                    pass


# ── _SafeDriverWrapper tests ────────────────────────────────────────


def _get_wrapper_class():
    """Return the module-level ``_SafeDriverWrapper`` class.

    Since the cognitive-complexity refactor it lives at module scope in
    ``twindb_lightrag_memgraph.patches.registry`` (previously nested inside
    ``_patch_builtin_memgraph_storage`` and extracted from the closure).
    """
    from twindb_lightrag_memgraph.patches.registry import _SafeDriverWrapper

    return _SafeDriverWrapper


class TestSafeDriverWrapperEnterprise:
    """Tests for _SafeDriverWrapper.session() on Enterprise."""

    async def test_use_database_skipped_for_default_memgraph(self):
        """Wrapper skips USE DATABASE when database is 'memgraph' (Community default)."""
        wrapper_cls = _get_wrapper_class()
        mock_session = _make_mock_session(enterprise=True)
        mock_real_driver = _make_mock_driver(mock_session)

        wrapper = wrapper_cls(mock_real_driver, "memgraph", use_routing=False)

        async with wrapper.session(database="memgraph") as session:
            pass

        mock_session.run.assert_not_called()

    async def test_use_database_custom(self):
        """Wrapper must send USE DATABASE for custom names."""
        wrapper_cls = _get_wrapper_class()
        mock_session = _make_mock_session(enterprise=True)
        mock_real_driver = _make_mock_driver(mock_session)

        wrapper = wrapper_cls(mock_real_driver, "prod_db", use_routing=False)

        async with wrapper.session(database="prod_db") as session:
            pass

        mock_session.run.assert_called_once_with("USE DATABASE prod_db")

    async def test_use_database_skipped_when_empty(self):
        """Wrapper must NOT send USE DATABASE when database is empty."""
        wrapper_cls = _get_wrapper_class()
        mock_session = _make_mock_session(enterprise=True)
        mock_real_driver = _make_mock_driver(mock_session)

        wrapper = wrapper_cls(mock_real_driver, "", use_routing=False)

        async with wrapper.session() as session:
            pass

        mock_session.run.assert_not_called()

    async def test_strips_database_kwarg(self):
        """Wrapper must strip database= from kwargs before calling real driver."""
        wrapper_cls = _get_wrapper_class()
        mock_session = _make_mock_session(enterprise=True)

        received_kwargs = {}

        real_driver = AsyncMock()

        def capture_session(**kwargs):
            received_kwargs.update(kwargs)
            return mock_session

        real_driver.session = capture_session

        wrapper = wrapper_cls(real_driver, "memgraph", use_routing=False)

        async with wrapper.session(
            database="memgraph", default_access_mode="READ"
        ) as session:
            pass

        assert "database" not in received_kwargs
        assert received_kwargs.get("default_access_mode") == "READ"


class TestSafeDriverWrapperCommunity:
    """Tests for _SafeDriverWrapper.session() on Community."""

    async def test_community_skipped_for_default_memgraph(self):
        """Wrapper skips USE DATABASE entirely for database='memgraph'."""
        wrapper_cls = _get_wrapper_class()
        mock_session = _make_mock_session(enterprise=False)
        mock_real_driver = _make_mock_driver(mock_session)

        wrapper = wrapper_cls(mock_real_driver, "memgraph", use_routing=False)

        async with wrapper.session(database="memgraph") as session:
            pass

        mock_session.run.assert_not_called()
        assert wrapper._enterprise_supported is None

    async def test_custom_database_on_community_refuses_the_session(self):
        """The graph-pool wrapper is fail-closed like the shared pool."""
        from twindb_lightrag_memgraph._pool import MemgraphDatabaseUnavailableError

        wrapper_cls = _get_wrapper_class()
        mock_session = _make_mock_session(enterprise=False)
        mock_real_driver = _make_mock_driver(mock_session)

        wrapper = wrapper_cls(mock_real_driver, "custom_db", use_routing=False)

        with pytest.raises(MemgraphDatabaseUnavailableError, match="graph pool"):
            async with wrapper.session(database="custom_db"):
                pass  # pragma: no cover

        assert wrapper._enterprise_supported is None

    async def test_refusal_is_not_cached_on_the_wrapper(self):
        """Second session re-issues USE DATABASE and is refused again."""
        from twindb_lightrag_memgraph._pool import MemgraphDatabaseUnavailableError

        wrapper_cls = _get_wrapper_class()
        mock_session = _make_mock_session(enterprise=False)
        mock_real_driver = _make_mock_driver(mock_session)

        wrapper = wrapper_cls(mock_real_driver, "custom_db", use_routing=False)

        for _ in range(2):
            with pytest.raises(MemgraphDatabaseUnavailableError):
                async with wrapper.session(database="custom_db"):
                    pass  # pragma: no cover

        use_calls = [
            c
            for c in mock_session.run.call_args_list
            if str(c.args[0]).startswith("USE DATABASE")
        ]
        assert len(use_calls) == 2
