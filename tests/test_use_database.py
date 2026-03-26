"""
Tests for USE DATABASE behavior across pool and graph wrapper.

Enterprise: USE DATABASE is sent and succeeds.
Community:  USE DATABASE fails with "enterprise feature" error → detected
            once, skipped on all subsequent sessions.

OFFLINE — no Memgraph needed. All driver calls are mocked.
"""

from unittest.mock import AsyncMock, patch

import pytest
from neo4j.exceptions import ClientError as Neo4jClientError

import twindb_lightrag_memgraph._pool as _pool_module


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

    async def test_community_detected_on_custom_database(self):
        """First session with a custom database detects Community."""
        mock_session = _make_mock_session(enterprise=False)
        mock_driver = _make_mock_driver(mock_session)

        with patch(
            "twindb_lightrag_memgraph._pool.get_driver",
            return_value=(mock_driver, "custom_db"),
        ):
            from twindb_lightrag_memgraph._pool import get_session

            async with get_session():
                pass

        assert _pool_module._enterprise_supported is False

    async def test_community_skips_use_database_after_detection(self):
        """After detecting Community, USE DATABASE is not attempted again."""
        mock_session = _make_mock_session(enterprise=False)
        mock_driver = _make_mock_driver(mock_session)

        with patch(
            "twindb_lightrag_memgraph._pool.get_driver",
            return_value=(mock_driver, "custom_db"),
        ):
            from twindb_lightrag_memgraph._pool import get_session

            # First session — probes and fails
            async with get_session():
                pass

            mock_session.run.reset_mock()

            # Second session — should skip USE DATABASE entirely
            async with get_session():
                pass

        mock_session.run.assert_not_called()

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
    """Extract _SafeDriverWrapper from the patched MemgraphStorage.initialize.

    The class is defined inside _patch_builtin_memgraph_storage() so we
    trigger the patch, then inspect the initialize closure to find it.
    """
    import twindb_lightrag_memgraph

    twindb_lightrag_memgraph._registered = False
    twindb_lightrag_memgraph.register()

    from lightrag.kg.memgraph_impl import MemgraphStorage

    init_fn = MemgraphStorage.initialize
    if init_fn.__closure__:
        for cell in init_fn.__closure__:
            val = cell.cell_contents
            if isinstance(val, type) and val.__name__ == "_SafeDriverWrapper":
                return val

    raise RuntimeError("Could not find _SafeDriverWrapper in closure")


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

    async def test_community_detected_on_custom_database(self):
        """Wrapper detects Community on first session with custom database."""
        wrapper_cls = _get_wrapper_class()
        mock_session = _make_mock_session(enterprise=False)
        mock_real_driver = _make_mock_driver(mock_session)

        wrapper = wrapper_cls(mock_real_driver, "custom_db", use_routing=False)

        async with wrapper.session(database="custom_db") as session:
            pass

        assert wrapper._enterprise_supported is False

    async def test_community_skips_after_detection(self):
        """After detecting Community, USE DATABASE is not attempted again."""
        wrapper_cls = _get_wrapper_class()
        mock_session = _make_mock_session(enterprise=False)
        mock_real_driver = _make_mock_driver(mock_session)

        wrapper = wrapper_cls(mock_real_driver, "custom_db", use_routing=False)

        # First session — probes and fails
        async with wrapper.session(database="custom_db"):
            pass

        mock_session.run.reset_mock()

        # Second session — should skip
        async with wrapper.session(database="custom_db"):
            pass

        mock_session.run.assert_not_called()
