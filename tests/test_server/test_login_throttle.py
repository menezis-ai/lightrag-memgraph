"""Regression: /login brute-force throttle (audit 2026-08-06, R-04).

Measured during the audit: 288 req/s against /login with no 429, no delay,
no lockout — a 100k dictionary was ~6 minutes single-threaded. The fix adds
a per-IP sliding-window limit (429 + Retry-After), a per-account
exponential backoff on consecutive failures, and a sev=critical audit
event past the alert threshold (purple-team reducer: the login_failure
telemetry existed but nothing consumed it).
"""

from __future__ import annotations

import pytest
from fastapi import Response
from httpx import ASGITransport, AsyncClient

from twindb_lightrag_memgraph.server import auth
from twindb_lightrag_memgraph.server.app import create_app
from twindb_lightrag_memgraph.server.auth import LoginRequest, configure_auth, login
from twindb_lightrag_memgraph.server.settings import LightRAGServerSettings

JWT_SECRET = "y" * 48


@pytest.fixture(autouse=True)
def _auth_config():
    configure_auth(jwt_secret=JWT_SECRET, jwt_password="s3cret")
    yield
    configure_auth(api_key=None, jwt_secret=None)


def _make_settings() -> LightRAGServerSettings:
    return LightRAGServerSettings(
        working_dir="/tmp/lightrag_login_throttle_test",
        workspace="cib",
        enable_langsmith_tracing=False,
        api_key=None,
        jwt_secret=JWT_SECRET,
        jwt_password="s3cret",
        enable_webui_routes=False,
    )


class TestPerIpRateLimit:
    async def test_sixth_attempt_in_window_is_429(self, monkeypatch):
        """The audit regression bar: 6 consecutive attempts → the 6th is 429."""
        # Backoff is covered separately — keep its real sleeps out of these
        # HTTP-level runs (they would add minutes to the suite).
        monkeypatch.setenv("TWIN_LOGIN_BACKOFF_MAX_SECONDS", "0")
        app = create_app(_make_settings())
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            codes = [
                (
                    await client.post(
                        "/login",
                        json={"username": "admin", "password": "wrong"},
                    )
                ).status_code
                for _ in range(6)
            ]
        assert codes[:5] == [401] * 5
        assert codes[5] == 429

    async def test_429_carries_retry_after(self, monkeypatch):
        monkeypatch.setenv("TWIN_LOGIN_BACKOFF_MAX_SECONDS", "0")
        app = create_app(_make_settings())
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            for _ in range(5):
                await client.post(
                    "/login", json={"username": "admin", "password": "wrong"}
                )
            resp = await client.post(
                "/login", json={"username": "admin", "password": "wrong"}
            )
        assert resp.status_code == 429
        assert int(resp.headers["Retry-After"]) >= 1

    async def test_rate_limit_disabled_with_zero(self, monkeypatch):
        # create_app re-runs configure_auth from settings; the env var is
        # the deployment-facing knob (configure_auth defers to it).
        monkeypatch.setenv("TWIN_LOGIN_RATE_LIMIT_PER_MINUTE", "0")
        monkeypatch.setenv("TWIN_LOGIN_BACKOFF_MAX_SECONDS", "0")
        app = create_app(_make_settings())
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            codes = [
                (
                    await client.post(
                        "/login",
                        json={"username": "admin", "password": "wrong"},
                    )
                ).status_code
                for _ in range(8)
            ]
        assert codes == [401] * 8

    async def test_window_counts_attempts_regardless_of_outcome(self, monkeypatch):
        """The limiter is deliberately blind to outcomes (no oracle on which
        attempt succeeded); 5/min is far above human login cadence, so the
        6th request in the window is 429 even when earlier ones passed."""
        monkeypatch.setenv("TWIN_LOGIN_BACKOFF_MAX_SECONDS", "0")
        app = create_app(_make_settings())
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            codes = [
                (
                    await client.post(
                        "/login",
                        json={"username": "admin", "password": "s3cret"},
                    )
                ).status_code
                for _ in range(6)
            ]
        assert codes[:5] == [200] * 5
        assert codes[5] == 429


class TestPerAccountBackoff:
    def test_no_delay_below_threshold(self):
        auth._reset_login_throttle()
        assert auth._login_backoff_seconds("admin") == 0.0
        auth._record_login_failure("admin")
        auth._record_login_failure("admin")
        assert auth._login_backoff_seconds("admin") == 0.0

    def test_exponential_growth_capped(self):
        auth._reset_login_throttle()
        delays = []
        for _ in range(12):
            auth._record_login_failure("admin")
            delays.append(auth._login_backoff_seconds("admin"))
        # threshold is 3: failure #3 → 1s, #4 → 2s, #5 → 4s … capped at 30s
        assert delays[2] == 1.0
        assert delays[3] == 2.0
        assert delays[4] == 4.0
        assert delays[-1] == 30.0
        assert max(delays) <= 30.0

    def test_success_resets_the_counter(self):
        auth._reset_login_throttle()
        for _ in range(5):
            auth._record_login_failure("admin")
        auth._login_failures_by_account.pop("admin", None)  # what login() does
        assert auth._login_backoff_seconds("admin") == 0.0

    async def test_failed_login_sleeps_the_backoff(self, monkeypatch):
        from fastapi import HTTPException

        slept: list[float] = []

        async def fake_sleep(seconds):
            slept.append(seconds)

        async def failed_login():
            bad = LoginRequest(username="admin", password="wrong")
            out = Response()
            with pytest.raises(HTTPException):
                await login(bad, out)

        monkeypatch.setattr(auth.asyncio, "sleep", fake_sleep)
        await failed_login()
        await failed_login()
        assert slept == []  # below threshold: no delay yet
        await failed_login()
        assert slept == [1.0]


class TestCriticalAlert:
    async def test_threshold_crossing_emits_critical_event(self, monkeypatch):
        from fastapi import HTTPException

        emitted: list[dict] = []

        async def fake_emit(**kwargs):
            emitted.append(kwargs)

        async def fake_sleep(_seconds):
            return None

        monkeypatch.setattr(
            "twindb_lightrag_memgraph.server.activity_events.emit_auth_event",
            fake_emit,
        )
        # The alert threshold sits deep into backoff territory — keep the
        # real delay out of the test wall-clock.
        monkeypatch.setattr(auth.asyncio, "sleep", fake_sleep)
        auth._reset_login_throttle()
        for _ in range(auth._LOGIN_ALERT_THRESHOLD - 1):
            auth._record_login_failure("admin")
        assert not emitted

        bad = LoginRequest(username="admin", password="wrong")
        out = Response()
        with pytest.raises(HTTPException):
            await login(bad, out)

        critical = [e for e in emitted if e["sev"] == "critical"]
        assert len(critical) == 1
        assert critical[0]["action"] == "login_failure_threshold"
        assert (
            critical[0]["meta"]["consecutive_failures"] == auth._LOGIN_ALERT_THRESHOLD
        )

    def test_tracking_maps_are_bounded(self):
        auth._reset_login_throttle()
        for i in range(auth._LOGIN_TRACKING_MAX_KEYS + 5):
            auth._record_login_failure(f"user-{i}")
        assert len(auth._login_failures_by_account) <= auth._LOGIN_TRACKING_MAX_KEYS + 5
