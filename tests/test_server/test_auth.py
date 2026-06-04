"""Tests for auth module (static key + JWT)."""

import pytest

from twindb_lightrag_memgraph.server.auth import (
    LoginRequest,
    LoginResponse,
    configure_auth,
    _create_jwt,
    _decode_jwt,
    require_auth,
    login,
    _auth_enabled,
)


class TestConfigureAuth:
    def test_disabled_when_no_keys(self):
        configure_auth(api_key=None, jwt_secret=None)
        from twindb_lightrag_memgraph.server import auth

        assert auth._auth_enabled is False

    def test_enabled_with_api_key(self):
        configure_auth(api_key="test-key")
        from twindb_lightrag_memgraph.server import auth

        assert auth._auth_enabled is True
        assert auth._static_api_key == "test-key"

    def test_enabled_with_jwt(self):
        configure_auth(jwt_secret="secret-123", jwt_password="test-password")
        from twindb_lightrag_memgraph.server import auth

        assert auth._auth_enabled is True
        assert auth._jwt_secret == "secret-123"

    def test_enabled_with_both(self):
        configure_auth(api_key="key", jwt_secret="secret", jwt_password="test-password")
        from twindb_lightrag_memgraph.server import auth

        assert auth._auth_enabled is True

    def test_jwt_secret_rejects_default_password(self):
        with pytest.raises(ValueError, match="changeme"):
            configure_auth(jwt_secret="secret-123")


class TestJWT:
    def setup_method(self):
        configure_auth(
            jwt_secret="test-jwt-secret",
            jwt_password="test-password",
            jwt_expiration_hours=4,
        )

    def test_create_and_decode(self):
        token = _create_jwt({"sub": "admin"})
        payload = _decode_jwt(token)
        assert payload["sub"] == "admin"
        assert "iat" in payload
        assert "exp" in payload

    def test_expired_token(self):
        import jwt as pyjwt
        from datetime import datetime, timedelta, timezone
        from fastapi import HTTPException

        now = datetime.now(timezone.utc)
        expired_payload = {
            "sub": "admin",
            "iat": now - timedelta(hours=5),
            "exp": now - timedelta(hours=1),
        }
        token = pyjwt.encode(expired_payload, "test-jwt-secret", algorithm="HS256")
        with pytest.raises(HTTPException) as exc_info:
            _decode_jwt(token)
        assert exc_info.value.status_code == 401
        assert "expired" in exc_info.value.detail.lower()

    def test_invalid_token(self):
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            _decode_jwt("not-a-valid-jwt-token")
        assert exc_info.value.status_code == 401


class TestRequireAuth:
    async def test_disabled_returns_none(self):
        configure_auth(api_key=None, jwt_secret=None)
        result = await require_auth(credentials=None)
        assert result is None

    async def test_missing_credentials(self):
        from fastapi import HTTPException

        configure_auth(api_key="my-key")
        with pytest.raises(HTTPException) as exc_info:
            await require_auth(credentials=None)
        assert exc_info.value.status_code == 401

    async def test_valid_api_key(self):
        configure_auth(api_key="my-key")
        from fastapi.security import HTTPAuthorizationCredentials

        creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="my-key")
        result = await require_auth(credentials=creds)
        assert result == "api_key"

    async def test_invalid_api_key_no_jwt(self):
        from fastapi import HTTPException
        from fastapi.security import HTTPAuthorizationCredentials

        configure_auth(api_key="my-key", jwt_secret=None)
        creds = HTTPAuthorizationCredentials(
            scheme="Bearer", credentials="wrong-key"
        )
        with pytest.raises(HTTPException) as exc_info:
            await require_auth(credentials=creds)
        assert exc_info.value.status_code == 401

    async def test_jwt_fallback(self):
        configure_auth(
            api_key="my-key", jwt_secret="secret", jwt_password="test-password"
        )
        token = _create_jwt({"sub": "user1"})
        from fastapi.security import HTTPAuthorizationCredentials

        creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
        result = await require_auth(credentials=creds)
        assert result == "user1"


class TestLoginEndpoint:
    async def test_login_success(self):
        configure_auth(jwt_secret="secret", jwt_username="admin", jwt_password="pass")
        resp = await login(LoginRequest(username="admin", password="pass"))
        assert resp.access_token
        assert resp.token_type == "bearer"
        assert resp.expires_in == 4 * 3600

    async def test_login_bad_password(self):
        from fastapi import HTTPException

        configure_auth(jwt_secret="secret", jwt_username="admin", jwt_password="pass")
        with pytest.raises(HTTPException) as exc_info:
            await login(LoginRequest(username="admin", password="wrong"))
        assert exc_info.value.status_code == 401

    async def test_login_jwt_not_configured(self):
        from fastapi import HTTPException

        configure_auth(api_key="key", jwt_secret=None)
        with pytest.raises(HTTPException) as exc_info:
            await login(LoginRequest(username="admin", password="pass"))
        assert exc_info.value.status_code == 501


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


class TestRequireAuthEdgeCases:
    """Edge cases for the dual-mode (api_key + JWT) authentication flow."""

    async def test_require_auth_wrong_key_falls_through_to_jwt(self):
        """When both api_key and jwt_secret are configured, a valid JWT that
        does not match the static key should authenticate via the JWT path."""
        configure_auth(
            api_key="static-key",
            jwt_secret="jwt-secret",
            jwt_password="test-password",
        )
        token = _create_jwt({"sub": "jwt-user"})
        from fastapi.security import HTTPAuthorizationCredentials

        # token != "static-key", so the static-key check fails;
        # require_auth should fall through to JWT decoding and succeed.
        creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
        result = await require_auth(credentials=creds)
        assert result == "jwt-user"

    async def test_require_auth_wrong_key_and_invalid_jwt(self):
        """When both api_key and jwt_secret are configured, providing a token
        that is neither the static key nor a valid JWT must return 401."""
        from fastapi import HTTPException

        configure_auth(
            api_key="static-key",
            jwt_secret="jwt-secret",
            jwt_password="test-password",
        )
        from fastapi.security import HTTPAuthorizationCredentials

        creds = HTTPAuthorizationCredentials(
            scheme="Bearer", credentials="neither-key-nor-jwt"
        )
        with pytest.raises(HTTPException) as exc_info:
            await require_auth(credentials=creds)
        assert exc_info.value.status_code == 401

    async def test_require_auth_jwt_only_no_api_key(self):
        """When only jwt_secret is configured (no api_key), a valid JWT
        authenticates and a random string fails with 401."""
        from fastapi import HTTPException
        from fastapi.security import HTTPAuthorizationCredentials

        configure_auth(
            api_key=None,
            jwt_secret="only-jwt-secret",
            jwt_password="test-password",
        )

        # Valid JWT should succeed.
        token = _create_jwt({"sub": "jwt-only-user"})
        creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
        result = await require_auth(credentials=creds)
        assert result == "jwt-only-user"

        # Random string should fail.
        bad_creds = HTTPAuthorizationCredentials(
            scheme="Bearer", credentials="random-garbage"
        )
        with pytest.raises(HTTPException) as exc_info:
            await require_auth(credentials=bad_creds)
        assert exc_info.value.status_code == 401

    async def test_jwt_payload_missing_sub(self):
        """A JWT with no 'sub' claim should authenticate but return 'unknown'."""
        configure_auth(
            api_key=None,
            jwt_secret="sub-test-secret",
            jwt_password="test-password",
        )
        from fastapi.security import HTTPAuthorizationCredentials

        # Create a JWT with no "sub" key at all.
        token = _create_jwt({"role": "viewer"})
        creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
        result = await require_auth(credentials=creds)
        assert result == "unknown"


class TestLoginEdgeCases:
    """Edge cases for the /login endpoint."""

    async def test_login_wrong_username(self):
        """LoginRequest with a wrong username (correct password) must return 401."""
        from fastapi import HTTPException

        configure_auth(
            jwt_secret="secret", jwt_username="admin", jwt_password="pass"
        )
        with pytest.raises(HTTPException) as exc_info:
            await login(LoginRequest(username="not-admin", password="pass"))
        assert exc_info.value.status_code == 401
        assert "invalid" in exc_info.value.detail.lower()


class TestJWTEdgeCases:
    """Edge cases for JWT creation, decoding, and algorithm handling."""

    def test_create_jwt_custom_expiration(self):
        """Configure jwt_expiration_hours=1, create a token, and verify the
        exp claim is approximately 1 hour from now."""
        from datetime import datetime, timezone

        configure_auth(
            jwt_secret="exp-test-secret",
            jwt_password="test-password",
            jwt_expiration_hours=1,
        )
        now_before = datetime.now(timezone.utc)
        token = _create_jwt({"sub": "expiry-test"})
        now_after = datetime.now(timezone.utc)

        payload = _decode_jwt(token)
        exp_ts = payload["exp"]

        # exp should be ~1 hour (3600s) after "now". Allow a generous 5s
        # window on each side to avoid flakiness.
        expected_low = now_before.timestamp() + 3600 - 5
        expected_high = now_after.timestamp() + 3600 + 5
        assert expected_low <= exp_ts <= expected_high

    def test_decode_jwt_wrong_secret(self):
        """Creating a token with one secret and decoding with a different
        secret must raise 401."""
        import jwt as pyjwt
        from datetime import datetime, timedelta, timezone
        from fastapi import HTTPException

        # Create token with secret "alpha".
        configure_auth(jwt_secret="alpha", jwt_password="test-password")
        token = _create_jwt({"sub": "user"})

        # Reconfigure with a different secret "beta".
        configure_auth(jwt_secret="beta", jwt_password="test-password")

        with pytest.raises(HTTPException) as exc_info:
            _decode_jwt(token)
        assert exc_info.value.status_code == 401

    def test_configure_auth_custom_algorithm(self):
        """Configure HS384 algorithm, create and decode a JWT successfully."""
        configure_auth(
            jwt_secret="algo-test-secret",
            jwt_password="test-password",
            jwt_algorithm="HS384",
        )
        token = _create_jwt({"sub": "algo-user"})

        # Decode should work with the same algorithm.
        payload = _decode_jwt(token)
        assert payload["sub"] == "algo-user"

        # Verify the token is actually HS384 by decoding the header.
        import jwt as pyjwt

        header = pyjwt.get_unverified_header(token)
        assert header["alg"] == "HS384"
