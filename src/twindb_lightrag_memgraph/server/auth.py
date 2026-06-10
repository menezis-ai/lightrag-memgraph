"""Authentication middleware -- dual mode: static API key + JWT.

Supports two modes simultaneously (compatible with CFT agent):

1. **Static API key** (``Authorization: Bearer <key>``):
   Simple, stateless, ideal for agents.  Configured via ``LIGHTRAG_API_KEY``.

2. **JWT** (``Authorization: Bearer <jwt_token>``):
   Obtained via ``POST /login`` with username/password.
   Tokens expire every 4 hours (configurable).
   Agent auto-refreshes on 401.

If neither ``LIGHTRAG_API_KEY`` nor ``LIGHTRAG_JWT_SECRET`` is set,
authentication is disabled (open access).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

logger = logging.getLogger(__name__)

auth_router = APIRouter(tags=["auth"])
_security = HTTPBearer(auto_error=False)

DEFAULT_JWT_USERNAME = "admin"
DEFAULT_JWT_PASSWORD = "changeme"

# Module-level config -- set by configure_auth()
_static_api_key: str | None = None
_jwt_secret: str | None = None
_jwt_algorithm: str = "HS256"
_jwt_expiration_hours: int = 4
_jwt_username: str = DEFAULT_JWT_USERNAME
_jwt_password: str = DEFAULT_JWT_PASSWORD
_auth_accounts: dict[str, str] = {}
_auth_enabled: bool = False
_local_jwt_cookie_name = "twin_local_token"
# When True, require_auth lets anonymous requests through with the
# identity ``anonymous-open-access``. Set by configure_auth(allow_open_access=)
# which itself honours TWIN_ALLOW_OPEN_ACCESS=1. Boot-time check in
# ensure_auth_backend_configured() raises RuntimeError otherwise.
_open_access_allowed: bool = False


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds


class AuthStatusResponse(BaseModel):
    auth_enabled: bool
    authenticated: bool
    user: str | None = None
    expires_at: str | None = None
    login_required: bool


def _parse_auth_accounts(raw: str | None) -> dict[str, str]:
    """Parse LightRAG-compatible AUTH_ACCOUNTS.

    Accepted shape: ``user:password,user2:password2``. Passwords may contain
    additional ``:`` characters; empty entries are ignored.
    """
    if not raw:
        return {}
    accounts: dict[str, str] = {}
    for entry in raw.split(","):
        item = entry.strip()
        if not item:
            continue
        if ":" not in item:
            logger.warning("AUTH_ACCOUNTS entry without ':' ignored")
            continue
        username, password = item.split(":", 1)
        username = username.strip()
        if not username:
            logger.warning("AUTH_ACCOUNTS entry with empty username ignored")
            continue
        accounts[username] = password
    return accounts


def configure_auth(
    *,
    api_key: str | None = None,
    jwt_secret: str | None = None,
    jwt_algorithm: str = "HS256",
    jwt_expiration_hours: int = 4,
    jwt_username: str = DEFAULT_JWT_USERNAME,
    jwt_password: str = DEFAULT_JWT_PASSWORD,
    auth_accounts: str | dict[str, str] | None = None,
    local_jwt_cookie_name: str = "twin_local_token",
    allow_open_access: bool = False,
) -> None:
    """Configure auth parameters.  Called once at startup."""
    global _static_api_key, _jwt_secret, _jwt_algorithm
    global _jwt_expiration_hours, _jwt_username, _jwt_password, _auth_accounts
    global _auth_enabled, _local_jwt_cookie_name, _open_access_allowed

    accounts = (
        dict(auth_accounts)
        if isinstance(auth_accounts, dict)
        else _parse_auth_accounts(auth_accounts)
    )

    if jwt_secret:
        if jwt_password == DEFAULT_JWT_PASSWORD:
            raise ValueError(
                "LIGHTRAG_JWT_SECRET enables /login, but LIGHTRAG_JWT_PASSWORD "
                "is still the insecure default 'changeme'"
            )
        offenders = sorted(
            user for user, pwd in accounts.items() if pwd == DEFAULT_JWT_PASSWORD
        )
        if offenders:
            raise ValueError(
                "AUTH_ACCOUNTS contains the insecure default password "
                f"'changeme' for: {', '.join(offenders)}"
            )

    _static_api_key = api_key
    _jwt_secret = jwt_secret
    _jwt_algorithm = jwt_algorithm
    _jwt_expiration_hours = jwt_expiration_hours
    _jwt_username = jwt_username
    _jwt_password = jwt_password
    _auth_accounts = accounts
    _local_jwt_cookie_name = local_jwt_cookie_name
    _auth_enabled = bool(api_key or jwt_secret)
    _open_access_allowed = bool(allow_open_access)

    if not _auth_enabled:
        if _open_access_allowed:
            logger.warning(
                "TWIN_ALLOW_OPEN_ACCESS=1 -- server accepting anonymous "
                "requests (NEVER use in production)"
            )
        else:
            logger.warning("No API_KEY or JWT_SECRET configured -- auth DISABLED")
    else:
        modes = []
        if api_key:
            modes.append("static-key")
        if jwt_secret:
            modes.append("JWT")
        if accounts:
            modes.append("multi-account-login")
        logger.info("Auth enabled: %s", " + ".join(modes))


def _create_jwt(payload: dict[str, Any]) -> str:
    """Create a signed JWT token."""
    import jwt as pyjwt

    now = datetime.now(timezone.utc)
    payload.update(
        {
            "iat": now,
            "exp": now + timedelta(hours=_jwt_expiration_hours),
        }
    )
    return pyjwt.encode(payload, _jwt_secret, algorithm=_jwt_algorithm)


def _decode_jwt(token: str) -> dict[str, Any]:
    """Decode and verify a JWT token."""
    import jwt as pyjwt

    try:
        return pyjwt.decode(token, _jwt_secret, algorithms=[_jwt_algorithm])
    except pyjwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired",
        )
    except pyjwt.InvalidTokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {exc}",
        )


def _jwt_exp_to_iso(payload: dict[str, Any]) -> str | None:
    exp = payload.get("exp")
    if exp is None:
        return None
    try:
        return datetime.fromtimestamp(float(exp), timezone.utc).isoformat()
    except (TypeError, ValueError, OSError):
        return None


async def require_auth(
    request: Request = None,  # type: ignore[assignment]
    credentials: HTTPAuthorizationCredentials | None = Depends(_security),
) -> str | None:
    """FastAPI dependency: validate auth (IdP JWT cookie, static key, or
    legacy local JWT).

    Resolution order:
      1. IdP JWT — when ``TWIN_IDP_JWKS_URL`` is configured. Token may
         come from the configured HttpOnly cookie or the ``Authorization``
         header. Returns the verified ``sso_subject``.
      2. Static API key (``LIGHTRAG_API_KEY``) carried via Authorization.
      3. Legacy local JWT (``LIGHTRAG_JWT_SECRET``) carried via
         Authorization — kept for the CFT agent until it migrates to the
         IdP.

    Returns the authenticated identity (username, sso_subject, or
    ``"api_key"``) or ``None`` if no auth path is configured.
    """
    from . import idp_jwt

    # 1. IdP JWT (Couche 3 §3.3) — active when JWKS URL is set.
    #
    # An IdP-shaped cookie is treated as authoritative: present →
    # verify or 401. We deliberately do NOT silently fall through to
    # the legacy paths when an IdP cookie is rejected, so a stale
    # static key can't shadow a refused session.
    #
    # An ``Authorization: Bearer`` header with a JWT-shaped value also
    # routes through IdP verification. Non-JWT bearer values (e.g. the
    # ``LIGHTRAG_API_KEY`` literal carried by the CFT agent) are left
    # for the legacy branches below.
    idp_config = idp_jwt.get_active_config()
    if idp_config is not None and request is not None:
        cookie_token = request.cookies.get(idp_config.cookie_name)
        if cookie_token:
            user = idp_jwt.require_idp_user(request)
            if user is not None:
                return user.get("sso_subject") or user.get("sub") or "idp_user"
        auth_header = request.headers.get("authorization") or ""
        if auth_header.lower().startswith("bearer "):
            bearer = auth_header.split(" ", 1)[1].strip()
            # JWTs always carry exactly two ``.`` separators (header,
            # payload, signature). Anything else is treated as a
            # legacy bearer and falls through.
            if bearer.count(".") == 2:
                user = idp_jwt.require_idp_user(request)
                if user is not None:
                    return user.get("sso_subject") or user.get("sub") or "idp_user"

    if not _auth_enabled:
        if idp_config is not None and request is not None:
            idp_jwt.require_idp_user(request)
            # require_idp_user raises on missing/invalid token. If it
            # returns, the request is authenticated against the IdP.
            return "idp_user"
        if _open_access_allowed:
            return "anonymous-open-access"
        # No auth backend configured AND no explicit open-access opt-in:
        # boot should have already raised, but if a test setup or hot
        # reconfigure bypassed it, fail closed here too (defense in depth).
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Server has no auth backend configured. Set LIGHTRAG_API_KEY, "
                "LIGHTRAG_JWT_SECRET, or TWIN_IDP_JWKS_URL."
            ),
        )

    if credentials is None and _jwt_secret and request is not None:
        cookie_token = request.cookies.get(_local_jwt_cookie_name)
        if cookie_token:
            payload = _decode_jwt(cookie_token)
            return payload.get("sub", "unknown")

    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials

    # 2. Static API key
    if _static_api_key and token == _static_api_key:
        return "api_key"

    # 3. Legacy local JWT
    if _jwt_secret:
        payload = _decode_jwt(token)
        return payload.get("sub", "unknown")

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )


@auth_router.get("/auth-status", response_model=AuthStatusResponse)
async def auth_status(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_security),
) -> AuthStatusResponse:
    if not _auth_enabled:
        return AuthStatusResponse(
            auth_enabled=False,
            authenticated=True,
            login_required=False,
        )

    token: str | None = None
    if _jwt_secret:
        token = request.cookies.get(_local_jwt_cookie_name)
    if token is None and credentials is not None:
        token = credentials.credentials

    if not token:
        return AuthStatusResponse(
            auth_enabled=True,
            authenticated=False,
            login_required=True,
        )

    if _static_api_key and token == _static_api_key:
        return AuthStatusResponse(
            auth_enabled=True,
            authenticated=True,
            user="api_key",
            login_required=False,
        )

    if _jwt_secret:
        try:
            payload = _decode_jwt(token)
        except HTTPException:
            return AuthStatusResponse(
                auth_enabled=True,
                authenticated=False,
                login_required=True,
            )
        return AuthStatusResponse(
            auth_enabled=True,
            authenticated=True,
            user=str(payload.get("sub", "unknown")),
            expires_at=_jwt_exp_to_iso(payload),
            login_required=False,
        )

    return AuthStatusResponse(
        auth_enabled=True,
        authenticated=False,
        login_required=True,
    )


@auth_router.post("/login", response_model=LoginResponse)
async def login(body: LoginRequest, response: Response) -> LoginResponse:
    """Authenticate with username/password and receive a JWT token."""
    if not _jwt_secret:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="JWT auth not configured on this server",
        )

    expected_password = (
        _auth_accounts.get(body.username)
        if _auth_accounts
        else _jwt_password
        if body.username == _jwt_username
        else None
    )
    if expected_password is None or body.password != expected_password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )

    token = _create_jwt({"sub": body.username})
    expires_in = _jwt_expiration_hours * 3600
    response.set_cookie(
        _local_jwt_cookie_name,
        token,
        max_age=expires_in,
        httponly=True,
        secure=True,
        samesite="lax",
        path="/",
    )

    return LoginResponse(access_token=token, expires_in=expires_in)


@auth_router.post("/logout")
async def logout(response: Response) -> dict[str, bool]:
    response.delete_cookie(_local_jwt_cookie_name, path="/")
    return {"ok": True}


def ensure_auth_backend_configured(
    *,
    api_key: str | None,
    jwt_secret: str | None,
    idp_configured: bool,
    allow_open_access: bool,
) -> None:
    """Raise RuntimeError when no auth backend is configured.

    Called at boot from both the standalone app factory (``server/app.py``)
    and the plugin entry point (``__init__.py:_mount_twin_subapp``).
    LightRAG natively boots wide open when no backend is set; Twin
    refuses that posture by default.

    Args:
        api_key: Static API key (LIGHTRAG_API_KEY).
        jwt_secret: Local JWT secret (LIGHTRAG_JWT_SECRET / TOKEN_SECRET).
        idp_configured: True when ``TWIN_IDP_JWKS_URL`` is set
            (``IdpConfig.from_env() is not None``).
        allow_open_access: When True (TWIN_ALLOW_OPEN_ACCESS=1), log a
            loud warning and let the server boot. Use only in dev/CI.
    """
    if api_key or jwt_secret or idp_configured:
        return
    if allow_open_access:
        logger.warning(
            "Boot: no auth backend configured but TWIN_ALLOW_OPEN_ACCESS=1 "
            "-- server starting WIDE OPEN (dev/CI only)"
        )
        return
    raise RuntimeError(
        "Refusing to start: no auth backend configured. Set one of "
        "LIGHTRAG_API_KEY, LIGHTRAG_JWT_SECRET, or TWIN_IDP_JWKS_URL. "
        "Dev escape (NEVER in prod): TWIN_ALLOW_OPEN_ACCESS=1."
    )
