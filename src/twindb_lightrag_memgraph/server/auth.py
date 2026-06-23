"""Authentication middleware -- dual mode: static API key + JWT.

Supports two modes simultaneously (compatible with CFT agent):

1. **Static API key** (``Authorization: Bearer <key>``):
   Simple, stateless, ideal for agents.  Configured via ``LIGHTRAG_API_KEY``.

2. **JWT** (``Authorization: Bearer <jwt_token>``):
   Obtained via ``POST /login`` with username/password.
   Tokens expire every 4 hours (configurable).
   Agent auto-refreshes on 401.

If neither ``LIGHTRAG_API_KEY`` nor ``LIGHTRAG_JWT_SECRET`` is set,
authentication is disabled (open access) unless production auth is
explicitly required by the application factory.
"""

from __future__ import annotations

import asyncio
import hmac
import logging
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

logger = logging.getLogger(__name__)

auth_router = APIRouter(tags=["auth"])
_security = HTTPBearer(auto_error=False)

DEFAULT_JWT_USERNAME = "admin"
DEFAULT_JWT_PASSWORD = "".join(("change", "me"))
_DUMMY_PASSWORD = "\0" * 32

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
_api_key_mark_used_tasks: set[asyncio.Task[Any]] = set()


class AuthConfigurationError(ValueError):
    """Raised when an explicitly strict auth posture is misconfigured."""


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
        password = password.strip()
        if not username:
            logger.warning("AUTH_ACCOUNTS entry with empty username ignored")
            continue
        accounts[username] = password
    return accounts


def _secret_equal(provided: str, expected: str) -> bool:
    """Compare secrets without early-exit timing leaks."""
    return hmac.compare_digest(
        provided.encode("utf-8"),
        expected.encode("utf-8"),
    )


def _warn_default_passwords(
    jwt_secret: str | None, accounts: dict[str, str], jwt_password: str
) -> None:
    """Emit loud SECURITY warnings when the default 'changeme' password is live."""
    if not jwt_secret:
        return
    if not accounts and jwt_password == DEFAULT_JWT_PASSWORD:
        logger.warning(
            "SECURITY: /login is enabled with the default password "
            "'changeme' (set LIGHTRAG_JWT_PASSWORD or AUTH_ACCOUNTS "
            "before exposing this server)"
        )
    offenders = sorted(
        user for user, pwd in accounts.items() if pwd == DEFAULT_JWT_PASSWORD
    )
    if offenders:
        logger.warning(
            "SECURITY: AUTH_ACCOUNTS uses the default password "
            "'changeme' for: %s",
            ", ".join(offenders),
        )


def _log_auth_mode(
    api_key: str | None, jwt_secret: str | None, accounts: dict[str, str]
) -> None:
    if not (api_key or jwt_secret):
        logger.warning(
            "No API_KEY or JWT_SECRET configured -- auth DISABLED "
            "(open access, LightRAG-parity default)"
        )
        return
    modes = []
    if api_key:
        modes.append("static-key")
    if jwt_secret:
        modes.append("JWT")
    if accounts:
        modes.append("multi-account-login")
    logger.info("Auth enabled: %s", " + ".join(modes))


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
    production_mode: bool = False,
    idp_enabled: bool = False,
) -> None:
    """Configure auth parameters.  Called once at startup.

    LightRAG-parity posture (product decision 2026-06-10): insecure
    defaults are tolerated with a loud warning by default. Operators
    can opt into a fail-closed production posture through ``create_app``.
    """
    global _static_api_key, _jwt_secret, _jwt_algorithm
    global _jwt_expiration_hours, _jwt_username, _jwt_password, _auth_accounts
    global _auth_enabled, _local_jwt_cookie_name

    accounts = (
        dict(auth_accounts)
        if isinstance(auth_accounts, dict)
        else _parse_auth_accounts(auth_accounts)
    )

    if production_mode:
        _validate_production_auth_config(
            api_key=api_key,
            jwt_secret=jwt_secret,
            jwt_algorithm=jwt_algorithm,
            jwt_password=jwt_password,
            auth_accounts=accounts,
            idp_enabled=idp_enabled,
        )

    _warn_default_passwords(jwt_secret, accounts, jwt_password)

    _static_api_key = api_key
    _jwt_secret = jwt_secret
    _jwt_algorithm = jwt_algorithm
    _jwt_expiration_hours = jwt_expiration_hours
    _jwt_username = jwt_username
    _jwt_password = jwt_password
    _auth_accounts = accounts
    _local_jwt_cookie_name = local_jwt_cookie_name
    _auth_enabled = bool(api_key or jwt_secret)

    _log_auth_mode(api_key, jwt_secret, accounts)


def _validate_production_auth_config(
    *,
    api_key: str | None,
    jwt_secret: str | None,
    jwt_algorithm: str,
    jwt_password: str,
    auth_accounts: dict[str, str],
    idp_enabled: bool,
) -> None:
    """Fail closed for explicitly production-mode deployments."""
    if not (api_key or jwt_secret or idp_enabled):
        raise AuthConfigurationError(
            "Production auth requires LIGHTRAG_API_KEY, LIGHTRAG_JWT_SECRET, "
            "TOKEN_SECRET, or TWIN_IDP_JWKS_URL"
        )

    if not jwt_secret:
        return

    if jwt_algorithm.upper().startswith("HS") and len(jwt_secret.encode("utf-8")) < 32:
        raise AuthConfigurationError(
            "Production JWT auth requires an HMAC secret of at least 32 bytes"
        )

    if not auth_accounts and jwt_password == DEFAULT_JWT_PASSWORD:
        raise AuthConfigurationError(
            "Production JWT auth cannot use the default LIGHTRAG_JWT_PASSWORD"
        )

    offenders = sorted(
        user for user, password in auth_accounts.items() if password == DEFAULT_JWT_PASSWORD
    )
    if offenders:
        raise AuthConfigurationError(
            "Production AUTH_ACCOUNTS cannot use the default password for: "
            + ", ".join(offenders)
        )


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


def _resolve_idp_identity(request, idp_config) -> str | None:
    """IdP JWT resolution (Couche 3 §3.3) — active when JWKS URL is set.

    An IdP-shaped cookie is authoritative: present → verify or 401. We
    deliberately do NOT silently fall through to the legacy paths when an IdP
    cookie is rejected, so a stale static key can't shadow a refused session.
    An ``Authorization: Bearer`` header with a JWT-shaped value (exactly two
    ``.`` separators) also routes through IdP verification; non-JWT bearers
    (e.g. the ``LIGHTRAG_API_KEY`` literal) fall through. Returns the verified
    identity, or ``None`` when no IdP token applies.
    """
    from . import idp_jwt

    if idp_config is None or request is None:
        return None
    cookie_token = request.cookies.get(idp_config.cookie_name)
    if cookie_token:
        user = idp_jwt.require_idp_user(request)
        if user is not None:
            return user.get("sso_subject") or user.get("sub") or "idp_user"
    auth_header = request.headers.get("authorization") or ""
    if auth_header.lower().startswith("bearer "):
        bearer = auth_header.split(" ", 1)[1].strip()
        if bearer.count(".") == 2:
            user = idp_jwt.require_idp_user(request)
            if user is not None:
                return user.get("sso_subject") or user.get("sub") or "idp_user"
    return None


async def _consume_operator_key(token: str) -> str | None:
    """Validate a per-operator ``twk_`` bearer and return its identity.

    Returns ``f"api_key:{key_id}"`` on success, or ``None`` when the token is
    not operator-prefixed, the store rejects it, or the store glitches (auth
    must never break on a store error). Schedules a fire-and-forget last-used
    bump so the request returns before the write completes.
    """
    from . import api_key_store
    from .._constants import resolve_workspace

    if not token.startswith(api_key_store.KEY_PREFIX):
        return None
    try:
        workspace = resolve_workspace()
        entry = await api_key_store.validate_bearer(workspace, token)
    except Exception:  # noqa: BLE001 — never break auth on store glitch
        logger.exception("[auth] api_key_store.validate_bearer crashed")
        entry = None
    if entry is None:
        return None
    key_id = str(entry.get("id"))
    try:
        task = asyncio.create_task(api_key_store.mark_used(workspace, key_id))
        _api_key_mark_used_tasks.add(task)
        task.add_done_callback(_api_key_mark_used_tasks.discard)
    except Exception:  # noqa: BLE001
        logger.exception("[auth] mark_used schedule failed")
    return f"api_key:{key_id}"


async def _resolve_open_access(request, credentials, idp_config) -> str | None:
    """Auth resolution when no env auth backend is configured (open access).

    Preserves v1.0.x storage-only / LightRAG-native parity (2026-06-10): when
    the IdP is configured it fails closed; a ``twk_``-prefixed bearer opts into
    the per-operator key contract (validate or 401); anonymous passes through.
    """
    from . import api_key_store

    if idp_config is not None and request is not None:
        from . import idp_jwt

        idp_jwt.require_idp_user(request)
        # require_idp_user raises on missing/invalid token; if it returns,
        # the request is authenticated against the IdP.
        return "idp_user"
    if credentials is not None:
        token = credentials.credentials
        if token.startswith(api_key_store.KEY_PREFIX):
            identity = await _consume_operator_key(token)
            if identity is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid credentials",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            return identity
    return None


async def require_auth(
    request: Request = None,  # type: ignore[assignment]
    credentials: Annotated[
        HTTPAuthorizationCredentials | None,
        Depends(_security),
    ] = None,
) -> str | None:
    """FastAPI dependency: validate auth (IdP JWT cookie, static key, or
    legacy local JWT).

    Resolution order:
      1. IdP JWT — when ``TWIN_IDP_JWKS_URL`` is configured (cookie or
         ``Authorization`` header). Returns the verified ``sso_subject``.
      2. Static API key (``LIGHTRAG_API_KEY``) carried via Authorization.
      3. Per-operator API key (``twk_``) minted via Settings → API keys.
      4. Legacy local JWT (``LIGHTRAG_JWT_SECRET``) carried via Authorization.

    Returns the authenticated identity (username, sso_subject, or
    ``"api_key"``) or ``None`` if no auth path is configured.
    """
    from . import idp_jwt

    idp_config = idp_jwt.get_active_config()
    identity = _resolve_idp_identity(request, idp_config)
    if identity is not None:
        return identity

    if not _auth_enabled:
        return await _resolve_open_access(request, credentials, idp_config)

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

    # 2. Static API key (env-set, infra root key — never exposed via UI).
    if _static_api_key and _secret_equal(token, _static_api_key):
        return "api_key"

    # 3. Per-operator API keys minted via Settings → API keys.
    identity = await _consume_operator_key(token)
    if identity is not None:
        return identity

    # 4. Legacy local JWT
    if _jwt_secret:
        payload = _decode_jwt(token)
        return payload.get("sub", "unknown")

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def _operator_key_status(token: str, auth_enabled: bool) -> AuthStatusResponse | None:
    """Return an authenticated status for a valid ``twk_`` bearer, else None.

    None means the token is not operator-prefixed, the store rejected it, or
    the store glitched — callers decide what that means for their branch.
    """
    from . import api_key_store
    from .._constants import resolve_workspace

    if not token.startswith(api_key_store.KEY_PREFIX):
        return None
    try:
        workspace = resolve_workspace()
        entry = await api_key_store.validate_bearer(workspace, token)
    except Exception:  # noqa: BLE001
        entry = None
    if entry is None:
        return None
    return AuthStatusResponse(
        auth_enabled=auth_enabled,
        authenticated=True,
        user=f"api_key:{entry.get('id')}",
        login_required=False,
    )


def _auth_status_idp(request, credentials, idp_config) -> AuthStatusResponse | None:
    """IdP branch of /auth-status. None → no IdP token present, fall through."""
    from . import idp_jwt

    if idp_config is None:
        return None
    bearer = credentials.credentials if credentials is not None else None
    has_idp_cookie = bool(request.cookies.get(idp_config.cookie_name))
    has_idp_bearer = bool(bearer and bearer.count(".") == 2)
    if not (has_idp_cookie or has_idp_bearer or not _auth_enabled):
        return None
    try:
        user = idp_jwt.require_idp_user(request)
    except HTTPException:
        return AuthStatusResponse(
            auth_enabled=True, authenticated=False, login_required=True
        )
    return AuthStatusResponse(
        auth_enabled=True,
        authenticated=True,
        user=str(user.get("sso_subject") or user.get("sub") or "idp_user")
        if user
        else "idp_user",
        login_required=False,
    )


async def _auth_status_open(credentials) -> AuthStatusResponse:
    """Open-access /auth-status: mirrors require_auth's open-access resolution.

    A ``twk_`` bearer opts into per-operator key validation; anonymous (or any
    non-prefixed bearer) falls through as authenticated=true (LightRAG parity).
    """
    from . import api_key_store

    bearer = credentials.credentials if credentials is not None else None
    if bearer and bearer.startswith(api_key_store.KEY_PREFIX):
        resp = await _operator_key_status(bearer, auth_enabled=False)
        if resp is not None:
            return resp
        return AuthStatusResponse(
            auth_enabled=False, authenticated=False, login_required=False
        )
    return AuthStatusResponse(
        auth_enabled=False, authenticated=True, login_required=False
    )


@auth_router.get("/auth-status")
async def auth_status(
    request: Request,
    credentials: Annotated[
        HTTPAuthorizationCredentials | None,
        Depends(_security),
    ] = None,
) -> AuthStatusResponse:
    from . import idp_jwt

    idp_config = idp_jwt.get_active_config()
    idp_resp = _auth_status_idp(request, credentials, idp_config)
    if idp_resp is not None:
        return idp_resp

    if not _auth_enabled:
        return await _auth_status_open(credentials)

    token: str | None = None
    if _jwt_secret:
        token = request.cookies.get(_local_jwt_cookie_name)
    if token is None and credentials is not None:
        token = credentials.credentials

    if not token:
        return AuthStatusResponse(
            auth_enabled=True, authenticated=False, login_required=True
        )

    if _static_api_key and _secret_equal(token, _static_api_key):
        return AuthStatusResponse(
            auth_enabled=True,
            authenticated=True,
            user="api_key",
            login_required=False,
        )

    key_resp = await _operator_key_status(token, auth_enabled=True)
    if key_resp is not None:
        return key_resp

    if _jwt_secret:
        try:
            payload = _decode_jwt(token)
        except HTTPException:
            return AuthStatusResponse(
                auth_enabled=True, authenticated=False, login_required=True
            )
        return AuthStatusResponse(
            auth_enabled=True,
            authenticated=True,
            user=str(payload.get("sub", "unknown")),
            expires_at=_jwt_exp_to_iso(payload),
            login_required=False,
        )

    return AuthStatusResponse(
        auth_enabled=True, authenticated=False, login_required=True
    )


@auth_router.post("/login")
async def login(body: LoginRequest, response: Response) -> LoginResponse:  # NOSONAR - async contract.
    """Authenticate with username/password and receive a JWT token."""
    if not _jwt_secret:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="JWT auth not configured on this server",
        )

    if _auth_accounts:
        expected_password = _auth_accounts.get(body.username)
    elif body.username == _jwt_username:
        expected_password = _jwt_password
    else:
        expected_password = None
    password_to_check = (
        expected_password if expected_password is not None else _DUMMY_PASSWORD
    )
    password_matches = _secret_equal(body.password, password_to_check)
    if expected_password is None or not password_matches:
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
def logout(response: Response) -> dict[str, bool]:
    response.delete_cookie(_local_jwt_cookie_name, path="/")
    return {"ok": True}
