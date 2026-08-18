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
import os
from collections import deque
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

# --- Login brute-force throttle (audit 2026-08-06, R-04) -------------------
# Three layered, dependency-free controls on POST /login:
#   1. per-IP sliding-window rate limit (429 + Retry-After),
#   2. per-account exponential backoff (only failed attempts pay it),
#   3. a sev=critical audit event + SECURITY log once an account crosses the
#      consecutive-failure alert threshold (purple-team: the login_failure
#      telemetry existed but nothing consumed it).
# The warning-only default-password posture (product decision 2026-06-10) is
# NOT revisited here — these controls encadre it instead.
_LOGIN_RATE_LIMIT_DEFAULT = 5  # attempts per minute per source IP; 0 disables
_LOGIN_BACKOFF_THRESHOLD = 3  # consecutive failures before the first delay
_LOGIN_BACKOFF_MAX_DEFAULT = 30  # seconds; env TWIN_LOGIN_BACKOFF_MAX_SECONDS
_LOGIN_ALERT_THRESHOLD = 10  # consecutive failures → critical audit event
_LOGIN_TRACKING_MAX_KEYS = 10_000  # crude memory cap; state is best-effort

_login_attempts_by_ip: dict[str, deque[float]] = {}
_login_failures_by_account: dict[str, int] = {}
_login_rate_limit_per_minute: int = _LOGIN_RATE_LIMIT_DEFAULT
_login_backoff_max_seconds: int = _LOGIN_BACKOFF_MAX_DEFAULT

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
            "SECURITY: AUTH_ACCOUNTS uses the default password 'changeme' for: %s",
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
    login_rate_limit_per_minute: int | None = None,
    login_backoff_max_seconds: int | None = None,
) -> None:
    """Configure auth parameters.  Called once at startup.

    LightRAG-parity posture (product decision 2026-06-10): insecure
    defaults are tolerated with a loud warning by default. Operators
    can opt into a fail-closed production posture through ``create_app``.

    Login throttle knobs (audit 2026-08-06, R-04): ``None`` defers to the
    ``TWIN_LOGIN_RATE_LIMIT_PER_MINUTE`` / ``TWIN_LOGIN_BACKOFF_MAX_SECONDS``
    env vars, then to the module defaults. ``0`` disables the per-IP rate
    limit (the per-account backoff stays; set the max to 0 to disable it
    too). Reconfiguring auth resets the throttle state.
    """
    global _static_api_key, _jwt_secret, _jwt_algorithm
    global _jwt_expiration_hours, _jwt_username, _jwt_password, _auth_accounts
    global _auth_enabled, _local_jwt_cookie_name
    global _login_rate_limit_per_minute, _login_backoff_max_seconds

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

    _login_rate_limit_per_minute = _resolve_login_throttle_knob(
        login_rate_limit_per_minute,
        "TWIN_LOGIN_RATE_LIMIT_PER_MINUTE",
        _LOGIN_RATE_LIMIT_DEFAULT,
    )
    _login_backoff_max_seconds = _resolve_login_throttle_knob(
        login_backoff_max_seconds,
        "TWIN_LOGIN_BACKOFF_MAX_SECONDS",
        _LOGIN_BACKOFF_MAX_DEFAULT,
    )
    _reset_login_throttle()

    _log_auth_mode(api_key, jwt_secret, accounts)


def _resolve_login_throttle_knob(
    explicit: int | None, env_name: str, default: int
) -> int:
    if explicit is not None:
        return max(int(explicit), 0)
    raw = (os.environ.get(env_name) or "").strip()
    if raw:
        try:
            return max(int(raw), 0)
        except ValueError:
            logger.warning("Invalid %s=%r ignored (integer expected)", env_name, raw)
    return default


def _reset_login_throttle() -> None:
    """Drop all login-throttle state (called by configure_auth and tests)."""
    _login_attempts_by_ip.clear()
    _login_failures_by_account.clear()


def _login_client_ip(request: Request | None) -> str:
    """Socket peer for throttle accounting; X-Forwarded-For is NOT trusted."""
    if request is not None and request.client and request.client.host:
        return request.client.host
    return "unknown"


def _login_rate_limited(ip: str) -> tuple[int, bool]:
    """Record one attempt for *ip*.

    Returns ``(retry_after_seconds, just_tripped)`` — ``just_tripped`` is
    True only for the attempt that first fills the window, so the caller can
    audit the limit hit once instead of per blocked request (the audit feed
    flooding was itself an R-04 observation).
    """
    if _login_rate_limit_per_minute <= 0:
        return 0, False
    if len(_login_attempts_by_ip) > _LOGIN_TRACKING_MAX_KEYS:
        _login_attempts_by_ip.clear()
    now = datetime.now(timezone.utc).timestamp()
    window = _login_attempts_by_ip.setdefault(ip, deque())
    while window and now - window[0] > 60.0:
        window.popleft()
    if len(window) >= _login_rate_limit_per_minute:
        just_tripped = len(window) == _login_rate_limit_per_minute
        return max(1, int(60.0 - (now - window[0]))), just_tripped
    window.append(now)
    return 0, False


def _login_backoff_seconds(username: str) -> float:
    """Exponential delay owed after the current consecutive-failure count."""
    if _login_backoff_max_seconds <= 0:
        return 0.0
    failures = _login_failures_by_account.get(username, 0)
    if failures < _LOGIN_BACKOFF_THRESHOLD:
        return 0.0
    return float(
        min(2 ** (failures - _LOGIN_BACKOFF_THRESHOLD), _login_backoff_max_seconds)
    )


def _record_login_failure(username: str) -> int:
    """Increment the account's consecutive-failure count; return the total."""
    if len(_login_failures_by_account) > _LOGIN_TRACKING_MAX_KEYS:
        _login_failures_by_account.clear()
    failures = _login_failures_by_account.get(username, 0) + 1
    _login_failures_by_account[username] = failures
    return failures


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
        user
        for user, password in auth_accounts.items()
        if password == DEFAULT_JWT_PASSWORD
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
    """Decode and verify a JWT token.

    Audit 2026-08-06, R-05: the client-facing message is deliberately
    constant — PyJWT internals (codec bytes, signature-mismatch reasons)
    are an error oracle that helps token crafting. The detail is logged
    server-side only.
    """
    import jwt as pyjwt

    try:
        return pyjwt.decode(token, _jwt_secret, algorithms=[_jwt_algorithm])
    except pyjwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired",
        )
    except pyjwt.InvalidTokenError as exc:
        logger.debug("JWT rejected: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
        )


def _jwt_exp_to_iso(payload: dict[str, Any]) -> str | None:
    exp = payload.get("exp")
    if exp is None:
        return None
    try:
        return datetime.fromtimestamp(float(exp), timezone.utc).isoformat()
    except (TypeError, ValueError, OSError):
        return None


async def _emit_login_activity(
    *,
    username: str,
    success: bool,
    reason: str | None = None,
) -> None:
    from .activity_events import emit_auth_event

    await emit_auth_event(
        action="login_success" if success else "login_failure",
        sev="info" if success else "warning",
        actor=username or "anonymous",
        target_type="auth",
        target_label="login",
        summary=(
            f"login succeeded for {username or 'anonymous'}"
            if success
            else f"login failed for {username or 'anonymous'}"
        ),
        meta={
            "username": username or "anonymous",
            "success": success,
            "reason": reason if reason else None,
        },
    )


async def _emit_login_alert(*, username: str, failures: int, source_ip: str) -> None:
    """sev=critical alert when an account crosses the failure threshold.

    Purple-team gap (audit 2026-08-06): ``login_failure`` warnings existed
    but no consumer ever escalated. This is the reducer — one critical event
    per threshold multiple, plus the SECURITY log line ops hooks scrape.
    """
    from .activity_events import emit_auth_event

    logger.warning(
        "SECURITY: %d consecutive login failures for account %r from %s",
        failures,
        username,
        source_ip,
    )
    await emit_auth_event(
        action="login_failure_threshold",
        sev="critical",
        actor=username or "anonymous",
        target_type="auth",
        target_label="login",
        summary=(
            f"{failures} consecutive login failures for "
            f"{username or 'anonymous'} (source {source_ip})"
        ),
        meta={
            "username": username or "anonymous",
            "success": False,
            "reason": "consecutive_failure_threshold",
            "consecutive_failures": failures,
        },
    )


def _actor_from_idp_user(user: dict[str, Any] | None) -> str | None:
    if not isinstance(user, dict):
        return None
    for key in ("sso_subject", "email", "sub", "name"):
        value = user.get(key)
        if value:
            return str(value)
    return None


def _resolve_idp_actor(request: Request) -> str | None:
    try:
        from . import idp_jwt

        if idp_jwt.get_active_config() is None:
            return None
        return _actor_from_idp_user(idp_jwt.require_idp_user(request))
    except HTTPException:
        return None
    except Exception:  # noqa: BLE001 - actor attribution must not break auth.
        logger.debug("[auth] IdP actor resolution failed", exc_info=True)
        return None


def _bearer_token_from_request(request: Request) -> str | None:
    auth_header = request.headers.get("authorization") or ""
    if not auth_header.lower().startswith("bearer "):
        return None
    return auth_header.split(" ", 1)[1].strip() or None


def _resolve_local_jwt_actor(request: Request, bearer_token: str | None) -> str | None:
    if not _jwt_secret:
        return None
    token = request.cookies.get(_local_jwt_cookie_name) or bearer_token
    if not token:
        return None
    try:
        payload = _decode_jwt(token)
        return str(payload.get("sub") or "unknown")
    except HTTPException:
        return None


def is_infrastructure_root_request(request: Request | None) -> bool:
    """Return whether *request* presents the configured static root secret.

    Authorization gates must use this credential predicate rather than compare
    the human-readable actor label returned by :func:`resolve_auth_actor`.
    A local JWT subject is caller-controlled configuration and may legitimately
    be named ``api_key``; treating that string as a capability would let the
    account collide with the infrastructure-root audit label.
    """
    if request is None or not _static_api_key:
        return False
    bearer_token = _bearer_token_from_request(request)
    native_api_key = request.headers.get("x-api-key") or ""
    return any(
        token and _secret_equal(token, _static_api_key)
        for token in (bearer_token, native_api_key)
    )


def resolve_auth_actor(request: Request | None) -> str | None:
    """Best-effort actor resolution for auth Activity events.

    Never raises and never returns secrets. The order mirrors auth itself:
    IdP user first, then local JWT cookie/bearer, then static API key label.
    """
    if request is None:
        return None

    actor = _resolve_idp_actor(request)
    if actor:
        return actor

    # Keep actor attribution aligned with ``require_auth``: the separately
    # managed static key is the infrastructure root credential and is checked
    # before legacy JWT decoding.  Besides producing accurate audit records,
    # this distinction is used by dormant-IdP authorization gates.
    # LightRAG's native document routes carry the static root credential in
    # ``X-API-Key`` while Twin routes use ``Authorization: Bearer``. Folder
    # binding runs before either router dependency, so it must recognise both
    # transports as the same separately managed infrastructure identity.
    if is_infrastructure_root_request(request):
        return "api_key"

    bearer_token = _bearer_token_from_request(request)
    actor = _resolve_local_jwt_actor(request, bearer_token)
    if actor:
        return actor
    return None


async def emit_logout_activity(request: Request | None = None) -> None:
    from .activity_events import emit_auth_event

    actor = resolve_auth_actor(request) or "unknown"
    await emit_auth_event(
        action="logout",
        sev="info",
        actor=actor,
        target_type="auth",
        target_label="logout",
        summary=f"logout for {actor}",
        meta={"success": True},
    )


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

        def _cleanup_mark_used(done: asyncio.Task[Any]) -> None:
            _api_key_mark_used_tasks.discard(done)

        task.add_done_callback(_cleanup_mark_used)
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
      2. Static API key (``LIGHTRAG_API_KEY``) carried via Authorization or
         the LightRAG-native ``X-API-Key`` transport.
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

    # Static infrastructure root. Keep both transports aligned with the
    # credential predicate used by dormant-IdP admin and folder gates.
    if is_infrastructure_root_request(request):
        return "api_key"

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

    # Preserve the dependency's direct-call contract used by integrations and
    # tests that supply parsed HTTPBearer credentials without a Request object.
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


async def _operator_key_status(
    token: str, auth_enabled: bool
) -> AuthStatusResponse | None:
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
        user=(
            str(user.get("sso_subject") or user.get("sub") or "idp_user")
            if user
            else "idp_user"
        ),
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


@auth_router.get(
    "/auth-status",
    summary="Session status",
    # Anonymous access is allowed (the bearer is optional and only enriches
    # the report). openapi_extra list values are CONCATENATED onto the
    # generated operation, so appending the empty requirement turns the
    # auto-emitted [{HTTPBearer}] into [{HTTPBearer}, {}] — OpenAPI's
    # encoding for "bearer or anonymous".
    openapi_extra={"security": [{}]},
)
async def auth_status(
    request: Request,
    credentials: Annotated[
        HTTPAuthorizationCredentials | None,
        Depends(_security),
    ] = None,
) -> AuthStatusResponse:
    """Report whether authentication is enabled on this deployment and
    whether the caller is currently authenticated (via session cookie,
    `Bearer` token or API key). `login_required: true` means the caller
    must go through `POST /login` before using protected endpoints."""
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


@auth_router.post(
    "/login",
    summary="Log in with username and password",
    responses={
        401: {"description": "Invalid username or password"},
        429: {"description": "Too many login attempts (rate limited)"},
        501: {"description": "Password login is not configured on this deployment"},
    },
)
async def login(  # NOSONAR - async contract.
    body: LoginRequest,
    response: Response,
    # Bare ``Request`` annotation (not ``Request | None``) so FastAPI keeps
    # special-casing the injection; the None default preserves the direct
    # two-arg call contract used by tests and integrations (same pattern as
    # ``require_auth`` below).
    request: Request = None,  # type: ignore[assignment]
) -> LoginResponse:
    """Authenticate with username and password. On success the response
    carries a `Bearer` token (`access_token`) and also sets it as an
    HttpOnly session cookie, so both API clients and the browser UI can
    use it. `expires_in` is the token lifetime in seconds.

    Brute-force posture (audit 2026-08-06, R-04): per-IP sliding-window
    rate limit (429 + `Retry-After`), per-account exponential backoff on
    consecutive failures, and a `sev=critical` audit event past the alert
    threshold."""
    if not _jwt_secret:
        await _emit_login_activity(
            username=body.username,
            success=False,
            reason="jwt_not_configured",
        )
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="JWT auth not configured on this server",
        )

    source_ip = _login_client_ip(request)
    retry_after, just_tripped = _login_rate_limited(source_ip)
    if retry_after:
        if just_tripped:
            await _emit_login_activity(
                username=body.username,
                success=False,
                reason="rate_limited",
            )
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many login attempts; retry later",
            headers={"Retry-After": str(retry_after)},
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
        failures = _record_login_failure(body.username or "anonymous")
        if failures % _LOGIN_ALERT_THRESHOLD == 0:
            await _emit_login_alert(
                username=body.username,
                failures=failures,
                source_ip=source_ip,
            )
        backoff = _login_backoff_seconds(body.username or "anonymous")
        await _emit_login_activity(
            username=body.username,
            success=False,
            reason="invalid_credentials",
        )
        if backoff > 0:
            await asyncio.sleep(backoff)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )

    _login_failures_by_account.pop(body.username or "anonymous", None)
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

    await _emit_login_activity(username=body.username, success=True)
    return LoginResponse(access_token=token, expires_in=expires_in)


async def logout(
    response: Response,
    request: Request | None = None,
) -> dict[str, bool]:
    response.delete_cookie(_local_jwt_cookie_name, path="/")
    await emit_logout_activity(request)
    return {"ok": True}


@auth_router.post(
    "/logout",
    summary="Log out (clear the session cookie)",
)
async def logout_route(request: Request, response: Response) -> dict[str, bool]:
    """Clear the local session cookie and record the logout in the audit
    feed. Previously issued `Bearer` tokens are not revoked — they expire
    on their own schedule."""
    return await logout(response, request)
