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

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

logger = logging.getLogger(__name__)

auth_router = APIRouter(tags=["auth"])
_security = HTTPBearer(auto_error=False)

# Module-level config -- set by configure_auth()
_static_api_key: str | None = None
_jwt_secret: str | None = None
_jwt_algorithm: str = "HS256"
_jwt_expiration_hours: int = 4
_jwt_username: str = "admin"
_jwt_password: str = "changeme"
_auth_enabled: bool = False


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds


def configure_auth(
    *,
    api_key: str | None = None,
    jwt_secret: str | None = None,
    jwt_algorithm: str = "HS256",
    jwt_expiration_hours: int = 4,
    jwt_username: str = "admin",
    jwt_password: str = "changeme",
) -> None:
    """Configure auth parameters.  Called once at startup."""
    global _static_api_key, _jwt_secret, _jwt_algorithm
    global _jwt_expiration_hours, _jwt_username, _jwt_password, _auth_enabled

    _static_api_key = api_key
    _jwt_secret = jwt_secret
    _jwt_algorithm = jwt_algorithm
    _jwt_expiration_hours = jwt_expiration_hours
    _jwt_username = jwt_username
    _jwt_password = jwt_password
    _auth_enabled = bool(api_key or jwt_secret)

    if not _auth_enabled:
        logger.warning("No API_KEY or JWT_SECRET configured -- auth DISABLED")
    else:
        modes = []
        if api_key:
            modes.append("static-key")
        if jwt_secret:
            modes.append("JWT")
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
        # IdP is the only auth source: when it's not active and no
        # legacy mode is configured either, we let the request through
        # to preserve the v1.0.x storage-only behaviour. The
        # ``TWIN_IDP_JWKS_URL`` setting is the explicit opt-in for
        # production gating.
        return None

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


@auth_router.post("/login", response_model=LoginResponse)
async def login(body: LoginRequest) -> LoginResponse:
    """Authenticate with username/password and receive a JWT token."""
    if not _jwt_secret:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="JWT auth not configured on this server",
        )

    if body.username != _jwt_username or body.password != _jwt_password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )

    token = _create_jwt({"sub": body.username})
    expires_in = _jwt_expiration_hours * 3600

    return LoginResponse(access_token=token, expires_in=expires_in)
