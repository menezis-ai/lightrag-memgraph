"""IdP JWT middleware — Couche 3 §3.3.

Wraps an external Identity Provider (BNP MyAccess / Keycloak / generic
OIDC) so the Twin FastAPI app can:

1. **Extract** a bearer token from a cookie (HttpOnly, SameSite=Lax) or
   an ``Authorization: Bearer …`` header.
2. **Verify** the JWT signature against the IdP's JWKS endpoint, with
   a TTL-bounded cache so cold lookups don't block every request.
3. **Validate** the standard registered claims (``iss``, ``aud``,
   ``exp``, ``nbf`` if present) and surface clear ``401`` reasons via
   the ``WWW-Authenticate`` header so the React port can render an
   honest expiry message.
4. **Project** the verified claim set into the
   ``AuthenticatedUser`` shape the React port expects
   (``lightrag_webui_twin/src/types/auth.ts``). Mapping is fully env-
   configurable so BNP-specific claim names don't require a code
   change.

The module is intentionally side-effect free at import time — the
``IdpConfig.from_env()`` factory is the one place that reads
``os.environ``. Tests inject configs directly.

## Activation contract

- **No env**: the module is dormant and ``require_idp_user`` returns
  ``None`` (auth falls back to the static API key / legacy JWT
  branches in ``auth.py``). ``_build_runtime_config`` keeps the
  ``debugUser`` shim alive so dev / OVH standalone stays usable.
- **`TWIN_IDP_JWKS_URL` set**: the middleware activates. Requests
  without a valid token get ``401`` instead of silently bypassing.
  ``debugUser`` is stripped from the runtime config so React stops
  using it as a fallback.

## Env vars

| Var | Default | Purpose |
|---|---|---|
| ``TWIN_IDP_JWKS_URL`` | unset | OIDC JWKS endpoint. Setting this activates the middleware. |
| ``TWIN_IDP_ISSUER`` | unset | Expected ``iss`` claim. ``None`` skips the check. |
| ``TWIN_IDP_AUDIENCE`` | unset | Expected ``aud`` claim. ``None`` skips the check. |
| ``TWIN_IDP_ALGORITHMS`` | ``RS256`` | Comma-separated allow-list. |
| ``TWIN_IDP_NAME`` | ``keycloak`` | Display identifier rendered in Settings → Profile. |
| ``TWIN_IDP_REALM`` | ``twin`` | Realm name rendered in Settings → Profile. |
| ``TWIN_IDP_COOKIE_NAME`` | ``twin_idp_token`` | HttpOnly cookie carrying the JWT. |
| ``TWIN_IDP_JWKS_CACHE_TTL`` | ``300`` | Seconds before the next JWKS fetch. |
| ``TWIN_IDP_CLAIM_SUBJECT`` | ``sub`` | Maps to ``sso_subject``. |
| ``TWIN_IDP_CLAIM_EMAIL`` | ``email`` | Maps to ``email``. |
| ``TWIN_IDP_CLAIM_NAME`` | ``name`` | Maps to ``name``. |
| ``TWIN_IDP_CLAIM_GROUPS`` | ``groups`` | Group/role list used for palier mapping. |
| ``TWIN_IDP_CLAIM_WORKSPACES`` | ``twin_spaces`` | List of space ids the user can switch into. |
| ``TWIN_IDP_CLAIM_SCOPES`` | ``scope`` | Space-delimited string OR list. |
| ``TWIN_IDP_GROUP_TO_PALIER_JSON`` | built-in | JSON mapping group → palier level. |
| ``TWIN_IDP_ADMIN_GROUPS`` | ``twin-admin,twin-steward`` | CSV of MyAccess groups whose members get ``admin:spaces`` in ``gateway_scopes``. Set to an empty string to deny admin to everyone. |

The built-in group→palier fallback maps:
``{"twin-steward": 3, "twin-contributor": 2, "twin-reader": 1}``.
Louis can drop in the BNP MyAccess group names without touching code.

Admin gating is orthogonal to palier: the default ``admin_groups`` set
includes ``twin-steward`` to mirror the doctrine *Steward = admin by
default*, but BNP can configure either dimension independently via env.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Iterable

from fastapi import HTTPException, Request, status

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_DEFAULT_GROUP_TO_PALIER: dict[str, int] = {
    "twin-steward": 3,
    "twin-contributor": 2,
    "twin-reader": 1,
}

_PALIER_LABEL: dict[int, str] = {
    1: "Reader",
    2: "Contributor",
    3: "Steward",
}

# Default Twin-internal palier capability tokens (mirror the
# `debugUser.palier.scopes` shim in `_build_runtime_config`).
_DEFAULT_PALIER_SCOPES: dict[int, list[str]] = {
    1: ["twin:read"],
    2: ["twin:read", "twin:write"],
    3: ["twin:read", "twin:write", "twin:approve"],
}

# Default admin groups. Mirrors the Steward-equals-admin doctrine while
# leaving room for a dedicated ``twin-admin`` group.
_DEFAULT_ADMIN_GROUPS: frozenset[str] = frozenset({"twin-admin", "twin-steward"})

# Gateway scope token granted when a user is in any of the configured
# admin groups. The Space CRUD routes gate on this; the React port
# reads it via ``user.gateway_scopes`` (cf. ``canManageSpaces`` in
# ``lightrag_webui_twin/src/lib/permissions.ts``).
ADMIN_SPACES_SCOPE = "admin:spaces"


@dataclass(frozen=True)
class IdpConfig:
    """Static config for the IdP middleware.

    All values are typically derived from env vars; tests construct
    instances directly to pin behaviour without polluting ``os.environ``.
    """

    jwks_url: str
    issuer: str | None = None
    audience: str | None = None
    algorithms: tuple[str, ...] = ("RS256",)
    idp_name: str = "keycloak"
    idp_realm: str = "twin"
    cookie_name: str = "twin_idp_token"
    jwks_cache_ttl: int = 300
    claim_subject: str = "sub"
    claim_email: str = "email"
    claim_name: str = "name"
    claim_groups: str = "groups"
    claim_workspaces: str = "twin_spaces"
    claim_scopes: str = "scope"
    group_to_palier: dict[str, int] = field(
        default_factory=lambda: dict(_DEFAULT_GROUP_TO_PALIER)
    )
    admin_groups: frozenset[str] = field(
        default_factory=lambda: frozenset(_DEFAULT_ADMIN_GROUPS)
    )

    @property
    def enabled(self) -> bool:
        return bool(self.jwks_url)

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "IdpConfig | None":
        """Build an `IdpConfig` from env vars, or return ``None`` if
        the JWKS URL isn't set (middleware stays dormant)."""
        env = env if env is not None else os.environ  # type: ignore[assignment]
        jwks_url = (env.get("TWIN_IDP_JWKS_URL") or "").strip()
        if not jwks_url:
            return None
        algorithms_raw = env.get("TWIN_IDP_ALGORITHMS", "RS256")
        algorithms = tuple(
            a.strip() for a in algorithms_raw.split(",") if a.strip()
        ) or ("RS256",)
        try:
            ttl = int(env.get("TWIN_IDP_JWKS_CACHE_TTL", "300"))
        except ValueError:
            ttl = 300
        group_to_palier = dict(_DEFAULT_GROUP_TO_PALIER)
        raw_map = env.get("TWIN_IDP_GROUP_TO_PALIER_JSON")
        if raw_map:
            try:
                parsed = json.loads(raw_map)
                if isinstance(parsed, dict):
                    group_to_palier = {
                        str(k): int(v)
                        for k, v in parsed.items()
                        if isinstance(v, (int, float)) and int(v) in (1, 2, 3)
                    }
            except (ValueError, TypeError):
                logger.exception(
                    "TWIN_IDP_GROUP_TO_PALIER_JSON invalid; using built-in default"
                )
        admin_groups_raw = env.get("TWIN_IDP_ADMIN_GROUPS")
        if admin_groups_raw is None:
            admin_groups: frozenset[str] = frozenset(_DEFAULT_ADMIN_GROUPS)
        else:
            # An explicit empty string is meaningful: "no admin group →
            # admin gating denies everyone". Useful for paranoid deploys
            # that wire admin via JWT scope directly and never via group.
            admin_groups = frozenset(
                part.strip()
                for part in admin_groups_raw.split(",")
                if part.strip()
            )
        return cls(
            jwks_url=jwks_url,
            issuer=(env.get("TWIN_IDP_ISSUER") or None),
            audience=(env.get("TWIN_IDP_AUDIENCE") or None),
            algorithms=algorithms,
            idp_name=env.get("TWIN_IDP_NAME", "keycloak"),
            idp_realm=env.get("TWIN_IDP_REALM", "twin"),
            cookie_name=env.get("TWIN_IDP_COOKIE_NAME", "twin_idp_token"),
            jwks_cache_ttl=max(0, ttl),
            claim_subject=env.get("TWIN_IDP_CLAIM_SUBJECT", "sub"),
            claim_email=env.get("TWIN_IDP_CLAIM_EMAIL", "email"),
            claim_name=env.get("TWIN_IDP_CLAIM_NAME", "name"),
            claim_groups=env.get("TWIN_IDP_CLAIM_GROUPS", "groups"),
            claim_workspaces=env.get("TWIN_IDP_CLAIM_WORKSPACES", "twin_spaces"),
            claim_scopes=env.get("TWIN_IDP_CLAIM_SCOPES", "scope"),
            group_to_palier=group_to_palier,
            admin_groups=admin_groups,
        )


# ---------------------------------------------------------------------------
# JWKS cache
# ---------------------------------------------------------------------------


class JwksCache:
    """In-process JWKS cache with TTL-based refresh.

    Lookups call ``signing_key_for(token)`` which returns the PyJWT
    ``PyJWK`` matching the token's ``kid`` header. A cache miss does
    NOT bypass verification — we fail closed by raising the error,
    forcing the request to a 401.
    """

    def __init__(self, config: IdpConfig, *, fetcher=None) -> None:
        self._config = config
        self._lock = Lock()
        self._fetched_at: float = 0.0
        self._client: Any | None = None
        # `fetcher` is injectable for tests so they can plug in a
        # PyJWKClient backed by an in-memory key set rather than HTTP.
        self._fetcher = fetcher or _default_fetcher

    def _is_fresh(self) -> bool:
        if self._client is None:
            return False
        return (time.time() - self._fetched_at) < max(
            1, self._config.jwks_cache_ttl
        )

    def refresh(self, *, force: bool = False) -> None:
        with self._lock:
            if not force and self._is_fresh():
                return
            try:
                self._client = self._fetcher(self._config.jwks_url)
                self._fetched_at = time.time()
            except Exception:
                logger.exception(
                    "idp_jwt: JWKS fetch failed for %s", self._config.jwks_url
                )
                # Don't blank out a previously-good cache on transient
                # failure; the next request gets a chance to retry.
                if self._client is None:
                    raise

    def signing_key_for(self, token: str) -> Any:
        if not self._is_fresh():
            self.refresh()
        if self._client is None:
            raise RuntimeError("JWKS cache is empty after refresh")
        return self._client.get_signing_key_from_jwt(token).key

    @property
    def fetched_at(self) -> float:
        return self._fetched_at


def _default_fetcher(jwks_url: str):
    """Default JWKS fetcher — wraps ``jwt.PyJWKClient`` so a single
    HTTP roundtrip per refresh cycle hits the IdP."""
    import jwt as pyjwt

    return pyjwt.PyJWKClient(jwks_url)


# ---------------------------------------------------------------------------
# Token extraction + decode
# ---------------------------------------------------------------------------


def extract_bearer_token(request: Request, config: IdpConfig) -> str | None:
    """Pull the JWT from either the configured HttpOnly cookie or the
    ``Authorization: Bearer …`` header. Cookie wins so a re-deployment
    that forgets to clear an Authorization header doesn't shadow the
    session cookie."""
    cookie_token = request.cookies.get(config.cookie_name)
    if cookie_token:
        return cookie_token.strip() or None
    auth_header = request.headers.get("authorization") or ""
    if auth_header.lower().startswith("bearer "):
        token = auth_header.split(" ", 1)[1].strip()
        return token or None
    return None


class IdpAuthError(HTTPException):
    """401 wrapper that always sets a ``WWW-Authenticate`` header
    matching RFC 6750 §3 so the client can render an honest error."""

    def __init__(self, *, error: str, description: str) -> None:
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=description,
            headers={
                "WWW-Authenticate": (
                    f'Bearer error="{error}", error_description="{description}"'
                )
            },
        )


def decode_idp_token(
    token: str, config: IdpConfig, jwks_cache: JwksCache
) -> dict[str, Any]:
    """Verify + decode a JWT against the configured IdP, raising
    ``IdpAuthError`` on any failure."""
    import jwt as pyjwt

    try:
        key = jwks_cache.signing_key_for(token)
    except Exception as exc:
        logger.exception("idp_jwt: signing key lookup failed")
        raise IdpAuthError(
            error="invalid_token",
            description=f"Cannot resolve signing key: {exc}",
        ) from exc

    options: dict[str, Any] = {}
    if not config.audience:
        options["verify_aud"] = False
    try:
        return pyjwt.decode(
            token,
            key=key,
            algorithms=list(config.algorithms),
            audience=config.audience,
            issuer=config.issuer,
            options=options,
        )
    except pyjwt.ExpiredSignatureError as exc:
        raise IdpAuthError(
            error="expired",
            description="Token expired",
        ) from exc
    except pyjwt.InvalidAudienceError as exc:
        raise IdpAuthError(
            error="invalid_token",
            description=f"Wrong audience: {exc}",
        ) from exc
    except pyjwt.InvalidIssuerError as exc:
        raise IdpAuthError(
            error="invalid_token",
            description=f"Wrong issuer: {exc}",
        ) from exc
    except pyjwt.InvalidSignatureError as exc:
        raise IdpAuthError(
            error="invalid_token",
            description="Signature mismatch",
        ) from exc
    except pyjwt.InvalidTokenError as exc:
        raise IdpAuthError(
            error="invalid_token",
            description=f"Invalid token: {exc}",
        ) from exc


# ---------------------------------------------------------------------------
# Claims → AuthenticatedUser
# ---------------------------------------------------------------------------


def _coerce_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        # Space-delimited (OAuth2 ``scope``) OR comma-delimited.
        parts = value.replace(",", " ").split()
        return [p for p in parts if p]
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if v]
    return []


def claims_to_user(
    claims: dict[str, Any], config: IdpConfig
) -> dict[str, Any]:
    """Project the verified claim dict into the ``AuthenticatedUser``
    shape the React port consumes (mirrors
    ``lightrag_webui_twin/src/types/auth.ts``)."""
    sso = str(
        claims.get(config.claim_subject)
        or claims.get("sub")
        or claims.get("email")
        or ""
    )
    email = str(claims.get(config.claim_email) or "")
    name = str(claims.get(config.claim_name) or email or sso or "Unknown")
    groups = _coerce_list(claims.get(config.claim_groups))
    workspaces = _coerce_list(claims.get(config.claim_workspaces))
    scopes = _coerce_list(claims.get(config.claim_scopes))

    palier_level = _resolve_palier_level(groups, config.group_to_palier)
    palier_label = _PALIER_LABEL[palier_level]
    palier_scopes = _DEFAULT_PALIER_SCOPES[palier_level]

    if config.admin_groups and not set(groups).isdisjoint(config.admin_groups):
        if ADMIN_SPACES_SCOPE not in scopes:
            scopes = [*scopes, ADMIN_SPACES_SCOPE]

    exp = claims.get("exp")
    session_expires: str
    if isinstance(exp, (int, float)):
        session_expires = (
            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(int(exp)))
        )
    else:
        session_expires = ""

    return {
        "sso_subject": sso,
        "email": email,
        "name": name,
        "palier": {
            "level": palier_level,
            "label": palier_label,
            "scopes": palier_scopes,
        },
        "workspaces": workspaces,
        "idp": config.idp_name,
        "idp_realm": config.idp_realm,
        "sub": str(claims.get("sub") or sso),
        "session_expires": session_expires,
        "gateway_scopes": scopes,
    }


def _resolve_palier_level(
    groups: Iterable[str], group_to_palier: dict[str, int]
) -> int:
    """Highest palier from any matching group; defaults to 1 (Reader)."""
    levels = [
        group_to_palier[g] for g in groups if g in group_to_palier
    ]
    if not levels:
        return 1
    return max(levels)


# ---------------------------------------------------------------------------
# Module-level state + FastAPI hooks
# ---------------------------------------------------------------------------

_active_config: IdpConfig | None = None
_active_cache: JwksCache | None = None


def configure_idp(config: IdpConfig | None) -> None:
    """Activate or reset the IdP middleware. Called once at startup
    (or repeatedly from tests)."""
    global _active_config, _active_cache
    _active_config = config
    _active_cache = JwksCache(config) if config and config.enabled else None
    if config and config.enabled:
        logger.info(
            "idp_jwt: middleware active (jwks=%s, issuer=%s, audience=%s)",
            config.jwks_url,
            config.issuer or "<unverified>",
            config.audience or "<unverified>",
        )
    else:
        logger.info("idp_jwt: middleware dormant (no JWKS URL configured)")


def get_active_config() -> IdpConfig | None:
    return _active_config


def require_idp_user(request: Request) -> dict[str, Any] | None:
    """FastAPI dependency: return an ``AuthenticatedUser`` dict when
    the request carries a valid IdP JWT, or ``None`` when the
    middleware is dormant. Raises 401 when activated but the token is
    missing / invalid."""
    if _active_config is None or _active_cache is None:
        return None
    token = extract_bearer_token(request, _active_config)
    if token is None:
        raise IdpAuthError(
            error="missing_token",
            description="Missing IdP credentials",
        )
    claims = decode_idp_token(token, _active_config, _active_cache)
    return claims_to_user(claims, _active_config)


def require_admin_user(request: Request) -> dict[str, Any] | None:
    """FastAPI dependency: only let through requests whose IdP token
    carries the ``admin:spaces`` gateway scope.

    Activation contract mirrors ``require_idp_user``:

    - **Dormant IdP** (``TWIN_IDP_JWKS_URL`` unset) → returns ``None``
      without raising. Dev / OVH standalone / maquette stay usable.
    - **Active IdP** → resolves the user via ``require_idp_user`` (401
      on missing/invalid token), then raises 403 unless the user's
      ``gateway_scopes`` contains :data:`ADMIN_SPACES_SCOPE`. The scope
      is injected by :func:`claims_to_user` whenever the user's
      ``groups`` intersect ``IdpConfig.admin_groups``.
    """
    if _active_config is None:
        return None
    user = require_idp_user(request)
    if user is None:
        # Defensive: should be unreachable when ``_active_config`` is
        # set, since ``require_idp_user`` raises rather than returns
        # ``None`` in that branch. Treat as 401 just in case.
        raise IdpAuthError(
            error="missing_token",
            description="Missing IdP credentials",
        )
    if ADMIN_SPACES_SCOPE not in (user.get("gateway_scopes") or []):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Admin scope '{ADMIN_SPACES_SCOPE}' required",
        )
    return user


__all__ = [
    "ADMIN_SPACES_SCOPE",
    "IdpAuthError",
    "IdpConfig",
    "JwksCache",
    "claims_to_user",
    "configure_idp",
    "decode_idp_token",
    "extract_bearer_token",
    "get_active_config",
    "require_admin_user",
    "require_idp_user",
]
