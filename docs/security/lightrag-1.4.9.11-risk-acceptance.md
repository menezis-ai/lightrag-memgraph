# LightRAG 1.4.9.11 CVE Risk Acceptance

Date: 2026-06-21

Owner: Julien / Twincore Team

Review date: 2026-07-21

Scope: `lightrag-hku==1.4.9.11` production target.

## Decision

The project temporarily accepts the residual risk of the known LightRAG
1.4.9.11 JWT advisories and its `python-jose` / `ecdsa` transitive crypto
advisory while keeping the production pin on
`lightrag-hku==1.4.9.11`.

Accepted advisories:

| Advisory | Affected surface | Local posture |
| --- | --- | --- |
| `CVE-2026-30762` / `GHSA-mcww-4hxq-hfr3` | LightRAG native hardcoded JWT secret | Accepted temporarily with compensating controls |
| `CVE-2026-39413` / `GHSA-8ffj-4hx4-9pgf` | LightRAG native JWT algorithm confusion | Accepted temporarily with compensating controls |
| `PYSEC-2026-1325` | `ecdsa==0.19.2`, transitive through `python-jose==3.5.0` required by LightRAG 1.4.9.11 | Accepted temporarily; no fixed version is reported by `pip-audit` for the pinned dependency set |

This is not a permanent waiver. The intended remediation remains a LightRAG
upgrade after compatibility validation.

## Rationale

`lightrag-hku==1.4.9.11` is the current production baseline and must remain
pinned for this release train. Moving the pin now would reopen compatibility
risk across the Memgraph storage patches, native route shims, query envelope
projection, and WebUI route parity.

The vulnerable upstream LightRAG surface is authentication-related. The Twin KMS
overlay has already added local controls around that area:

- production mode fails closed when `TWIN_ENV=production` or
  `TWIN_REQUIRE_AUTH=true`;
- production boot requires `LIGHTRAG_API_KEY`, `LIGHTRAG_JWT_SECRET`,
  `TOKEN_SECRET`, or `TWIN_IDP_JWKS_URL`;
- production local HS JWT secrets shorter than 32 bytes are rejected;
- production local JWT auth rejects the default `changeme` password;
- corporate IdP/JWKS can be used instead of local JWT auth;
- `LIGHTRAG_CORS_ALLOWED_ORIGINS='*'` with credentials is rejected;
- runtime `pipmaster` installs are blocked by the security baseline.

Residual risk remains if an operator exposes LightRAG native authentication
without the production posture above. That deployment mode is not accepted for
production.

`PYSEC-2026-1325` is carried by `ecdsa==0.19.2` through the upstream
`python-jose` dependency. Twin KMS does not use `python-jose` for local or IdP
token validation; the Twin server uses `PyJWT` for that surface. The preferred
remediation remains removing or replacing upstream `python-jose` when the
LightRAG production pin can be upgraded or vendor-patched.

## CI Posture

The Forgejo `pip-audit` gate keeps the accepted advisories as explicit ignores
and continues to fail on every other production dependency advisory.

The ignores must not be broadened beyond:

```text
CVE-2026-30762
CVE-2026-39413
PYSEC-2026-1325
```

## Revisit Criteria

Revisit this acceptance no later than 2026-07-21, or earlier if any of the
following happens:

- the deployment needs LightRAG native auth directly exposed;
- a public exploit materially changes the likelihood/impact assessment;
- compatibility testing passes on a LightRAG version that fixes or removes the
  accepted advisories;
- a release candidate is promoted beyond the current temporary acceptance.

Expected follow-up: test and plan the upgrade to a LightRAG version that fixes
the native JWT advisories and removes the `python-jose` / `ecdsa` exposure while
preserving the Memgraph storage and WebUI overlay contracts.
