# LightRAG 1.4.9.11 CVE Risk Acceptance

Date: 2026-06-21

Owner: Julien / Twincore Team

Review date: 2026-07-21

Scope: `lightrag-hku==1.4.9.11` production target.

## Decision

The project temporarily accepts the residual risk of the two known LightRAG
1.4.9.11 JWT advisories while keeping the production pin on
`lightrag-hku==1.4.9.11`.

Accepted advisories:

| Advisory | Affected surface | Local posture |
| --- | --- | --- |
| `CVE-2026-30762` / `GHSA-mcww-4hxq-hfr3` | LightRAG native hardcoded JWT secret | Accepted temporarily with compensating controls |
| `CVE-2026-39413` / `GHSA-8ffj-4hx4-9pgf` | LightRAG native JWT algorithm confusion | Accepted temporarily with compensating controls |
| `PYSEC-2026-1325` / `CVE-2024-23342` / `GHSA-wj6h-64fc-37mp` | `python-jose` -> `ecdsa` Minerva timing attack (ECDSA P-256 signing/keygen) | Not exploitable here; unused transitive, see rationale |

This is not a permanent waiver. The intended remediation remains a LightRAG
upgrade after compatibility validation.

## Rationale

`lightrag-hku==1.4.9.11` is the current production baseline and must remain
pinned for this release train. Moving the pin now would reopen compatibility
risk across the Memgraph storage patches, native route shims, query envelope
projection, and WebUI route parity.

The vulnerable upstream surface is authentication-related. The Twin KMS overlay
has already added local controls around that area:

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

### `python-jose` / `ecdsa` (PYSEC-2026-1325)

`python-jose[cryptography]` is declared by `lightrag-hku[api]` and drags in
`ecdsa`, whose only reported advisory is the Minerva P-256 **timing** attack
(`CVE-2024-23342`). Exposure here is effectively nil:

- neither Twin nor LightRAG import `jose` anywhere — both sign/verify JWTs with
  **PyJWT** (`cryptography` backend), so `python-ecdsa` is never executed;
- the advisory affects ECDSA *signing* / keygen / ECDH only — signature
  *verification* is unaffected — and requires a local timing side channel;
- upstream is **WONTFIX** (the `python-ecdsa` project considers side channels
  out of scope), so there is no fix version to upgrade to.

Preferred remediation is dropping the unused package (e.g. `pip uninstall
python-jose ecdsa` in the image, or an upstream LightRAG change to its `api`
extra). Until then the advisory is ignored in the gate.

## CI Posture

The Forgejo `pip-audit` gate keeps these advisories as explicit ignores and
continues to fail on every other production dependency advisory.

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
- compatibility testing passes on a LightRAG version that fixes both advisories;
- a release candidate is promoted beyond the current temporary acceptance.

Expected follow-up: test and plan the upgrade to a LightRAG version that fixes
both advisories while preserving the Memgraph storage and WebUI overlay
contracts.
