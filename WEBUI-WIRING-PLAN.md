# WebUI Wiring Plan

Entry point. The plan was split for maintainability — see:

- **[WEBUI-WIRING-WIRED.md](WEBUI-WIRING-WIRED.md)** — as-built, what's already implemented and on `stable/0.6.x`.
- **[WEBUI-WIRING-TO-WIRE.md](WEBUI-WIRING-TO-WIRE.md)** — remaining work + PO-gated items.

## Status table (2026-06-07)

| Area | Status |
|---|---|
| React WebUI port | Wired |
| Runtime Twin spaces config | Wired |
| `X-Twin-Space` frontend/backend contract | Wired |
| Twin overlay router + per-space stores | Wired |
| Memgraph tag / activity / notification stores | Wired |
| Native LightRAG route shims (`/documents`, `/health`, `/pipeline_status`, etc.) | Wired |
| Admin Space CRUD + `admin:spaces` gating | Wired (`a62b4b4`) |
| Mock-kill remediation (F1+F2+F3+F5+F6) | Wired (`731f0d1`) |
| Real `/twin/api/query` + streaming + advanced controls | Wired (`524b2a8`) |
| `tag_filter` end-to-end on retrieval | Wired (`a6ff23a`) |
| Tag delete cascade (sum graph + seed) | Wired (`7302023`) |
| Real MyAccess / IdP JWT enforcement | To wire (Priority 1, PO-gated Louis HORVAT) |
| Deployment smoke on OVH twin-real | To wire (Priority 2) |
| Retention / sweep policy for tags + activity | Deferred by PO/compliance |
| BNP MIP classification + hook | **PO-gated / not in current scope** |

## Doctrine

Twin is a governance overlay on LightRAG (RAG + ontology + tags + audit), not a document management system. Source of truth for any document remains its origin system (SharePoint, internal DMS, mail). Twin does not durably store the raw body of a source, and does not expose source download to the operator via the UI.

Any future feature that re-opens a "View raw" route or a source download endpoint requires explicit BNP Compliance sign-off, not a unilateral technical decision.
