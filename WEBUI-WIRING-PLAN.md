# WebUI Wiring Plan

Current entry point for the Twin WebUI wiring state. The historical split is
still kept for context, but those files describe older checkpoints and must not
be read as the live contract without this page:

- **[WEBUI-WIRING-WIRED.md](WEBUI-WIRING-WIRED.md)** — historical as-built notes.
- **[WEBUI-WIRING-TO-WIRE.md](WEBUI-WIRING-TO-WIRE.md)** — historical backlog notes.

The live contract is Folder-based (`TWIN_DEFAULT_FOLDER`, `TWIN_FOLDERS_JSON`,
`X-Twin-Folder`, `/twin/api/folders`) and is enforced by the Forgejo CI in
`.forgejo/workflows/ci.yml`.

## Status table (2026-06-20)

| Area | Status |
|---|---|
| React WebUI port | Wired |
| Runtime Twin folders config | Wired |
| `X-Twin-Folder` frontend/backend contract | Wired |
| Twin overlay router + per-folder stores | Wired |
| Memgraph tag / activity / notification stores | Wired |
| Native LightRAG route shims (`/documents`, `/health`, `/pipeline_status`, etc.) | Wired |
| Admin Folder CRUD + `admin:folders` gating | Wired |
| Mock-kill remediation (F1+F2+F3+F5+F6) | Wired (`731f0d1`) |
| Real `/twin/api/query` + streaming + advanced controls | Wired (`524b2a8`) |
| `tag_filter` end-to-end on retrieval | Wired (`a6ff23a`) |
| Tag delete cascade (sum graph + seed) | Wired (`7302023`) |
| Real MyAccess / IdP JWT endpoint wiring | Ops/deployment pending |
| Deployment smoke on OVH twin-real | Pending deployment lane/runbook |
| Retention / sweep policy for tags + activity | Deferred by PO/compliance |
| BNP MIP classification + hook | **PO-gated / not in current scope** |

## Doctrine

Twin is a governance overlay on LightRAG (RAG + ontology + tags + audit), not a document management system. Source of truth for any document remains its origin system (SharePoint, internal DMS, mail). Twin does not durably store the raw body of a source, and does not expose source download to the operator via the UI.

Any future feature that re-opens a "View raw" route or a source download endpoint requires explicit BNP Compliance sign-off, not a unilateral technical decision.
