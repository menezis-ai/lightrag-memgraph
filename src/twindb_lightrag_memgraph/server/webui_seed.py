"""In-memory seed for the WebUI phase-1 endpoints.

This data mirrors the TS fixtures in ``lightrag_webui_twin/src/fixtures/`` so
running the WebUI against ``VITE_API_BASE_URL=…`` gives the same demo
experience as the MSW-stubbed dev path.

Phase-1 scope: the seed lives in-process (rebuilt per server boot) so the
endpoints work end-to-end without depending on a separate tag/workspace
store. A later slice will swap this for real persistence (Memgraph for
docs+graph, a dedicated KV for tag governance, an events table for
activity, etc.).
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------

DOCUMENTS: list[dict[str, Any]] = [
    {
        "id": "d1",
        "type": "file",
        "source": "oracle-restart-procedure.pdf",
        "summary": "Step-by-step guide for restarting Oracle DB on RHEL 9 in CIB prod",
        "tags": ["rman", "oracle"],
        "status": "completed",
        "chunks": 418,
        "updated": "2h ago",
        "visibility": "private",
        "workspace": "cib",
    },
    {
        "id": "d2",
        "type": "confluence",
        "source": "/cib/runbooks/oracle-pga-tuning",
        "summary": "Oracle PGA memory tuning recommendations and worked examples",
        "tags": ["rman"],
        "status": "completed",
        "chunks": 54,
        "updated": "1d ago",
        "visibility": "private",
        "workspace": "cib",
    },
    {
        "id": "d3",
        "type": "sharepoint",
        "source": "huge-archive.zip",
        "summary": "Failed ingest — unsupported MIME",
        "tags": [],
        "status": "failed",
        "chunks": 0,
        "updated": "30m ago",
        "visibility": "private",
        "workspace": "cib",
    },
    {
        "id": "d4",
        "type": "file",
        "source": "memgraph-mage-3.8-release-notes.md",
        "summary": "Memgraph MAGE 3.8 release notes — vector_search improvements",
        "tags": ["memgraph", "mage"],
        "status": "completed",
        "chunks": 42,
        "updated": "25m ago",
        "visibility": "private",
        "workspace": "cib",
    },
    {
        "id": "d5",
        "type": "confluence",
        "source": "/cib/incidents/2026-04-prod-outage",
        "summary": "Postmortem: 2026-04 prod outage, Oracle PGA OOM cascade",
        "tags": ["incident", "oracle", "production"],
        "status": "completed",
        "chunks": 156,
        "updated": "3d ago",
        "visibility": "private",
        "workspace": "cib",
    },
    {
        "id": "d6",
        "type": "file",
        "source": "pending-review-doc.pdf",
        "summary": "Awaiting palier-2 review before publication",
        "tags": ["pending-review"],
        "status": "processing",
        "chunks": 21,
        "updated": "5h ago",
        "visibility": "private",
        "workspace": "cib",
    },
]

# ---------------------------------------------------------------------------
# Workspaces
# ---------------------------------------------------------------------------

WORKSPACES: list[dict[str, Any]] = [
    {"id": "cib", "kb": "CIB KB", "visibility": "private", "sources": 247, "role": "admin / steward", "current": True},
    {"id": "cib-edge", "kb": "CIB Edge KB", "visibility": "private", "sources": 82, "role": "admin", "current": False},
    {"id": "payments", "kb": "Payments KB", "visibility": "internal", "sources": 1318, "role": "reader", "current": False},
    {"id": "infra", "kb": "Infra Runbooks", "visibility": "internal", "sources": 612, "role": "steward", "current": False},
    {"id": "sandbox", "kb": "Personal sandbox", "visibility": "private", "sources": 9, "role": "owner", "current": False},
]

# ---------------------------------------------------------------------------
# Notifications
# ---------------------------------------------------------------------------

NOTIFICATIONS: list[dict[str, Any]] = [
    {"id": "n_001", "kind": "tag-mutation", "title": "Tag", "tagname": "rman", "suffix": "applied", "sub": "oracle-restart-procedure.pdf · 418 chunks", "rel": "12m ago", "read": False},
    {"id": "n_002", "kind": "source-failed", "title": "Ingestion failed", "sub": "huge-archive.zip · unsupported MIME", "rel": "30m ago", "read": False},
    {"id": "n_003", "kind": "source-ready", "title": "Source ready", "sub": "memgraph-mage-3.8-release-notes.md · 42 chunks", "rel": "25m ago", "read": False},
    {"id": "n_004", "kind": "pipeline-warning", "title": "Pipeline warning", "sub": "LLM extraction retrying · attempt 2/3", "rel": "57m ago", "read": True},
    {"id": "n_005", "kind": "tag-mutation", "title": "Tag", "tagname": "iso20022", "suffix": "added", "sub": "palier 2 · category payment", "rel": "22h ago", "read": True},
]

# ---------------------------------------------------------------------------
# Thesaurus
# ---------------------------------------------------------------------------

THESAURUS: list[dict[str, Any]] = [
    {"tag": "rman", "category": "oracle", "def": "Recovery Manager — Oracle backup and recovery tool"},
    {"tag": "rmf-validated", "category": "governance", "def": "Reviewed by risk management framework"},
    {"tag": "oracle", "category": "oracle", "def": "Oracle Database engine and ecosystem"},
    {"tag": "production", "category": "lifecycle", "def": "Document applies to production environments"},
    {"tag": "incident", "category": "lifecycle", "def": "Postmortem or active incident material"},
    {"tag": "vmware", "category": "infra", "def": "VMware vSphere and ESXi runtime"},
    {"tag": "memgraph", "category": "infra", "def": "Memgraph graph database (MAGE extensions)"},
    {"tag": "network", "category": "network", "def": "Networking, routing, firewall topics"},
    {"tag": "cft", "category": "network", "def": "Cross-platform file transfer (Axway CFT)"},
    {"tag": "swift", "category": "payment", "def": "SWIFT messaging and payment flows"},
    {"tag": "governance", "category": "governance", "def": "Internal governance, audit, or charter"},
    {"tag": "deprecated", "category": "lifecycle", "def": "Marked for archival, do not act on"},
    {"tag": "pending-review", "category": "lifecycle", "def": "Awaiting risk review before publish"},
]

# ---------------------------------------------------------------------------
# Tag governance — categories + full tag entries
# ---------------------------------------------------------------------------

TAG_CATEGORIES: list[dict[str, Any]] = [
    {"id": "oracle", "label": "Oracle", "color": "#B85A1E"},
    {"id": "infra", "label": "Infrastructure", "color": "#5A7FB4"},
    {"id": "network", "label": "Network", "color": "#1F8A7A"},
    {"id": "payment", "label": "Payment", "color": "#7B5BB8"},
    {"id": "lifecycle", "label": "Lifecycle", "color": "#8A5C0E"},
    {"id": "governance", "label": "Governance", "color": "#2C3E50"},
]


def _tag(
    tag: str,
    tier: int | str,
    category: str,
    status: str,
    def_: str,
    *,
    aliases: list[str] | None = None,
    deprecates: list[str] | None = None,
    sources_count: int = 0,
    chunks_count: int = 0,
    query_freq_30d: int = 0,
    created: dict[str, str] | None = None,
    last_edit: dict[str, str] | None = None,
    related: list[dict[str, Any]] | None = None,
    examples: list[str] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "tag": tag,
        "tier": tier,
        "category": category,
        "status": status,
        "def": def_,
        "aliases": aliases or [],
        "deprecates": deprecates or [],
        "sources_count": sources_count,
        "chunks_count": chunks_count,
        "query_freq_30d": query_freq_30d,
        "created": created or {"by": "system", "at": "2025-08-01"},
        "last_edit": last_edit or created or {"by": "system", "at": "2025-08-01"},
        "related": related or [],
        "examples": examples or [],
    }
    out.update(extra)
    return out


TAGS: list[dict[str, Any]] = [
    _tag(
        "rman", 1, "oracle", "active",
        "Oracle Recovery Manager — the supported backup and recovery toolchain for Oracle Database. Use for any source dealing with rman backups, archive log management, restore procedures, or PITR.",
        aliases=["recovery-manager"], sources_count=47, chunks_count=1842, query_freq_30d=312,
        created={"by": "claire.benoit", "at": "2025-09-12"},
        last_edit={"by": "claire.benoit", "at": "2026-05-11", "action": "promoted to Tier 1"},
        related=[{"tag": "oracle", "strength": 0.92}, {"tag": "production", "strength": 0.71}, {"tag": "rhel9", "strength": 0.44}],
        examples=["oracle-restart-procedure.pdf", "/cib/runbooks/rman-restore-cookbook"],
    ),
    _tag(
        "oracle", 1, "oracle", "active",
        "Oracle Database engine and ecosystem — covers any source about RDBMS configuration, tuning, RAC, ASM, Data Guard, etc.",
        aliases=["oracle-db", "ora"], sources_count=89, chunks_count=4421, query_freq_30d=504,
        last_edit={"by": "claire.benoit", "at": "2026-03-18", "action": "added alias ora"},
        related=[{"tag": "rman", "strength": 0.92}, {"tag": "rhel9", "strength": 0.61}, {"tag": "memgraph", "strength": 0.12}],
        examples=["oracle-restart-procedure.pdf", "/cib/runbooks/oracle-pga-tuning"],
    ),
    _tag(
        "vmware", 1, "infra", "active",
        "VMware vSphere and ESXi runtime — virtualization platform topics: clusters, vMotion, DRS, storage policies.",
        sources_count=31, chunks_count=982, query_freq_30d=87,
        related=[{"tag": "network", "strength": 0.34}, {"tag": "production", "strength": 0.68}],
        examples=["vmware-best-practices-2026.pdf"],
    ),
    _tag(
        "memgraph", 1, "infra", "active",
        "Memgraph graph database, including MAGE extensions, vector_search procedure, and Cypher dialect specifics.",
        aliases=["mage"], sources_count=8, chunks_count=142, query_freq_30d=41,
        created={"by": "yann.dubois", "at": "2025-11-04"},
        last_edit={"by": "yann.dubois", "at": "2026-02-10", "action": "added alias mage"},
        related=[{"tag": "graphrag", "strength": 0.81}],
        examples=["memgraph-mage-3.8-release-notes.md"],
    ),
    _tag(
        "rhel9", 2, "infra", "active",
        "Red Hat Enterprise Linux 9 — OS-level configuration, kernel tuning, systemd, SELinux specific to RHEL 9.",
        aliases=["redhat-9", "el9"], deprecates=["rhel8"], sources_count=12, chunks_count=287, query_freq_30d=22,
        created={"by": "yann.dubois", "at": "2026-01-22"},
        last_edit={"by": "claire.benoit", "at": "2026-04-15", "action": "approved for Tier 2"},
        related=[{"tag": "oracle", "strength": 0.61}, {"tag": "rman", "strength": 0.44}],
        examples=["rhel9-kernel-tuning.pdf"],
    ),
    _tag(
        "network", 1, "network", "active",
        "Networking topics — routing, firewall rules, load balancers, DNS, TLS/PKI, observability of network flows.",
        sources_count=24, chunks_count=612, query_freq_30d=58,
        related=[{"tag": "cft", "strength": 0.42}, {"tag": "vmware", "strength": 0.34}],
    ),
    _tag(
        "cft", 1, "network", "active",
        "Cross-platform File Transfer (Axway CFT) — partner connectivity, flow definitions, monitoring.",
        aliases=["axway-cft"], sources_count=9, chunks_count=218, query_freq_30d=12,
        created={"by": "philippe.marchand", "at": "2025-12-05"},
        last_edit={"by": "philippe.marchand", "at": "2025-12-05", "action": "created"},
        related=[{"tag": "network", "strength": 0.42}, {"tag": "swift", "strength": 0.21}],
        examples=["cft-network-architecture.docx"],
    ),
    _tag(
        "swift", 1, "payment", "active",
        "SWIFT messaging — MT/MX, FIN, GPI, ISO 20022 migration. Use for any source discussing SWIFT flows or compliance.",
        sources_count=18, chunks_count=542, query_freq_30d=94,
        last_edit={"by": "claire.benoit", "at": "2026-05-10", "action": "tag iso20022 split out"},
        related=[{"tag": "iso20022", "strength": 0.78}, {"tag": "cft", "strength": 0.21}],
        examples=["swift-iso20022-migration.pdf"],
    ),
    _tag(
        "iso20022", 2, "payment", "active",
        'ISO 20022 message standard — pacs, pain, camt families. Distinct from generic "swift" for cases where standard precision matters.',
        sources_count=4, chunks_count=88, query_freq_30d=17,
        created={"by": "claire.benoit", "at": "2026-05-10"},
        last_edit={"by": "claire.benoit", "at": "2026-05-10", "action": "created as Tier 2"},
        related=[{"tag": "swift", "strength": 0.78}],
        examples=["swift-iso20022-migration.pdf"],
    ),
    _tag(
        "production", 1, "lifecycle", "active",
        "Source applies to production environments — exclude from default retrieval if user filters to non-prod context.",
        aliases=["prod"], sources_count=64, chunks_count=2918, query_freq_30d=281,
        related=[{"tag": "oracle", "strength": 0.71}, {"tag": "vmware", "strength": 0.68}],
        examples=["oracle-restart-procedure.pdf", "vmware-best-practices-2026.pdf"],
    ),
    _tag(
        "incident", 1, "lifecycle", "active",
        "Postmortem or active incident material — surfaced with critical visual emphasis to remind users this is high-stakes content.",
        aliases=["postmortem"], sources_count=14, chunks_count=318, query_freq_30d=47,
        related=[{"tag": "production", "strength": 0.62}],
        examples=["/cib/incidents/2026-Q1-postmortems"],
    ),
    _tag(
        "deprecated", 1, "lifecycle", "active",
        "Source is marked for archival and excluded from default retrieval. Apply when superseding content has been ingested.",
        aliases=["archived"], sources_count=7, chunks_count=184, query_freq_30d=3,
        examples=["/cib/runbooks/rman-restore-cookbook"],
    ),
    _tag(
        "pending-review", 1, "lifecycle", "active",
        "Awaiting risk review before being surfaced in unrestricted retrieval. Default-excluded from queries without explicit override.",
        sources_count=3, chunks_count=62, query_freq_30d=1,
        created={"by": "claire.benoit", "at": "2025-10-18"},
        last_edit={"by": "claire.benoit", "at": "2025-10-18", "action": "created"},
        examples=["/cib/governance/tagging-charter"],
    ),
    _tag(
        "governance", 1, "governance", "active",
        "Internal governance, audit, or charter material — policies, RACI, taxonomy definitions.",
        sources_count=11, chunks_count=142, query_freq_30d=8,
        examples=["/cib/governance/tagging-charter"],
    ),
    _tag(
        "rmf-validated", 1, "governance", "active",
        "Material that has been formally reviewed and signed off by the Risk Management Framework board.",
        aliases=["rmf-approved"], sources_count=6, chunks_count=88, query_freq_30d=2,
        created={"by": "claire.benoit", "at": "2025-10-01"},
        last_edit={"by": "claire.benoit", "at": "2025-10-01", "action": "created"},
        related=[{"tag": "governance", "strength": 0.55}],
    ),
    _tag(
        "graphrag", 2, "infra", "pending-promotion",
        "Knowledge graph + RAG hybrid retrieval — covers any source describing graph-enhanced retrieval architectures.",
        aliases=["graph-rag"], sources_count=5, chunks_count=96, query_freq_30d=28,
        created={"by": "yann.dubois", "at": "2026-02-14"},
        last_edit={"by": "yann.dubois", "at": "2026-05-02", "action": "promotion requested"},
        related=[{"tag": "memgraph", "strength": 0.81}],
        examples=["memgraph-mage-3.8-release-notes.md"],
    ),
    _tag(
        "k8s", 2, "infra", "active",
        "Kubernetes — cluster topology, workload manifests, operators, GitOps. Scoped to Platform team usage.",
        aliases=["kubernetes"], sources_count=7, chunks_count=154, query_freq_30d=18,
        created={"by": "yann.dubois", "at": "2026-03-20"},
        last_edit={"by": "yann.dubois", "at": "2026-03-20", "action": "created as Tier 2"},
    ),
    _tag(
        "vault", 3, "infra", "active",
        "HashiCorp Vault — secret management. User-proposed leaf, not yet validated by the Platform tier-2 process.",
        sources_count=2, chunks_count=38, query_freq_30d=4,
        created={"by": "marc.berthier", "at": "2026-04-29"},
        last_edit={"by": "marc.berthier", "at": "2026-04-29", "action": "created as Tier 3"},
    ),
    _tag(
        "ansible", 3, "infra", "active",
        "Ansible playbooks and roles — config management automation. Tier 3 leaf.",
        sources_count=3, chunks_count=71, query_freq_30d=7,
        created={"by": "marc.berthier", "at": "2026-04-12"},
        last_edit={"by": "marc.berthier", "at": "2026-04-12", "action": "created as Tier 3"},
    ),
    _tag(
        "argocd", "requested", "infra", "pending-review",
        "Proposed: ArgoCD — GitOps continuous-delivery controller. Awaiting Tier 3 acceptance.",
        created={"by": "marc.berthier", "at": "2026-05-09"},
        last_edit={"by": "marc.berthier", "at": "2026-05-09", "action": "requested"},
        related=[{"tag": "k8s", "strength": 0.7}],
        requested_by="marc.berthier", requested_at="2026-05-09",
        justification="Used in 4 new sources scheduled for ingestion this sprint.",
    ),
    _tag(
        "pacs008", "requested", "payment", "pending-review",
        "Proposed: pacs.008 — ISO 20022 customer credit transfer message family.",
        created={"by": "philippe.marchand", "at": "2026-05-10"},
        last_edit={"by": "philippe.marchand", "at": "2026-05-10", "action": "requested"},
        related=[{"tag": "iso20022", "strength": 0.85}],
        requested_by="philippe.marchand", requested_at="2026-05-10",
        justification="Granularity needed below iso20022; we have 3 sources discussing pacs.008 specifically.",
    ),
]

# ---------------------------------------------------------------------------
# Activity audit feed (pinned demo "now" so range filters stay deterministic)
# ---------------------------------------------------------------------------

import datetime  # noqa: E402

ACTIVITY_NOW_ISO = "2026-05-11T10:00:00Z"
ACTIVITY_NOW_MS = int(
    datetime.datetime.fromisoformat(ACTIVITY_NOW_ISO.replace("Z", "+00:00")).timestamp()
    * 1000
)


def _evt(id_: str, ts: str, rel: str, day: str, kind: str, sev: str, actor_user: str, actor_role: str, target_type: str, target_label: str, summary: str, meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": id_, "ts": ts, "rel": rel, "day": day, "kind": kind, "sev": sev,
        "actor": {"user": actor_user, "role": actor_role},
        "target": {"type": target_type, "label": target_label},
        "summary": summary, "meta": meta,
    }


ACTIVITY: list[dict[str, Any]] = [
    _evt("evt_01HX9Z7Q", "2026-05-11T09:42:18Z", "12m ago", "Today", "retrieval", "info", "marc.berthier", "DBA", "query", "How to restart Oracle on RHEL 9?",
         "Retrieval · hybrid · top_k=60 · 5 sources cited · 1.4s",
         {"mode": "hybrid", "top_k": 60, "tag_filter": ["rman", "oracle"], "latency_ms": 1412, "tokens_in": 312, "tokens_out": 488}),
    _evt("evt_01HX9Z7M", "2026-05-11T09:35:02Z", "19m ago", "Today", "tag-mutation", "info", "claire.benoit", "KB Admin", "source", "oracle-restart-procedure.pdf",
         "Added tag rman · removed tag pending-review · 418 chunks re-tagged",
         {"added": ["rman"], "removed": ["pending-review"], "chunks_affected": 418, "propagation_ms": 1830}),
    _evt("evt_01HX9Z7K", "2026-05-11T09:29:51Z", "25m ago", "Today", "source-ready", "info", "system", "pipeline", "source", "memgraph-mage-3.8-release-notes.md",
         "Indexing completed · 42 chunks · 3 entities · 7 relations",
         {"chunks": 42, "entities": 3, "relations": 7, "duration_ms": 8421, "embed_model": "text-embedding-3-large"}),
    _evt("evt_01HX9Z7H", "2026-05-11T09:24:11Z", "30m ago", "Today", "source-failed", "error", "claire.benoit", "KB Admin", "source", "huge-archive.zip",
         "Ingestion rejected — unsupported MIME type & 312 MB > 50 MB limit",
         {"mime": "application/zip", "size_bytes": 326962482, "error_code": "E_UNSUPPORTED_FORMAT"}),
    _evt("evt_01HX9Z7B", "2026-05-11T09:11:33Z", "44m ago", "Today", "retrieval", "info", "yann.dubois", "SRE", "query", "RMAN restore point-in-time procedure",
         "Retrieval · local · top_k=40 · 8 sources cited · 0.9s",
         {"mode": "local", "top_k": 40, "tag_filter": ["rman"], "latency_ms": 902, "tokens_in": 198, "tokens_out": 612}),
    _evt("evt_01HX9Z76", "2026-05-11T08:58:02Z", "57m ago", "Today", "pipeline-warning", "warning", "system", "pipeline", "source", "cib-incidents-2026-Q1-postmortems",
         "LLM extraction timeout on chunk 78/124 · retrying (attempt 2/3)",
         {"provider": "openai", "model": "gpt-4o-mini", "chunk_index": 78, "attempt": 2, "error": "ReadTimeout(60s)"}),
    _evt("evt_01HX9Z71", "2026-05-11T08:42:19Z", "1h ago", "Today", "source-uploaded", "info", "claire.benoit", "KB Admin", "source", "swift-iso20022-migration.pdf",
         "Uploaded · 14.2 MB · queued for ingestion",
         {"size_bytes": 14883941, "mime": "application/pdf", "batch_tags": ["swift", "production"]}),
    _evt("evt_01HX9Z6Y", "2026-05-11T08:30:00Z", "1h ago", "Today", "auth", "info", "marc.berthier", "DBA", "session", "Bearer token issued",
         "Login successful · scope: read:documents read:query · expires in 24h",
         {"ip": "10.42.7.118", "user_agent": "twin-cli/0.4.1", "scope": "read:documents read:query", "ttl_h": 24}),
    _evt("evt_01HX8VK3", "2026-05-10T17:14:08Z", "16h ago", "Yesterday", "tag-mutation", "info", "claire.benoit", "KB Admin", "bulk", "Bulk retag · 9 sources",
         "Replaced tag rman-archived → rman across 9 sources · 847 chunks",
         {"added": ["rman"], "removed": ["rman-archived"], "sources_count": 9, "chunks_affected": 847}),
    _evt("evt_01HX8V2P", "2026-05-10T15:02:51Z", "18h ago", "Yesterday", "retrieval", "info", "philippe.marchand", "Architect", "query", "VMware vSphere 8 best practices for banking workloads",
         "Retrieval · global · top_k=80 · 12 sources cited · 2.1s",
         {"mode": "global", "top_k": 80, "tag_filter": ["vmware", "production"], "latency_ms": 2104, "tokens_in": 278, "tokens_out": 1024}),
    _evt("evt_01HX8TWA", "2026-05-10T11:48:00Z", "22h ago", "Yesterday", "settings", "info", "claire.benoit", "KB Admin", "workspace", "cib · thesaurus",
         "Added new tag iso20022 (palier 2) · category payment",
         {"tag": "iso20022", "category": "payment", "tier": 2, "requested_by": "marc.berthier"}),
    _evt("evt_01HX8T03", "2026-05-10T09:32:12Z", "1d ago", "Yesterday", "source-failed", "error", "system", "pipeline", "source", "cft-vendor-api-spec.pdf",
         "Embedding provider returned 429 after 3 retries · source marked failed",
         {"provider": "openai", "model": "text-embedding-3-large", "error_code": "E_PROVIDER_RATE_LIMIT", "retries": 3}),
    _evt("evt_01HX5LM2", "2026-05-08T16:21:09Z", "3d ago", "Earlier this week", "source-ready", "info", "system", "pipeline", "source", "vmware-best-practices-2026.pdf",
         "Indexing completed · 281 chunks · 47 entities · 92 relations",
         {"chunks": 281, "entities": 47, "relations": 92, "duration_ms": 54213, "embed_model": "text-embedding-3-large"}),
    _evt("evt_01HX5KGH", "2026-05-08T10:08:44Z", "3d ago", "Earlier this week", "auth", "warning", "external.client@partner.com", "external", "session", "Login attempt rejected",
         "401 Unauthorized · IP not in allow-list · 10.214.x.x",
         {"ip": "10.214.99.4", "error_code": "E_IP_NOT_ALLOWLISTED", "attempts_24h": 7}),
    _evt("evt_01HX4XQT", "2026-05-07T14:55:18Z", "4d ago", "Earlier this week", "tag-mutation", "warning", "claire.benoit", "KB Admin", "source", "rman-restore-cookbook",
         "Added tag deprecated · 132 chunks marked excluded from default retrieval",
         {"added": ["deprecated"], "chunks_affected": 132, "excluded_from_default": True}),
    _evt("evt_01HX4WJM", "2026-05-07T11:22:01Z", "4d ago", "Earlier this week", "retrieval", "info", "marc.berthier", "DBA", "query", "Oracle PGA tuning for OLTP workload",
         "Retrieval · hybrid · top_k=60 · 6 sources cited · 1.1s",
         {"mode": "hybrid", "top_k": 60, "tag_filter": ["oracle"], "latency_ms": 1132}),
]

# ---------------------------------------------------------------------------
# OpenAPI surface (curated view of the underlying LightRAG endpoints)
# ---------------------------------------------------------------------------

OPENAPI_VERSION = "v1.4.12/0279"

OPENAPI_GROUPS: list[dict[str, Any]] = [
    {
        "id": "documents", "name": "documents", "desc": "Source ingestion, listing and lifecycle.",
        "endpoints": [
            {"m": "POST", "p": "/documents/upload", "s": "Upload Document"},
            {"m": "POST", "p": "/documents/text", "s": "Insert Text"},
            {"m": "POST", "p": "/documents/texts", "s": "Insert Texts"},
            {"m": "POST", "p": "/documents/scan", "s": "Scan For New Documents"},
            {"m": "GET", "p": "/documents", "s": "List Documents"},
            {"m": "GET", "p": "/documents/pipeline_status", "s": "Get Pipeline Status"},
            {"m": "DELETE", "p": "/documents", "s": "Clear Documents"},
            {"m": "DELETE", "p": "/documents/delete_document", "s": "Delete Document"},
            {"m": "POST", "p": "/documents/clear_cache", "s": "Clear Cache"},
        ],
    },
    {
        "id": "query", "name": "query", "desc": "Retrieval + LLM synthesis endpoints.",
        "endpoints": [
            {"m": "POST", "p": "/query", "s": "Query Text"},
            {"m": "POST", "p": "/query/stream", "s": "Query Text Stream"},
        ],
    },
    {
        "id": "graph", "name": "graph", "desc": "Knowledge-graph CRUD and label browsing.",
        "endpoints": [
            {"m": "GET", "p": "/graph/label/list", "s": "Get Graph Labels"},
            {"m": "GET", "p": "/graph/label/popular", "s": "Get Popular Labels"},
            {"m": "GET", "p": "/graph/label/search", "s": "Search Labels"},
            {"m": "GET", "p": "/graphs", "s": "Get Knowledge Graph"},
            {"m": "GET", "p": "/graph/entity/exists", "s": "Check Entity Exists"},
            {"m": "POST", "p": "/graph/entity/edit", "s": "Update Entity"},
            {"m": "POST", "p": "/graph/relation/edit", "s": "Update Relation"},
            {"m": "POST", "p": "/graph/entity/create", "s": "Create Entity"},
            {"m": "POST", "p": "/graph/relation/create", "s": "Create Relation"},
        ],
    },
    {
        "id": "ollama", "name": "ollama", "desc": "Drop-in Ollama-compatible chat & generate surface.",
        "endpoints": [
            {"m": "GET", "p": "/api/version", "s": "Get Version"},
            {"m": "GET", "p": "/api/tags", "s": "Get Tags"},
            {"m": "GET", "p": "/api/ps", "s": "Get Running Models"},
            {"m": "POST", "p": "/api/generate", "s": "Generate"},
            {"m": "POST", "p": "/api/chat", "s": "Chat"},
        ],
    },
    {
        "id": "default", "name": "default", "desc": "Auth, health and root.",
        "endpoints": [
            {"m": "GET", "p": "/", "s": "Redirect To Webui"},
            {"m": "GET", "p": "/auth-status", "s": "Get Auth Status"},
            {"m": "POST", "p": "/login", "s": "Login"},
            {"m": "GET", "p": "/health", "s": "Get system health and configuration status"},
        ],
    },
]

# ---------------------------------------------------------------------------
# Knowledge graph teaser — entities + relations
# ---------------------------------------------------------------------------

GRAPH_ENTITIES: list[dict[str, Any]] = [
    {"id": "e_oracle", "name": "Oracle Database", "type": "PRODUCT", "x": 240, "y": 200, "mentions": 412, "sources": 47, "summary": "Relational database engine; primary OLTP backing store for CIB workloads."},
    {"id": "e_rman", "name": "RMAN", "type": "TECHNOLOGY", "x": 130, "y": 290, "mentions": 318, "sources": 31, "summary": "Oracle Recovery Manager — supported backup/restore toolchain."},
    {"id": "e_archlog", "name": "Archive Log", "type": "CONCEPT", "x": 80, "y": 160, "mentions": 142, "sources": 24, "summary": "Redo log archive used for PITR and standby replication."},
    {"id": "e_rhel", "name": "RHEL 9", "type": "PRODUCT", "x": 340, "y": 320, "mentions": 198, "sources": 38, "summary": "Red Hat Enterprise Linux 9 — certified OS for Oracle 19c+."},
    {"id": "e_pga", "name": "PGA tuning", "type": "CONCEPT", "x": 380, "y": 130, "mentions": 64, "sources": 9, "summary": "Program Global Area sizing for OLTP workload concurrency."},
    {"id": "e_vmware", "name": "VMware vSphere 8", "type": "PRODUCT", "x": 540, "y": 240, "mentions": 287, "sources": 22, "summary": "Hypervisor stack; banking-grade configuration baseline."},
    {"id": "e_esxi", "name": "ESXi host", "type": "PRODUCT", "x": 640, "y": 320, "mentions": 122, "sources": 14, "summary": "Bare-metal hypervisor node."},
    {"id": "e_vmotion", "name": "vMotion", "type": "TECHNOLOGY", "x": 700, "y": 200, "mentions": 58, "sources": 9, "summary": "Live migration of running VMs across ESXi hosts."},
    {"id": "e_memgraph", "name": "Memgraph", "type": "PRODUCT", "x": 470, "y": 480, "mentions": 156, "sources": 19, "summary": "Graph DB backing LightRAG entity/relation storage."},
    {"id": "e_mage", "name": "MAGE 3.8", "type": "PRODUCT", "x": 560, "y": 560, "mentions": 84, "sources": 7, "summary": "Memgraph Algorithm Extensions Engine — vector_search modules."},
    {"id": "e_cypher", "name": "Cypher", "type": "TECHNOLOGY", "x": 360, "y": 540, "mentions": 109, "sources": 12, "summary": "Graph query language used for pre-filter retrieval."},
    {"id": "e_lightrag", "name": "LightRAG", "type": "PRODUCT", "x": 250, "y": 460, "mentions": 274, "sources": 28, "summary": "Open-source retrieval framework forked into Twin RAG."},
    {"id": "e_swift", "name": "SWIFT", "type": "ORG", "x": 820, "y": 130, "mentions": 198, "sources": 17, "summary": "Society for Worldwide Interbank Financial Telecommunication."},
    {"id": "e_iso20022", "name": "ISO 20022", "type": "CONCEPT", "x": 880, "y": 230, "mentions": 142, "sources": 14, "summary": "XML messaging standard for financial transactions; SWIFT migration target."},
    {"id": "e_cft", "name": "CFT", "type": "PRODUCT", "x": 780, "y": 330, "mentions": 92, "sources": 11, "summary": "Cross File Transfer middleware — Axway product."},
    {"id": "e_marc", "name": "Marc Berthier", "type": "PERSON", "x": 100, "y": 420, "mentions": 28, "sources": 12, "summary": "DBA — primary author on Oracle restart procedures."},
    {"id": "e_claire", "name": "Claire Benoit", "type": "PERSON", "x": 160, "y": 580, "mentions": 41, "sources": 18, "summary": "KB Admin / Tier 3 steward for CIB workspace."},
    {"id": "e_paris", "name": "DC Paris", "type": "LOCATION", "x": 700, "y": 440, "mentions": 37, "sources": 8, "summary": "Primary datacenter; active site of the dual-DC topology."},
    {"id": "e_aubervil", "name": "DC Aubervilliers", "type": "LOCATION", "x": 820, "y": 520, "mentions": 31, "sources": 7, "summary": "Secondary datacenter; standby site."},
]

GRAPH_RELATIONS: list[dict[str, Any]] = [
    {"id": "r_01", "source": "e_rman", "target": "e_oracle", "label": "BACKS_UP", "strength": 0.92},
    {"id": "r_02", "source": "e_rman", "target": "e_archlog", "label": "MANAGES", "strength": 0.74},
    {"id": "r_03", "source": "e_oracle", "target": "e_rhel", "label": "RUNS_ON", "strength": 0.88},
    {"id": "r_04", "source": "e_oracle", "target": "e_pga", "label": "TUNED_VIA", "strength": 0.61},
    {"id": "r_05", "source": "e_archlog", "target": "e_oracle", "label": "GENERATED_BY", "strength": 0.55},
    {"id": "r_06", "source": "e_esxi", "target": "e_vmware", "label": "PART_OF", "strength": 0.90},
    {"id": "r_07", "source": "e_vmotion", "target": "e_vmware", "label": "FEATURE_OF", "strength": 0.78},
    {"id": "r_08", "source": "e_oracle", "target": "e_vmware", "label": "HOSTED_ON", "strength": 0.66},
    {"id": "r_09", "source": "e_esxi", "target": "e_paris", "label": "DEPLOYED_AT", "strength": 0.70},
    {"id": "r_10", "source": "e_esxi", "target": "e_aubervil", "label": "DEPLOYED_AT", "strength": 0.62},
    {"id": "r_11", "source": "e_lightrag", "target": "e_memgraph", "label": "USES", "strength": 0.89},
    {"id": "r_12", "source": "e_lightrag", "target": "e_cypher", "label": "QUERIES_WITH", "strength": 0.71},
    {"id": "r_13", "source": "e_memgraph", "target": "e_mage", "label": "EXTENDED_BY", "strength": 0.83},
    {"id": "r_14", "source": "e_memgraph", "target": "e_cypher", "label": "SPEAKS", "strength": 0.80},
    {"id": "r_15", "source": "e_swift", "target": "e_iso20022", "label": "MIGRATING_TO", "strength": 0.85},
    {"id": "r_16", "source": "e_cft", "target": "e_swift", "label": "TRANSPORTS_FOR", "strength": 0.64},
    {"id": "r_17", "source": "e_iso20022", "target": "e_oracle", "label": "PERSISTED_IN", "strength": 0.42},
    {"id": "r_18", "source": "e_marc", "target": "e_oracle", "label": "AUTHORED_ON", "strength": 0.79},
    {"id": "r_19", "source": "e_marc", "target": "e_rman", "label": "AUTHORED_ON", "strength": 0.82},
    {"id": "r_20", "source": "e_claire", "target": "e_lightrag", "label": "ADMINISTERS", "strength": 0.68},
    {"id": "r_21", "source": "e_claire", "target": "e_rman", "label": "TAGGED", "strength": 0.55},
]
