// Mock data for the Twin RAG prototype

window.MOCK_DOCUMENTS = [
  {
    id: "d1",
    type: "file",
    source: "oracle-restart-procedure.pdf",
    summary: "Step-by-step guide for restarting Oracle DB on RHEL 9 in CIB prod",
    tags: ["rman", "oracle"],
    status: "completed",
    chunks: 418,
    updated: "2h ago",
    visibility: "private",
    workspace: "cib"
  },
  {
    id: "d2",
    type: "confluence",
    source: "/cib/runbooks/oracle-pga-tuning",
    summary: "Oracle PGA memory tuning recommendations and worked examples",
    tags: ["rman"],
    status: "completed",
    chunks: 54,
    updated: "1d ago",
    visibility: "private",
    workspace: "cib"
  },
  {
    id: "d3",
    type: "sharepoint",
    source: "/cib/incidents/2026-Q1-postmortems",
    summary: "Quarterly retro of Q1 incidents — root causes and remediation actions",
    tags: ["incident"],
    status: "processing",
    chunks: null,
    updated: "5m ago",
    visibility: "private",
    workspace: "cib"
  },
  {
    id: "d4",
    type: "file",
    source: "vmware-best-practices-2026.pdf",
    summary: "VMware vSphere 8 best practices for banking workloads",
    tags: ["vmware", "production"],
    status: "completed",
    chunks: 281,
    updated: "3d ago",
    visibility: "private",
    workspace: "cib"
  },
  {
    id: "d5",
    type: "confluence",
    source: "/cib/runbooks/rman-restore-cookbook",
    summary: "RMAN restore scenarios from point-in-time to tablespace recovery",
    tags: ["rman", "oracle", "deprecated"],
    status: "completed",
    chunks: 132,
    updated: "1w ago",
    visibility: "private",
    workspace: "cib"
  },
  {
    id: "d6",
    type: "file",
    source: "cft-network-architecture.docx",
    summary: "CFT cross-platform file-transfer network architecture and flows",
    tags: ["cft", "network"],
    status: "completed",
    chunks: 92,
    updated: "2w ago",
    visibility: "private",
    workspace: "cib"
  },
  {
    id: "d7",
    type: "file",
    source: "huge-archive.zip",
    summary: "Exceeds 50 MB · unsupported type",
    tags: [],
    status: "failed",
    chunks: null,
    updated: "10m ago",
    visibility: "private",
    workspace: "cib"
  },
  {
    id: "d8",
    type: "sharepoint",
    source: "/cib/onboarding/dba-handbook",
    summary: "Onboarding handbook for new DBA team members — environments and tools",
    tags: ["governance"],
    status: "pending",
    chunks: null,
    updated: "30s ago",
    visibility: "private",
    workspace: "cib"
  },
  {
    id: "d9",
    type: "file",
    source: "memgraph-mage-3.8-release-notes.md",
    summary: "Memgraph MAGE 3.8 release — vector_search behavior changes and bug fixes",
    tags: ["memgraph", "production"],
    status: "completed",
    chunks: 42,
    updated: "4d ago",
    visibility: "private",
    workspace: "cib"
  },
  {
    id: "d10",
    type: "confluence",
    source: "/cib/governance/tagging-charter",
    summary: "Tagging charter — Reader/Contributor/Steward governance for the Twin thesaurus",
    tags: ["governance", "pending-review"],
    status: "completed",
    chunks: 21,
    updated: "5d ago",
    visibility: "private",
    workspace: "cib"
  },
  {
    id: "d11",
    type: "file",
    source: "rhel9-kernel-tuning.pdf",
    summary: "RHEL 9 kernel parameter tuning for high-IO Oracle workloads",
    tags: [],
    status: "completed",
    chunks: 67,
    updated: "1w ago",
    visibility: "private",
    workspace: "cib"
  },
  {
    id: "d12",
    type: "file",
    source: "swift-iso20022-migration.pdf",
    summary: "SWIFT MX migration plan and impact analysis for payment flows",
    tags: ["swift", "production"],
    status: "completed",
    chunks: 184,
    updated: "2w ago",
    visibility: "private",
    workspace: "cib"
  },
  // ─── Pending review (palier-3 validation queue) ─────────────────────
  {
    id: "d13",
    type: "file",
    source: "cft-vendor-api-spec-draft.pdf",
    summary: "Axway CFT vendor API specification — third-party submission, pending steward validation",
    tags: ["cft", "network"],
    status: "completed",
    chunks: 47,
    updated: "12m ago",
    visibility: "private",
    workspace: "cib",
    review: {
      state: "pending-review",
      requested_by: "marc.berthier",
      requested_at: "2026-05-20",
      justification: "Vendor-provided spec — needs sign-off by a steward before retrieval. Confidence sourcing uncertain."
    }
  },
  {
    id: "d14",
    type: "confluence",
    source: "/cib/runbooks/incident-2026-Q2-postmortem-draft",
    summary: "Q2 2026 incident postmortem draft — sensitive content, awaiting steward approval",
    tags: ["incident", "production"],
    status: "completed",
    chunks: 89,
    updated: "1h ago",
    visibility: "private",
    workspace: "cib",
    review: {
      state: "pending-review",
      requested_by: "yann.dubois",
      requested_at: "2026-05-20",
      justification: "Contains client-impact figures — steward review required before exposure to broader retrieval."
    }
  }
];

window.MOCK_TAG_CATEGORY = {
  rman: "oracle",
  oracle: "oracle",
  vmware: "infra",
  production: "lifecycle",
  incident: "lifecycle",
  cft: "network",
  network: "network",
  memgraph: "infra",
  swift: "payment",
  governance: "governance",
  deprecated: "lifecycle",
  "pending-review": "lifecycle",
  "rmf-validated": "governance",
  "rman-archived": "oracle"
};

window.MOCK_TAG_SEMANTICS = {
  incident: "critical",
  "pending-review": "warning",
  deprecated: "warning"
};

window.MOCK_THESAURUS = [
  { tag: "rman", category: "oracle", def: "Recovery Manager — Oracle backup and recovery tool" },
  { tag: "rmf-validated", category: "governance", def: "Reviewed by risk management framework" },
  { tag: "oracle", category: "oracle", def: "Oracle Database engine and ecosystem" },
  { tag: "production", category: "lifecycle", def: "Document applies to production environments" },
  { tag: "incident", category: "lifecycle", def: "Postmortem or active incident material" },
  { tag: "vmware", category: "infra", def: "VMware vSphere and ESXi runtime" },
  { tag: "memgraph", category: "infra", def: "Memgraph graph database (MAGE extensions)" },
  { tag: "network", category: "network", def: "Networking, routing, firewall topics" },
  { tag: "cft", category: "network", def: "Cross-platform file transfer (Axway CFT)" },
  { tag: "swift", category: "payment", def: "SWIFT messaging and payment flows" },
  { tag: "governance", category: "governance", def: "Internal governance, audit, or charter" },
  { tag: "deprecated", category: "lifecycle", def: "Marked for archival, do not act on" },
  { tag: "pending-review", category: "lifecycle", def: "Awaiting risk review before publish" }
];

window.MOCK_RETRIEVAL_SOURCES = [
  { n: 1, type: "file", name: "oracle-restart-procedure.pdf", meta: "p.4", score: 0.94 },
  { n: 2, type: "file", name: "oracle-restart-procedure.pdf", meta: "p.7", score: 0.89 },
  { n: 3, type: "confluence", name: "/cib/runbooks/oracle-pga-tuning", meta: null, score: 0.82 },
  { n: 4, type: "confluence", name: "/cib/runbooks/rman-restore-cookbook", meta: null, score: 0.78 },
  { n: 5, type: "file", name: "rhel9-kernel-tuning.pdf", meta: "p.12", score: 0.71 }
];

// Streamed answer with inline citation markers — split on [n]
window.MOCK_ANSWER_TOKENS = [
  "To restart Oracle on RHEL 9 in a CIB production setup, follow the documented runbook ",
  "{cite:1} ",
  "which mandates first stopping listeners and pending RMAN jobs before issuing a graceful ",
  "`shutdown immediate` ",
  "from SQL*Plus. ",
  "After the database is down, verify the PGA configuration matches the tuning recommendations ",
  "{cite:3} ",
  "— this avoids the memory contention seen in incidents Q4 2025. ",
  "Restart the listener (`lsnrctl start`), then bring the DB back up with `startup`, ",
  "and finally re-enable RMAN backup jobs. ",
  "If a restore is required during this maintenance window, prefer point-in-time recovery as described in the RMAN cookbook ",
  "{cite:4} ",
  "rather than a full database restore. ",
  "Kernel parameters (`vm.swappiness`, `vm.dirty_ratio`) should already match the RHEL 9 hardening profile ",
  "{cite:5}",
  "."
];

window.MOCK_FORMAT_CATEGORIES = [
  { cat: "Documents", fmts: "TXT MD MDX DOCX PDF PPTX XLSX RTF ODT EPUB" },
  { cat: "Markup", fmts: "HTML HTM TEX" },
  { cat: "Data", fmts: "JSON XML YAML YML CSV LOG CONF INI PROPERTIES SQL" },
  { cat: "Code", fmts: "SH BAT C H CPP HPP PY JAVA JS TS SWIFT GO RB PHP CSS SCSS LESS" },
  { cat: "Images", fmts: "PNG JPG SVG · coming soon", future: true }
];

// ===== Activity events =====
window.MOCK_ACTIVITY = [
  // ---- Today ----
  { id: "evt_01HX9ZRV1", ts: "2026-05-11T09:55:12Z", rel: "2m ago", day: "Today",
    kind: "doc-review", sev: "info", actor: { user: "claire.benoit", role: "KB Admin" },
    target: { type: "source", label: "swift-iso20022-migration.pdf" },
    summary: "Document approved · entered active retrieval set",
    meta: { doc_id: "d12", from_state: "pending-review", to_state: "approved", review_duration_s: 1820 } },

  { id: "evt_01HX9ZRV2", ts: "2026-05-11T09:48:47Z", rel: "7m ago", day: "Today",
    kind: "doc-review", sev: "warning", actor: { user: "claire.benoit", role: "KB Admin" },
    target: { type: "source", label: "outdated-runbook-v2-draft.pdf" },
    summary: "Document rejected · superseded by v3 already in retrieval",
    meta: { doc_id: "d_legacy_42", from_state: "pending-review", to_state: "rejected", reason: "superseded by v3" } },

  { id: "evt_01HX9Z7Q", ts: "2026-05-11T09:42:18Z", rel: "12m ago", day: "Today",
    kind: "retrieval", sev: "info", actor: { user: "marc.berthier", role: "DBA" },
    target: { type: "query", label: "How to restart Oracle on RHEL 9?" },
    summary: "Retrieval · hybrid · top_k=60 · 5 sources cited · 1.4s",
    meta: { mode: "hybrid", top_k: 60, tag_filter: ["rman", "oracle"], latency_ms: 1412, tokens_in: 312, tokens_out: 488 } },

  { id: "evt_01HX9Z7M", ts: "2026-05-11T09:35:02Z", rel: "19m ago", day: "Today",
    kind: "tag-mutation", sev: "info", actor: { user: "claire.benoit", role: "KB Admin" },
    target: { type: "source", label: "oracle-restart-procedure.pdf" },
    summary: "Added tag rman · removed tag pending-review · 418 chunks re-tagged",
    meta: { added: ["rman"], removed: ["pending-review"], chunks_affected: 418, propagation_ms: 1830 } },

  { id: "evt_01HX9Z7K", ts: "2026-05-11T09:29:51Z", rel: "25m ago", day: "Today",
    kind: "source-ready", sev: "info", actor: { user: "system", role: "pipeline" },
    target: { type: "source", label: "memgraph-mage-3.8-release-notes.md" },
    summary: "Indexing completed · 42 chunks · 3 entities · 7 relations",
    meta: { chunks: 42, entities: 3, relations: 7, duration_ms: 8421, embed_model: "text-embedding-3-large" } },

  { id: "evt_01HX9Z7H", ts: "2026-05-11T09:24:11Z", rel: "30m ago", day: "Today",
    kind: "source-failed", sev: "error", actor: { user: "claire.benoit", role: "KB Admin" },
    target: { type: "source", label: "huge-archive.zip" },
    summary: "Ingestion rejected — unsupported MIME type & 312 MB > 50 MB limit",
    meta: { mime: "application/zip", size_bytes: 326962482, error_code: "E_UNSUPPORTED_FORMAT" } },

  { id: "evt_01HX9Z7B", ts: "2026-05-11T09:11:33Z", rel: "44m ago", day: "Today",
    kind: "retrieval", sev: "info", actor: { user: "yann.dubois", role: "SRE" },
    target: { type: "query", label: "RMAN restore point-in-time procedure" },
    summary: "Retrieval · local · top_k=40 · 8 sources cited · 0.9s",
    meta: { mode: "local", top_k: 40, tag_filter: ["rman"], latency_ms: 902, tokens_in: 198, tokens_out: 612 } },

  { id: "evt_01HX9Z76", ts: "2026-05-11T08:58:02Z", rel: "57m ago", day: "Today",
    kind: "pipeline-warning", sev: "warning", actor: { user: "system", role: "pipeline" },
    target: { type: "source", label: "cib-incidents-2026-Q1-postmortems" },
    summary: "LLM extraction timeout on chunk 78/124 · retrying (attempt 2/3)",
    meta: { provider: "openai", model: "gpt-4o-mini", chunk_index: 78, attempt: 2, error: "ReadTimeout(60s)" } },

  { id: "evt_01HX9Z71", ts: "2026-05-11T08:42:19Z", rel: "1h ago", day: "Today",
    kind: "source-uploaded", sev: "info", actor: { user: "claire.benoit", role: "KB Admin" },
    target: { type: "source", label: "swift-iso20022-migration.pdf" },
    summary: "Uploaded · 14.2 MB · queued for ingestion",
    meta: { size_bytes: 14883941, mime: "application/pdf", batch_tags: ["swift", "production"] } },

  { id: "evt_01HX9Z6Y", ts: "2026-05-11T08:30:00Z", rel: "1h ago", day: "Today",
    kind: "auth", sev: "info", actor: { user: "marc.berthier", role: "DBA" },
    target: { type: "session", label: "Bearer token issued" },
    summary: "Login successful · scope: read:documents read:query · expires in 24h",
    meta: { ip: "10.42.7.118", user_agent: "twin-cli/0.4.1", scope: "read:documents read:query", ttl_h: 24 } },

  // ---- Yesterday ----
  { id: "evt_01HX8VK3", ts: "2026-05-10T17:14:08Z", rel: "16h ago", day: "Yesterday",
    kind: "tag-mutation", sev: "info", actor: { user: "claire.benoit", role: "KB Admin" },
    target: { type: "bulk", label: "Bulk retag · 9 sources" },
    summary: "Replaced tag rman-archived → rman across 9 sources · 847 chunks",
    meta: { added: ["rman"], removed: ["rman-archived"], sources_count: 9, chunks_affected: 847 } },

  { id: "evt_01HX8V2P", ts: "2026-05-10T15:02:51Z", rel: "18h ago", day: "Yesterday",
    kind: "retrieval", sev: "info", actor: { user: "philippe.marchand", role: "Architect" },
    target: { type: "query", label: "VMware vSphere 8 best practices for banking workloads" },
    summary: "Retrieval · global · top_k=80 · 12 sources cited · 2.1s",
    meta: { mode: "global", top_k: 80, tag_filter: ["vmware", "production"], latency_ms: 2104, tokens_in: 278, tokens_out: 1024 } },

  { id: "evt_01HX8TWA", ts: "2026-05-10T11:48:00Z", rel: "22h ago", day: "Yesterday",
    kind: "settings", sev: "info", actor: { user: "claire.benoit", role: "KB Admin" },
    target: { type: "workspace", label: "cib · thesaurus" },
    summary: "Added new tag iso20022 (Contributor) · category payment",
    meta: { tag: "iso20022", category: "payment", tier: 2, requested_by: "marc.berthier" } },

  { id: "evt_01HX8T03", ts: "2026-05-10T09:32:12Z", rel: "1d ago", day: "Yesterday",
    kind: "source-failed", sev: "error", actor: { user: "system", role: "pipeline" },
    target: { type: "source", label: "cft-vendor-api-spec.pdf" },
    summary: "Embedding provider returned 429 after 3 retries · source marked failed",
    meta: { provider: "openai", model: "text-embedding-3-large", error_code: "E_PROVIDER_RATE_LIMIT", retries: 3 } },

  // ---- Earlier this week ----
  { id: "evt_01HX5LM2", ts: "2026-05-08T16:21:09Z", rel: "3d ago", day: "Earlier this week",
    kind: "source-ready", sev: "info", actor: { user: "system", role: "pipeline" },
    target: { type: "source", label: "vmware-best-practices-2026.pdf" },
    summary: "Indexing completed · 281 chunks · 47 entities · 92 relations",
    meta: { chunks: 281, entities: 47, relations: 92, duration_ms: 54213, embed_model: "text-embedding-3-large" } },

  { id: "evt_01HX5KGH", ts: "2026-05-08T10:08:44Z", rel: "3d ago", day: "Earlier this week",
    kind: "auth", sev: "warning", actor: { user: "external.client@partner.com", role: "external" },
    target: { type: "session", label: "Login attempt rejected" },
    summary: "401 Unauthorized · IP not in allow-list · 10.214.x.x",
    meta: { ip: "10.214.99.4", error_code: "E_IP_NOT_ALLOWLISTED", attempts_24h: 7 } },

  { id: "evt_01HX4XQT", ts: "2026-05-07T14:55:18Z", rel: "4d ago", day: "Earlier this week",
    kind: "tag-mutation", sev: "warning", actor: { user: "claire.benoit", role: "KB Admin" },
    target: { type: "source", label: "rman-restore-cookbook" },
    summary: "Added tag deprecated · 132 chunks marked excluded from default retrieval",
    meta: { added: ["deprecated"], chunks_affected: 132, excluded_from_default: true } },

  { id: "evt_01HX4WJM", ts: "2026-05-07T11:22:01Z", rel: "4d ago", day: "Earlier this week",
    kind: "retrieval", sev: "info", actor: { user: "marc.berthier", role: "DBA" },
    target: { type: "query", label: "Oracle PGA tuning for OLTP workload" },
    summary: "Retrieval · hybrid · top_k=60 · 6 sources cited · 1.1s",
    meta: { mode: "hybrid", top_k: 60, tag_filter: ["oracle"], latency_ms: 1132 } }
];

// ===== Tags / Thesaurus governance =====
// tier: 1 (Trunk, gov-validated), 2 (Branch, dept-scoped), 3 (Leaf, user-proposed), "requested" (pending review)
window.MOCK_TAGS_FULL = [
  { tag: "rman", tier: 1, category: "oracle", status: "active",
    def: "Oracle Recovery Manager — the supported backup and recovery toolchain for Oracle Database. Use for any source dealing with rman backups, archive log management, restore procedures, or PITR.",
    aliases: ["recovery-manager"], deprecates: [],
    sources_count: 47, chunks_count: 1842, query_freq_30d: 312,
    created: { by: "claire.benoit", at: "2025-09-12" },
    last_edit: { by: "claire.benoit", at: "2026-05-11", action: "promoted to Tier 1" },
    related: [{ tag: "oracle", strength: 0.92 }, { tag: "production", strength: 0.71 }, { tag: "rhel9", strength: 0.44 }],
    examples: ["oracle-restart-procedure.pdf", "/cib/runbooks/rman-restore-cookbook"]
  },
  { tag: "oracle", tier: 1, category: "oracle", status: "active",
    def: "Oracle Database engine and ecosystem — covers any source about RDBMS configuration, tuning, RAC, ASM, Data Guard, etc.",
    aliases: ["oracle-db", "ora"], deprecates: [],
    sources_count: 89, chunks_count: 4421, query_freq_30d: 504,
    created: { by: "system", at: "2025-08-01" },
    last_edit: { by: "claire.benoit", at: "2026-03-18", action: "added alias ora" },
    related: [{ tag: "rman", strength: 0.92 }, { tag: "rhel9", strength: 0.61 }, { tag: "memgraph", strength: 0.12 }],
    examples: ["oracle-restart-procedure.pdf", "/cib/runbooks/oracle-pga-tuning"]
  },
  { tag: "vmware", tier: 1, category: "infra", status: "active",
    def: "VMware vSphere and ESXi runtime — virtualization platform topics: clusters, vMotion, DRS, storage policies.",
    aliases: [], deprecates: [],
    sources_count: 31, chunks_count: 982, query_freq_30d: 87,
    created: { by: "system", at: "2025-08-01" },
    last_edit: { by: "system", at: "2025-08-01", action: "created" },
    related: [{ tag: "network", strength: 0.34 }, { tag: "production", strength: 0.68 }],
    examples: ["vmware-best-practices-2026.pdf"]
  },
  { tag: "memgraph", tier: 1, category: "infra", status: "active",
    def: "Memgraph graph database, including MAGE extensions, vector_search procedure, and Cypher dialect specifics.",
    aliases: ["mage"], deprecates: [],
    sources_count: 8, chunks_count: 142, query_freq_30d: 41,
    created: { by: "yann.dubois", at: "2025-11-04" },
    last_edit: { by: "yann.dubois", at: "2026-02-10", action: "added alias mage" },
    related: [{ tag: "graphrag", strength: 0.81 }],
    examples: ["memgraph-mage-3.8-release-notes.md"]
  },
  { tag: "rhel9", tier: 2, category: "infra", status: "active",
    def: "Red Hat Enterprise Linux 9 — OS-level configuration, kernel tuning, systemd, SELinux specific to RHEL 9.",
    aliases: ["redhat-9", "el9"], deprecates: ["rhel8"],
    sources_count: 12, chunks_count: 287, query_freq_30d: 22,
    created: { by: "yann.dubois", at: "2026-01-22" },
    last_edit: { by: "claire.benoit", at: "2026-04-15", action: "approved for Tier 2" },
    related: [{ tag: "oracle", strength: 0.61 }, { tag: "rman", strength: 0.44 }],
    examples: ["rhel9-kernel-tuning.pdf"]
  },
  { tag: "network", tier: 1, category: "network", status: "active",
    def: "Networking topics — routing, firewall rules, load balancers, DNS, TLS/PKI, observability of network flows.",
    aliases: [], deprecates: [],
    sources_count: 24, chunks_count: 612, query_freq_30d: 58,
    created: { by: "system", at: "2025-08-01" },
    last_edit: { by: "system", at: "2025-08-01", action: "created" },
    related: [{ tag: "cft", strength: 0.42 }, { tag: "vmware", strength: 0.34 }],
    examples: []
  },
  { tag: "cft", tier: 1, category: "network", status: "active",
    def: "Cross-platform File Transfer (Axway CFT) — partner connectivity, flow definitions, monitoring.",
    aliases: ["axway-cft"], deprecates: [],
    sources_count: 9, chunks_count: 218, query_freq_30d: 12,
    created: { by: "philippe.marchand", at: "2025-12-05" },
    last_edit: { by: "philippe.marchand", at: "2025-12-05", action: "created" },
    related: [{ tag: "network", strength: 0.42 }, { tag: "swift", strength: 0.21 }],
    examples: ["cft-network-architecture.docx"]
  },
  { tag: "swift", tier: 1, category: "payment", status: "active",
    def: "SWIFT messaging — MT/MX, FIN, GPI, ISO 20022 migration. Use for any source discussing SWIFT flows or compliance.",
    aliases: [], deprecates: [],
    sources_count: 18, chunks_count: 542, query_freq_30d: 94,
    created: { by: "system", at: "2025-08-01" },
    last_edit: { by: "claire.benoit", at: "2026-05-10", action: "tag iso20022 split out" },
    related: [{ tag: "iso20022", strength: 0.78 }, { tag: "cft", strength: 0.21 }],
    examples: ["swift-iso20022-migration.pdf"]
  },
  { tag: "iso20022", tier: 2, category: "payment", status: "active",
    def: "ISO 20022 message standard — pacs, pain, camt families. Distinct from generic 'swift' for cases where standard precision matters.",
    aliases: [], deprecates: [],
    sources_count: 4, chunks_count: 88, query_freq_30d: 17,
    created: { by: "claire.benoit", at: "2026-05-10" },
    last_edit: { by: "claire.benoit", at: "2026-05-10", action: "created as Tier 2" },
    related: [{ tag: "swift", strength: 0.78 }],
    examples: ["swift-iso20022-migration.pdf"]
  },
  { tag: "production", tier: 1, category: "lifecycle", status: "active",
    def: "Source applies to production environments — exclude from default retrieval if user filters to non-prod context.",
    aliases: ["prod"], deprecates: [],
    sources_count: 64, chunks_count: 2918, query_freq_30d: 281,
    created: { by: "system", at: "2025-08-01" },
    last_edit: { by: "system", at: "2025-08-01", action: "created" },
    related: [{ tag: "oracle", strength: 0.71 }, { tag: "vmware", strength: 0.68 }],
    examples: ["oracle-restart-procedure.pdf", "vmware-best-practices-2026.pdf"]
  },
  { tag: "incident", tier: 1, category: "lifecycle", status: "active",
    def: "Postmortem or active incident material — surfaced with critical visual emphasis to remind users this is high-stakes content.",
    aliases: ["postmortem"], deprecates: [],
    sources_count: 14, chunks_count: 318, query_freq_30d: 47,
    created: { by: "system", at: "2025-08-01" },
    last_edit: { by: "system", at: "2025-08-01", action: "created" },
    related: [{ tag: "production", strength: 0.62 }],
    examples: ["/cib/incidents/2026-Q1-postmortems"]
  },
  { tag: "deprecated", tier: 1, category: "lifecycle", status: "active",
    def: "Source is marked for archival and excluded from default retrieval. Apply when superseding content has been ingested.",
    aliases: ["archived"], deprecates: [],
    sources_count: 7, chunks_count: 184, query_freq_30d: 3,
    created: { by: "system", at: "2025-08-01" },
    last_edit: { by: "system", at: "2025-08-01", action: "created" },
    related: [],
    examples: ["/cib/runbooks/rman-restore-cookbook"]
  },
  { tag: "pending-review", tier: 1, category: "lifecycle", status: "active",
    def: "Awaiting risk review before being surfaced in unrestricted retrieval. Default-excluded from queries without explicit override.",
    aliases: [], deprecates: [],
    sources_count: 3, chunks_count: 62, query_freq_30d: 1,
    created: { by: "claire.benoit", at: "2025-10-18" },
    last_edit: { by: "claire.benoit", at: "2025-10-18", action: "created" },
    related: [],
    examples: ["/cib/governance/tagging-charter"]
  },
  { tag: "governance", tier: 1, category: "governance", status: "active",
    def: "Internal governance, audit, or charter material — policies, RACI, taxonomy definitions.",
    aliases: [], deprecates: [],
    sources_count: 11, chunks_count: 142, query_freq_30d: 8,
    created: { by: "system", at: "2025-08-01" },
    last_edit: { by: "system", at: "2025-08-01", action: "created" },
    related: [],
    examples: ["/cib/governance/tagging-charter"]
  },
  { tag: "rmf-validated", tier: 1, category: "governance", status: "active",
    def: "Material that has been formally reviewed and signed off by the Risk Management Framework board.",
    aliases: ["rmf-approved"], deprecates: [],
    sources_count: 6, chunks_count: 88, query_freq_30d: 2,
    created: { by: "claire.benoit", at: "2025-10-01" },
    last_edit: { by: "claire.benoit", at: "2025-10-01", action: "created" },
    related: [{ tag: "governance", strength: 0.55 }],
    examples: []
  },
  // ----- Tier 2 (department-scoped, awaiting Tier-1 promotion) -----
  { tag: "graphrag", tier: 2, category: "infra", status: "pending-promotion",
    def: "Knowledge graph + RAG hybrid retrieval — covers any source describing graph-enhanced retrieval architectures.",
    aliases: ["graph-rag"], deprecates: [],
    sources_count: 5, chunks_count: 96, query_freq_30d: 28,
    created: { by: "yann.dubois", at: "2026-02-14" },
    last_edit: { by: "yann.dubois", at: "2026-05-02", action: "promotion requested" },
    related: [{ tag: "memgraph", strength: 0.81 }],
    examples: ["memgraph-mage-3.8-release-notes.md"]
  },
  { tag: "k8s", tier: 2, category: "infra", status: "active",
    def: "Kubernetes — cluster topology, workload manifests, operators, GitOps. Scoped to Platform team usage.",
    aliases: ["kubernetes"], deprecates: [],
    sources_count: 7, chunks_count: 154, query_freq_30d: 18,
    created: { by: "yann.dubois", at: "2026-03-20" },
    last_edit: { by: "yann.dubois", at: "2026-03-20", action: "created as Tier 2" },
    related: [],
    examples: []
  },
  // ----- Tier 3 (user-proposed leaves) -----
  { tag: "vault", tier: 3, category: "infra", status: "active",
    def: "HashiCorp Vault — secret management. User-proposed leaf, not yet validated by the Platform tier-2 process.",
    aliases: [], deprecates: [],
    sources_count: 2, chunks_count: 38, query_freq_30d: 4,
    created: { by: "marc.berthier", at: "2026-04-29" },
    last_edit: { by: "marc.berthier", at: "2026-04-29", action: "created as Tier 3" },
    related: [],
    examples: []
  },
  { tag: "ansible", tier: 3, category: "infra", status: "active",
    def: "Ansible playbooks and roles — config management automation. Tier 3 leaf.",
    aliases: [], deprecates: [],
    sources_count: 3, chunks_count: 71, query_freq_30d: 7,
    created: { by: "marc.berthier", at: "2026-04-12" },
    last_edit: { by: "marc.berthier", at: "2026-04-12", action: "created as Tier 3" },
    related: [],
    examples: []
  },
  // ----- Requested (not yet a tag) -----
  { tag: "argocd", tier: "requested", category: "infra", status: "pending-review",
    def: "Proposed: ArgoCD — GitOps continuous-delivery controller. Awaiting Tier 3 acceptance.",
    aliases: [], deprecates: [],
    sources_count: 0, chunks_count: 0, query_freq_30d: 0,
    requested_by: "marc.berthier", requested_at: "2026-05-09",
    justification: "Used in 4 new sources scheduled for ingestion this sprint.",
    created: { by: "marc.berthier", at: "2026-05-09" },
    last_edit: { by: "marc.berthier", at: "2026-05-09", action: "requested" },
    related: [{ tag: "k8s", strength: 0.7 }],
    examples: []
  },
  { tag: "pacs008", tier: "requested", category: "payment", status: "pending-review",
    def: "Proposed: pacs.008 — ISO 20022 customer credit transfer message family.",
    aliases: [], deprecates: [],
    sources_count: 0, chunks_count: 0, query_freq_30d: 0,
    requested_by: "philippe.marchand", requested_at: "2026-05-10",
    justification: "Granularity needed below iso20022; we have 3 sources discussing pacs.008 specifically.",
    created: { by: "philippe.marchand", at: "2026-05-10" },
    last_edit: { by: "philippe.marchand", at: "2026-05-10", action: "requested" },
    related: [{ tag: "iso20022", strength: 0.85 }],
    examples: []
  },
  // Rejected request — kept in the thesaurus so the "Rejected" status filter
  // has something to show. Stewards may revisit later if justification firms up.
  { tag: "legacy-mq", tier: 3, category: "network", status: "rejected",
    def: "(rejected) IBM MQ legacy adapter — overlaps with existing `messaging` tag; not adopted.",
    aliases: [], deprecates: [],
    sources_count: 0, chunks_count: 0, query_freq_30d: 0,
    requested_by: "marc.berthier", requested_at: "2026-04-22",
    justification: "Granularity below `messaging` was requested; rejected as duplicative.",
    created: { by: "marc.berthier", at: "2026-04-22" },
    last_edit: { by: "claire.benoit", at: "2026-04-24", action: "rejected — overlaps with `messaging`" },
    related: [{ tag: "messaging", strength: 0.78 }],
    examples: []
  }
];

window.MOCK_TAG_CATEGORIES = [
  { id: "oracle",     label: "Oracle",         color: "#B85A1E" },
  { id: "infra",      label: "Infrastructure", color: "#5A7FB4" },
  { id: "network",    label: "Network",        color: "#1F8A7A" },
  { id: "payment",    label: "Payment",        color: "#7B5BB8" },
  { id: "lifecycle",  label: "Lifecycle",      color: "#8A5C0E" },
  { id: "governance", label: "Governance",     color: "#2C3E50" }
];

// ===== Retrieval conversation threads (seeded for sidebar demo) =====
window.MOCK_THREADS = [
  {
    id: "th_seed_1",
    title: "RMAN backup restart procedure",
    created: Date.now() - 3600e3 * 6,
    updated: Date.now() - 3600e3 * 6,
    messages: [
      { role: "user", text: "How do I restart Oracle RMAN after a failed backup?" },
      { role: "assistant", tokens: ["To", " restart", " RMAN", "…"], sources: [] }
    ]
  },
  {
    id: "th_seed_2",
    title: "CFT troubleshooting checklist",
    created: Date.now() - 86400e3 * 2,
    updated: Date.now() - 86400e3 * 2,
    messages: []
  }
];

// ===== Workspaces (switcher dropdown) =====
window.MOCK_WORKSPACES = [
  { id: "cib",      kb: "CIB KB",          visibility: "private",  sources: 247, role: "admin / steward", current: true  },
  { id: "cib-edge", kb: "CIB Edge KB",     visibility: "private",  sources:  82, role: "admin",            current: false },
  { id: "payments", kb: "Payments KB",     visibility: "internal", sources: 1318, role: "reader",          current: false },
  { id: "infra",    kb: "Infra Runbooks",  visibility: "internal", sources: 612, role: "steward",          current: false },
  { id: "sandbox",  kb: "Personal sandbox", visibility: "private", sources:   9, role: "owner",            current: false }
];

// ===== Notifications (seeded for the bell popover; runtime toasts append on top) =====
window.MOCK_NOTIFICATIONS = [
  // ── Pending governance — surfaced for stewards (palier 3) so they can
  // act from the bell without hunting through Documents/Tags tabs. Same
  // items also show up in their respective "Pending review" sections.
  { id: "n_p01", kind: "doc-review",  title: "Document needs review", sub: "cft-vendor-api-spec-draft.pdf · submitted by marc.berthier", rel: "47m ago", read: false },
  { id: "n_p02", kind: "doc-review",  title: "Document needs review", sub: "/cib/runbooks/incident-2026-Q2-postmortem-draft · submitted by yann.dubois", rel: "1h ago", read: false },
  { id: "n_p03", kind: "tag-request", title: "Tag request", tagname: "argocd", suffix: "awaiting steward approval", sub: "requested by marc.berthier · category infra", rel: "3d ago", read: false },
  { id: "n_p04", kind: "tag-request", title: "Tag request", tagname: "pacs008", suffix: "awaiting steward approval", sub: "requested by philippe.marchand · category payment", rel: "2d ago", read: false },

  { id: "n_001", kind: "tag-mutation", title: "Tag", tagname: "rman", suffix: "applied", sub: "oracle-restart-procedure.pdf · 418 chunks", rel: "12m ago", read: false },
  { id: "n_002", kind: "source-failed", title: "Ingestion failed", sub: "huge-archive.zip · unsupported MIME", rel: "30m ago", read: false },
  { id: "n_003", kind: "source-ready", title: "Source ready", sub: "memgraph-mage-3.8-release-notes.md · 42 chunks", rel: "25m ago", read: false },
  { id: "n_004", kind: "pipeline-warning", title: "Pipeline warning", sub: "LLM extraction retrying · attempt 2/3", rel: "57m ago", read: true },
  { id: "n_005", kind: "tag-mutation", title: "Tag", tagname: "iso20022", suffix: "added", sub: "Contributor · category payment", rel: "22h ago", read: true }
];

// ===== Knowledge Graph (LightRAG-extracted entities + relations) =====
// Pre-computed layout in a 1000×680 viewBox so the SVG renders deterministically
// without running a force sim in the browser. Types match LightRAG's default
// extraction prompt (PRODUCT, ORG, PERSON, TECHNOLOGY, CONCEPT, LOCATION).
window.MOCK_GRAPH_ENTITIES = [
  // Oracle cluster
  { id: "e_oracle", tags: ["oracle","production"],   name: "Oracle Database",     type: "PRODUCT",    x: 240, y: 200, mentions: 412, sources: 47, summary: "Relational database engine; primary OLTP backing store for CIB workloads." },
  { id: "e_rman", tags: ["oracle","rman"],     name: "RMAN",                type: "TECHNOLOGY", x: 130, y: 290, mentions: 318, sources: 31, summary: "Oracle Recovery Manager — supported backup/restore toolchain." },
  { id: "e_archlog", tags: ["oracle","rman"],  name: "Archive Log",         type: "CONCEPT",    x: 80,  y: 160, mentions: 142, sources: 24, summary: "Redo log archive used for PITR and standby replication." },
  { id: "e_rhel", tags: ["oracle","production"],     name: "RHEL 9",              type: "PRODUCT",    x: 340, y: 320, mentions: 198, sources: 38, summary: "Red Hat Enterprise Linux 9 — certified OS for Oracle 19c+." },
  { id: "e_pga", tags: ["oracle"],      name: "PGA tuning",          type: "CONCEPT",    x: 380, y: 130, mentions:  64, sources:  9, summary: "Program Global Area sizing for OLTP workload concurrency." },
  // Virt
  { id: "e_vmware", tags: ["vmware","production"],   name: "VMware vSphere 8",    type: "PRODUCT",    x: 540, y: 240, mentions: 287, sources: 22, summary: "Hypervisor stack; banking-grade configuration baseline." },
  { id: "e_esxi", tags: ["vmware","production"],     name: "ESXi host",           type: "PRODUCT",    x: 640, y: 320, mentions: 122, sources: 14, summary: "Bare-metal hypervisor node." },
  { id: "e_vmotion", tags: ["vmware"],  name: "vMotion",             type: "TECHNOLOGY", x: 700, y: 200, mentions:  58, sources:  9, summary: "Live migration of running VMs across ESXi hosts." },
  // Memgraph / RAG
  { id: "e_memgraph", tags: ["memgraph"], name: "Memgraph",            type: "PRODUCT",    x: 470, y: 480, mentions: 156, sources: 19, summary: "Graph DB backing LightRAG entity/relation storage." },
  { id: "e_mage", tags: ["memgraph"],     name: "MAGE 3.8",            type: "PRODUCT",    x: 560, y: 560, mentions:  84, sources:  7, summary: "Memgraph Algorithm Extensions Engine — vector_search modules." },
  { id: "e_cypher", tags: ["memgraph"],   name: "Cypher",              type: "TECHNOLOGY", x: 360, y: 540, mentions: 109, sources: 12, summary: "Graph query language used for pre-filter retrieval." },
  { id: "e_lightrag", tags: ["memgraph"], name: "LightRAG",            type: "PRODUCT",    x: 250, y: 460, mentions: 274, sources: 28, summary: "Open-source retrieval framework forked into Twin RAG." },
  // Payment / SWIFT
  { id: "e_swift", tags: ["swift","payment"],    name: "SWIFT",               type: "ORG",        x: 820, y: 130, mentions: 198, sources: 17, summary: "Society for Worldwide Interbank Financial Telecommunication." },
  { id: "e_iso20022", tags: ["swift","iso20022","payment"], name: "ISO 20022",           type: "CONCEPT",    x: 880, y: 230, mentions: 142, sources: 14, summary: "XML messaging standard for financial transactions; SWIFT migration target." },
  { id: "e_cft", tags: ["cft","payment","production"],      name: "CFT",                 type: "PRODUCT",    x: 780, y: 330, mentions:  92, sources: 11, summary: "Cross File Transfer middleware — Axway product." },
  // People / process
  { id: "e_marc", tags: ["oracle","rman"],     name: "Marc Berthier",       type: "PERSON",     x: 100, y: 420, mentions:  28, sources: 12, summary: "DBA — primary author on Oracle restart procedures." },
  { id: "e_claire", tags: ["governance"],   name: "Claire Benoit",       type: "PERSON",     x: 160, y: 580, mentions:  41, sources: 18, summary: "KB Admin / Tier 3 steward for CIB workspace." },
  // Location
  { id: "e_paris", tags: ["production"],    name: "DC Paris",            type: "LOCATION",   x: 700, y: 440, mentions:  37, sources:  8, summary: "Primary datacenter; active site of the dual-DC topology." },
  { id: "e_aubervil", tags: ["production"], name: "DC Aubervilliers",    type: "LOCATION",   x: 820, y: 520, mentions:  31, sources:  7, summary: "Secondary datacenter; standby site." }
];

window.MOCK_GRAPH_RELATIONS = [
  // Oracle cluster
  { id: "r_01", source: "e_rman",     target: "e_oracle",   label: "BACKS_UP",         strength: 0.92 },
  { id: "r_02", source: "e_rman",     target: "e_archlog",  label: "MANAGES",          strength: 0.74 },
  { id: "r_03", source: "e_oracle",   target: "e_rhel",     label: "RUNS_ON",          strength: 0.88 },
  { id: "r_04", source: "e_oracle",   target: "e_pga",      label: "TUNED_VIA",        strength: 0.61 },
  { id: "r_05", source: "e_archlog",  target: "e_oracle",   label: "GENERATED_BY",     strength: 0.55 },
  // Virt
  { id: "r_06", source: "e_esxi",     target: "e_vmware",   label: "PART_OF",          strength: 0.90 },
  { id: "r_07", source: "e_vmotion",  target: "e_vmware",   label: "FEATURE_OF",       strength: 0.78 },
  { id: "r_08", source: "e_oracle",   target: "e_vmware",   label: "HOSTED_ON",        strength: 0.66 },
  { id: "r_09", source: "e_esxi",     target: "e_paris",    label: "DEPLOYED_AT",      strength: 0.70 },
  { id: "r_10", source: "e_esxi",     target: "e_aubervil", label: "DEPLOYED_AT",      strength: 0.62 },
  // RAG / graph
  { id: "r_11", source: "e_lightrag", target: "e_memgraph", label: "USES",             strength: 0.89 },
  { id: "r_12", source: "e_lightrag", target: "e_cypher",   label: "QUERIES_WITH",     strength: 0.71 },
  { id: "r_13", source: "e_memgraph", target: "e_mage",     label: "EXTENDED_BY",      strength: 0.83 },
  { id: "r_14", source: "e_memgraph", target: "e_cypher",   label: "SPEAKS",           strength: 0.80 },
  // Payment
  { id: "r_15", source: "e_swift",    target: "e_iso20022", label: "MIGRATING_TO",     strength: 0.85 },
  { id: "r_16", source: "e_cft",      target: "e_swift",    label: "TRANSPORTS_FOR",   strength: 0.64 },
  { id: "r_17", source: "e_iso20022", target: "e_oracle",   label: "PERSISTED_IN",     strength: 0.42 },
  // People
  { id: "r_18", source: "e_marc",     target: "e_oracle",   label: "AUTHORED_ON",      strength: 0.79 },
  { id: "r_19", source: "e_marc",     target: "e_rman",     label: "AUTHORED_ON",      strength: 0.82 },
  { id: "r_20", source: "e_claire",   target: "e_lightrag", label: "ADMINISTERS",      strength: 0.68 },
  { id: "r_21", source: "e_claire",   target: "e_rman",     label: "TAGGED",           strength: 0.55 }
];

// LightRAG entity type palette (kept colorblind-friendly).
window.GRAPH_TYPE_COLORS = {
  PRODUCT:    "#3871B4",
  TECHNOLOGY: "#6A4FB6",
  CONCEPT:    "#8A5C0E",
  ORG:        "#1F8A7A",
  PERSON:     "#B03060",
  LOCATION:   "#2C3E50"
};


// ===== Settings: API tokens =====
// Personal access tokens — separate identity from OIDC session bearer.
window.MOCK_API_TOKENS = [
  { id: "tok_a1b2",   name: "Local dev",         scopes: ["read:documents","read:query"],                 last_used: "2026-05-18T14:22:00Z", created: "2026-03-01T09:30:00Z", prefix: "tw_pat_a1b2" },
  { id: "tok_c3d4",   name: "ci · ingest job",   scopes: ["read:documents","write:documents","read:query"], last_used: "2026-05-19T03:15:00Z", created: "2026-02-14T11:02:00Z", prefix: "tw_pat_c3d4" },
  { id: "tok_e5f6",   name: "Grafana exporter",  scopes: ["read:activity"],                                last_used: "2026-04-30T18:01:00Z", created: "2025-12-20T15:45:00Z", prefix: "tw_pat_e5f6" }
];

// ===== Settings: Workspace members =====
window.MOCK_MEMBERS = [
  { email: "claire.benoit@bnpparibas.com",       name: "Claire Benoit",      palier: 3, role: "admin / steward",   joined: "2025-09-12", last_seen: "now",     status: "active" },
  { email: "marc.berthier@bnpparibas.com",       name: "Marc Berthier",      palier: 2, role: "DBA",               joined: "2025-09-14", last_seen: "12m ago", status: "active" },
  { email: "philippe.marchand@bnpparibas.com",   name: "Philippe Marchand",  palier: 2, role: "Architect",         joined: "2025-10-01", last_seen: "18h ago", status: "active" },
  { email: "yann.dubois@bnpparibas.com",         name: "Yann Dubois",        palier: 2, role: "SRE",               joined: "2025-10-22", last_seen: "44m ago", status: "active" },
  { email: "amine.kassi@bnpparibas.com",         name: "Amine Kassi",        palier: 1, role: "Reader",            joined: "2026-01-08", last_seen: "3d ago",  status: "active" },
  { email: "leo.tessier@bnpparibas.com",         name: "Léo Tessier",        palier: 1, role: "Reader",            joined: "2026-02-19", last_seen: "—",       status: "invited" }
];

// ===== Settings: Providers (env-controlled in prod; display only here) =====
window.MOCK_PROVIDERS = {
  llm: {
    provider: "openai", model: "gpt-4o-mini",
    base_url: "https://api.openai.com/v1",
    key_ref: "secret://twin/cib/openai_api_key",
    rate_limit_rpm: 500, monthly_quota_usd: 2000, monthly_spend_usd: 487.32
  },
  embedder: {
    provider: "openai", model: "text-embedding-3-large",
    dims: 3072,
    base_url: "https://api.openai.com/v1",
    key_ref: "secret://twin/cib/openai_api_key",
    rate_limit_rpm: 3000
  },
  reranker: {
    provider: "bge", model: "bge-reranker-v2-m3",
    base_url: "https://cib-bge.twin.internal/v1",
    enabled: true
  }
};

// ===== Settings: Workspace retention (mirrors Activity Clear modal) =====
window.MOCK_RETENTION = [
  { area: "Source mgmt",     ttl: "90d",  note: "uploads, deletes, re-ingests" },
  { area: "Tag mgmt",        ttl: "90d",  note: "requests, approvals, deprecations" },
  { area: "Retrieval",       ttl: "30d",  note: "queries + cited sources" },
  { area: "Admin",           ttl: "1y",   note: "workspace + provider changes" },
  { area: "Auth",            ttl: "1y",   note: "logins, token mints" },
  { area: "Policy / System", ttl: "7y",   note: "policy violations, system actions" }
];

// ===== Settings: current user (was inline in tags.jsx) =====
window.MOCK_CURRENT_USER = {
  name: "Claire Benoit",
  email: "claire.benoit@bnpparibas.com",
  palier: 3,
  role: "admin / steward",
  scopes: ["read:documents", "write:documents", "read:query", "read:activity", "admin:tags", "admin:workspace"],
  sso: "keycloak · twin-cib · sub=clb-7f4e",
  session_expires: "2026-05-19T23:59:00Z"
};


// ===== External sync connections (Confluence / SharePoint / URL feeds) =====
window.MOCK_CONNECTIONS = [
  {
    id: "conn_conf_runbooks",
    kind: "confluence",
    name: "CIB Runbooks",
    url: "https://confluence.bnp/spaces/CIBRUN",
    space_key: "CIBRUN",
    status: "ok",
    health: "ok",
    sources_tracked: 142,
    last_sync_at: "2026-05-19T08:14:00Z",
    last_sync_duration_ms: 18420,
    next_sync_at: "2026-05-19T20:00:00Z",
    schedule: "every 12h",
    oauth_account: "svc-twin-sync@bnpparibas.com",
    scopes: ["read:pages", "read:attachments"],
    pages_added_7d: 8,
    pages_changed_7d: 23,
    pages_deleted_7d: 1,
    default_tags: ["cib", "runbook"],
    visibility: "private",
    connected_at: "2025-11-04",
    connected_by: "claire.benoit@bnpparibas.com"
  },
  {
    id: "conn_sp_incidents",
    kind: "sharepoint",
    name: "Incidents — postmortems",
    url: "https://erwin-labs.sharepoint.com/sites/cib-incidents",
    site_id: "cib-incidents",
    status: "token-expired",
    health: "warn",
    sources_tracked: 87,
    last_sync_at: "2026-05-16T22:01:00Z",
    last_sync_duration_ms: 9210,
    next_sync_at: null,
    schedule: "every 6h",
    oauth_account: "svc-twin-sync@bnpparibas.com",
    scopes: ["Sites.Read.All", "Files.Read.All"],
    pages_added_7d: 0,
    pages_changed_7d: 0,
    pages_deleted_7d: 0,
    default_tags: ["incidents"],
    visibility: "private",
    connected_at: "2025-09-22",
    connected_by: "claire.benoit@bnpparibas.com",
    error: "Microsoft refresh token expired (90d limit). Re-authorize to resume sync."
  },
  {
    id: "conn_conf_archi",
    kind: "confluence",
    name: "Architecture decisions",
    url: "https://confluence.bnp/spaces/ARCHCIB",
    space_key: "ARCHCIB",
    status: "sync-failed",
    health: "error",
    sources_tracked: 31,
    last_sync_at: "2026-05-19T06:00:00Z",
    last_sync_duration_ms: 1840,
    next_sync_at: "2026-05-19T18:00:00Z",
    schedule: "every 12h",
    oauth_account: "svc-twin-sync@bnpparibas.com",
    scopes: ["read:pages"],
    pages_added_7d: 0,
    pages_changed_7d: 4,
    pages_deleted_7d: 0,
    default_tags: ["architecture"],
    visibility: "private",
    connected_at: "2025-12-11",
    connected_by: "philippe.marchand@bnpparibas.com",
    error: "Permission denied on 12 pages — service account lost group membership 'archi-readers' on 2026-05-15."
  },
  {
    id: "conn_url_oracle",
    kind: "url",
    name: "Oracle Data Guard docs",
    url: "https://docs.cib/oracle/dataguard",
    status: "ok",
    health: "ok",
    sources_tracked: 23,
    last_sync_at: "2026-05-19T07:30:00Z",
    last_sync_duration_ms: 4210,
    next_sync_at: "2026-05-20T07:30:00Z",
    schedule: "daily",
    oauth_account: null,
    scopes: [],
    pages_added_7d: 2,
    pages_changed_7d: 5,
    pages_deleted_7d: 0,
    default_tags: ["oracle", "dataguard"],
    visibility: "private",
    connected_at: "2026-02-08",
    connected_by: "marc.berthier@bnpparibas.com"
  }
];

// Recent sync events (used by mini-history under the connection list).
window.MOCK_SYNC_HISTORY = [
  { id: "syn_1", conn_id: "conn_conf_runbooks", at: "2026-05-19T08:14:00Z", outcome: "ok",    summary: "8 added · 23 changed · 1 deleted",         duration_ms: 18420 },
  { id: "syn_2", conn_id: "conn_url_oracle",    at: "2026-05-19T07:30:00Z", outcome: "ok",    summary: "2 added · 5 changed · 0 deleted",           duration_ms: 4210 },
  { id: "syn_3", conn_id: "conn_conf_archi",    at: "2026-05-19T06:00:00Z", outcome: "error", summary: "Permission denied on 12 pages · abort", duration_ms: 1840 },
  { id: "syn_4", conn_id: "conn_conf_runbooks", at: "2026-05-18T20:00:00Z", outcome: "ok",    summary: "3 added · 19 changed",                       duration_ms: 16880 },
  { id: "syn_5", conn_id: "conn_sp_incidents",  at: "2026-05-16T22:01:00Z", outcome: "ok",    summary: "1 added · 7 changed",                        duration_ms: 9210 },
  { id: "syn_6", conn_id: "conn_conf_runbooks", at: "2026-05-18T08:00:00Z", outcome: "ok",    summary: "5 added · 14 changed",                       duration_ms: 17120 }
];
