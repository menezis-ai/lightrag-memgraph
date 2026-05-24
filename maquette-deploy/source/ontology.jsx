// Ontology Studio — n8n-style visual editor for the LightRAG knowledge graph.
//
// Three-pane layout (Inbox · Canvas · Inspector). The canvas is a hand-rolled
// React Flow-like surface (SVG bezier edges + abs-positioned div nodes) so we
// don't have to bring in the @xyflow ESM bundle through Babel. The visual
// vocabulary stays Lagom — same neutrals/borders/accent as the rest of the app,
// just with rectangular cards instead of force-directed circles.
//
// Editing model: every node/edge has a `status` of "validated" (audited),
// "pending" (steward-created but awaiting a second-pair review) or "proposed"
// (AI candidate sitting in the Inbox until dropped). The inspector exposes the
// audit trail + type override controls; the canvas exposes drag-to-pin +
// drag-from-handle to create new edges.
//
// Also hosts the Wargame mode: a toggle on the canvas turns the surface into
// an impact-analysis flood-fill (BFS from a chosen origin node along edges,
// depth → color). The reusable widget lives in wargame.jsx so the same impact
// panel can be rendered in the Retrieval answer stream.

const {
  useState: _osUseState, useEffect: _osUseEffect, useMemo: _osUseMemo,
  useRef: _osUseRef, useCallback: _osUseCallback
} = React;

// ─── Extra type palette (extends GRAPH_TYPE_COLORS for ontology-only kinds) ───
const ONTOLOGY_TYPE_COLORS = {
  ...(window.GRAPH_TYPE_COLORS || {}),
  INFRA:       "#4A6072",  // slate — routers, firewalls, network gear
  APPLICATION: "#0F7A6E"   // teal — business apps (SWIFT-Payment, CFT…)
};

const ONTOLOGY_TYPE_LABEL = {
  PRODUCT: "Product", TECHNOLOGY: "Technology", CONCEPT: "Concept",
  ORG: "Org", PERSON: "Person", LOCATION: "Location",
  INFRA: "Infrastructure", APPLICATION: "Application"
};

// ─── Seed ontology — subset of MOCK_GRAPH_ENTITIES laid out on a wider canvas,
// plus a payment cluster for the Wargame act of the demo. All starting nodes
// are status:"validated" (this is the steady-state ontology).
window.ONTOLOGY_SEED_NODES = [
  // Oracle stack
  { id: "e_oracle",    name: "Oracle Database",   type: "PRODUCT",    x: 360, y: 240, status: "validated", summary: "Relational OLTP backing store for CIB workloads." },
  { id: "e_rman",      name: "RMAN",              type: "TECHNOLOGY", x: 150, y: 290, status: "validated", summary: "Oracle Recovery Manager — backup/restore." },
  { id: "e_archlog",   name: "Archive Log",       type: "CONCEPT",    x: 100, y: 160, status: "validated", summary: "Redo archive used for PITR and standby." },
  { id: "e_rhel",      name: "RHEL 9",            type: "PRODUCT",    x: 470, y: 360, status: "validated", summary: "Red Hat 9 — certified OS for Oracle 19c+." },
  { id: "e_pga",       name: "PGA tuning",        type: "CONCEPT",    x: 530, y: 140, status: "validated", summary: "PGA sizing for OLTP concurrency." },

  // Virt
  { id: "e_vmware",    name: "VMware vSphere 8",  type: "PRODUCT",    x: 700, y: 470, status: "validated", summary: "Hypervisor stack." },
  { id: "e_esxi",      name: "ESXi host",         type: "PRODUCT",    x: 850, y: 540, status: "validated", summary: "Bare-metal hypervisor node." },

  // Payment + infra (new — staged for the wargame demo)
  { id: "e_router02",  name: "R-CORE-02",         type: "INFRA",      x: 920, y: 130, status: "validated", summary: "Core router · DC Aubervilliers — north-south traffic.", kind: "Core router" },
  { id: "e_swift_pay", name: "SWIFT-Payment",     type: "APPLICATION",x: 1140, y: 260, status: "validated", summary: "MX message generator, ISO 20022 pipeline." },
  { id: "e_cft_pay",   name: "CFT-Payment",       type: "APPLICATION",x: 1140, y: 420, status: "validated", summary: "Cross-File Transfer payment bridge." },
  { id: "e_pra_swift", name: "PRA · SWIFT",       type: "CONCEPT",    x: 1340, y: 350, status: "validated", summary: "Disaster Recovery Plan — SWIFT failover runbook." },

  // Locations
  { id: "e_aubervil",  name: "DC Aubervilliers",  type: "LOCATION",   x: 1000, y: 620, status: "validated", summary: "Secondary datacenter; standby site." },

  // People / governance
  { id: "e_marc",      name: "Marc Berthier",     type: "PERSON",     x: 130, y: 480, status: "validated", summary: "DBA — primary author on Oracle restart procedures." },
  { id: "e_claire",    name: "Claire Benoit",     type: "PERSON",     x: 260, y: 620, status: "validated", summary: "KB Admin · Tier 3 steward · CIB workspace." }
];

window.ONTOLOGY_SEED_EDGES = [
  // Oracle
  { id: "r_01", source: "e_rman",     target: "e_oracle",    label: "BACKS_UP",          strength: 0.92, status: "validated" },
  { id: "r_02", source: "e_rman",     target: "e_archlog",   label: "MANAGES",           strength: 0.74, status: "validated" },
  { id: "r_03", source: "e_oracle",   target: "e_rhel",      label: "RUNS_ON",           strength: 0.88, status: "validated" },
  { id: "r_04", source: "e_oracle",   target: "e_pga",       label: "TUNED_VIA",         strength: 0.61, status: "validated" },
  // Virt + hosting
  { id: "r_05", source: "e_oracle",   target: "e_vmware",    label: "HOSTED_ON",         strength: 0.66, status: "validated" },
  { id: "r_06", source: "e_esxi",     target: "e_vmware",    label: "PART_OF",           strength: 0.90, status: "validated" },
  { id: "r_07", source: "e_esxi",     target: "e_aubervil",  label: "DEPLOYED_AT",       strength: 0.70, status: "validated" },
  // Payment topology — note: R-CORE-02 → SWIFT-Payment is INTENTIONALLY MISSING
  // so the demo can show the steward creating it live in Act 2.
  { id: "r_10", source: "e_cft_pay",  target: "e_router02",  label: "ROUTES_VIA",        strength: 0.78, status: "validated" },
  { id: "r_11", source: "e_router02", target: "e_aubervil",  label: "LOCATED_AT",        strength: 0.82, status: "validated" },
  { id: "r_12", source: "e_swift_pay",target: "e_pra_swift", label: "COVERED_BY",        strength: 0.71, status: "validated" },
  { id: "r_13", source: "e_cft_pay",  target: "e_pra_swift", label: "COVERED_BY",        strength: 0.64, status: "validated" },
  // People
  { id: "r_20", source: "e_marc",     target: "e_oracle",    label: "AUTHORED_ON",       strength: 0.79, status: "validated" },
  { id: "r_21", source: "e_marc",     target: "e_rman",      label: "AUTHORED_ON",       strength: 0.82, status: "validated" },
  { id: "r_22", source: "e_claire",   target: "e_oracle",    label: "STEWARDS",          strength: 0.55, status: "validated" }
];

// AI-proposed candidates sitting in the Inbox until a steward drops them on
// the canvas (or rejects). Mixed kinds: new entities, new relations between
// existing nodes, and type-correction proposals.
window.ONTOLOGY_INBOX_SEED = [
  { id: "inb_dg", kind: "entity",   name: "Data Guard",        type: "TECHNOLOGY", confidence: 0.84,
    evidence: "runbook-oracle-dr.pdf · p.3",
    rationale: "Mentioned 17× in 4 DR runbooks; pattern matches Oracle replication tooling." },
  { id: "inb_ax", kind: "entity",   name: "Axway",             type: "ORG",        confidence: 0.66,
    evidence: "cft-config.md · §2 · p.1",
    rationale: "Recognized as vendor for the CFT product family." },
  { id: "inb_pra_or", kind: "entity", name: "PRA · Oracle",    type: "CONCEPT",    confidence: 0.79,
    evidence: "pra-2026-q1.pdf · p.12",
    rationale: "DR plan section; sibling of existing PRA · SWIFT node." },
  { id: "inb_rel_sap", kind: "relation",
    source: "e_router02", target: "e_swift_pay", label: "CRITICAL_DEPENDENCY",
    confidence: 0.72, evidence: "incident-2026-04-12-postmortem.md · §3",
    rationale: "SWIFT-Payment lost network during R-CORE-02 maintenance window. Causal chain confirmed by postmortem." },
  { id: "inb_typefix_rman", kind: "entity-type-fix",
    subject: "e_rman", existing_type: "TECHNOLOGY", proposed_type: "TECHNOLOGY",
    note: "Confirmed — one-off misclassification as PERSON during a 2025-12 re-extraction; fixed by the ML curation worker.",
    confidence: 0.95, evidence: "ml-curator-log/2025-12-19.jsonl" },
  { id: "inb_rel_cft_swift", kind: "relation",
    source: "e_cft_pay", target: "e_swift_pay", label: "FEEDS",
    confidence: 0.61, evidence: "swift-iso20022-migration.pdf · p.5",
    rationale: "CFT pipeline ingests into SWIFT-Payment for outbound MX messages." }
];

// Default audit trail — pre-seeded with steward activity so the inspector
// isn't empty on first load.
const SEED_AUDIT = [
  { ts: "today · 09:12", who: "claire.benoit", action: "Node created", target: "PRA · SWIFT", kind: "create-node" },
  { ts: "today · 08:47", who: "marc.berthier", action: "Relation approved", target: "Oracle Database → RHEL 9 (RUNS_ON)", kind: "approve" },
  { ts: "yesterday · 18:02", who: "ml-curator", action: "Reclassification automatique de RMAN", target: "PERSON → TECHNOLOGY", kind: "type-fix" },
  { ts: "yesterday · 14:33", who: "marc.berthier", action: "Pinned", target: "Oracle Database", kind: "pin" }
];

// ─── Geometry helpers ───────────────────────────────────────────────────
const NODE_W = 184;
const NODE_H = 56;

// Compute handle anchor points (top/right/bottom/left) in canvas coords.
function handlePoint(node, side) {
  const cx = node.x + NODE_W / 2;
  const cy = node.y + NODE_H / 2;
  if (side === "left")   return { x: node.x,            y: cy };
  if (side === "right")  return { x: node.x + NODE_W,   y: cy };
  if (side === "top")    return { x: cx,                y: node.y };
  if (side === "bottom") return { x: cx,                y: node.y + NODE_H };
  return { x: cx, y: cy };
}

// Pick the best handle pair (lowest cartesian distance + side coherence)
// for an edge between two nodes. Keeps bezier curves readable instead of
// always exiting from the right port.
function autoAnchors(a, b) {
  const sides = ["left", "right", "top", "bottom"];
  let best = null;
  for (const sa of sides) {
    for (const sb of sides) {
      const pa = handlePoint(a, sa);
      const pb = handlePoint(b, sb);
      const dx = pb.x - pa.x;
      const dy = pb.y - pa.y;
      const dist = Math.hypot(dx, dy);
      // Penalize edges leaving a node in the opposite direction of the target.
      let penalty = 0;
      if (sa === "right" && dx < 0) penalty += 60;
      if (sa === "left"  && dx > 0) penalty += 60;
      if (sa === "top"   && dy > 0) penalty += 40;
      if (sa === "bottom"&& dy < 0) penalty += 40;
      if (sb === "right" && pb.x > pa.x) penalty += 40;
      if (sb === "left"  && pb.x < pa.x) penalty += 40;
      const score = dist + penalty;
      if (!best || score < best.score) best = { sa, sb, pa, pb, score };
    }
  }
  return best;
}

// Bezier path between two anchor points. Curvature depends on the side they
// exit so bottom/top exits curve vertically and left/right curve horizontally.
function bezierPath(pa, pb, sa, sb) {
  const dx = pb.x - pa.x;
  const dy = pb.y - pa.y;
  const horiz = (sa === "left" || sa === "right") || (sb === "left" || sb === "right");
  let c1, c2;
  if (horiz) {
    const off = Math.max(40, Math.abs(dx) * 0.4);
    c1 = { x: pa.x + (sa === "right" ? off : sa === "left" ? -off : 0), y: pa.y };
    c2 = { x: pb.x + (sb === "right" ? off : sb === "left" ? -off : 0), y: pb.y };
  } else {
    const off = Math.max(40, Math.abs(dy) * 0.4);
    c1 = { x: pa.x, y: pa.y + (sa === "bottom" ? off : sa === "top" ? -off : 0) };
    c2 = { x: pb.x, y: pb.y + (sb === "bottom" ? off : sb === "top" ? -off : 0) };
  }
  return `M ${pa.x},${pa.y} C ${c1.x},${c1.y} ${c2.x},${c2.y} ${pb.x},${pb.y}`;
}

// Midpoint along a cubic bezier (for placing the edge type pill).
function bezierMid(pa, pb, sa, sb) {
  // Cheap approximation: average. Close enough at our zoom levels.
  return { x: (pa.x + pb.x) / 2, y: (pa.y + pb.y) / 2 };
}

// ─── Inbox panel ────────────────────────────────────────────────────────
function InboxPanel({ items, onDrop, onReject, onSelect, selectedId, filter, setFilter, collapsed, onToggleCollapse, count }) {
  const filtered = items.filter(it => {
    if (filter === "all") return true;
    if (filter === "entities") return it.kind === "entity";
    if (filter === "relations") return it.kind === "relation";
    if (filter === "fixes") return it.kind === "entity-type-fix";
    return true;
  });

  if (collapsed) {
    return (
      <aside className="onto-inbox is-collapsed" onClick={onToggleCollapse} title="Expand validation queue">
        <div className="onto-inbox-collapsed-rail">
          <span className="onto-inbox-collapsed-label">Validation queue</span>
          <span className="onto-inbox-badge">{count}</span>
        </div>
      </aside>
    );
  }

  return (
    <aside className="onto-inbox">
      <div className="onto-inbox-h">
        <div className="onto-pane-title">
          <span>Validation queue</span>
          <span className="onto-inbox-badge">{count}</span>
        </div>
        <button className="onto-collapse" onClick={onToggleCollapse} title="Collapse">
          <Icon name="chevron-left" size={11} />
        </button>
      </div>
      <div className="onto-inbox-sub">ML proposals · awaiting steward review</div>
      <div className="onto-inbox-filters">
        {[
          { id: "all",       label: "All" },
          { id: "entities",  label: "Entities" },
          { id: "relations", label: "Relations" },
          { id: "fixes",     label: "Type fixes" }
        ].map(f => (
          <button
            key={f.id}
            className={`onto-inbox-filter${filter === f.id ? " is-on" : ""}`}
            onClick={() => setFilter(f.id)}
          >{f.label}</button>
        ))}
      </div>
      <div className="onto-inbox-list">
        {filtered.length === 0 && (
          <div className="onto-inbox-empty">
            <Icon name="circle-check" size={14} color="var(--twin-green-700)" />
            <span>Backlog clear — nothing to review.</span>
          </div>
        )}
        {filtered.map(it => (
          <InboxCard
            key={it.id}
            item={it}
            onDrop={onDrop}
            onReject={onReject}
            onSelect={onSelect}
            selected={selectedId === it.id}
          />
        ))}
      </div>
      <div className="onto-inbox-foot">
        <Icon name="info-circle" size={10} />
        <span>Glisser sur le canvas pour valider · cliquer pour inspecter</span>
      </div>
    </aside>
  );
}

function InboxCard({ item, onDrop, onReject, onSelect, selected }) {
  const colors = ONTOLOGY_TYPE_COLORS;
  const conf = Math.round((item.confidence || 0) * 100);
  const confTone = item.confidence >= 0.8 ? "high" : item.confidence >= 0.6 ? "mid" : "low";

  const onDragStart = (e) => {
    e.dataTransfer.setData("text/twin-inbox-id", item.id);
    e.dataTransfer.effectAllowed = "copy";
  };

  if (item.kind === "entity") {
    return (
      <div
        className={`onto-inbox-card${selected ? " is-selected" : ""}`}
        draggable
        onDragStart={onDragStart}
        onClick={() => onSelect(item.id)}
      >
        <div className="onto-inbox-card-h">
          <span className="onto-card-swatch" style={{ background: colors[item.type] }} />
          <span className="onto-card-name">{item.name}</span>
          <span className={`onto-card-conf is-${confTone}`}>{conf}%</span>
        </div>
        <div className="onto-card-meta">
          <span className="onto-kind-tag">entity</span>
          <span>{ONTOLOGY_TYPE_LABEL[item.type]}</span>
        </div>
        <div className="onto-card-evidence" title={item.rationale}>{item.evidence}</div>
        <div className="onto-card-actions" onClick={e => e.stopPropagation()}>
          <button className="onto-mini-btn primary" onClick={() => onDrop(item.id, null)}>Approve</button>
          <button className="onto-mini-btn ghost" onClick={() => onReject(item.id)} title="Reject">Reject</button>
        </div>
      </div>
    );
  }

  if (item.kind === "relation") {
    return (
      <div
        className={`onto-inbox-card is-relation${selected ? " is-selected" : ""}`}
        draggable
        onDragStart={onDragStart}
        onClick={() => onSelect(item.id)}
      >
        <div className="onto-inbox-card-h">
          <span className="onto-kind-tag is-rel">relation</span>
          <span className={`onto-card-conf is-${confTone}`}>{conf}%</span>
        </div>
        <div className="onto-card-rel-body">
          <code className="onto-card-rel-end">{nodeNameOf(item.source)}</code>
          <span className="onto-card-rel-arrow">→</span>
          <code className="onto-card-rel-label">{item.label}</code>
          <span className="onto-card-rel-arrow">→</span>
          <code className="onto-card-rel-end">{nodeNameOf(item.target)}</code>
        </div>
        <div className="onto-card-evidence" title={item.rationale}>{item.evidence}</div>
        <div className="onto-card-actions" onClick={e => e.stopPropagation()}>
          <button className="onto-mini-btn primary" onClick={() => onDrop(item.id, null)}>Accepter</button>
          <button className="onto-mini-btn ghost" onClick={() => onReject(item.id)}>Reject</button>
        </div>
      </div>
    );
  }

  if (item.kind === "entity-type-fix") {
    const nm = nodeNameOf(item.subject);
    return (
      <div
        className={`onto-inbox-card is-fix${selected ? " is-selected" : ""}`}
        onClick={() => onSelect(item.id)}
      >
        <div className="onto-inbox-card-h">
          <span className="onto-kind-tag is-fix">correction typage</span>
          <span className={`onto-card-conf is-${confTone}`}>{conf}%</span>
        </div>
        <div className="onto-card-fix-body">
          <code>{nm}</code>
          <span className="onto-card-fix-arrow">{item.existing_type} → {item.proposed_type}</span>
        </div>
        <div className="onto-card-evidence" title={item.note}>{item.evidence}</div>
        <div className="onto-card-actions" onClick={e => e.stopPropagation()}>
          <button className="onto-mini-btn primary" onClick={() => onDrop(item.id, null)}>Appliquer</button>
          <button className="onto-mini-btn ghost" onClick={() => onReject(item.id)}>Reject</button>
        </div>
      </div>
    );
  }
  return null;
}

function nodeNameOf(id) {
  const all = [...(window.ONTOLOGY_SEED_NODES || [])];
  const n = all.find(x => x.id === id);
  return n ? n.name : id;
}

// Export helpers + global hooks ────────────────────────────────────────
Object.assign(window, {
  ONTOLOGY_TYPE_COLORS, ONTOLOGY_TYPE_LABEL,
  handlePoint, autoAnchors, bezierPath, bezierMid,
  InboxPanel, InboxCard, nodeNameOf, NODE_W, NODE_H, SEED_AUDIT
});
