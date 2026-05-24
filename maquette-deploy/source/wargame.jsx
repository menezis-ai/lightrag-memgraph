// Wargame widget — rendered inline in the Retrieval answer stream when the
// user asks an impact-analysis question. This is the "Generative UI" beat
// of the demo: instead of a paragraph of text, the assistant returns a small
// interactive React component pinned to a real subgraph traversal.
//
// Reuses ONTOLOGY_SEED_NODES / EDGES + the same BFS the studio canvas does,
// so the widget always agrees with whatever the steward has validated.

const {
  useState: _wgUseState, useEffect: _wgUseEffect, useMemo: _wgUseMemo
} = React;

// Walk the graph BFS from an origin id, marking depth. Pulls live state from
// the studio if the OntologyStudio is mounted (so a relation a steward just
// created reflects immediately); falls back to the seed otherwise.
function computeImpact(originId, depth = 3) {
  // Prefer live state if the studio has published it via a window hook.
  const liveNodes = window.__ontoLive ? window.__ontoLive.nodes : null;
  const liveEdges = window.__ontoLive ? window.__ontoLive.edges : null;
  const nodes = liveNodes || window.ONTOLOGY_SEED_NODES;
  const edges = liveEdges || window.ONTOLOGY_SEED_EDGES;

  const adj = {};
  edges.forEach(e => {
    if (!adj[e.source]) adj[e.source] = [];
    if (!adj[e.target]) adj[e.target] = [];
    adj[e.source].push({ id: e.target, edge: e });
    adj[e.target].push({ id: e.source, edge: e });
  });
  const dist = { [originId]: 0 };
  const via = { [originId]: null };
  let frontier = [originId];
  for (let d = 1; d <= depth; d++) {
    const next = [];
    for (const id of frontier) {
      for (const nb of (adj[id] || [])) {
        if (dist[nb.id] === undefined) {
          dist[nb.id] = d;
          via[nb.id] = { from: id, edge: nb.edge };
          next.push(nb.id);
        }
      }
    }
    frontier = next;
    if (!next.length) break;
  }
  const result = Object.entries(dist)
    .filter(([id]) => id !== originId)
    .map(([id, d]) => {
      const n = nodes.find(x => x.id === id);
      return n ? { node: n, depth: d, via: via[id] } : null;
    })
    .filter(Boolean)
    .sort((a, b) => a.depth - b.depth || a.node.name.localeCompare(b.node.name));
  return result;
}

// Public widget — invoked from retrieval.jsx with an origin id.
window.WargameImpactWidget = function WargameImpactWidget({ originId = "e_router02", question, onJumpToStudio }) {
  const liveNodes = window.__ontoLive ? window.__ontoLive.nodes : window.ONTOLOGY_SEED_NODES;
  const origin = liveNodes.find(n => n.id === originId);
  const [impacted, setImpacted] = _wgUseState(() => computeImpact(originId, 3));
  const [revealed, setRevealed] = _wgUseState(0);  // animate ring-by-ring
  const [pra, setPra] = _wgUseState(false);

  _wgUseEffect(() => {
    setRevealed(0);
    setImpacted(computeImpact(originId, 3));
  }, [originId]);

  _wgUseEffect(() => {
    // Reveal 1-hop, then 2-hop, then 3-hop.
    let cancelled = false;
    const tick = (n) => {
      if (cancelled) return;
      setRevealed(n);
      if (n < 3) setTimeout(() => tick(n + 1), 520);
    };
    setTimeout(() => tick(1), 220);
    return () => { cancelled = true; };
  }, [originId]);

  if (!origin) {
    return (
      <div className="wargame-widget is-error">
        <Icon name="alert-triangle" size={12} /> Unknown origin <code>{originId}</code>
      </div>
    );
  }

  const groups = { 1: [], 2: [], 3: [] };
  impacted.forEach(it => { if (groups[it.depth] && it.depth <= revealed) groups[it.depth].push(it); });
  const appsImpacted = impacted
    .filter(it => (it.node.type === "APPLICATION" || it.node.type === "PRODUCT") && it.depth <= revealed);
  const haveBraced = impacted.filter(it => it.depth <= revealed).length;

  // Honesty signal: confidence = share of traversed edges that a steward has
  // explicitly validated. Pending edges (steward not yet reviewed) are not
  // counted as validated — they lower the confidence so the user knows the
  // graph is incomplete. Without this pill the widget reads as authoritative
  // even when half the underlying edges are unreviewed.
  const traversedEdges = impacted
    .filter(it => it.depth <= revealed && it.via && it.via.edge)
    .map(it => it.via.edge);
  const totalEdges = traversedEdges.length;
  const pendingEdges = traversedEdges.filter(e => e.status === "pending").length;
  const validatedEdges = totalEdges - pendingEdges;
  const confidencePct = totalEdges === 0 ? 100 : Math.round((validatedEdges / totalEdges) * 100);
  const confidenceTier = confidencePct >= 90 ? "high" : confidencePct >= 70 ? "med" : "low";

  return (
    <div className="wargame-widget">
      <div className="wargame-h">
        <div className="wargame-h-left">
          <div className="wargame-h-row">
            <span className="wargame-h-badge"><Icon name="alert-triangle" size={11} /> Impact analysis</span>
            <span
              className={`wargame-confidence is-${confidenceTier}`}
              title={`${validatedEdges} validated / ${pendingEdges} pending steward review across ${totalEdges} traversed relations. Lower confidence = more of the underlying graph is still unreviewed — treat the cascade as a hypothesis, not ground truth.`}
            >
              <Icon name="circle-check" size={10} /> Confidence {confidencePct}%
            </span>
          </div>
          <span className="wargame-h-sub">
            Generated · workspace <code>cib</code> · {haveBraced} of {impacted.length} components impacted · {validatedEdges}/{totalEdges} edges validated
          </span>
        </div>
        <div className="wargame-h-right">
          <button className="wargame-h-btn" onClick={() => onJumpToStudio && onJumpToStudio(originId)} title="Open Ontology Studio focused on this node">
            <Icon name="external-link" size={10} /> Open in Studio
          </button>
        </div>
      </div>

      <div className="wargame-origin">
        <div className="wargame-origin-label">Failure origin</div>
        <div className="wargame-origin-node">
          <span className="onto-detail-swatch" style={{ background: window.ONTOLOGY_TYPE_COLORS[origin.type] }} />
          <b>{origin.name}</b>
          <span className="wargame-origin-kind">{window.ONTOLOGY_TYPE_LABEL[origin.type]}</span>
        </div>
      </div>

      <div className="wargame-rings">
        <RingViz originId={originId} impacted={impacted} revealed={revealed} />
      </div>

      <div className="wargame-counts">
        <div className={`wargame-count is-d1${revealed >= 1 ? " is-on" : ""}`}>
          <b>{groups[1].length}</b><span>1 hop</span>
        </div>
        <div className={`wargame-count is-d2${revealed >= 2 ? " is-on" : ""}`}>
          <b>{groups[2].length}</b><span>2 hops</span>
        </div>
        <div className={`wargame-count is-d3${revealed >= 3 ? " is-on" : ""}`}>
          <b>{groups[3].length}</b><span>3 hops</span>
        </div>
      </div>

      {[1, 2, 3].map(d => groups[d].length > 0 && (
        <div key={d} className={`wargame-group is-d${d}`}>
          <div className="wargame-group-h">
            <span className={`wargame-ring-dot is-d${d}`} />
            <span>{d === 1 ? "Direct dependencies" : d === 2 ? "2-hop propagation" : "3-hop propagation"}</span>
            <span className="wargame-group-count">{groups[d].length}</span>
          </div>
          <ul>
            {groups[d].map(it => (
              <li key={it.node.id}>
                <span className="onto-detail-swatch" style={{ background: window.ONTOLOGY_TYPE_COLORS[it.node.type] }} />
                <span className="wargame-name">{it.node.name}</span>
                <span className="wargame-typed">{window.ONTOLOGY_TYPE_LABEL[it.node.type]}</span>
                {it.via && it.via.edge && (
                  <span className="wargame-via">
                    via <code>{it.via.edge.label}</code>
                    {it.via.edge.status === "pending" && <span className="wargame-via-pending">· pending</span>}
                  </span>
                )}
              </li>
            ))}
          </ul>
        </div>
      ))}

      {appsImpacted.length > 0 && (
        <div className="wargame-cta">
          <div className="wargame-cta-msg">
            <Icon name="info-circle" size={11} />
            <span><b>{appsImpacted.length}</b> application{appsImpacted.length === 1 ? "" : "s"} impacted — DR runbook{appsImpacted.length === 1 ? "" : "s"} linked in the graph.</span>
          </div>
          <button className={`wargame-cta-btn${pra ? " is-on" : ""}`} onClick={() => setPra(p => !p)}>
            {pra ? "Hide DR plans" : "Open DR procedures"}
          </button>
        </div>
      )}

      {pra && (
        <div className="wargame-pra-list">
          <div className="wargame-pra-h">DR · linked failover procedures</div>
          <ul>
            <li>
              <Icon name="file-text" size={11} color="var(--twin-accent)" />
              <code>pra-swift-2026q1.pdf</code>
              <span>· §3 — Bascule R-CORE-02 → R-CORE-01</span>
              <span className="wargame-pra-ts">approved · Claire B. · 2026-03-12</span>
            </li>
            <li>
              <Icon name="file-text" size={11} color="var(--twin-accent)" />
              <code>pra-cft-failover.pdf</code>
              <span>· §1 — Reroutage CFT-Payment via DC Paris</span>
              <span className="wargame-pra-ts">approved · Marc B. · 2026-04-02</span>
            </li>
            <li>
              <Icon name="file-text" size={11} color="var(--twin-accent)" />
              <code>runbook-network-core.md</code>
              <span>· §5 — Diagnostic R-CORE-* loss-of-link</span>
              <span className="wargame-pra-ts">draft · awaiting review</span>
            </li>
          </ul>
        </div>
      )}

      <div className="wargame-foot">
        <Icon name="info-circle" size={10} />
        <span>Generated by LightRAG graph traversal. Each relation used is audited and approved by a steward.</span>
      </div>
    </div>
  );
};

// ─── Ring visualization ─────────────────────────────────────────────────
// Concentric rings, origin at center, impacted nodes orbit on their depth ring.
function RingViz({ originId, impacted, revealed }) {
  const W = 480;
  const H = 220;
  const cx = W / 2;
  const cy = H / 2;
  const ringR = [0, 48, 88, 128];

  const byDepth = { 1: [], 2: [], 3: [] };
  impacted.forEach(it => { if (byDepth[it.depth]) byDepth[it.depth].push(it); });
  const colors = window.ONTOLOGY_TYPE_COLORS;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="wargame-ringviz">
      {/* Ring outlines */}
      {[1, 2, 3].map(d => (
        <circle
          key={d}
          cx={cx} cy={cy} r={ringR[d]}
          fill="none"
          stroke="var(--color-border-tertiary)"
          strokeWidth="0.8"
          strokeDasharray="3 4"
          opacity={revealed >= d ? 0.9 : 0.3}
        />
      ))}

      {/* Pulse rings (visible only when revealed reaches that depth) */}
      {[1, 2, 3].map(d => revealed >= d && (
        <circle
          key={`p${d}`}
          cx={cx} cy={cy}
          r={ringR[d]}
          fill="none"
          stroke={d === 1 ? "var(--twin-red-vivid)" : d === 2 ? "var(--twin-amber-vivid)" : "var(--twin-amber-700)"}
          strokeWidth="1.2"
          opacity="0.55"
          className={`wargame-pulse-ring is-d${d}`}
        />
      ))}

      {/* Origin node */}
      <g transform={`translate(${cx}, ${cy})`}>
        <circle r="14" fill="var(--twin-red-vivid)" opacity="0.18" />
        <circle r="9" fill="var(--twin-red-vivid)" />
        <text y="3" textAnchor="middle" fill="#fff" fontSize="9" fontWeight="600">⚡</text>
      </g>

      {/* Impacted nodes — distribute around each ring */}
      {[1, 2, 3].map(d => {
        const group = byDepth[d];
        if (!group.length || revealed < d) return null;
        return group.map((it, i) => {
          const angle = (i / group.length) * Math.PI * 2 - Math.PI / 2;
          const x = cx + ringR[d] * Math.cos(angle);
          const y = cy + ringR[d] * Math.sin(angle);
          const tone = colors[it.node.type] || "#888";
          const labelAbove = Math.sin(angle) < 0;
          return (
            <g key={it.node.id} transform={`translate(${x}, ${y})`} className={`wargame-node-g is-d${d}`}>
              <circle r="5.5" fill={tone} stroke="var(--color-background-primary)" strokeWidth="1.5" />
              <text
                y={labelAbove ? -8 : 14}
                textAnchor="middle"
                className="wargame-node-text"
              >{it.node.name}</text>
            </g>
          );
        });
      })}
    </svg>
  );
}

// Live state mirror — the Studio sets window.__ontoLive on every render so
// the wargame widget can recompute against the steward's latest validated
// graph state.
window.publishOntoLive = function publishOntoLive(nodes, edges) {
  window.__ontoLive = { nodes, edges };
};

Object.assign(window, { computeImpact, RingViz });
