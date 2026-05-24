// Ontology Studio main shell. Owns nodes/edges state, the inbox, the inspector,
// and toggles to wargame mode.

const {
  useState: _osmUseState, useEffect: _osmUseEffect, useMemo: _osmUseMemo,
  useRef: _osmUseRef, useCallback: _osmUseCallback
} = React;

window.OntologyStudio = function OntologyStudio({ layoutVariant, presentation, demoBus }) {
  // ── State ────────────────────────────────────────────────────────────
  // State survives tab unmount/remount by reading from window.__ontoLive if
  // a previous mount published into it (App keys the tab-pane by tab id, so
  // we get a fresh component instance on every tab switch).
  const [nodes, setNodes] = _osmUseState(() => (window.__ontoLive && window.__ontoLive.nodes)
    ? window.__ontoLive.nodes.map(n => ({ ...n }))
    : window.ONTOLOGY_SEED_NODES.map(n => ({ ...n })));
  const [edges, setEdges] = _osmUseState(() => (window.__ontoLive && window.__ontoLive.edges)
    ? window.__ontoLive.edges.map(e => ({ ...e }))
    : window.ONTOLOGY_SEED_EDGES.map(e => ({ ...e })));
  const [inbox, setInbox] = _osmUseState(() => (window.__ontoInbox)
    ? window.__ontoInbox.map(i => ({ ...i }))
    : window.ONTOLOGY_INBOX_SEED.map(i => ({ ...i })));
  const [audit, setAudit] = _osmUseState(() => (window.__ontoAudit ? window.__ontoAudit.slice() : window.SEED_AUDIT.slice()));

  const [selectedId, setSelectedId] = _osmUseState("e_oracle");
  const [inboxSelectedId, setInboxSelectedId] = _osmUseState(null);
  const [hoverNodeId, setHoverNodeId] = _osmUseState(null);
  const [inboxFilter, setInboxFilter] = _osmUseState("all");
  const [inboxCollapsed, setInboxCollapsed] = _osmUseState(false);
  const [inspectorCollapsed, setInspectorCollapsed] = _osmUseState(false);

  // Wargame state
  const [wargame, setWargame] = _osmUseState({ active: false, originId: null, depth: 3 });
  const [wargameRunId, setWargameRunId] = _osmUseState(0);

  // Ephemeral animations
  const [ephemeral, setEphemeral] = _osmUseState({ flashEdges: new Set(), flashNodes: new Set() });

  // Publish live state to window so the Wargame widget (rendered in the
  // Retrieval tab) always sees the steward's latest validated graph, even
  // after the Studio component unmounts on tab switch.
  _osmUseEffect(() => {
    if (window.publishOntoLive) window.publishOntoLive(nodes, edges);
  }, [nodes, edges]);
  _osmUseEffect(() => { window.__ontoInbox = inbox; }, [inbox]);
  _osmUseEffect(() => { window.__ontoAudit = audit; }, [audit]);

  // ── Mutations ────────────────────────────────────────────────────────
  const logAudit = (entry) => setAudit(a => [{ ts: "just now", who: "marc.berthier", ...entry }, ...a].slice(0, 32));

  const pushFlash = (kind, ids, ms = 1200) => {
    const set = new Set(ids);
    setEphemeral(s => ({ ...s, [kind]: set }));
    setTimeout(() => {
      setEphemeral(s => {
        const next = new Set(s[kind]);
        ids.forEach(i => next.delete(i));
        return { ...s, [kind]: next };
      });
    }, ms);
  };

  const dropInboxItem = (inboxId, pos) => {
    const it = inbox.find(x => x.id === inboxId);
    if (!it) return;

    if (it.kind === "entity") {
      const nid = "e_" + Math.random().toString(36).slice(2, 8);
      const x = pos ? pos.x : 600;
      const y = pos ? pos.y : 400;
      setNodes(ns => [...ns, { id: nid, name: it.name, type: it.type, x, y, status: "validated", summary: it.rationale || "" }]);
      setInbox(ix => ix.filter(x => x.id !== inboxId));
      pushFlash("flashNodes", [nid], 1400);
      setSelectedId(nid);
      logAudit({ action: "Entity approved", target: `${it.name} (${it.type})`, kind: "create-node" });
    }

    if (it.kind === "relation") {
      const rid = "r_" + Math.random().toString(36).slice(2, 8);
      setEdges(es => [...es, {
        id: rid, source: it.source, target: it.target,
        label: it.label, strength: it.confidence, status: "validated"
      }]);
      setInbox(ix => ix.filter(x => x.id !== inboxId));
      pushFlash("flashEdges", [rid], 1600);
      logAudit({ action: "Relation accepted", target: `${window.nodeNameOf(it.source)} → ${it.label} → ${window.nodeNameOf(it.target)}`, kind: "create-edge" });
    }

    if (it.kind === "entity-type-fix") {
      setNodes(ns => ns.map(n => n.id === it.subject ? { ...n, type: it.proposed_type } : n));
      setInbox(ix => ix.filter(x => x.id !== inboxId));
      pushFlash("flashNodes", [it.subject], 1400);
      logAudit({ action: "Type fix applied", target: `${window.nodeNameOf(it.subject)} → ${it.proposed_type}`, kind: "type-fix" });
    }
  };

  const rejectInboxItem = (inboxId) => {
    const it = inbox.find(x => x.id === inboxId);
    if (!it) return;
    setInbox(ix => ix.filter(x => x.id !== inboxId));
    const label = it.kind === "entity" ? it.name : (it.kind === "relation" ? it.label : `correction sur ${window.nodeNameOf(it.subject)}`);
    logAudit({ action: "Proposal rejected", target: label, kind: "reject" });
  };

  const createRelation = ({ source, target, label, status = "validated", quiet = false }) => {
    const rid = "r_" + Math.random().toString(36).slice(2, 8);
    setEdges(es => [...es, { id: rid, source, target, label, strength: 0.9, status }]);
    if (!quiet) {
      pushFlash("flashEdges", [rid], 2000);
      logAudit({ action: "Relation created", target: `${window.nodeNameOf(source)} → ${label} → ${window.nodeNameOf(target)}`, kind: "create-edge" });
    }
    return rid;
  };

  // Right-click on a node → "Run Wargame from here"
  _osmUseEffect(() => {
    const onWar = (e) => {
      setWargame({ active: true, originId: e.detail.nodeId, depth: 3 });
      setWargameRunId(r => r + 1);
    };
    window.addEventListener("onto-wargame-from", onWar);
    return () => window.removeEventListener("onto-wargame-from", onWar);
  }, []);

  // ── Demo bus integration ─────────────────────────────────────────────
  _osmUseEffect(() => {
    if (!demoBus) return;
    const off = demoBus.on("studio", async (msg) => {
      if (msg.kind === "focus-pair") {
        // Fly the viewport so two specific nodes are centered.
        const a = nodes.find(n => n.id === msg.a);
        const b = nodes.find(n => n.id === msg.b);
        if (!a || !b) return;
        if (window.__ontoCanvasViewSet) {
          const wrap = document.querySelector(".onto-canvas");
          const w = wrap ? wrap.clientWidth : 1200;
          const h = wrap ? wrap.clientHeight : 600;
          const cx = (a.x + b.x + window.NODE_W) / 2;
          const cy = (a.y + b.y + window.NODE_H) / 2;
          const k = 0.95;
          window.__ontoCanvasViewSet({ x: w / 2 - cx * k, y: h / 2 - cy * k, k });
        }
        setSelectedId(msg.a);
        pushFlash("flashNodes", [msg.a, msg.b], 1000);
      }
      if (msg.kind === "highlight") {
        pushFlash("flashNodes", [msg.id], 1200);
        setSelectedId(msg.id);
      }
      if (msg.kind === "create-edge") {
        createRelation({ source: msg.source, target: msg.target, label: msg.label, status: msg.status || "validated" });
      }
      if (msg.kind === "set-wargame") {
        setWargame({ active: !!msg.active, originId: msg.originId || null, depth: msg.depth ?? 3 });
        setWargameRunId(r => r + 1);
      }
    });
    return off;
  }, [demoBus, nodes]);

  // ── Selected entity (node or edge) ───────────────────────────────────
  const selectedNode = nodes.find(n => n.id === selectedId);
  const selectedEdge = edges.find(e => e.id === selectedId);
  const selectedInbox = inbox.find(i => i.id === inboxSelectedId);

  const wargameImpacted = _osmUseMemo(() => {
    if (!wargame.active || !wargame.originId) return null;
    // Walk BFS just to make a tidy list for the panel.
    const adj = {};
    edges.forEach(e => {
      if (!adj[e.source]) adj[e.source] = [];
      if (!adj[e.target]) adj[e.target] = [];
      adj[e.source].push(e.target);
      adj[e.target].push(e.source);
    });
    const depth = { [wargame.originId]: 0 };
    let frontier = [wargame.originId];
    const max = wargame.depth ?? 3;
    for (let d = 1; d <= max; d++) {
      const next = [];
      for (const id of frontier) {
        for (const nb of (adj[id] || [])) {
          if (depth[nb] === undefined) { depth[nb] = d; next.push(nb); }
        }
      }
      frontier = next;
      if (!next.length) break;
    }
    return Object.entries(depth)
      .filter(([id]) => id !== wargame.originId)
      .map(([id, d]) => ({ node: nodes.find(n => n.id === id), depth: d }))
      .filter(x => x.node)
      .sort((a, b) => a.depth - b.depth || a.node.name.localeCompare(b.node.name));
  }, [wargame, edges, nodes]);

  // ── Render ───────────────────────────────────────────────────────────
  return (
    <div className={`onto-shell layout-${layoutVariant || "split"}${presentation ? " is-presentation" : ""}`}>
      {!presentation && (
        <div className="onto-header">
          <div className="onto-header-left">
            <h1>Ontology Studio</h1>
            <div className="onto-sub">
              <span>{nodes.length} entities · {edges.length} relations · {inbox.length} pending</span>
              <span className="dot-sep">·</span>
              <span className="onto-tier-note" title="Visual editor for the LightRAG ontology. Validated by stewards; immediate impact on retrieval graph traversals.">
                <Icon name="info-circle" size={11} /> visual editing · steward-driven
              </span>
            </div>
          </div>
          <div className="onto-header-actions">
            <button
              className={`btn${wargame.active ? " primary" : ""}`}
              onClick={() => {
                if (wargame.active) setWargame({ active: false, originId: null, depth: 3 });
                else setWargame({ active: true, originId: selectedNode ? selectedNode.id : nodes[0].id, depth: 3 });
                setWargameRunId(r => r + 1);
              }}
              title="Toggle resilience stress-test"
            >
              <Icon name="alert-triangle" size={12} /> {wargame.active ? "Resilience on" : "Resilience test"}
            </button>
            <button className="btn subtle" onClick={() => {
              setNodes(window.ONTOLOGY_SEED_NODES.map(n => ({ ...n })));
              setEdges(window.ONTOLOGY_SEED_EDGES.map(e => ({ ...e })));
              setInbox(window.ONTOLOGY_INBOX_SEED.map(i => ({ ...i })));
              setAudit(window.SEED_AUDIT.slice());
              setWargame({ active: false, originId: null, depth: 3 });
            }}>
              <Icon name="refresh" size={12} /> Reset
            </button>
          </div>
        </div>
      )}

      <div className="onto-body">
        {layoutVariant !== "canvas" && (
          <window.InboxPanel
            items={inbox}
            onDrop={dropInboxItem}
            onReject={rejectInboxItem}
            onSelect={(id) => { setInboxSelectedId(id); setSelectedId(null); }}
            selectedId={inboxSelectedId}
            filter={inboxFilter}
            setFilter={setInboxFilter}
            count={inbox.length}
            collapsed={inboxCollapsed || presentation}
            onToggleCollapse={() => setInboxCollapsed(c => !c)}
          />
        )}

        <window.StudioCanvas
          nodes={nodes}
          edges={edges}
          setNodes={setNodes}
          setEdges={setEdges}
          selectedId={selectedId}
          setSelectedId={(id) => { setSelectedId(id); setInboxSelectedId(null); }}
          hoverNodeId={hoverNodeId}
          setHoverNodeId={setHoverNodeId}
          wargame={wargame}
          wargameRunId={wargameRunId}
          onCreateRelation={({ source, target, label }) => createRelation({ source, target, label, status: "validated" })}
          onDropInbox={dropInboxItem}
          ephemeral={ephemeral}
          setEphemeral={setEphemeral}
          fullscreenInspector={layoutVariant === "canvas"}
          presentation={presentation}
        />

        {layoutVariant !== "canvas" && layoutVariant !== "bottom" && !presentation && (
          <InspectorPanel
            node={selectedNode}
            edge={selectedEdge}
            inbox={selectedInbox}
            audit={audit}
            wargame={wargame}
            wargameImpacted={wargameImpacted}
            onTypeChange={(t) => {
              if (!selectedNode) return;
              setNodes(ns => ns.map(n => n.id === selectedNode.id ? { ...n, type: t } : n));
              logAudit({ action: "Type changed", target: `${selectedNode.name} → ${t}`, kind: "type-fix" });
            }}
            onRunWargame={(id) => { setWargame({ active: true, originId: id, depth: 3 }); setWargameRunId(r => r + 1); }}
            onExitWargame={() => setWargame({ active: false, originId: null, depth: 3 })}
            onSelect={(id) => setSelectedId(id)}
            nodes={nodes}
            collapsed={inspectorCollapsed}
            onToggleCollapse={() => setInspectorCollapsed(c => !c)}
          />
        )}
      </div>

      {layoutVariant === "bottom" && !presentation && (
        <BottomInspector
          node={selectedNode}
          edge={selectedEdge}
          audit={audit}
          wargame={wargame}
          wargameImpacted={wargameImpacted}
          nodes={nodes}
          onSelect={(id) => setSelectedId(id)}
        />
      )}

      {presentation && wargame.active && wargameImpacted && (
        <PresentationOverlay
          origin={nodes.find(n => n.id === wargame.originId)}
          impacted={wargameImpacted}
          onClose={() => setWargame({ active: false, originId: null, depth: 3 })}
        />
      )}
    </div>
  );
};

// ─── Inspector (right rail) ─────────────────────────────────────────────
function InspectorPanel({
  node, edge, inbox, audit, wargame, wargameImpacted, onTypeChange,
  onRunWargame, onExitWargame, onSelect, nodes, collapsed, onToggleCollapse
}) {
  const colors = window.ONTOLOGY_TYPE_COLORS;

  if (collapsed) {
    return (
      <aside className="onto-inspector is-collapsed" onClick={onToggleCollapse} title="Expand Inspector">
        <div className="onto-inspector-collapsed-rail">
          <span>Inspector</span>
        </div>
      </aside>
    );
  }

  return (
    <aside className="onto-inspector">
      <div className="onto-inspector-h">
        <div className="onto-pane-title"><span>Inspector</span></div>
        <button className="onto-collapse" onClick={onToggleCollapse} title="Collapse Inspector">
          <Icon name="chevron-right" size={11} />
        </button>
      </div>

      {/* Wargame summary takes priority */}
      {wargame.active && wargameImpacted && (
        <WargameSummary
          origin={nodes.find(n => n.id === wargame.originId)}
          impacted={wargameImpacted}
          onSelect={onSelect}
          onExit={onExitWargame}
        />
      )}

      {!wargame.active && node && (
        <NodeInspector node={node} colors={colors} onTypeChange={onTypeChange} onRunWargame={onRunWargame} />
      )}
      {!wargame.active && edge && !node && (
        <EdgeInspector edge={edge} nodes={nodes} onSelect={onSelect} />
      )}
      {!wargame.active && inbox && !node && !edge && (
        <InboxItemInspector item={inbox} />
      )}
      {!wargame.active && !node && !edge && !inbox && (
        <div className="onto-inspector-empty">
          <Icon name="circle-dot" size={20} color="var(--color-text-tertiary)" />
          <div>Select a node, edge, or inbox item.</div>
        </div>
      )}

      <div className="onto-audit">
        <div className="section-label"><span>Audit trail</span></div>
        <ul className="onto-audit-list">
          {audit.slice(0, 8).map((a, i) => (
            <li key={i} className={`onto-audit-row kind-${a.kind}`}>
              <span className="onto-audit-ts">{a.ts}</span>
              <span className="onto-audit-who"><code>{a.who}</code></span>
              <span className="onto-audit-action">{a.action}</span>
              <span className="onto-audit-target" title={a.target}>{a.target}</span>
            </li>
          ))}
        </ul>
      </div>
    </aside>
  );
}

function NodeInspector({ node, colors, onTypeChange, onRunWargame }) {
  return (
    <div className="onto-inspector-content">
      <div className="onto-inspector-title">
        <span className="onto-detail-swatch" style={{ background: colors[node.type] }} />
        <h2>{node.name}</h2>
      </div>
      <div className="onto-inspector-row">
        <span className="onto-inspector-label">Type</span>
        <select
          className="onto-type-select"
          value={node.type}
          onChange={e => onTypeChange(e.target.value)}
          style={{ borderColor: colors[node.type] }}
        >
          {Object.keys(window.ONTOLOGY_TYPE_LABEL).map(t => (
            <option key={t} value={t}>{window.ONTOLOGY_TYPE_LABEL[t]}</option>
          ))}
        </select>
      </div>
      <div className="onto-inspector-row">
        <span className="onto-inspector-label">Status</span>
        <span className={`onto-status-pill is-${node.status}`}>{node.status}</span>
      </div>
      {node.summary && (
        <div className="onto-inspector-summary">{node.summary}</div>
      )}
      <div className="onto-inspector-actions">
        <button className="btn primary" onClick={() => onRunWargame(node.id)}>
          <Icon name="alert-triangle" size={12} /> Run resilience test from here
        </button>
        <button className="btn subtle" title="Open documents that mention this entity">
          <Icon name="external-link" size={12} /> View sources
        </button>
      </div>
    </div>
  );
}

function EdgeInspector({ edge, nodes, onSelect }) {
  const src = nodes.find(n => n.id === edge.source);
  const tgt = nodes.find(n => n.id === edge.target);
  return (
    <div className="onto-inspector-content">
      <div className="onto-inspector-title">
        <code style={{ fontSize: 12, padding: "2px 6px", background: "var(--twin-accent-soft-bg)", color: "var(--twin-accent-soft-text)", borderRadius: 4 }}>{edge.label}</code>
      </div>
      <div className="onto-inspector-edgepair">
        <button onClick={() => onSelect(edge.source)}><span className="onto-detail-swatch" style={{ background: window.ONTOLOGY_TYPE_COLORS[src && src.type] }} />{src && src.name}</button>
        <span className="onto-edgepair-arrow">→</span>
        <button onClick={() => onSelect(edge.target)}><span className="onto-detail-swatch" style={{ background: window.ONTOLOGY_TYPE_COLORS[tgt && tgt.type] }} />{tgt && tgt.name}</button>
      </div>
      <div className="onto-inspector-row">
        <span className="onto-inspector-label">Status</span>
        <span className={`onto-status-pill is-${edge.status}`}>{edge.status}</span>
      </div>
      <div className="onto-inspector-row">
        <span className="onto-inspector-label">Confidence</span>
        <div className="onto-conf-bar"><div style={{ width: `${(edge.strength || 0) * 100}%` }} /><span>{Math.round((edge.strength || 0) * 100)}%</span></div>
      </div>
    </div>
  );
}

function InboxItemInspector({ item }) {
  if (item.kind === "entity") {
    return (
      <div className="onto-inspector-content">
        <div className="onto-inspector-title">
          <span className="onto-detail-swatch" style={{ background: window.ONTOLOGY_TYPE_COLORS[item.type] }} />
          <h2>{item.name}</h2>
          <span className="onto-kind-tag" style={{ marginLeft: "auto" }}>entity · proposed</span>
        </div>
        <div className="onto-inspector-row"><span className="onto-inspector-label">Type</span><span>{window.ONTOLOGY_TYPE_LABEL[item.type]}</span></div>
        <div className="onto-inspector-row"><span className="onto-inspector-label">Confidence</span><div className="onto-conf-bar"><div style={{ width: `${item.confidence * 100}%` }} /><span>{Math.round(item.confidence * 100)}%</span></div></div>
        <div className="onto-inspector-row"><span className="onto-inspector-label">Evidence</span><code className="onto-evidence">{item.evidence}</code></div>
        <div className="onto-inspector-summary">{item.rationale}</div>
      </div>
    );
  }
  if (item.kind === "relation") {
    return (
      <div className="onto-inspector-content">
        <div className="onto-inspector-title"><code style={{ fontSize: 13 }}>{item.label}</code><span className="onto-kind-tag is-rel" style={{ marginLeft: "auto" }}>relation · proposed</span></div>
        <div className="onto-inspector-edgepair">
          <span><span className="onto-detail-swatch" /> {window.nodeNameOf(item.source)}</span>
          <span className="onto-edgepair-arrow">→</span>
          <span><span className="onto-detail-swatch" /> {window.nodeNameOf(item.target)}</span>
        </div>
        <div className="onto-inspector-row"><span className="onto-inspector-label">Confidence</span><div className="onto-conf-bar"><div style={{ width: `${item.confidence * 100}%` }} /><span>{Math.round(item.confidence * 100)}%</span></div></div>
        <div className="onto-inspector-row"><span className="onto-inspector-label">Evidence</span><code className="onto-evidence">{item.evidence}</code></div>
        <div className="onto-inspector-summary">{item.rationale}</div>
      </div>
    );
  }
  if (item.kind === "entity-type-fix") {
    return (
      <div className="onto-inspector-content">
        <div className="onto-inspector-title"><h2>Type fix</h2><span className="onto-kind-tag is-fix" style={{ marginLeft: "auto" }}>type fix</span></div>
        <div className="onto-inspector-summary">{item.note}</div>
        <div className="onto-inspector-row"><span className="onto-inspector-label">Subject</span><code>{window.nodeNameOf(item.subject)}</code></div>
        <div className="onto-inspector-row"><span className="onto-inspector-label">Change</span><span>{item.existing_type} → <b>{item.proposed_type}</b></span></div>
        <div className="onto-inspector-row"><span className="onto-inspector-label">Evidence</span><code className="onto-evidence">{item.evidence}</code></div>
      </div>
    );
  }
  return null;
}

// ─── Wargame summary panel ──────────────────────────────────────────────
function WargameSummary({ origin, impacted, onSelect, onExit }) {
  if (!origin) return null;
  const groups = { 1: [], 2: [], 3: [] };
  impacted.forEach(it => { if (groups[it.depth]) groups[it.depth].push(it); });
  const apps = impacted.filter(it => it.node.type === "APPLICATION" || it.node.type === "PRODUCT");
  return (
    <div className="onto-wargame-summary">
      <div className="onto-wargame-h">
        <span className="onto-wargame-icon"><Icon name="alert-triangle" size={12} /></span>
        <span>Impact analysis</span>
        <button className="onto-wargame-exit" onClick={onExit}><Icon name="x" size={10} /></button>
      </div>
      <div className="onto-wargame-origin">
        <div className="onto-wargame-origin-label">Failure origin</div>
        <button className="onto-wargame-origin-node" onClick={() => onSelect(origin.id)}>
          <span className="onto-detail-swatch" style={{ background: window.ONTOLOGY_TYPE_COLORS[origin.type] }} />
          {origin.name}
        </button>
      </div>
      <div className="onto-wargame-counts">
        <div><b>{groups[1].length}</b><span>1 hop</span></div>
        <div><b>{groups[2].length}</b><span>2-hop</span></div>
        <div><b>{groups[3].length}</b><span>3-hop</span></div>
      </div>
      {[1, 2, 3].map(d => groups[d].length > 0 && (
        <div key={d} className={`onto-wargame-group is-d${d}`}>
          <div className="onto-wargame-group-h">
            <span className={`onto-wargame-dot is-d${d}`} />
            <span>{d === 1 ? "Direct dependencies" : d === 2 ? "2-hop propagation" : "3-hop propagation"}</span>
            <span className="onto-wargame-group-count">{groups[d].length}</span>
          </div>
          <ul>
            {groups[d].map(it => (
              <li key={it.node.id} onClick={() => onSelect(it.node.id)}>
                <span className="onto-detail-swatch" style={{ background: window.ONTOLOGY_TYPE_COLORS[it.node.type] }} />
                <span className="onto-wargame-name">{it.node.name}</span>
                <span className="onto-wargame-typed">{window.ONTOLOGY_TYPE_LABEL[it.node.type]}</span>
              </li>
            ))}
          </ul>
        </div>
      ))}
      {apps.length > 0 && (
        <div className="onto-wargame-pra">
          <Icon name="info-circle" size={11} />
          <span>{apps.length} application{apps.length === 1 ? "" : "s"} touched — PRA fallback runbooks linked in the graph.</span>
        </div>
      )}
    </div>
  );
}

// ─── Bottom inspector variant (compact horizontal strip) ────────────────
function BottomInspector({ node, edge, audit, wargame, wargameImpacted, nodes, onSelect }) {
  return (
    <div className="onto-bottom-inspector">
      <div className="onto-bottom-inspector-inner">
        {wargame.active && wargameImpacted ? (
          <div className="onto-bottom-wargame">
            <div className="onto-bottom-h">
              <Icon name="alert-triangle" size={11} color="var(--twin-red-vivid)" />
              <span>Impact depuis <b>{nodes.find(n => n.id === wargame.originId) ? nodes.find(n => n.id === wargame.originId).name : "—"}</b></span>
            </div>
            <ul className="onto-bottom-impact-list">
              {wargameImpacted.slice(0, 12).map(it => (
                <li key={it.node.id} className={`is-d${it.depth}`} onClick={() => onSelect(it.node.id)}>
                  <span className="onto-detail-swatch" style={{ background: window.ONTOLOGY_TYPE_COLORS[it.node.type] }} />
                  {it.node.name}<span className="onto-bottom-depth">·{it.depth}</span>
                </li>
              ))}
            </ul>
          </div>
        ) : node ? (
          <div className="onto-bottom-node">
            <div className="onto-bottom-h">
              <span className="onto-detail-swatch" style={{ background: window.ONTOLOGY_TYPE_COLORS[node.type] }} />
              <b>{node.name}</b>
              <span className="onto-bottom-type">{window.ONTOLOGY_TYPE_LABEL[node.type]}</span>
              <span className={`onto-status-pill is-${node.status}`}>{node.status}</span>
            </div>
            <div className="onto-bottom-summary">{node.summary || "—"}</div>
          </div>
        ) : (
          <div className="onto-bottom-empty">Select a node or relation to inspect</div>
        )}
        <div className="onto-bottom-audit">
          <div className="onto-bottom-audit-h">Recent activity</div>
          <ul>
            {audit.slice(0, 3).map((a, i) => (
              <li key={i}>
                <span className="onto-audit-ts">{a.ts}</span>
                <span className="onto-audit-action">{a.action}</span>
                <span className="onto-audit-target">{a.target}</span>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}

// ─── Presentation-mode overlay (clean wargame display, no chrome) ───────
function PresentationOverlay({ origin, impacted, onClose }) {
  if (!origin) return null;
  return (
    <div className="onto-presentation-overlay">
      <div className="onto-presentation-card">
        <div className="onto-presentation-h">
          <Icon name="alert-triangle" size={14} color="var(--twin-red-vivid)" />
          <span>Critical dependency identified</span>
          <button onClick={onClose}><Icon name="x" size={12} /></button>
        </div>
        <div className="onto-presentation-origin">Failure origin · <b>{origin.name}</b></div>
        <div className="onto-presentation-counts">
          <div><b>{impacted.filter(i => i.depth === 1).length}</b> direct</div>
          <div><b>{impacted.filter(i => i.depth === 2).length}</b> 2-hop</div>
          <div><b>{impacted.filter(i => i.depth === 3).length}</b> 3-hop</div>
        </div>
      </div>
    </div>
  );
}

Object.assign(window, {
  InspectorPanel, NodeInspector, EdgeInspector, InboxItemInspector,
  WargameSummary, BottomInspector, PresentationOverlay
});
