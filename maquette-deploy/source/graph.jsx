// Knowledge Graph tab — LightRAG entity extraction surfaced with a live force-sim layout,
// tag-aware filtering (KG ↔ thesaurus pivot), search→focus+zoom, drag-to-pin nodes, and
// PNG / JSON sub-graph export. Read-only by tier (Twin RAG); :MENTIONED_IN traversal
// remains gated behind Twin Graph.

const { useState: _useStateKG, useMemo: _useMemoKG, useRef: _useRefKG, useEffect: _useEffectKG, useCallback: _useCallbackKG } = React;

const KG_TYPE_LABEL = {
  PRODUCT: "Product", TECHNOLOGY: "Technology", CONCEPT: "Concept",
  ORG: "Org", PERSON: "Person", LOCATION: "Location"
};

// ─── Force-directed layout hook ─────────────────────────────────────────
// Custom O(n²) sim with edge springs + Coulomb repulsion + soft center pull.
// Runs in rAF for `maxIters` then settles. Re-runs when the node/edge set changes
// (filter applied) so the picture re-arranges instead of looking broken.
// Positions are kept in a ref to avoid React re-renders on every tick — instead
// we bump a `tick` counter once per frame to force a re-render of just the SVG.
function useForceLayout(nodes, edges, opts) {
  const {
    cx = 500, cy = 340,
    repulsion = 14000,
    spring = 0.012,
    restLength = 110,
    center = 0.008,
    damping = 0.84,
    maxIters = 220,
    seed
  } = opts;

  const posRef = _useRefKG({});
  const fixedRef = _useRefKG({});
  const draggingRef = _useRefKG(null);
  const [_, force] = _useStateKG(0);
  const tickRender = () => force(t => (t + 1) % 1e9);

  // Use a stable filter signature to detect when the visible node/edge set changes.
  const sig = nodes.map(n => n.id).join(",") + "|" + edges.map(e => e.id).join(",");

  _useEffectKG(() => {
    if (!nodes.length) return;
    // Seed positions: keep existing ones, initialize new nodes near the seed
    // hint (their precomputed x/y) so the initial run doesn't fly across the
    // canvas. After the first run, dragged positions are preserved.
    nodes.forEach(n => {
      if (!posRef.current[n.id]) {
        const sx = (seed && seed[n.id] && typeof seed[n.id].x === "number") ? seed[n.id].x : cx + (Math.random() - 0.5) * 300;
        const sy = (seed && seed[n.id] && typeof seed[n.id].y === "number") ? seed[n.id].y : cy + (Math.random() - 0.5) * 240;
        posRef.current[n.id] = { x: sx, y: sy, vx: 0, vy: 0 };
      }
    });
    // Drop positions for nodes no longer in the set so they don't haunt the sim.
    for (const id of Object.keys(posRef.current)) {
      if (!nodes.find(n => n.id === id)) delete posRef.current[id];
    }

    let iters = 0;
    let raf;
    const step = () => {
      const pos = posRef.current;

      // Pairwise repulsion — gentle, capped so close nodes don't explode.
      for (let i = 0; i < nodes.length; i++) {
        const a = pos[nodes[i].id]; if (!a) continue;
        for (let j = i + 1; j < nodes.length; j++) {
          const b = pos[nodes[j].id]; if (!b) continue;
          let dx = b.x - a.x;
          let dy = b.y - a.y;
          let dist2 = dx * dx + dy * dy + 1;
          const dist = Math.sqrt(dist2);
          // Cap dist2 so very close pairs don't get infinite force
          const f = repulsion / Math.max(dist2, 400);
          const fx = (dx / dist) * f;
          const fy = (dy / dist) * f;
          if (!fixedRef.current[nodes[i].id] && draggingRef.current !== nodes[i].id) { a.vx -= fx; a.vy -= fy; }
          if (!fixedRef.current[nodes[j].id] && draggingRef.current !== nodes[j].id) { b.vx += fx; b.vy += fy; }
        }
      }

      // Edge springs toward rest length.
      edges.forEach(e => {
        const a = pos[e.source], b = pos[e.target];
        if (!a || !b) return;
        const dx = b.x - a.x, dy = b.y - a.y;
        const dist = Math.sqrt(dx * dx + dy * dy) + 0.01;
        const force = spring * (dist - restLength);
        const fx = (dx / dist) * force;
        const fy = (dy / dist) * force;
        if (!fixedRef.current[e.source] && draggingRef.current !== e.source) { a.vx += fx; a.vy += fy; }
        if (!fixedRef.current[e.target] && draggingRef.current !== e.target) { b.vx -= fx; b.vy -= fy; }
      });

      // Weak pull toward viewport center to prevent drift off-canvas.
      nodes.forEach(n => {
        const p = pos[n.id]; if (!p) return;
        if (fixedRef.current[n.id] || draggingRef.current === n.id) return;
        p.vx += (cx - p.x) * center;
        p.vy += (cy - p.y) * center;
      });

      // Integrate with velocity damping.
      nodes.forEach(n => {
        const p = pos[n.id]; if (!p) return;
        if (fixedRef.current[n.id] || draggingRef.current === n.id) { p.vx = 0; p.vy = 0; return; }
        p.vx *= damping;
        p.vy *= damping;
        p.x += p.vx;
        p.y += p.vy;
      });

      tickRender();
      iters++;
      if (iters < maxIters) {
        raf = requestAnimationFrame(step);
      }
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sig]);

  const startDrag = _useCallbackKG((id) => { draggingRef.current = id; }, []);
  const endDrag = _useCallbackKG(() => { draggingRef.current = null; }, []);
  const moveDragTo = _useCallbackKG((id, x, y) => {
    const p = posRef.current[id];
    if (!p) return;
    p.x = x; p.y = y; p.vx = 0; p.vy = 0;
    tickRender();
  }, []);
  const pin = _useCallbackKG((id, isPinned) => {
    fixedRef.current[id] = isPinned;
  }, []);
  const positionOf = (id) => posRef.current[id];

  return { positionOf, startDrag, endDrag, moveDragTo, pin, fixedRef };
}

// ─── Seed metadata for the most-mentioned entities + relations. Lives here
// (not in data.js) so it stays scoped to the editor without polluting the
// retrieval/graph layout fixtures.
const KG_SEED_ENTITY_PROPS = {
  e_oracle:    { version: "19c",      license: "Enterprise Edition", owner_team: "DBA-CIB",   last_audit: "2026-03-12", critical_tier: "1" },
  e_rman:      { backup_strategy: "incremental + archive log", retention_days: "30", cron: "0 2 * * *" },
  e_archlog:   { format: "Oracle binary", rotation_min: "30" },
  e_rhel:      { kernel: "5.14.0-362", subscription: "Premium",  cis_baseline: "1.0.0" },
  e_vmware:    { build: "8.0 U2",  ha_enabled: "yes", drs_enabled: "yes" },
  e_memgraph:  { version: "2.18",  ha_cluster: "3 nodes",     replication: "sync" },
  e_swift:     { bic: "—",         network: "FIN+InterAct",   iso_target: "20022 · 2025-11" },
  e_iso20022:  { migration_phase: "2", deadline: "2025-11-22", impacted_messages: "MT103, MT202, MT202COV" },
  e_cft:       { vendor: "Axway",  protocol: "PeSIT, SFTP",   tps: "120 msg/s" },
  e_paris:     { lat: "48.8566",   lon: "2.3522",  power_kw: "1200", redundancy: "N+1" },
  e_aubervil:  { lat: "48.9136",   lon: "2.3833",  power_kw: "950",  redundancy: "N+1",  role: "DR / standby" },
  e_marc:      { palier: "2",      joined: "2018-04-03", focus: "Oracle · RMAN" },
  e_claire:    { palier: "3",      joined: "2015-09-12", focus: "Governance · Tags" }
};
const KG_SEED_REL_PROPS = {
  r_01: { validated_by: "marc.berthier", validated_at: "2026-01-08", evidence: "oracle-rman-procedure.pdf §2" },
  r_03: { certified: "Oracle ACS",    min_version: "RHEL 9.1", notes: "kernel.shmmax must be set" },
  r_06: { confirmed: "vSphere cluster manifest" },
  r_15: { migration_date: "2025-11-22", message_types: "MT → MX", iso_focus: "pacs.008.001.08" },
  r_18: { since: "2021-02-15" }
};

// ─── Main component ────────────────────────────────────────────────────
window.GraphTab = function GraphTab() {
  const baseEntities  = window.MOCK_GRAPH_ENTITIES  || [];
  const baseRelations = window.MOCK_GRAPH_RELATIONS || [];
  const thesaurus = window.MOCK_THESAURUS || [];
  const COLORS = window.GRAPH_TYPE_COLORS;

  // URL-persisted filter state
  const [q, setQ] = window.useUrlParam("gq", "");
  const [activeTypes, setActiveTypes] = window.useUrlArrayParam("gtype", Object.keys(KG_TYPE_LABEL));
  const [activeTags, setActiveTags] = window.useUrlArrayParam("gtag", []);
  const [selectedId, setSelectedId] = window.useUrlParam("gent", "e_oracle");
  // Right-rail selection can also be a relation — when the user clicks an edge
  // row in the detail panel. Mutually exclusive with entity selection at the
  // panel level (the panel decides which to show).
  const [selectedRelId, setSelectedRelId] = _useStateKG(null);
  // Local edits, keyed by id. The original mocks are immutable so we layer
  // changes on top — that lets the user edit, save, and even re-export via the
  // existing PNG/JSON export with metadata included.
  const [entityOverrides, setEntityOverrides] = _useStateKG({});
  const [relationOverrides, setRelationOverrides] = _useStateKG({});

  // Merge: seed props → base mock → user override. The shallow merge on
  // `properties` is intentional so removing a key in the override actually
  // removes it from the effective object.
  const entities = _useMemoKG(() => baseEntities.map(e => {
    const ov = entityOverrides[e.id];
    const seed = KG_SEED_ENTITY_PROPS[e.id] || {};
    if (!ov) return { ...e, properties: { ...seed } };
    return {
      ...e,
      ...ov,
      properties: ov.properties !== undefined ? { ...ov.properties } : { ...seed }
    };
  }), [baseEntities, entityOverrides]);
  const relations = _useMemoKG(() => baseRelations.map(r => {
    const ov = relationOverrides[r.id];
    const seed = KG_SEED_REL_PROPS[r.id] || {};
    if (!ov) return { ...r, properties: { ...seed } };
    return {
      ...r,
      ...ov,
      properties: ov.properties !== undefined ? { ...ov.properties } : { ...seed }
    };
  }), [baseRelations, relationOverrides]);

  // Mutation helpers exposed to the detail panel.
  const saveEntity = _useCallbackKG((id, patch) => {
    setEntityOverrides(prev => ({ ...prev, [id]: { ...(prev[id] || {}), ...patch } }));
  }, []);
  const saveRelation = _useCallbackKG((id, patch) => {
    setRelationOverrides(prev => ({ ...prev, [id]: { ...(prev[id] || {}), ...patch } }));
  }, []);
  const [hoverId, setHoverId] = _useStateKG(null);
  const [zoom, setZoom] = _useStateKG(1);
  const [pan, setPan] = _useStateKG({ x: 0, y: 0 });
  const [tagInput, setTagInput] = _useStateKG("");
  const [exportOpen, setExportOpen] = _useStateKG(false);
  const [pinnedIds, setPinnedIds] = _useStateKG(() => new Set());

  const svgRef = _useRefKG(null);
  const canvasRef = _useRefKG(null);
  const exportMenuRef = _useRefKG(null);

  // ── Filtering ────────────────────────────────────────────────────────
  const typeCounts = _useMemoKG(() => {
    const c = {};
    entities.forEach(e => { c[e.type] = (c[e.type] || 0) + 1; });
    return c;
  }, [entities]);

  const tagCounts = _useMemoKG(() => {
    const c = {};
    entities.forEach(e => (e.tags || []).forEach(t => { c[t] = (c[t] || 0) + 1; }));
    return c;
  }, [entities]);

  const matches = _useMemoKG(() => {
    const needle = q.trim().toLowerCase();
    return entities.filter(e => {
      if (!activeTypes.includes(e.type)) return false;
      if (activeTags.length > 0) {
        const ets = e.tags || [];
        if (!activeTags.every(t => ets.includes(t))) return false;
      }
      if (!needle) return true;
      return e.name.toLowerCase().includes(needle) || (e.summary || "").toLowerCase().includes(needle);
    });
  }, [entities, q, activeTypes, activeTags]);

  const visibleIds = _useMemoKG(() => new Set(matches.map(e => e.id)), [matches]);
  const visibleRels = _useMemoKG(
    () => relations.filter(r => visibleIds.has(r.source) && visibleIds.has(r.target)),
    [relations, visibleIds]
  );

  // Force layout — seed with precomputed positions from mock so the first
  // settle doesn't fly all over the place; subsequent filter changes re-run.
  const layout = useForceLayout(matches, visibleRels, {
    cx: 500, cy: 340,
    seed: _useMemoKG(() => {
      const s = {};
      entities.forEach(e => { s[e.id] = { x: e.x, y: e.y }; });
      return s;
    }, [entities])
  });

  // Selected entity / neighbor highlight set
  const selected = entities.find(e => e.id === selectedId) || matches[0] || null;
  const neighbors = _useMemoKG(() => {
    if (!selected) return { rels: [], nodes: [] };
    const rels = relations.filter(r => r.source === selected.id || r.target === selected.id);
    const ids = new Set();
    rels.forEach(r => { ids.add(r.source); ids.add(r.target); });
    ids.delete(selected.id);
    return { rels, nodes: entities.filter(e => ids.has(e.id)) };
  }, [selected, entities, relations]);
  const highlightIds = _useMemoKG(() => {
    const ids = new Set([selected ? selected.id : null]);
    neighbors.nodes.forEach(n => ids.add(n.id));
    return ids;
  }, [selected, neighbors]);

  // ── Search → focus + zoom-to-node ────────────────────────────────────
  // When the search narrows to a single match, gently pan the viewport so it
  // sits at the canvas centre and select it. Bounce a short pulse via CSS.
  _useEffectKG(() => {
    if (q.trim().length < 2) return;
    if (matches.length !== 1) return;
    const m = matches[0];
    setSelectedId(m.id);
    // Allow the sim to position the node, then center on it.
    const settle = setTimeout(() => {
      const p = layout.positionOf(m.id);
      if (!p) return;
      const z = Math.min(2, Math.max(1.4, zoom));
      setZoom(z);
      // pan such that node lands at center of svg coordinate space (500, 340) in viewbox terms
      setPan({ x: (500 - p.x) * z, y: (340 - p.y) * z });
    }, 320);
    return () => clearTimeout(settle);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [q, matches.length]);

  // ── Pan / zoom ───────────────────────────────────────────────────────
  const dragRef = _useRefKG(null);
  const onCanvasMouseDown = (e) => {
    // Right click / middle click should never start node drag; this handler
    // sits on the SVG so it only fires for the empty canvas (nodes stop
    // propagation in their own onMouseDown).
    if (e.button !== 0) return;
    dragRef.current = { x: e.clientX, y: e.clientY, panX: pan.x, panY: pan.y };
  };
  _useEffectKG(() => {
    const onMove = (e) => {
      if (!dragRef.current) return;
      setPan({
        x: dragRef.current.panX + (e.clientX - dragRef.current.x),
        y: dragRef.current.panY + (e.clientY - dragRef.current.y)
      });
    };
    const onUp = () => { dragRef.current = null; };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    return () => { window.removeEventListener("mousemove", onMove); window.removeEventListener("mouseup", onUp); };
  }, []);
  const onWheel = (e) => {
    e.preventDefault();
    const dz = e.deltaY < 0 ? 1.1 : 0.9;
    setZoom(z => Math.max(0.4, Math.min(3, z * dz)));
  };
  const resetView = () => { setZoom(1); setPan({ x: 0, y: 0 }); };

  // Node drag — convert viewport delta to SVG coordinate delta via the inverse
  // of the active transform (translate + scale).
  const nodeDragStateRef = _useRefKG(null);
  const startNodeDrag = (e, id) => {
    e.stopPropagation();
    e.preventDefault();
    layout.startDrag(id);
    nodeDragStateRef.current = { id, startClientX: e.clientX, startClientY: e.clientY };
  };
  _useEffectKG(() => {
    const onMove = (e) => {
      const st = nodeDragStateRef.current;
      if (!st) return;
      const svg = svgRef.current;
      if (!svg) return;
      // Map client coords → SVG viewBox coords accounting for the wrapping
      // <g transform="translate(pan) scale(zoom)">.
      const rect = svg.getBoundingClientRect();
      const vbW = 1000, vbH = 680;
      const scaleX = vbW / rect.width;
      const scaleY = vbH / rect.height;
      const localX = (e.clientX - rect.left) * scaleX;
      const localY = (e.clientY - rect.top) * scaleY;
      const x = (localX - pan.x) / zoom;
      const y = (localY - pan.y) / zoom;
      layout.moveDragTo(st.id, x, y);
    };
    const onUp = () => {
      const st = nodeDragStateRef.current;
      if (!st) return;
      layout.endDrag();
      nodeDragStateRef.current = null;
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    return () => { window.removeEventListener("mousemove", onMove); window.removeEventListener("mouseup", onUp); };
  }, [zoom, pan.x, pan.y]); // re-bind when transform changes so math stays right

  const togglePin = (id) => {
    setPinnedIds(prev => {
      const next = new Set(prev);
      if (next.has(id)) { next.delete(id); layout.pin(id, false); }
      else { next.add(id); layout.pin(id, true); }
      return next;
    });
  };

  // ── Type / tag filter toggles ────────────────────────────────────────
  const toggleType = (t) => {
    if (activeTypes.includes(t)) setActiveTypes(activeTypes.filter(x => x !== t));
    else setActiveTypes([...activeTypes, t]);
  };
  const addTagFilter = (tag) => {
    if (!tag || activeTags.includes(tag)) return;
    setActiveTags([...activeTags, tag]);
    setTagInput("");
  };
  const removeTagFilter = (tag) => setActiveTags(activeTags.filter(x => x !== tag));

  const tagSuggestions = _useMemoKG(() => {
    const v = tagInput.toLowerCase();
    return thesaurus
      .filter(t => !activeTags.includes(t.tag))
      .filter(t => tagCounts[t.tag])
      .filter(t => !v || t.tag.includes(v))
      .slice(0, 6);
  }, [tagInput, activeTags, thesaurus, tagCounts]);

  // ── Export ───────────────────────────────────────────────────────────
  _useEffectKG(() => {
    if (!exportOpen) return;
    const onDown = (e) => { if (exportMenuRef.current && !exportMenuRef.current.contains(e.target)) setExportOpen(false); };
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, [exportOpen]);

  const exportJson = () => {
    const sub = {
      exported_at: new Date().toISOString(),
      workspace: "cib",
      filters: { q, types: activeTypes, tags: activeTags },
      nodes: matches.map(e => ({
        id: e.id, name: e.name, type: e.type,
        tags: e.tags || [], mentions: e.mentions, sources: e.sources, summary: e.summary
      })),
      edges: visibleRels.map(r => ({ id: r.id, source: r.source, target: r.target, label: r.label, strength: r.strength }))
    };
    const blob = new Blob([JSON.stringify(sub, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `twin-kg-subgraph-${new Date().toISOString().slice(0, 10)}.json`;
    document.body.appendChild(a); a.click();
    setTimeout(() => { URL.revokeObjectURL(url); a.remove(); }, 0);
    setExportOpen(false);
  };

  const exportPng = () => {
    const svg = svgRef.current;
    if (!svg) return;
    const xml = new XMLSerializer().serializeToString(svg);
    // Inline currentColor → explicit value so the rasterized PNG isn't blank
    // outside the iframe document context. We re-serialize after wrapping
    // styles inline. Simpler approach: rely on inline attributes already
    // present on nodes (fill/stroke are inline), and skip css-derived stroke.
    const blob = new Blob([xml], { type: "image/svg+xml" });
    const url = URL.createObjectURL(blob);
    const img = new Image();
    img.onload = () => {
      const c = document.createElement("canvas");
      c.width = 2000; c.height = 1360;
      const ctx = c.getContext("2d");
      ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue("--color-background-primary") || "#fff";
      ctx.fillRect(0, 0, c.width, c.height);
      ctx.drawImage(img, 0, 0, c.width, c.height);
      c.toBlob((b) => {
        const pngUrl = URL.createObjectURL(b);
        const a = document.createElement("a");
        a.href = pngUrl;
        a.download = `twin-kg-subgraph-${new Date().toISOString().slice(0, 10)}.png`;
        document.body.appendChild(a); a.click();
        setTimeout(() => { URL.revokeObjectURL(pngUrl); URL.revokeObjectURL(url); a.remove(); }, 0);
      }, "image/png");
    };
    img.src = url;
    setExportOpen(false);
  };

  return (
    <div className="kg">
      <div className="kg-header">
        <div>
          <h1>Knowledge Graph</h1>
          <div className="kg-sub">
            <span>{entities.length} entities · {relations.length} relations · workspace <code>cib</code></span>
            <span className="dot-sep">·</span>
            <span className="kg-tier-note" title="Read-only view of LightRAG entity extraction. :MENTIONED_IN traversal + tag-filtered graph reasoning are Twin Graph tier features.">
              <Icon name="info-circle" size={11} /> read-only · Twin Graph tier for traversal
            </span>
          </div>
        </div>
        <div className="kg-header-actions">
          <div className="kg-search">
            <Icon name="search" size={12} color="var(--color-text-tertiary)" />
            <input
              type="text"
              placeholder="Search entities…"
              value={q}
              onChange={e => setQ(e.target.value)}
            />
            {q && <button className="kg-search-clear" onClick={() => setQ("")} aria-label="Clear"><Icon name="x" size={11} /></button>}
          </div>
          <button className="ghost-btn" onClick={resetView} title="Reset pan + zoom">
            <Icon name="refresh" size={12} /> Reset view
          </button>
          <div ref={exportMenuRef} style={{ position: "relative" }}>
            <button className="ghost-btn" onClick={() => setExportOpen(o => !o)} aria-expanded={exportOpen}>
              <Icon name="external-link" size={12} /> Export
              <Icon name={exportOpen ? "chevron-up" : "chevron-down"} size={10} />
            </button>
            {exportOpen && (
              <div className="kg-export-menu" role="menu">
                <button onClick={exportPng}>
                  <Icon name="external-link" size={11} />
                  <span>Sub-graph PNG</span>
                  <span className="kg-export-meta">visible nodes + edges</span>
                </button>
                <button onClick={exportJson}>
                  <Icon name="external-link" size={11} />
                  <span>Sub-graph JSON</span>
                  <span className="kg-export-meta">filters + topology</span>
                </button>
              </div>
            )}
          </div>
        </div>
      </div>

      <div className="kg-body">
        <aside className="kg-rail">
          {/* Entity types */}
          <div>
            <div className="kg-rail-h">Entity types</div>
            <ul className="kg-type-list">
              {Object.keys(KG_TYPE_LABEL).map(t => {
                const on = activeTypes.includes(t);
                return (
                  <li key={t}>
                    <button
                      className={`kg-type-row${on ? " is-on" : ""}`}
                      onClick={() => toggleType(t)}
                      aria-pressed={on}
                    >
                      <span className="kg-type-swatch" style={{ background: COLORS[t] }} />
                      <span className="kg-type-name">{KG_TYPE_LABEL[t]}</span>
                      <span className="kg-type-count">{typeCounts[t] || 0}</span>
                    </button>
                  </li>
                );
              })}
            </ul>
          </div>

          {/* Tag filter — pivot to thesaurus */}
          <div className="kg-tag-filter">
            <div className="kg-rail-h">
              Tags <em>— Twin</em>
            </div>
            {activeTags.length > 0 && (
              <div className="tag-chips" style={{ marginBottom: 6 }}>
                {activeTags.map(t => <TagChip key={t} tag={t} removable onRemove={removeTagFilter} />)}
              </div>
            )}
            <input
              className="kg-tag-input"
              value={tagInput}
              onChange={e => setTagInput(e.target.value.toLowerCase())}
              onKeyDown={e => {
                if (e.key === "Enter" && tagSuggestions[0]) { e.preventDefault(); addTagFilter(tagSuggestions[0].tag); }
                if (e.key === "Escape") setTagInput("");
              }}
              placeholder={activeTags.length ? "Add another…" : "Filter by tag…"}
            />
            {tagInput && tagSuggestions.length > 0 && (
              <div className="kg-tag-sugg">
                {tagSuggestions.map((s, i) => (
                  <button key={s.tag} className={`kg-tag-sugg-row${i === 0 ? " is-focus" : ""}`} onMouseDown={() => addTagFilter(s.tag)}>
                    <code>{s.tag}</code>
                    <span className="kg-tag-sugg-count">{tagCounts[s.tag]} ent</span>
                  </button>
                ))}
              </div>
            )}
            {activeTags.length > 0 && (
              <div className="kg-tag-note">
                <Icon name="info-circle" size={10} />
                <span>Showing {matches.length} of {entities.length} entities</span>
              </div>
            )}
          </div>

          <div className="kg-legend">
            <div className="kg-rail-h">Legend</div>
            <ul>
              <li><span className="kg-legend-line" /> relation</li>
              <li><span className="kg-legend-line strong" /> high confidence</li>
              <li><span className="kg-legend-dot" /> node size = mentions</li>
              <li><span className="kg-legend-pin">📌</span> drag to pin</li>
            </ul>
          </div>
        </aside>

        <div
          className="kg-canvas"
          ref={canvasRef}
          onWheel={onWheel}
        >
          <svg
            ref={svgRef}
            viewBox="0 0 1000 680"
            preserveAspectRatio="xMidYMid meet"
            className="kg-svg"
            onMouseDown={onCanvasMouseDown}
          >
            <defs>
              <marker id="kg-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
                <path d="M0,0 L10,5 L0,10 z" fill="var(--color-text-tertiary)" opacity="0.55" />
              </marker>
              <marker id="kg-arrow-hi" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
                <path d="M0,0 L10,5 L0,10 z" fill="var(--twin-accent)" />
              </marker>
            </defs>
            <g transform={`translate(${pan.x}, ${pan.y}) scale(${zoom})`}>
              {/* Edges first so nodes paint on top */}
              {visibleRels.map(r => {
                const a = layout.positionOf(r.source);
                const b = layout.positionOf(r.target);
                if (!a || !b) return null;
                const hi = selected && (r.source === selected.id || r.target === selected.id);
                const dim = selected && !hi;
                const strong = r.strength >= 0.75;
                return (
                  <g key={r.id} className={`kg-edge${hi ? " is-hi" : ""}${dim ? " is-dim" : ""}`}>
                    <line
                      x1={a.x} y1={a.y} x2={b.x} y2={b.y}
                      stroke={hi ? "var(--twin-accent)" : "var(--color-text-tertiary)"}
                      strokeWidth={hi ? 1.6 : (strong ? 1.1 : 0.7)}
                      strokeOpacity={hi ? 0.9 : (dim ? 0.08 : 0.32)}
                      markerEnd={hi ? "url(#kg-arrow-hi)" : "url(#kg-arrow)"}
                    />
                    {hi && (
                      <text
                        x={(a.x + b.x) / 2}
                        y={(a.y + b.y) / 2 - 4}
                        textAnchor="middle"
                        className="kg-edge-label"
                      >{r.label}</text>
                    )}
                  </g>
                );
              })}

              {/* Nodes */}
              {matches.map(e => {
                const p = layout.positionOf(e.id);
                if (!p) return null;
                const r = 8 + Math.min(18, Math.sqrt(e.mentions) * 0.9);
                const isSelected = selected && selected.id === e.id;
                const isNeighbor = highlightIds.has(e.id) && !isSelected;
                const isDim = selected && !highlightIds.has(e.id);
                const isHover = hoverId === e.id;
                const isPinned = pinnedIds.has(e.id);
                const isSingleMatch = q.trim().length >= 2 && matches.length === 1;
                return (
                  <g
                    key={e.id}
                    className={`kg-node${isSelected ? " is-selected" : ""}${isDim ? " is-dim" : ""}${isSingleMatch ? " is-pulse" : ""}`}
                    transform={`translate(${p.x}, ${p.y})`}
                    onClick={(ev) => { ev.stopPropagation(); setSelectedRelId(null); setSelectedId(e.id); }}
                    onMouseEnter={() => setHoverId(e.id)}
                    onMouseLeave={() => setHoverId(null)}
                    onMouseDown={(ev) => startNodeDrag(ev, e.id)}
                    onDoubleClick={(ev) => { ev.stopPropagation(); togglePin(e.id); }}
                    style={{ cursor: nodeDragStateRef.current && nodeDragStateRef.current.id === e.id ? "grabbing" : "grab", opacity: isDim ? 0.35 : 1 }}
                  >
                    {isSelected && <circle r={r + 7} className="kg-node-halo" />}
                    {isNeighbor && <circle r={r + 4} className="kg-node-halo subtle" />}
                    {isSingleMatch && <circle r={r + 10} className="kg-node-pulse" />}
                    <circle
                      r={r}
                      fill={COLORS[e.type]}
                      stroke={isSelected ? "var(--twin-accent)" : "var(--color-background-primary)"}
                      strokeWidth={isSelected ? 2.5 : 1.5}
                    />
                    {isPinned && (
                      <circle r={3} cx={r - 4} cy={-r + 4} fill="var(--twin-accent)" stroke="var(--color-background-primary)" strokeWidth={1} />
                    )}
                    <text
                      y={r + 12}
                      textAnchor="middle"
                      className={`kg-node-label${isSelected ? " is-selected" : ""}`}
                      style={{ fontWeight: isSelected || isHover ? 600 : 500 }}
                    >{e.name}</text>
                  </g>
                );
              })}
            </g>
          </svg>

          {matches.length === 0 && (
            <div className="kg-empty">
              <Icon name="search" size={20} color="var(--color-text-tertiary)" />
              <div>No entities match the current filter.</div>
              <button className="ghost-btn" onClick={() => { setQ(""); setActiveTypes(Object.keys(KG_TYPE_LABEL)); setActiveTags([]); }}>Clear filters</button>
            </div>
          )}

          <div className="kg-zoom-pill">
            <button onClick={() => setZoom(z => Math.max(0.4, z * 0.85))} aria-label="Zoom out"><Icon name="minus" size={11} /></button>
            <span>{Math.round(zoom * 100)}%</span>
            <button onClick={() => setZoom(z => Math.min(3, z * 1.18))} aria-label="Zoom in"><Icon name="plus" size={11} /></button>
          </div>

          {matches.length > 0 && (
            <div className="kg-hint">
              <Icon name="info-circle" size={10} />
              <span>drag node to move · double-click to pin · scroll to zoom</span>
            </div>
          )}
        </div>

        <GraphDetailPanel
          entity={selected}
          neighbors={neighbors}
          colors={COLORS}
          onSelect={(id) => { setSelectedRelId(null); setSelectedId(id); }}
          onFilterByTag={(tag) => addTagFilter(tag)}
          pinned={selected ? pinnedIds.has(selected.id) : false}
          onTogglePin={() => selected && togglePin(selected.id)}
          selectedRelId={selectedRelId}
          onSelectRelation={(id) => setSelectedRelId(id)}
          onClearRelation={() => setSelectedRelId(null)}
          relations={relations}
          entities={entities}
          onSaveEntity={saveEntity}
          onSaveRelation={saveRelation}
          typeLabels={KG_TYPE_LABEL}
        />
      </div>
    </div>
  );
};

function GraphDetailPanel({
  entity, neighbors, colors, onSelect, onFilterByTag, pinned, onTogglePin,
  selectedRelId, onSelectRelation, onClearRelation,
  relations, entities, onSaveEntity, onSaveRelation, typeLabels
}) {
  // ── Relation detail editor ──────────────────────────────────────────
  // When the user clicks an outgoing/incoming row, we pivot the rail to a
  // relation editor instead of the entity panel. Back button returns.
  const selectedRel = relations && selectedRelId && relations.find(r => r.id === selectedRelId);
  if (selectedRel) {
    const src = entities.find(n => n.id === selectedRel.source);
    const tgt = entities.find(n => n.id === selectedRel.target);
    return (
      <aside className="kg-detail">
        <RelationEditor
          rel={selectedRel}
          src={src}
          tgt={tgt}
          colors={colors}
          onSave={(patch) => onSaveRelation(selectedRel.id, patch)}
          onSelectNode={onSelect}
          onBack={onClearRelation}
        />
      </aside>
    );
  }

  if (!entity) {
    return (
      <aside className="kg-detail">
        <div className="kg-detail-empty">
          <Icon name="circle-dot" size={20} color="var(--color-text-tertiary)" />
          <div>Select a node to inspect</div>
        </div>
      </aside>
    );
  }
  return (
    <aside className="kg-detail">
      <EntityEditor
        entity={entity}
        neighbors={neighbors}
        colors={colors}
        typeLabels={typeLabels}
        pinned={pinned}
        onTogglePin={onTogglePin}
        onFilterByTag={onFilterByTag}
        onSelect={onSelect}
        onSelectRelation={onSelectRelation}
        onSave={(patch) => onSaveEntity(entity.id, patch)}
      />
    </aside>
  );
}

// ─── Entity editor — view + edit metadata, manage tags + properties ────
function EntityEditor({
  entity, neighbors, colors, typeLabels, pinned, onTogglePin,
  onFilterByTag, onSelect, onSelectRelation, onSave
}) {
  const [editing, setEditing] = _useStateKG(false);
  const [draft, setDraft] = _useStateKG(null);

  // Reset edit mode when switching entities.
  _useEffectKG(() => { setEditing(false); setDraft(null); }, [entity.id]);

  const startEdit = () => {
    setDraft({
      name: entity.name,
      type: entity.type,
      summary: entity.summary || "",
      tags: (entity.tags || []).slice(),
      properties: { ...(entity.properties || {}) }
    });
    setEditing(true);
  };
  const cancel = () => { setEditing(false); setDraft(null); };
  const commit = () => {
    onSave({
      name: draft.name.trim() || entity.name,
      type: draft.type,
      summary: draft.summary,
      tags: draft.tags,
      properties: draft.properties
    });
    setEditing(false);
    setDraft(null);
  };

  const incoming = neighbors.rels.filter(r => r.target === entity.id);
  const outgoing = neighbors.rels.filter(r => r.source === entity.id);
  const props = entity.properties || {};
  const propEntries = Object.entries(props);

  return (
    <>
      <div className="kg-detail-h">
        <div className="kg-detail-title">
          <span className="kg-detail-swatch" style={{ background: colors[entity.type] }} />
          {editing ? (
            <input
              className="kg-edit-input kg-edit-name"
              value={draft.name}
              onChange={e => setDraft(d => ({ ...d, name: e.target.value }))}
              placeholder="Name"
              autoFocus
            />
          ) : (
            <h2>{entity.name}</h2>
          )}
          {!editing && (
            <button
              className="kg-pin-btn"
              onClick={startEdit}
              title="Edit metadata"
              style={{ marginLeft: "auto" }}
            >
              <Icon name="edit" size={11} /> Edit
            </button>
          )}
          <button
            className={`kg-pin-btn${pinned ? " is-on" : ""}`}
            onClick={onTogglePin}
            title={pinned ? "Unpin (release to the layout)" : "Pin in place"}
            aria-pressed={pinned}
            style={editing ? undefined : { marginLeft: 4 }}
          >
            <Icon name={pinned ? "lock" : "lock-open"} size={11} />
            {pinned ? "Pinned" : "Pin"}
          </button>
        </div>
        {editing ? (
          <div className="kg-edit-row">
            <span className="kg-edit-label">Type</span>
            <select
              className="kg-edit-select"
              value={draft.type}
              onChange={e => setDraft(d => ({ ...d, type: e.target.value }))}
              style={{ borderTopColor: colors[draft.type] }}
            >
              {Object.keys(typeLabels).map(t => <option key={t} value={t}>{typeLabels[t]}</option>)}
            </select>
          </div>
        ) : (
          <div className="kg-detail-type" style={{ color: colors[entity.type] }}>
            {typeLabels[entity.type]}
          </div>
        )}
        {editing ? (
          <textarea
            className="kg-edit-input kg-edit-summary"
            rows={3}
            value={draft.summary}
            onChange={e => setDraft(d => ({ ...d, summary: e.target.value }))}
            placeholder="Short description"
          />
        ) : (
          <p className="kg-detail-summary">{entity.summary || "—"}</p>
        )}
        {!editing && (
          <div className="kg-detail-stats">
            <div><span className="kg-stat-n">{entity.mentions}</span><span className="kg-stat-l">mentions</span></div>
            <div><span className="kg-stat-n">{entity.sources}</span><span className="kg-stat-l">sources</span></div>
            <div><span className="kg-stat-n">{neighbors.rels.length}</span><span className="kg-stat-l">relations</span></div>
          </div>
        )}
      </div>

      {/* Tags */}
      <div className="kg-detail-section">
        <div className="section-label">
          <span>Tags {editing ? <em>— edit</em> : <em>— click to filter graph</em>}</span>
        </div>
        {editing ? (
          <TagEditor
            tags={draft.tags}
            onChange={(tags) => setDraft(d => ({ ...d, tags }))}
          />
        ) : (
          (entity.tags && entity.tags.length > 0) ? (
            <div className="tag-chips">
              {entity.tags.map(t => (
                <button key={t} className="tag-chip kg-tag-clickable" onClick={() => onFilterByTag(t)} title={`Filter graph to entities tagged "${t}"`}>
                  {t}
                </button>
              ))}
            </div>
          ) : (
            <div className="muted-sm">No tags.</div>
          )
        )}
      </div>

      {/* Properties (custom metadata) */}
      <div className="kg-detail-section">
        <div className="section-label">
          <span>Properties {editing ? <em>— add / remove</em> : <em>— custom metadata</em>}</span>
          {!editing && propEntries.length > 0 && <span className="kg-prop-count">{propEntries.length}</span>}
        </div>
        {editing ? (
          <PropEditor
            properties={draft.properties}
            onChange={(properties) => setDraft(d => ({ ...d, properties }))}
          />
        ) : (
          propEntries.length === 0 ? (
            <div className="muted-sm">No custom properties. <button className="kg-inline-add" onClick={startEdit}>+ Add some</button></div>
          ) : (
            <dl className="kg-prop-list">
              {propEntries.map(([k, v]) => (
                <div key={k} className="kg-prop-row">
                  <dt>{k}</dt>
                  <dd>{String(v)}</dd>
                </div>
              ))}
            </dl>
          )
        )}
      </div>

      {/* Edit actions */}
      {editing && (
        <div className="kg-detail-section kg-edit-actions">
          <button className="ghost-btn" onClick={cancel}>Cancel</button>
          <button className="ghost-btn primary" onClick={commit}><Icon name="check" size={11} /> Save</button>
        </div>
      )}

      {/* Outgoing / Incoming — read-only, clickable to switch the panel to
          the relation editor. Hidden in edit mode so the form stays focused. */}
      {!editing && (
        <>
          <div className="kg-detail-section">
            <div className="section-label"><span>Outgoing ({outgoing.length}) <em>— click to edit</em></span></div>
            {outgoing.length === 0 ? <div className="muted-sm">No outgoing relations.</div> : (
              <ul className="kg-rel-list">
                {outgoing.map(r => {
                  const t = neighbors.nodes.find(n => n.id === r.target);
                  if (!t) return null;
                  return (
                    <li key={r.id} className="kg-rel-row" onClick={() => onSelectRelation(r.id)} role="button" tabIndex={0}>
                      <span className="kg-rel-arrow">→</span>
                      <code className="kg-rel-label">{r.label}</code>
                      <span className="kg-rel-target">
                        <span className="kg-rel-swatch" style={{ background: colors[t.type] }} />
                        {t.name}
                      </span>
                      <span className="kg-rel-strength" title={`strength ${r.strength.toFixed(2)}`}>{Math.round(r.strength * 100)}</span>
                    </li>
                  );
                })}
              </ul>
            )}
          </div>

          <div className="kg-detail-section">
            <div className="section-label"><span>Incoming ({incoming.length}) <em>— click to edit</em></span></div>
            {incoming.length === 0 ? <div className="muted-sm">No incoming relations.</div> : (
              <ul className="kg-rel-list">
                {incoming.map(r => {
                  const s = neighbors.nodes.find(n => n.id === r.source);
                  if (!s) return null;
                  return (
                    <li key={r.id} className="kg-rel-row" onClick={() => onSelectRelation(r.id)} role="button" tabIndex={0}>
                      <span className="kg-rel-target">
                        <span className="kg-rel-swatch" style={{ background: colors[s.type] }} />
                        {s.name}
                      </span>
                      <code className="kg-rel-label">{r.label}</code>
                      <span className="kg-rel-arrow">→</span>
                      <span className="kg-rel-strength" title={`strength ${r.strength.toFixed(2)}`}>{Math.round(r.strength * 100)}</span>
                    </li>
                  );
                })}
              </ul>
            )}
          </div>

          <div className="kg-detail-section kg-detail-cta">
            <button
              className="ghost-btn"
              onClick={() => {
                const p = new URLSearchParams(window.location.search);
                p.set("tab", "documents");
                p.set("q", entity.name);
                window.history.pushState(null, "", window.location.pathname + "?" + p.toString());
                window.dispatchEvent(new PopStateEvent("popstate"));
              }}
            >
              <Icon name="external-link" size={11} /> View {entity.sources} sources mentioning this entity
            </button>
            <div className="kg-detail-locked">
              <Icon name="lock" size={11} />
              <span>Traverse relations with tag filter — <b>Twin Graph</b></span>
            </div>
          </div>
        </>
      )}
    </>
  );
}

// ─── Relation editor — same shape as EntityEditor, for an edge ─────────
function RelationEditor({ rel, src, tgt, colors, onSave, onSelectNode, onBack }) {
  const [editing, setEditing] = _useStateKG(false);
  const [draft, setDraft] = _useStateKG(null);
  _useEffectKG(() => { setEditing(false); setDraft(null); }, [rel.id]);

  const startEdit = () => {
    setDraft({
      label: rel.label,
      strength: rel.strength,
      properties: { ...(rel.properties || {}) }
    });
    setEditing(true);
  };
  const cancel = () => { setEditing(false); setDraft(null); };
  const commit = () => {
    onSave({
      label: (draft.label || "").trim().toUpperCase().replace(/\s+/g, "_") || rel.label,
      strength: Math.max(0, Math.min(1, parseFloat(draft.strength) || 0)),
      properties: draft.properties
    });
    setEditing(false);
    setDraft(null);
  };

  const propEntries = Object.entries(rel.properties || {});

  return (
    <>
      <div className="kg-detail-h">
        <button className="kg-rel-back" onClick={onBack} title="Back to entity">
          <Icon name="chevron-left" size={11} /> Back
        </button>
        <div className="kg-detail-title" style={{ marginTop: 4 }}>
          {editing ? (
            <input
              className="kg-edit-input kg-edit-name"
              value={draft.label}
              onChange={e => setDraft(d => ({ ...d, label: e.target.value }))}
              placeholder="RELATION_LABEL"
              autoFocus
              style={{ fontFamily: "var(--font-mono)", textTransform: "uppercase" }}
            />
          ) : (
            <h2 style={{ fontFamily: "var(--font-mono)", fontSize: 14 }}>{rel.label}</h2>
          )}
          {!editing && (
            <button className="kg-pin-btn" onClick={startEdit} style={{ marginLeft: "auto" }}>
              <Icon name="edit" size={11} /> Edit
            </button>
          )}
        </div>
        <div className="kg-detail-type" style={{ marginTop: 2 }}>Relation</div>
        <div className="kg-rel-endpoints">
          <button className="kg-rel-endpoint" onClick={() => src && onSelectNode(src.id)}>
            <span className="kg-rel-swatch" style={{ background: src ? colors[src.type] : "#888" }} />
            {src ? src.name : "?"}
          </button>
          <span className="kg-rel-arrow">→</span>
          <button className="kg-rel-endpoint" onClick={() => tgt && onSelectNode(tgt.id)}>
            <span className="kg-rel-swatch" style={{ background: tgt ? colors[tgt.type] : "#888" }} />
            {tgt ? tgt.name : "?"}
          </button>
        </div>
      </div>

      <div className="kg-detail-section">
        <div className="section-label"><span>Strength {editing ? <em>— 0.00–1.00</em> : null}</span></div>
        {editing ? (
          <div className="kg-strength-edit">
            <input
              type="range"
              min="0" max="1" step="0.01"
              value={draft.strength}
              onChange={e => setDraft(d => ({ ...d, strength: e.target.value }))}
            />
            <code>{Number(draft.strength).toFixed(2)}</code>
          </div>
        ) : (
          <div className="kg-strength-view">
            <div className="kg-strength-bar"><div style={{ width: `${rel.strength * 100}%` }} /></div>
            <code>{Math.round(rel.strength * 100)}%</code>
          </div>
        )}
      </div>

      <div className="kg-detail-section">
        <div className="section-label">
          <span>Properties {editing ? <em>— add / remove</em> : <em>— custom metadata</em>}</span>
          {!editing && propEntries.length > 0 && <span className="kg-prop-count">{propEntries.length}</span>}
        </div>
        {editing ? (
          <PropEditor
            properties={draft.properties}
            onChange={(properties) => setDraft(d => ({ ...d, properties }))}
          />
        ) : (
          propEntries.length === 0 ? (
            <div className="muted-sm">No custom properties. <button className="kg-inline-add" onClick={startEdit}>+ Add some</button></div>
          ) : (
            <dl className="kg-prop-list">
              {propEntries.map(([k, v]) => (
                <div key={k} className="kg-prop-row">
                  <dt>{k}</dt>
                  <dd>{String(v)}</dd>
                </div>
              ))}
            </dl>
          )
        )}
      </div>

      {editing && (
        <div className="kg-detail-section kg-edit-actions">
          <button className="ghost-btn" onClick={cancel}>Cancel</button>
          <button className="ghost-btn primary" onClick={commit}><Icon name="check" size={11} /> Save</button>
        </div>
      )}
    </>
  );
}

// ─── Tag chip editor (used inside the form) ─────────────────────────────
function TagEditor({ tags, onChange }) {
  const [v, setV] = _useStateKG("");
  const add = () => {
    const t = v.trim().toLowerCase().replace(/\s+/g, "-");
    if (!t || tags.includes(t)) return;
    onChange([...tags, t]);
    setV("");
  };
  const remove = (t) => onChange(tags.filter(x => x !== t));
  return (
    <div className="kg-tag-editor">
      <div className="tag-chips">
        {tags.map(t => (
          <span key={t} className="tag-chip kg-tag-edit-chip">
            {t}
            <button onClick={() => remove(t)} aria-label={`Remove ${t}`}><Icon name="x" size={9} /></button>
          </span>
        ))}
      </div>
      <div className="kg-tag-add-row">
        <input
          value={v}
          onChange={e => setV(e.target.value)}
          onKeyDown={e => { if (e.key === "Enter") { e.preventDefault(); add(); } }}
          placeholder="Add tag…"
        />
        <button className="ghost-btn" onClick={add} disabled={!v.trim()}>Add</button>
      </div>
    </div>
  );
}

// ─── Properties editor — key/value list with add/remove ────────────────
function PropEditor({ properties, onChange }) {
  const entries = Object.entries(properties);
  const [draftKey, setDraftKey] = _useStateKG("");
  const [draftVal, setDraftVal] = _useStateKG("");

  const editValue = (k, newVal) => onChange({ ...properties, [k]: newVal });
  const renameKey = (oldK, newK) => {
    if (!newK || newK === oldK || properties[newK] !== undefined) return;
    const next = {};
    for (const [k, v] of Object.entries(properties)) {
      next[k === oldK ? newK : k] = v;
    }
    onChange(next);
  };
  const removeKey = (k) => {
    const next = { ...properties };
    delete next[k];
    onChange(next);
  };
  const addProp = () => {
    const k = draftKey.trim();
    if (!k || properties[k] !== undefined) return;
    onChange({ ...properties, [k]: draftVal });
    setDraftKey(""); setDraftVal("");
  };

  return (
    <div className="kg-prop-editor">
      {entries.length === 0 && <div className="muted-sm" style={{ marginBottom: 6 }}>No properties yet — add the first one below.</div>}
      {entries.map(([k, v]) => (
        <div key={k} className="kg-prop-edit-row">
          <input
            className="kg-prop-key"
            value={k}
            onChange={e => renameKey(k, e.target.value.trim())}
            placeholder="key"
          />
          <span className="kg-prop-sep">:</span>
          <input
            className="kg-prop-val"
            value={String(v)}
            onChange={e => editValue(k, e.target.value)}
            placeholder="value"
          />
          <button className="kg-prop-x" onClick={() => removeKey(k)} aria-label={`Remove ${k}`}><Icon name="x" size={10} /></button>
        </div>
      ))}
      <div className="kg-prop-add-row">
        <input
          className="kg-prop-key"
          value={draftKey}
          onChange={e => setDraftKey(e.target.value)}
          onKeyDown={e => { if (e.key === "Enter" && draftKey.trim()) { e.preventDefault(); addProp(); } }}
          placeholder="new key"
        />
        <span className="kg-prop-sep">:</span>
        <input
          className="kg-prop-val"
          value={draftVal}
          onChange={e => setDraftVal(e.target.value)}
          onKeyDown={e => { if (e.key === "Enter" && draftKey.trim()) { e.preventDefault(); addProp(); } }}
          placeholder="value"
        />
        <button className="kg-prop-add" onClick={addProp} disabled={!draftKey.trim() || properties[draftKey.trim()] !== undefined}>
          <Icon name="plus" size={10} /> Add
        </button>
      </div>
    </div>
  );
}
