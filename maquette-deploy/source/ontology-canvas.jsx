// Ontology Studio canvas + main component.
// Loaded AFTER ontology.jsx — relies on the helpers it exports to window.

const {
  useState: _ocUseState, useEffect: _ocUseEffect, useMemo: _ocUseMemo,
  useRef: _ocUseRef, useCallback: _ocUseCallback
} = React;

// ─── Canvas (custom React Flow-style surface) ────────────────────────────
function StudioCanvas({
  nodes, edges, setNodes, setEdges,
  selectedId, setSelectedId,
  hoverNodeId, setHoverNodeId,
  wargame, wargameRunId,
  onCreateRelation, onDropInbox,
  ephemeral, setEphemeral,
  fullscreenInspector,
  presentation
}) {
  const wrapRef = _ocUseRef(null);
  const svgRef = _ocUseRef(null);
  const [view, setView] = _ocUseState({ x: -120, y: -40, k: 0.85 });
  const dragNodeRef = _ocUseRef(null);
  const panRef = _ocUseRef(null);
  const handleDragRef = _ocUseRef(null);
  const [draftEdge, setDraftEdge] = _ocUseState(null);  // {from:{id,side}, to:{x,y}}
  const [newEdgePopover, setNewEdgePopover] = _ocUseState(null); // {sourceId, targetId, midX, midY, label}
  const [contextMenu, setContextMenu] = _ocUseState(null);

  const NODE_W = window.NODE_W;
  const NODE_H = window.NODE_H;
  const colors = window.ONTOLOGY_TYPE_COLORS;

  // ── Wargame: BFS depth map from an origin node ─────────────────────────
  const wargameMap = _ocUseMemo(() => {
    if (!wargame.active || !wargame.originId) return null;
    const adj = {};
    edges.forEach(e => {
      if (!adj[e.source]) adj[e.source] = [];
      if (!adj[e.target]) adj[e.target] = [];
      adj[e.source].push({ id: e.target, edge: e });
      adj[e.target].push({ id: e.source, edge: e });
    });
    const depth = { [wargame.originId]: 0 };
    let frontier = [wargame.originId];
    const maxDepth = wargame.depth ?? 3;
    for (let d = 1; d <= maxDepth; d++) {
      const next = [];
      for (const id of frontier) {
        for (const n of (adj[id] || [])) {
          if (depth[n.id] === undefined) {
            depth[n.id] = d;
            next.push(n.id);
          }
        }
      }
      frontier = next;
      if (!next.length) break;
    }
    // Mark edges that participate in a path.
    const edgeDepth = {};
    edges.forEach(e => {
      const a = depth[e.source], b = depth[e.target];
      if (a !== undefined && b !== undefined) {
        edgeDepth[e.id] = Math.max(a, b);
      }
    });
    return { depth, edgeDepth };
  }, [wargame.active, wargame.originId, wargame.depth, edges, wargameRunId]);

  // ── Pan / zoom on empty canvas ─────────────────────────────────────────
  const onCanvasMouseDown = (e) => {
    if (e.target.closest("[data-onto-node]")) return;
    if (e.target.closest("[data-onto-handle]")) return;
    if (e.button !== 0) return;
    setSelectedId(null);
    setContextMenu(null);
    panRef.current = { startX: e.clientX, startY: e.clientY, vx: view.x, vy: view.y };
  };
  _ocUseEffect(() => {
    const onMove = (e) => {
      if (panRef.current) {
        const { startX, startY, vx, vy } = panRef.current;
        setView(v => ({ ...v, x: vx + (e.clientX - startX), y: vy + (e.clientY - startY) }));
      }
    };
    const onUp = () => { panRef.current = null; };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    return () => { window.removeEventListener("mousemove", onMove); window.removeEventListener("mouseup", onUp); };
  }, []);
  const onWheel = (e) => {
    e.preventDefault();
    const rect = wrapRef.current.getBoundingClientRect();
    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;
    const dz = e.deltaY < 0 ? 1.08 : 1 / 1.08;
    setView(v => {
      const k2 = Math.max(0.35, Math.min(2.2, v.k * dz));
      // Zoom about cursor so the canvas doesn't pop sideways.
      const wx = (cx - v.x) / v.k;
      const wy = (cy - v.y) / v.k;
      return { x: cx - wx * k2, y: cy - wy * k2, k: k2 };
    });
  };

  // Expose viewport setter so the demo player can fly the canvas.
  _ocUseEffect(() => {
    window.__ontoCanvasViewSet = (v) => setView(prev => ({ ...prev, ...v }));
    return () => { delete window.__ontoCanvasViewSet; };
  }, []);

  // Convert screen → canvas coords.
  const screenToCanvas = (sx, sy) => {
    const rect = wrapRef.current.getBoundingClientRect();
    return { x: (sx - rect.left - view.x) / view.k, y: (sy - rect.top - view.y) / view.k };
  };

  // ── Node drag ──────────────────────────────────────────────────────────
  const startNodeDrag = (e, id) => {
    if (e.button !== 0) return;
    if (e.target.closest("[data-onto-handle]")) return;
    const n = nodes.find(x => x.id === id);
    if (!n) return;
    e.stopPropagation();
    dragNodeRef.current = {
      id,
      startClientX: e.clientX,
      startClientY: e.clientY,
      origX: n.x, origY: n.y
    };
    setSelectedId(id);
  };
  _ocUseEffect(() => {
    const onMove = (e) => {
      const d = dragNodeRef.current;
      if (!d) return;
      const dx = (e.clientX - d.startClientX) / view.k;
      const dy = (e.clientY - d.startClientY) / view.k;
      setNodes(ns => ns.map(n => n.id === d.id ? { ...n, x: d.origX + dx, y: d.origY + dy } : n));
    };
    const onUp = () => { dragNodeRef.current = null; };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    return () => { window.removeEventListener("mousemove", onMove); window.removeEventListener("mouseup", onUp); };
  }, [view.k, setNodes]);

  // ── Handle drag (edge creation) ────────────────────────────────────────
  const startHandleDrag = (e, sourceId, side) => {
    e.stopPropagation();
    e.preventDefault();
    const sourceNode = nodes.find(n => n.id === sourceId);
    if (!sourceNode) return;
    const anchor = window.handlePoint(sourceNode, side);
    handleDragRef.current = { sourceId, side, anchor };
    setDraftEdge({ from: { id: sourceId, side }, to: anchor });
  };
  _ocUseEffect(() => {
    const onMove = (e) => {
      if (!handleDragRef.current) return;
      const pt = screenToCanvas(e.clientX, e.clientY);
      setDraftEdge(d => d ? { ...d, to: pt } : null);
    };
    const onUp = (e) => {
      if (!handleDragRef.current) return;
      // Did we drop on a node?
      const el = document.elementFromPoint(e.clientX, e.clientY);
      const nodeEl = el && el.closest("[data-onto-node]");
      const targetId = nodeEl && nodeEl.getAttribute("data-onto-node");
      const { sourceId } = handleDragRef.current;
      handleDragRef.current = null;
      setDraftEdge(null);
      if (targetId && targetId !== sourceId) {
        // Compute approximate midpoint in screen coords for the popover.
        const src = nodes.find(n => n.id === sourceId);
        const tgt = nodes.find(n => n.id === targetId);
        if (src && tgt) {
          const a = window.handlePoint(src, "right");
          const b = window.handlePoint(tgt, "left");
          setNewEdgePopover({
            sourceId, targetId,
            midX: (a.x + b.x) / 2,
            midY: (a.y + b.y) / 2,
            label: ""
          });
        }
      }
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    return () => { window.removeEventListener("mousemove", onMove); window.removeEventListener("mouseup", onUp); };
  }, [nodes, view.k, view.x, view.y]);

  // ── Drop from Inbox onto canvas ────────────────────────────────────────
  const onCanvasDragOver = (e) => {
    if (e.dataTransfer.types.includes("text/twin-inbox-id")) {
      e.preventDefault();
      e.dataTransfer.dropEffect = "copy";
    }
  };
  const onCanvasDrop = (e) => {
    const id = e.dataTransfer.getData("text/twin-inbox-id");
    if (!id) return;
    e.preventDefault();
    const pt = screenToCanvas(e.clientX, e.clientY);
    onDropInbox(id, { x: pt.x - NODE_W / 2, y: pt.y - NODE_H / 2 });
  };

  // ── Submit / cancel edge popover ───────────────────────────────────────
  const submitNewEdge = (label) => {
    const trimmed = (label || "").trim().toUpperCase().replace(/\s+/g, "_");
    if (!trimmed) return;
    onCreateRelation({
      source: newEdgePopover.sourceId,
      target: newEdgePopover.targetId,
      label: trimmed
    });
    setNewEdgePopover(null);
  };

  // ── Render ─────────────────────────────────────────────────────────────
  // Compute extents for the SVG layer (large but bounded for crisp rendering).
  const xs = nodes.map(n => n.x);
  const ys = nodes.map(n => n.y);
  const minX = Math.min(...xs, 0) - 200;
  const minY = Math.min(...ys, 0) - 200;
  const maxX = Math.max(...xs, 0) + NODE_W + 400;
  const maxY = Math.max(...ys, 0) + NODE_H + 400;

  // Ephemeral animations (flash a newly-created edge / node)
  const flashEdgeIds = ephemeral.flashEdges || new Set();
  const flashNodeIds = ephemeral.flashNodes || new Set();

  return (
    <div
      className={`onto-canvas${wargame.active ? " is-wargame" : ""}${presentation ? " is-presentation" : ""}`}
      ref={wrapRef}
      onWheel={onWheel}
      onMouseDown={onCanvasMouseDown}
      onDragOver={onCanvasDragOver}
      onDrop={onCanvasDrop}
    >
      {/* Dot grid background */}
      <div className="onto-grid" style={{
        backgroundPosition: `${view.x % (24 * view.k)}px ${view.y % (24 * view.k)}px`,
        backgroundSize: `${24 * view.k}px ${24 * view.k}px`
      }} />

      <div
        className="onto-world"
        style={{ transform: `translate(${view.x}px, ${view.y}px) scale(${view.k})`, transformOrigin: "0 0" }}
      >
        {/* SVG edges */}
        <svg
          ref={svgRef}
          className="onto-edges"
          width={maxX - minX}
          height={maxY - minY}
          style={{ position: "absolute", left: minX, top: minY, overflow: "visible", pointerEvents: "none" }}
        >
          <defs>
            <marker id="onto-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
              <path d="M0,0 L10,5 L0,10 z" fill="var(--color-text-tertiary)" opacity="0.7" />
            </marker>
            <marker id="onto-arrow-hi" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
              <path d="M0,0 L10,5 L0,10 z" fill="var(--twin-accent)" />
            </marker>
            <marker id="onto-arrow-pending" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
              <path d="M0,0 L10,5 L0,10 z" fill="var(--twin-amber-vivid)" />
            </marker>
            <marker id="onto-arrow-blast" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
              <path d="M0,0 L10,5 L0,10 z" fill="var(--twin-red-vivid)" />
            </marker>
          </defs>

          {/* Each edge */}
          {edges.map(e => {
            const a = nodes.find(n => n.id === e.source);
            const b = nodes.find(n => n.id === e.target);
            if (!a || !b) return null;
            const an = window.autoAnchors(a, b);
            const d = window.bezierPath(an.pa, an.pb, an.sa, an.sb);
            const isFlash = flashEdgeIds.has(e.id);
            const isPending = e.status === "pending";
            const isHi = (selectedId === e.id) || (selectedId && (e.source === selectedId || e.target === selectedId));

            // Wargame coloring overrides
            let blastClass = "";
            if (wargameMap) {
              const dep = wargameMap.edgeDepth[e.id];
              if (dep === undefined) blastClass = " is-blast-dim";
              else if (dep === 1) blastClass = " is-blast-1";
              else if (dep === 2) blastClass = " is-blast-2";
              else if (dep >= 3) blastClass = " is-blast-3";
            }

            const stroke =
              wargameMap ? null :
              isPending ? "var(--twin-amber-vivid)" :
              isHi ? "var(--twin-accent)" :
              "var(--color-text-tertiary)";
            const marker =
              wargameMap ? "url(#onto-arrow-blast)" :
              isPending ? "url(#onto-arrow-pending)" :
              isHi ? "url(#onto-arrow-hi)" :
              "url(#onto-arrow)";

            const mid = window.bezierMid(an.pa, an.pb, an.sa, an.sb);
            // Translate svg coords from canvas (minX/minY based viewport).
            const tx = -minX, ty = -minY;
            return (
              <g key={e.id} transform={`translate(${tx}, ${ty})`} className={`onto-edge${isFlash ? " is-flash" : ""}${isPending ? " is-pending" : ""}${blastClass}`} style={{ pointerEvents: "auto" }}>
                <path
                  d={d}
                  fill="none"
                  stroke={stroke || undefined}
                  strokeWidth={isHi ? 2 : 1.4}
                  strokeOpacity={wargameMap && blastClass === " is-blast-dim" ? 0.12 : (isPending ? 0.95 : isHi ? 0.95 : 0.55)}
                  strokeDasharray={isPending ? "5 4" : undefined}
                  markerEnd={marker}
                  onClick={(ev) => { ev.stopPropagation(); setSelectedId(e.id); }}
                  style={{ cursor: "pointer" }}
                />
                {(isHi || isFlash || isPending) && (
                  <g transform={`translate(${mid.x}, ${mid.y})`} style={{ pointerEvents: "none" }}>
                    <rect x={-e.label.length * 3.2 - 8} y={-9} width={e.label.length * 6.4 + 16} height={18}
                      rx={4} ry={4}
                      fill="var(--color-background-primary)"
                      stroke={isPending ? "var(--twin-amber-vivid)" : "var(--twin-accent)"}
                      strokeWidth="1" />
                    <text x={0} y={3.5} textAnchor="middle" className="onto-edge-label-text"
                      fill={isPending ? "var(--twin-amber-700)" : "var(--twin-accent-soft-text)"}>{e.label}</text>
                  </g>
                )}
              </g>
            );
          })}

          {/* Draft edge being drawn from a handle */}
          {draftEdge && (() => {
            const src = nodes.find(n => n.id === draftEdge.from.id);
            if (!src) return null;
            const pa = window.handlePoint(src, draftEdge.from.side);
            const pb = draftEdge.to;
            const sa = draftEdge.from.side;
            // Guess sb from delta direction
            const dx = pb.x - pa.x, dy = pb.y - pa.y;
            const sb = Math.abs(dx) > Math.abs(dy) ? (dx > 0 ? "left" : "right") : (dy > 0 ? "top" : "bottom");
            const d = window.bezierPath(pa, pb, sa, sb);
            const tx = -minX, ty = -minY;
            return (
              <g transform={`translate(${tx}, ${ty})`}>
                <path d={d} fill="none" stroke="var(--twin-accent)" strokeWidth="1.8" strokeDasharray="6 4" opacity="0.85" />
                <circle cx={pb.x} cy={pb.y} r="4" fill="var(--twin-accent)" opacity="0.85" />
              </g>
            );
          })()}
        </svg>

        {/* Nodes (HTML for crisp text + handles) */}
        {nodes.map(n => {
          const isSel = selectedId === n.id;
          const isHover = hoverNodeId === n.id;
          const isFlash = flashNodeIds.has(n.id);
          const blast = wargameMap ? wargameMap.depth[n.id] : undefined;
          let blastClass = "";
          if (wargameMap) {
            if (blast === 0) blastClass = " is-blast-origin";
            else if (blast === 1) blastClass = " is-blast-1";
            else if (blast === 2) blastClass = " is-blast-2";
            else if (blast === 3) blastClass = " is-blast-3";
            else blastClass = " is-blast-dim";
          }
          const tone = colors[n.type] || "#888";
          return (
            <div
              key={n.id}
              data-onto-node={n.id}
              className={`onto-node${isSel ? " is-selected" : ""}${isHover ? " is-hover" : ""}${n.status === "pending" ? " is-pending" : ""}${isFlash ? " is-flash" : ""}${blastClass}`}
              style={{
                left: n.x, top: n.y, width: NODE_W, height: NODE_H,
                borderTopColor: tone
              }}
              onMouseDown={(e) => startNodeDrag(e, n.id)}
              onMouseEnter={() => setHoverNodeId(n.id)}
              onMouseLeave={() => setHoverNodeId(null)}
              onClick={(e) => { e.stopPropagation(); setSelectedId(n.id); }}
              onContextMenu={(e) => {
                e.preventDefault();
                setContextMenu({ x: e.clientX, y: e.clientY, nodeId: n.id });
                setSelectedId(n.id);
              }}
            >
              <div className="onto-node-body">
                <span className="onto-node-swatch" style={{ background: tone }} />
                <div className="onto-node-text">
                  <div className="onto-node-name">{n.name}</div>
                  <div className="onto-node-type">{window.ONTOLOGY_TYPE_LABEL[n.type] || n.type}</div>
                </div>
                {n.status === "pending" && <span className="onto-node-pending-pill" title="Pending review">●</span>}
                {wargameMap && blast === 0 && <span className="onto-node-origin-pill" title="Failure origin">⚡</span>}
              </div>
              {/* Handles */}
              {["top","right","bottom","left"].map(side => (
                <button
                  key={side}
                  data-onto-handle
                  className={`onto-handle is-${side}`}
                  onMouseDown={(e) => startHandleDrag(e, n.id, side)}
                  aria-label={`Connect from ${side}`}
                />
              ))}
            </div>
          );
        })}
      </div>

      {/* New-edge popover (positioned in screen space at the midpoint) */}
      {newEdgePopover && (() => {
        const cx = newEdgePopover.midX * view.k + view.x;
        const cy = newEdgePopover.midY * view.k + view.y;
        return (
          <div className="onto-newedge-popover" style={{ left: cx, top: cy }} onMouseDown={e => e.stopPropagation()}>
            <div className="onto-newedge-h">
              <span>New relation</span>
              <button className="onto-newedge-x" onClick={() => setNewEdgePopover(null)} aria-label="Cancel"><Icon name="x" size={10} /></button>
            </div>
            <div className="onto-newedge-body">
              <div className="onto-newedge-ends">
                <code>{window.nodeNameOf(newEdgePopover.sourceId)}</code>
                <span>→</span>
                <code>{window.nodeNameOf(newEdgePopover.targetId)}</code>
              </div>
              <NewEdgeInput
                value={newEdgePopover.label}
                onChange={(v) => setNewEdgePopover(p => ({ ...p, label: v }))}
                onSubmit={(v) => submitNewEdge(v)}
              />
              <div className="onto-newedge-suggestions">
                {["DEPENDS_ON","CRITICAL_DEPENDENCY","HOSTED_ON","ROUTES_VIA","COVERED_BY","FEEDS","REPLICATED_VIA"].map(s => (
                  <button key={s} className="onto-newedge-sugg" onClick={() => submitNewEdge(s)}>{s}</button>
                ))}
              </div>
            </div>
          </div>
        );
      })()}

      {/* Context menu on a node */}
      {contextMenu && (() => {
        const n = nodes.find(x => x.id === contextMenu.nodeId);
        if (!n) return null;
        return (
          <div
            className="onto-context-menu"
            style={{ left: contextMenu.x, top: contextMenu.y }}
            onMouseDown={e => e.stopPropagation()}
            onClick={() => setContextMenu(null)}
          >
            <button onClick={() => { setSelectedId(n.id); setEphemeral(s => ({ ...s, flashNodes: new Set([n.id]) })); setTimeout(() => setEphemeral(s => ({ ...s, flashNodes: new Set() })), 900); }}>
              <Icon name="focus" size={11} /> Focus
            </button>
            <button onClick={() => { window.dispatchEvent(new CustomEvent("onto-wargame-from", { detail: { nodeId: n.id } })); }}>
              <Icon name="alert-triangle" size={11} /> Run resilience test from here
            </button>
            <button onClick={() => { setNodes(ns => ns.filter(x => x.id !== n.id)); setEdges(es => es.filter(e => e.source !== n.id && e.target !== n.id)); }}>
              <Icon name="x" size={11} /> Delete
            </button>
          </div>
        );
      })()}

      {/* Viewport pill */}
      {!presentation && (
        <div className="onto-viewport-pill">
          <button onClick={() => setView(v => ({ ...v, k: Math.max(0.35, v.k * 0.88) }))} aria-label="Zoom out"><Icon name="minus" size={11} /></button>
          <span>{Math.round(view.k * 100)}%</span>
          <button onClick={() => setView(v => ({ ...v, k: Math.min(2.2, v.k * 1.14) }))} aria-label="Zoom in"><Icon name="plus" size={11} /></button>
          <span className="onto-vp-sep" />
          <button onClick={() => setView({ x: -120, y: -40, k: 0.85 })} title="Reset view">
            <Icon name="refresh" size={11} />
          </button>
        </div>
      )}

      {!presentation && (
        <div className="onto-canvas-hint">
          <Icon name="info-circle" size={10} />
          <span>Drag a node to move · drag from a handle to draw a relation · scroll to zoom · right-click for actions</span>
        </div>
      )}
    </div>
  );
}

function NewEdgeInput({ value, onChange, onSubmit }) {
  const ref = _ocUseRef(null);
  _ocUseEffect(() => { if (ref.current) ref.current.focus(); }, []);
  return (
    <input
      ref={ref}
      className="onto-newedge-input"
      value={value}
      onChange={e => onChange(e.target.value)}
      onKeyDown={e => {
        if (e.key === "Enter") { e.preventDefault(); onSubmit(value); }
        if (e.key === "Escape") { e.preventDefault(); onChange(""); }
      }}
      placeholder="Type the relation… (e.g. CRITICAL_DEPENDENCY)"
    />
  );
}

Object.assign(window, { StudioCanvas, NewEdgeInput });
