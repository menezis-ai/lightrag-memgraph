// Documents tab — table with status filter, tag filter, source rows
const { useState, useMemo, useEffect } = React;

const STATUS_LABELS = {
  all: "All",
  completed: "Completed",
  processing: "Processing",
  pending: "Pending",
  failed: "Failed"
};

// Current user palier (matches MOCK_CURRENT_USER). Palier ≥ 3 can approve/
// reject documents in the steward review queue (mirrors the tag governance
// flow in tags.jsx). Palier 2 (contributor) sees the queue read-only with
// an "Awaiting palier-3 review" caption — same UX as the Tag Pending
// requests block. Palier 1 (reader) doesn't see the section at all.
//
// Demo override: append `?palier=1|2|3` to the URL to switch perspective
// without rebuilding MOCK_CURRENT_USER. Lets us show Vihn what the
// contributor view looks like vs the steward view in the same session.
const _DOC_MOCK_USER = (typeof window !== "undefined" && window.MOCK_CURRENT_USER) || { palier: 3, name: "claire.benoit" };
// Demo override: ?palier=1|2|3 in the URL flips the perspective. Useful to
// show Vihn the contributor view vs the steward view in the same session.
// Each palier maps to a representative handle that owns the seeded pending
// submissions (marc.berthier owns d13 = cft-vendor-api-spec-draft.pdf,
// yann.dubois owns d14 = incident-2026-Q2-postmortem-draft) so the "your
// submission" badge actually lights up in palier-2 mode.
const _PALIER_HANDLE = { 1: "philippe.marchand", 2: "marc.berthier", 3: "claire.benoit" };
const _palierOverride = (() => {
  if (typeof window === "undefined") return null;
  const p = new URLSearchParams(window.location.search).get("palier");
  const n = parseInt(p, 10);
  return (n === 1 || n === 2 || n === 3) ? n : null;
})();
const _docPalier = _palierOverride != null ? _palierOverride : _DOC_MOCK_USER.palier;
const _docHandle = _palierOverride != null ? _PALIER_HANDLE[_palierOverride] : (_DOC_MOCK_USER.name || "claire.benoit").toLowerCase().replace(/\s+/g, ".");
const _DOC_CURRENT_USER = { palier: _docPalier, name: _docHandle };
const _docCanReview = _docPalier >= 3;
const _docCanSeeQueue = _docPalier >= 2;

window.DocumentsTab = function DocumentsTab({ docs, mutateDoc, isEmptyWorkspace, onOpenAdd, onOpenRetag, onOpenBulkRetag, onAddToast, onLoadDemo }) {
  const sys = window.useReadOnly ? window.useReadOnly() : { effectiveReadOnly: false, readOnlyReason: "" };
  const ro = sys.effectiveReadOnly;
  const roTitle = ro ? `Disabled — ${sys.readOnlyReason}` : undefined;
  const [selected, setSelected] = useState(() => new Set());
  const [statusFilter, setStatusFilter] = window.useUrlParam("status", "all", {
    validate: v => ["all","completed","processing","pending","failed"].includes(v)
  });
  const [search, setSearch] = window.useUrlParam("q", "");
  const [tagFilters, setTagFilters] = window.useUrlArrayParam("tag", ["rman"]);
  const [tagAddOpen, setTagAddOpen] = useState(false);
  const [tagAddVal, setTagAddVal] = useState("");
  const [selectedDoc, setSelectedDocRaw] = useState(null);
  const [detailFocus, setDetailFocus] = useState(null);
  const setSelectedDoc = (doc, opts) => {
    setSelectedDocRaw(doc);
    setDetailFocus(opts && opts.focus ? opts.focus : null);
  };
  const [pipelineOpen, setPipelineOpen] = useState(false);

  // Pending-review queue — docs flagged `review.state === "pending-review"`
  // are filtered out of the main grid and surfaced in a dedicated section at
  // the top of the tab (steward-only). Mirrors `Pending requests` in tags.jsx.
  const pendingDocs = useMemo(
    () => docs.filter(d => d.review && d.review.state === "pending-review"),
    [docs]
  );
  // Pending section collapsed by default — audit feedback: full-card
  // amber on first open reads as "alert" not "to-do". Choice persists in
  // localStorage per tab so the steward who already triaged doesn't have
  // to re-collapse on every load.
  const [pendingOpen, setPendingOpen] = useState(() => {
    try { return localStorage.getItem("twin.docsPending.open") === "true"; } catch (e) { return false; }
  });
  useEffect(() => {
    try { localStorage.setItem("twin.docsPending.open", String(pendingOpen)); } catch (e) {}
  }, [pendingOpen]);
  const [rejectDoc, setRejectDoc] = useState(null);
  const [rejectReason, setRejectReason] = useState("");
  // "View raw" modal — surfaced both from the pending-review card (so
  // the steward can read the source BEFORE deciding) and from the
  // DocDetailPanel footer (existing affordance). Without this, the
  // review queue is "trust me" governance — the steward needs eyes on
  // the actual content to make an informed call.
  const [rawDoc, setRawDoc] = useState(null);

  // Empty-workspace state: take over the whole pane with a focused CTA card
  // instead of rendering the filters + empty table. This is the first thing a
  // brand-new steward sees, so it has to read as an invitation, not as an error.
  if (docs.length === 0 && isEmptyWorkspace && window.EmptyWorkspaceCard) {
    return (
      <div className="docs">
        <div className="docs-header">
          <h1>Document management</h1>
        </div>
        <div className="empty-pane">
          <window.EmptyWorkspaceCard
            onAddSource={onOpenAdd}
            onLoadDemo={onLoadDemo}
          />
        </div>
      </div>
    );
  }

  const counts = useMemo(() => {
    // Pending-review + rejected docs surface only in the steward queue,
    // not in the main grid — exclude them from the status pill counters
    // so the numbers match the visible table.
    const reviewable = docs.filter(d =>
      !(d.review && (d.review.state === "pending-review" || d.review.state === "rejected"))
    );
    const c = { all: reviewable.length, completed: 0, processing: 0, pending: 0, failed: 0 };
    reviewable.forEach(d => { if (c[d.status] !== undefined) c[d.status]++; });
    return c;
  }, [docs]);
  const failedCount = counts.failed;

  const filtered = useMemo(() => {
    return docs.filter(d => {
      // Pending-review docs surface only in the steward queue (see above).
      if (d.review && d.review.state === "pending-review") return false;
      // Rejected docs are excluded from active retrieval set per spec.
      if (d.review && d.review.state === "rejected") return false;
      if (statusFilter !== "all" && d.status !== statusFilter) return false;
      if (search && !d.source.toLowerCase().includes(search.toLowerCase())) return false;
      if (tagFilters.length && !tagFilters.every(t => d.tags.includes(t))) return false;
      return true;
    });
  }, [docs, statusFilter, search, tagFilters]);

  // Approve / reject handlers — REAL state mutation (was toast-only in
  // the proto). mutateDoc lifts the change to app.jsx where sql.js persists
  // the docs array to IndexedDB, so the action survives a reload — table
  // stakes for a credible demo.
  const _stamp = () => new Date().toISOString().slice(0, 10);
  const approveDoc = (d) => {
    mutateDoc && mutateDoc(d.id, { review: { state: "approved", reviewed_by: _DOC_CURRENT_USER.name, reviewed_at: _stamp() } });
    onAddToast(
      `Document approved · ${d.source}`,
      `doc.approved · entered active retrieval set · ${d.review && d.review.requested_by ? "requested by " + d.review.requested_by : "audit recorded"}`
    );
  };
  const editAndApproveDoc = (d) => {
    mutateDoc && mutateDoc(d.id, { review: { state: "approved", reviewed_by: _DOC_CURRENT_USER.name, reviewed_at: _stamp(), edited: true } });
    onAddToast(
      `Document approved with edits · ${d.source}`,
      "doc.approved · steward acknowledged the request after a tag/summary tweak"
    );
  };
  const submitReject = () => {
    if (!rejectDoc) return;
    mutateDoc && mutateDoc(rejectDoc.id, { review: { state: "rejected", reviewed_by: _DOC_CURRENT_USER.name, reviewed_at: _stamp(), reason: rejectReason || "" } });
    onAddToast(
      `Document rejected · ${rejectDoc.source}`,
      `doc.rejected · reason: ${rejectReason || "(none provided)"} · requester notified`
    );
    setRejectDoc(null);
    setRejectReason("");
  };

  const removeTagFilter = (t) => setTagFilters(tagFilters.filter(x => x !== t));
  const addTagFilter = (t) => {
    if (t && !tagFilters.includes(t)) setTagFilters([...tagFilters, t]);
    setTagAddVal("");
    setTagAddOpen(false);
  };

  const thesaurusSuggestions = useMemo(() => {
    const v = tagAddVal.toLowerCase();
    return window.MOCK_THESAURUS
      .filter(t => !tagFilters.includes(t.tag))
      .filter(t => !v || t.tag.includes(v))
      .slice(0, 5);
  }, [tagAddVal, tagFilters]);

  const clickTagOnRow = (e, tag) => {
    e.stopPropagation();
    if (!tagFilters.includes(tag)) setTagFilters([...tagFilters, tag]);
  };

  const toggleRow = (id) => {
    const next = new Set(selected);
    if (next.has(id)) next.delete(id); else next.add(id);
    setSelected(next);
  };
  const filteredIds = filtered.map(d => d.id);
  const allFilteredSelected = filteredIds.length > 0 && filteredIds.every(id => selected.has(id));
  const someFilteredSelected = filteredIds.some(id => selected.has(id));
  const toggleAll = () => {
    const next = new Set(selected);
    if (allFilteredSelected) filteredIds.forEach(id => next.delete(id));
    else filteredIds.forEach(id => next.add(id));
    setSelected(next);
  };
  const clearSelection = () => setSelected(new Set());
  const selectedDocs = docs.filter(d => selected.has(d.id));
  const openBulk = () => onOpenBulkRetag && onOpenBulkRetag(selectedDocs);

  const hasFilters = statusFilter !== "all" || !!search || tagFilters.length > 0 || selected.size > 0;
  const clearAllFilters = () => {
    const summary = [
      statusFilter !== "all" && `status: ${statusFilter}`,
      search && `q: ${search}`,
      tagFilters.length > 0 && `tags: ${tagFilters.join(", ")}`,
      selected.size > 0 && `${selected.size} selected`
    ].filter(Boolean).join(" · ");
    setStatusFilter("all");
    setSearch("");
    setTagFilters([]);
    setSelected(new Set());
    if (summary) onAddToast("Filters cleared", summary);
  };

  return (
    <div className="docs">
      <div className="docs-header">
        <h1>Document management</h1>
        <div className="docs-header-actions">
          <button
            className={`btn${failedCount > 0 ? " btn-retry" : ""}`}
            disabled={ro}
            title={roTitle}
            onClick={() => onAddToast(
              failedCount > 0
                ? `Scan started · retrying ${failedCount} failed source${failedCount > 1 ? "s" : ""}`
                : "Pipeline scan started",
              failedCount > 0
                ? "POST /documents/scan?retry=failed · workers picking up now"
                : "POST /documents/scan · re-scanning 12 sources for changes"
            )}
          >
            <Icon name="refresh" size={14} />
            {failedCount > 0 ? "Scan / Retry" : "Scan"}
            {failedCount > 0 && (
              <span className="pipeline-badge" aria-label={`${failedCount} failed`}>{failedCount}</span>
            )}
          </button>
          <div style={{ position: "relative" }}>
            <button className={`btn${pipelineOpen ? " active" : ""}`} onClick={() => setPipelineOpen(o => !o)} aria-expanded={pipelineOpen} aria-haspopup="dialog">
              <Icon name="activity" size={14} /> Pipeline
              <span className="pipeline-badge" aria-label="3 sources processing">3</span>
            </button>
            {pipelineOpen && <PipelineStatusPopover onClose={() => setPipelineOpen(false)} onAddToast={onAddToast} />}
          </div>
          <button
            className="btn"
            onClick={clearAllFilters}
            disabled={!hasFilters}
            title={hasFilters ? "Clear status, search, tag filters and selection" : "No filters active"}
          >
            <Icon name="x" size={14} /> Clear
          </button>
          <button className="btn primary" onClick={onOpenAdd} disabled={ro} title={roTitle}>
            <Icon name="cloud-upload" size={14} /> Add source
          </button>
        </div>
      </div>

      {pendingDocs.length > 0 && _docCanSeeQueue && (
        <div className={"pending-section " + (pendingOpen ? "is-open" : "")}>
          <button className="pending-h" onClick={() => setPendingOpen(o => !o)}>
            <Icon name="alert-triangle" size={14} color="var(--twin-amber-vivid)" />
            <span className="pending-title">Pending review</span>
            <span className="pending-counts">
              <b>{pendingDocs.length}</b> document{pendingDocs.length > 1 ? "s" : ""}{" "}
              {_docCanReview ? "awaiting your sign-off" : "awaiting steward review"}
            </span>
            <Icon name="chevron-down" size={14} color="var(--color-text-tertiary)" style={{ transform: pendingOpen ? "none" : "rotate(-90deg)", transition: "transform .15s" }} />
          </button>
          {pendingOpen && (
            <div className="pending-grid">
              {pendingDocs.map(d => {
                const mine = d.review && d.review.requested_by === _DOC_CURRENT_USER.name;
                return (
                  <div key={d.id} className="pending-card requested">
                    <div className="pending-card-h">
                      <code className="pending-tagname">{d.source}</code>
                      {mine && !_docCanReview && (
                        <span className="status-badge sm" style={{ background: "var(--twin-accent-soft-bg)", color: "var(--twin-accent-soft-text)" }}>your submission</span>
                      )}
                    </div>
                    <div className="pending-justif">{d.review && d.review.justification}</div>
                    <div className="pending-meta">
                      Submitted by <b>{d.review && d.review.requested_by}</b> · {d.review && d.review.requested_at} · {d.chunks ? d.chunks + " chunks" : "—"} · tags <code>{(d.tags || []).join(", ") || "—"}</code>
                    </div>
                    {_docCanReview ? (
                      <div className="pending-actions">
                        <button className="ghost-btn small" onClick={() => setRawDoc(d)} title="Read the extracted source text before deciding">
                          <Icon name="eye" size={11} /> Read source
                        </button>
                        <span className="pending-actions-sep" />
                        <button className="primary-btn small" onClick={() => approveDoc(d)}>Approve</button>
                        <button className="ghost-btn small" onClick={() => editAndApproveDoc(d)}>Edit &amp; approve</button>
                        <button className="ghost-btn small danger" onClick={() => { setRejectDoc(d); setRejectReason(""); }}>Reject</button>
                      </div>
                    ) : (
                      <div className="pending-actions">
                        <button className="ghost-btn small" onClick={() => setRawDoc(d)}>
                          <Icon name="eye" size={11} /> Read source
                        </button>
                        <span className="pending-actions-sep" />
                        <span className="muted">Awaiting steward review · you'll be notified when a steward signs off</span>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}

      {rejectDoc && (
        <div className="modal-backdrop" onClick={() => setRejectDoc(null)}>
          <div className="modal small" role="dialog" aria-modal="true" aria-labelledby="doc-reject-title" onClick={e => e.stopPropagation()}>
            <div className="modal-h">
              <h3 id="doc-reject-title">Reject document</h3>
              <button className="modal-x" onClick={() => setRejectDoc(null)} aria-label="Close"><Icon name="x" size={14} /></button>
            </div>
            <div className="modal-body">
              <p className="muted" style={{ marginTop: 0, fontSize: 12 }}>
                <code>{rejectDoc.source}</code> will be removed from the steward queue and the requester ({rejectDoc.review && rejectDoc.review.requested_by}) will be notified.
                A <code>doc.rejected</code> event lands on the audit feed.
              </p>
              <label className="field-label" htmlFor="doc-reject-reason">Reason (visible to requester)</label>
              <textarea
                id="doc-reject-reason"
                className="text-input"
                rows="3"
                value={rejectReason}
                onChange={e => setRejectReason(e.target.value)}
                placeholder="e.g. Superseded by v3 — see /cib/runbooks/oracle-pga-tuning v3 already in retrieval."
                autoFocus
              />
            </div>
            <div className="modal-footer">
              <button className="ghost-btn" onClick={() => setRejectDoc(null)}>Cancel</button>
              <button className="primary-btn danger" onClick={submitReject} disabled={!rejectReason.trim()}>Reject document</button>
            </div>
          </div>
        </div>
      )}

      <div className="docs-filters">
        <span className="filter-label">Uploaded</span>
        <div className="filter-pills">
          {["all", "completed", "processing", "pending", "failed"].map(k => (
            <button
              key={k}
              className={`pill ${k}${statusFilter === k ? " active" : ""}`}
              onClick={() => setStatusFilter(k)}
            >
              {STATUS_LABELS[k]}
              <span className="pill-badge">{counts[k]}</span>
            </button>
          ))}
        </div>
        <span className="filter-divider" aria-hidden="true" />
        <span className="filter-label-tag">Tag<em>— Twin</em></span>
        <div className="tag-chips">
          {tagFilters.map(t => (
            <TagChip key={t} tag={t} removable onRemove={removeTagFilter} />
          ))}
          {tagAddOpen ? (
            <div style={{ position: "relative" }}>
              <input
                autoFocus
                value={tagAddVal}
                onChange={e => setTagAddVal(e.target.value)}
                onBlur={() => setTimeout(() => setTagAddOpen(false), 150)}
                onKeyDown={e => {
                  if (e.key === "Enter" && thesaurusSuggestions[0]) addTagFilter(thesaurusSuggestions[0].tag);
                  if (e.key === "Escape") setTagAddOpen(false);
                }}
                placeholder="tag…"
                style={{
                  fontFamily: "var(--font-mono)", fontSize: "11px",
                  padding: "3px 8px", border: "0.5px solid var(--twin-accent)",
                  borderRadius: "999px", width: 110, background: "var(--color-background-primary)"
                }}
              />
              {thesaurusSuggestions.length > 0 && (
                <div className="autocomplete" style={{ position: "absolute", top: "100%", left: 0, marginTop: 4, minWidth: 200, zIndex: 30 }}>
                  {thesaurusSuggestions.map((s, i) => (
                    <div
                      key={s.tag}
                      className={`autocomplete-row${i === 0 ? " focus" : ""}`}
                      onMouseDown={() => addTagFilter(s.tag)}
                    >
                      <div className="row1">
                        <span>{s.tag}</span>
                        <span className="badge">{s.category}</span>
                      </div>
                      <div className="def">{s.def}</div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          ) : (
            <button className="tag-add-btn" onClick={() => setTagAddOpen(true)}>+ Add tag</button>
          )}
        </div>
        <input
          className="search-source"
          placeholder="Search source name…"
          value={search}
          onChange={e => setSearch(e.target.value)}
        />
      </div>

      {selected.size > 0 && (
        <div className="bulk-bar" role="region" aria-label="Bulk actions">
          <span className="bulk-count">
            <b>{selected.size}</b> selected
            <span className="bulk-of">of {filtered.length}</span>
          </span>
          <button className="bulk-action primary" onClick={openBulk}>
            <Icon name="tags" size={13} /> Retag {selected.size} sources
          </button>
          <button className="bulk-action" onClick={() => onAddToast(
            `Re-processing ${selected.size} sources`,
            `Queued at pipeline · workers picking up now`
          )}>
            <Icon name="refresh" size={13} /> Re-process
          </button>
          <button className="bulk-clear" onClick={clearSelection}>
            <Icon name="x" size={12} /> Clear selection
          </button>
        </div>
      )}

      <div className="docs-table-wrap">
        <div className="docs-table">
          <div className="docs-row header has-select">
            <div className="cell-select">
              <input
                type="checkbox"
                className="row-check"
                checked={allFilteredSelected}
                ref={el => { if (el) el.indeterminate = !allFilteredSelected && someFilteredSelected; }}
                onChange={toggleAll}
                aria-label="Select all visible"
              />
            </div>
            <div>Source</div>
            <div>Summary</div>
            <div>Tags <span style={{ textTransform: "none", color: "var(--color-text-tertiary)", letterSpacing: 0, fontSize: 9.5 }}>— Twin</span></div>
            <div>Status</div>
            <div>Chunks</div>
            <div>Updated</div>
            <div></div>
          </div>
          {filtered.length === 0 && (
            <div style={{ padding: 30, textAlign: "center", color: "var(--color-text-tertiary)", fontSize: 12 }}>
              No documents match the current filters.
            </div>
          )}
          {filtered.map(d => (
            <DocRow
              key={d.id}
              doc={d}
              selected={selectedDoc && selectedDoc.id === d.id}
              checked={selected.has(d.id)}
              onToggle={toggleRow}
              onOpenRetag={onOpenRetag}
              onClickTag={clickTagOnRow}
              onSelect={setSelectedDoc}
            />
          ))}
        </div>
      </div>

      {selectedDoc && (
        <DocDetailPanel
          doc={selectedDoc}
          focus={detailFocus}
          onClose={() => setSelectedDocRaw(null)}
          onOpenRetag={onOpenRetag}
          onAddToast={onAddToast}
          onViewRaw={() => setRawDoc(selectedDoc)}
        />
      )}

      {rawDoc && <RawDocModal doc={rawDoc} onClose={() => setRawDoc(null)} />}
    </div>
  );
};

function DocRow({ doc, selected, checked, onToggle, onOpenRetag, onClickTag, onSelect }) {
  const isFail = doc.status === "failed";
  const mono = doc.type === "confluence" || doc.type === "sharepoint" || doc.type === "url";
  const visibleTags = doc.tags.slice(0, 2);
  const overflow = doc.tags.length - visibleTags.length;
  return (
    <div
      className={`docs-row has-select${selected ? " selected" : ""}${checked ? " is-checked" : ""}`}
      onClick={() => onSelect(doc)}
      onDoubleClick={() => onSelect(doc)}
    >
      <div className="cell-select" onClick={e => e.stopPropagation()}>
        <input
          type="checkbox"
          className="row-check"
          checked={!!checked}
          onChange={() => onToggle(doc.id)}
          aria-label={`Select ${doc.source}`}
        />
      </div>
      <div className="cell-source">
        <SourceIcon type={doc.type} size={15} />
        <span className={mono ? "name mono" : "name"}>{doc.source}</span>
      </div>
      <div className={isFail ? "cell-summary failed" : "cell-summary"}>{doc.summary}</div>
      <div className="cell-tags">
        {doc.tags.length === 0 && <span className="untagged-italic">untagged</span>}
        {visibleTags.map(t => (
          <span key={t} onClick={e => onClickTag(e, t)}><TagChip tag={t} /></span>
        ))}
        {overflow > 0 && <span className="more">+{overflow}</span>}
      </div>
      <div className="cell-status">
        <span className={`status-text ${doc.status}`}>
          {doc.status === "completed" && "Completed"}
          {doc.status === "processing" && "Processing"}
          {doc.status === "pending" && "Pending"}
          {doc.status === "failed" && "Failed"}
        </span>
      </div>
      <div className="cell-chunks">{doc.chunks !== null ? doc.chunks : "—"}</div>
      <div className="cell-updated">{doc.updated}</div>
      <div className="cell-action">
        {isFail ? (
          <button
            className="action-btn alert"
            onClick={e => { e.stopPropagation(); onSelect(doc, { focus: "error" }); }}
            title="View error details"
          >
            <Icon name="info-circle" size={13} />
          </button>
        ) : (
          <button
            className="action-btn retag"
            onClick={e => { e.stopPropagation(); onOpenRetag(doc); }}
            title="Retag this document"
          >
            Retag
          </button>
        )}
      </div>
    </div>
  );
}

// ─── Pipeline status popover ─────────────────────────────────────────────────
const PIPELINE_WORKERS = [
  { id: "extractor", label: "lightrag-extract", status: "ok", throughput: "12 chk/s", note: "v0.4.12" },
  { id: "embedder", label: "embedder · jina-v3", status: "warn", throughput: "4.2 chk/s", note: "rate-limited (60% quota)" },
  { id: "indexer", label: "graph-indexer", status: "ok", throughput: "31 chk/s", note: "lag 0.4s" },
  { id: "reranker", label: "reranker · bge-rr-v2", status: "ok", throughput: "—", note: "idle" }
];
const PIPELINE_QUEUE = [
  { source: "AWR_2026_Q1_report.pdf", state: "processing", progress: 62, eta: "1m 12s" },
  { source: "rman-backup-runbook.md", state: "processing", progress: 18, eta: "3m 40s" },
  { source: "https://docs.cib/oracle/dataguard", state: "processing", progress: 91, eta: "12s" },
  { source: "incident-2026-04-22.md", state: "queued", progress: 0, eta: "—" }
];

function PipelineStatusPopover({ onClose, onAddToast }) {
  const ref = React.useRef(null);
  const [paused, setPaused] = React.useState(false);
  const [confirmStop, setConfirmStop] = React.useState(false);
  React.useEffect(() => {
    const onDown = (e) => { if (ref.current && !ref.current.contains(e.target)) onClose(); };
    const onKey = (e) => { if (e.key === "Escape") onClose(); };
    setTimeout(() => document.addEventListener("mousedown", onDown), 0);
    document.addEventListener("keydown", onKey);
    return () => { document.removeEventListener("mousedown", onDown); document.removeEventListener("keydown", onKey); };
  }, [onClose]);

  const goToActivity = () => {
    onClose();
    window.location.search = "?tab=activity&kind=source-failed,source-completed,source-pending";
  };

  return (
    <div className="pipeline-popover" ref={ref} role="dialog" aria-label="Pipeline status">
      <header className="pp-header">
        <div className="pp-title">
          <Icon name="activity" size={14} />
          <span>Pipeline status</span>
          <span className={`pp-state-badge ${paused ? "paused" : "busy"}`}>
            <span className="pp-state-dot" />
            {paused ? "paused" : "busy"}
          </span>
        </div>
        <div className="pp-header-actions">
          {paused ? (
            <button
              className="btn small primary"
              onClick={() => { setPaused(false); onAddToast("Pipeline resumed", "POST /documents/pipeline_status · workers spinning up"); }}
            >
              <Icon name="refresh" size={12} /> Resume
            </button>
          ) : (
            <button className="btn small btn-stop-filled" onClick={() => setConfirmStop(true)}>
              <Icon name="x" size={12} /> Stop pipeline
            </button>
          )}
          <button className="icon-btn" onClick={onClose} aria-label="Close pipeline status"><Icon name="x" size={13} /></button>
        </div>
      </header>
      {confirmStop && (
        <div className="pp-confirm-banner">
          <span className="pp-confirm-text">Stop pipeline? In-flight chunks finish, queue holds.</span>
          <button className="btn small" onClick={() => setConfirmStop(false)}>Cancel</button>
          <button
            className="btn small danger"
            onClick={() => {
              setPaused(true); setConfirmStop(false);
              onAddToast("Pipeline stopped", "POST /documents/pipeline_status · 4 workers drained");
            }}
          >Stop</button>
        </div>
      )}

      <section className="pp-section">
        <h3>Workers</h3>
        <ul className="pp-workers">
          {PIPELINE_WORKERS.map(w => (
            <li key={w.id}>
              <span className={`pp-dot ${w.status}`} aria-hidden="true" />
              <span className="pp-w-label mono-meta">{w.label}</span>
              <span className="pp-w-thr mono-meta">{w.throughput}</span>
              <span className="pp-w-note">{w.note}</span>
            </li>
          ))}
        </ul>
      </section>

      <section className="pp-section">
        <div className="pp-section-head">
          <h3>Queue · {PIPELINE_QUEUE.filter(q => q.state === "processing").length} processing · {PIPELINE_QUEUE.filter(q => q.state === "queued").length} pending</h3>
        </div>
        <ul className="pp-queue">
          {PIPELINE_QUEUE.map((q, i) => (
            <li key={i} className={`pp-q-row ${q.state}`}>
              <span className="pp-q-name mono-meta" title={q.source}>{q.source}</span>
              <span className="pp-q-prog">
                {q.state === "processing" ? (
                  <span className="pp-q-bar"><span style={{ width: `${q.progress}%` }} /></span>
                ) : (
                  <span className="pp-q-state mono-meta">queued</span>
                )}
              </span>
              <span className="pp-q-eta mono-meta">{q.eta}</span>
            </li>
          ))}
        </ul>
      </section>

      <section className="pp-section pp-stats">
        <div><span className="pp-stat-num">142</span><span className="pp-stat-lbl">processed · 24h</span></div>
        <div><span className="pp-stat-num">3</span><span className="pp-stat-lbl">failed · 24h</span></div>
        <div><span className="pp-stat-num">0.4s</span><span className="pp-stat-lbl">queue lag</span></div>
      </section>

      <footer className="pp-footer">
        <button className="link-btn" onClick={goToActivity}>
          View pipeline events →
        </button>
      </footer>
    </div>
  );
}

// ─── Source detail side-panel ────────────────────────────────────────
const MOCK_LINEAGE = {
  uploadedBy: "m.ferrand@bnpparibas.com",
  uploadedAt: "2026-05-03 14:22 UTC",
  lastReingest: "2026-05-08 09:14 UTC",
  ingestDurationMs: 4820,
  reingestCount: 3,
  pipelineVersion: "lightrag/0.4.12+twin.0279",
  workspace: "cib-core",
  visibility: "internal",
  sha256: "7f4b9c…a82e1d",
  bytes: 184320
};

// Per-document extracted text for the "View raw" modal. The steward uses
// this to validate a pending document before approving — without it, the
// review queue is "trust me" governance. d13 + d14 are the seeded pending
// items the demo walks through; others fall back to a generic extract.
const MOCK_RAW_TEXT_BY_DOC = {
  d13: `=== CFT Vendor API specification (draft v0.7) ===
Vendor : Acme Payments Iberia SL · contract IBPAY-2026-014
Submitted by : marc.berthier · 2026-05-20

1. Overview
-----------
This document specifies the integration contract between the CIB
payment-orchestration layer and the Acme Payments Iberia gateway.
Coverage : SEPA Credit Transfer (SCT), SEPA Instant (SCT Inst) and
domestic Spain bizum-rail acknowledgments.

Production rollout target : 2026-Q3.
Confidence rating (vendor self-declared) : 70%.
Internal verification : pending steward sign-off (this review).

2. Authentication
-----------------
- mTLS, client cert issued by BNP InfoSec PKI (trust-store: cib-root-2024).
- API key in Authorization: Bearer header (rotated every 30 days).
- Optional HMAC-SHA256 body signature in X-Acme-Signature for high-value
  transfers (> EUR 100K). Rejected without 401 if missing.

3. Endpoints (vendor side)
--------------------------
POST   /v1/sct/credit-transfer        SCT initiation
POST   /v1/sct-inst/credit-transfer   SCT Inst initiation (<10s SLA)
GET    /v1/sct/{id}/status            Status enquiry, polling cap 1/s
POST   /v1/recall                     R-message (return / reject)
POST   /v1/bizum/ack                  Bizum acknowledgment relay

4. Idempotency
--------------
Required idempotency-key header on every POST. Acme retains the key for
24h. Duplicate requests within window return the original response with
HTTP 200; outside window return 409 Conflict.

5. Concerns flagged by Marc (submitter)
---------------------------------------
- Section 4 retention window (24h) is shorter than BNP guidance (72h).
  Recommend negotiation with Acme account manager.
- HMAC threshold at EUR 100K vs BNP CIB policy threshold at EUR 50K —
  policy mismatch, requires either contract amendment OR an internal
  override gateway rule.
- No explicit dispute-resolution endpoint; relies on R-message which
  doesn't cover all CIB business cases.

6. Open questions for steward
-----------------------------
[ ] Approve as-is and track the 3 concerns above in JIRA ?
[ ] Reject and request vendor revision (delays Q3 target by ~6 weeks) ?
[ ] Approve with edits (e.g. raise the HMAC threshold note in our
    operator runbook) ?

--- end of extracted text ---
chunks indexed: 47 · sha-256: 7f4b9c…a82e1d · bytes: 184320
`,

  d14: `=== Q2 2026 incident postmortem — DRAFT ===
Incident : INC-26-Q2-0414 · Severity: SEV-2 · Duration: 3h 41m
Service : CIB knowledge-base retrieval (TwinRAG production)
Submitted by : yann.dubois · 2026-05-20

1. Timeline (UTC)
-----------------
14:02 — quota-warn alert on llm provider (openai) at 85%.
14:14 — quota-exhausted at 100%. Retrieval halts on synthesis step.
14:18 — on-call paged via PagerDuty (yann.dubois).
14:31 — failover to anthropic-secondary attempted, rejected by gateway
        (cert chain not whitelisted in cib-router-2026).
14:55 — manual workaround : gateway whitelist updated via emergency
        change CHG-26-04188. Failover succeeds.
15:14 — secondary saturates (rate-limit 60 rpm at vendor side).
17:43 — incident closed after openai quota top-up + secondary release.

2. Client impact figures (sensitive)
------------------------------------
- 142 retrieval queries returned partial / no-context responses.
- 4 client-facing agents (treasury, corp-payments) degraded.
- 1 escalation from BNP Securities Services received at 15:48.
- No regulatory-reportable event (BCEN / ACPR threshold not crossed).

3. Root cause analysis
----------------------
Primary : quota planning at the openai vendor was based on Q1 traffic
patterns; Q2 onboarding of the corp-payments agent surfaced a 38%
month-over-month query growth not captured in the renewal forecast.

Contributing : secondary-provider failover hadn't been exercised since
2025-11; the cert-chain whitelist had silently drifted out of date.

4. Action items
---------------
- [done] Quarterly quota recalibration based on rolling 30-day usage.
- [done] Monthly failover drill (added to oncall rota).
- [in-progress] Multi-provider routing live, not just failover.
- [pending] Postmortem review with InfoSec on the gateway whitelist
  drift detection mechanism (this drift caused 24m of avoidable downtime).

5. Reason this is in review
---------------------------
Contains client-impact figures (section 2). Steward review required
before this document enters the active retrieval set, per the
sensitive-document policy in workspace cib-core.

--- end of extracted text ---
chunks indexed: 89 · sha-256: a82e1d…7f4b9c · bytes: 102400
`,
};

function _rawTextFor(doc) {
  if (MOCK_RAW_TEXT_BY_DOC[doc.id]) return MOCK_RAW_TEXT_BY_DOC[doc.id];
  // Generic fallback — synthesize from chunks + summary so the modal
  // always has something to show, even for unknown docs.
  return `=== ${doc.source || "Untitled"} ===
Type     : ${doc.type || "file"}
Summary  : ${doc.summary || "(no summary)"}
Tags     : ${(doc.tags || []).join(", ") || "—"}

--- extracted text (sampled chunks) ---
${MOCK_CHUNKS.map((c, i) => `[chunk ${i + 1} · ${c.tokens} tok · ${c.id}]\n${c.text}\n`).join("\n")}

--- end of sample ---
This document has ${doc.chunks ?? "?"} chunks total. The first three are
shown above as a representative sample for steward review.
`;
}

const MOCK_CHUNKS = [
  {
    id: "c_001",
    pos: "1 / 47",
    tokens: 412,
    text: "Oracle RMAN provides comprehensive backup and recovery for Oracle databases, supporting full, incremental and incrementally-updated backups. The RMAN catalog stores metadata about backups and is required for cross-instance recovery scenarios in the CIB topology."
  },
  {
    id: "c_002",
    pos: "2 / 47",
    tokens: 388,
    text: "Recovery scenarios documented here include: (a) datafile loss on primary, (b) full instance loss with Data Guard standby promotion, (c) corruption detected via DBVERIFY. Each scenario specifies the RTO/RPO target and the steward responsible for the runbook."
  },
  {
    id: "c_003",
    pos: "3 / 47",
    tokens: 401,
    text: "Backup retention follows the policy declared in the workspace `cib-core`: 14 days local, 90 days replicated to the secondary site. Any deviation requires an exception ticket and steward sign-off; the exception is propagated to the audit log via tag.deprecated event."
  }
];

// Translate a backend mutation row into a human-readable verb for the
// Audit timeline. Keeps the panel readable when payloads get nested.
function _formatAuditAction(m) {
  const p = m.payload || {};
  if (p.review) {
    const st = p.review.state;
    if (st === "approved") return p.review.edited ? "Approved with edits" : "Approved";
    if (st === "rejected") return "Rejected";
    if (st) return "Review → " + st;
  }
  if (m.action === "delete") return "Deleted";
  if (m.action === "patch") return "Patched";
  return m.action;
}

function DocDetailPanel({ doc, focus, onClose, onOpenRetag, onAddToast, onViewRaw }) {
  const [tab, setTab] = useState("overview");
  const errorRef = React.useRef(null);
  // Audit trail — live mutation log from the FastAPI backend, filtered
  // for this doc. Killer feature vs Kore.ai (audit is "in the logs, not
  // in the interface" per their own meeting transcription).
  const [audit, setAudit] = useState({ loading: false, entries: null, error: null });
  useEffect(() => {
    if (tab !== "audit") return;
    if (!window.twinDb || !doc) return;
    setAudit({ loading: true, entries: null, error: null });
    fetch("/api/mutations?limit=200")
      .then(r => r.ok ? r.json() : Promise.reject(new Error("HTTP " + r.status)))
      .then(rows => {
        // Backend returns ALL mutations; filter to this doc's audit trail
        // (cheap — dataset is small in the demo, would move server-side
        // with `?target_id=` query param if it grew).
        const mine = rows.filter(m => m.kind === "docs" && m.target_id === doc.id);
        setAudit({ loading: false, entries: mine, error: null });
      })
      .catch(err => setAudit({ loading: false, entries: null, error: err.message }));
  }, [tab, doc && doc.id]);

  useEffect(() => {
    if (focus === "error" && errorRef.current) {
      setTab("overview");
      const t = setTimeout(() => {
        const el = errorRef.current;
        if (!el) return;
        // Scroll within the detail-body container, not the page.
        const body = el.closest(".detail-body");
        if (body) body.scrollTop = el.offsetTop - 12;
        el.classList.add("is-flash");
        setTimeout(() => el.classList && el.classList.remove("is-flash"), 1400);
      }, 40);
      return () => clearTimeout(t);
    }
  }, [focus, doc && doc.id]);

  React.useEffect(() => {
    const handler = (e) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [onClose]);

  const reprocess = () => {
    onAddToast(`Re-processing ${doc.source}`, "Queued behind 2 sources · ETA 4 min");
    onClose();
  };

  const isFail = doc.status === "failed";
  const sizeKb = (MOCK_LINEAGE.bytes / 1024).toFixed(1);

  return (
    <>
      <div className="detail-scrim" onClick={onClose} />
      <aside className="detail-panel" role="dialog" aria-modal="true" aria-labelledby="detail-title">
        <header className="detail-header">
          <div className="detail-source">
            <SourceIcon type={doc.type} size={18} />
            <h2 id="detail-title">{doc.source}</h2>
          </div>
          <button className="icon-btn" onClick={onClose} aria-label="Close source detail">
            <Icon name="x" size={14} />
          </button>
        </header>

        <div className="detail-status-bar">
          <span className={`status-text ${doc.status}`}>
            {doc.status === "completed" && "Completed"}
            {doc.status === "processing" && "Processing"}
            {doc.status === "pending" && "Pending"}
            {doc.status === "failed" && "Failed"}
          </span>
          <span className="sep">·</span>
          <span className="mono-meta">{doc.chunks ?? "—"} chunks</span>
          <span className="sep">·</span>
          <span className="mono-meta">{sizeKb} KB</span>
          <span className="sep">·</span>
          <span className="mono-meta">updated {doc.updated}</span>
        </div>

        <nav className="detail-tabs">
          {["overview", "chunks", "lineage", "audit"].map(t => (
            <button
              key={t}
              className={`detail-tab${tab === t ? " active" : ""}`}
              onClick={() => setTab(t)}
            >
              {t === "overview" && "Overview"}
              {t === "chunks" && `Chunks (${doc.chunks ?? 0})`}
              {t === "lineage" && "Lineage"}
              {t === "audit" && <>Audit{audit.entries && audit.entries.length > 0 && <span className="audit-tab-count">{audit.entries.length}</span>}</>}
            </button>
          ))}
        </nav>

        <div className="detail-body">
          {tab === "overview" && (
            <>
              <section className="detail-section">
                <h3>Summary</h3>
                <p className={isFail ? "detail-summary failed" : "detail-summary"}>{doc.summary}</p>
              </section>

              <section className="detail-section">
                <div className="detail-section-head">
                  <h3>Tags <em>— Twin</em></h3>
                  <button className="link-btn" onClick={() => { onOpenRetag(doc); onClose(); }}>
                    Retag →
                  </button>
                </div>
                <div className="detail-tags">
                  {doc.tags.length === 0 && <span className="untagged-italic">untagged</span>}
                  {doc.tags.map(t => <TagChip key={t} tag={t} />)}
                </div>
              </section>

              {isFail && (
                <section className="detail-section detail-error" ref={errorRef}>
                  <h3>Error</h3>
                  <div className="detail-error-msg">
                    <Icon name="info-circle" size={13} />
                    <span>Embedder rejected payload (chunk 12/47): token count 8412 exceeds model limit 8192. Retry after adjusting chunk-size in workspace settings.</span>
                  </div>
                </section>
              )}

              <section className="detail-section">
                <h3>Quick info</h3>
                <dl className="detail-kv">
                  <dt>Workspace</dt><dd className="mono-meta">{MOCK_LINEAGE.workspace}</dd>
                  <dt>Visibility</dt><dd className="mono-meta">{MOCK_LINEAGE.visibility}</dd>
                  <dt>Uploaded by</dt><dd className="mono-meta">{MOCK_LINEAGE.uploadedBy}</dd>
                  <dt>SHA-256</dt><dd className="mono-meta">{MOCK_LINEAGE.sha256}</dd>
                </dl>
              </section>
            </>
          )}

          {tab === "chunks" && (
            <>
              <div className="chunks-toolbar">
                <span className="chunks-count">Showing 3 of {doc.chunks ?? 0}</span>
                <input className="chunks-search" placeholder="Search within chunks…" />
              </div>
              {MOCK_CHUNKS.map(c => (
                <article key={c.id} className="chunk-card">
                  <header className="chunk-head">
                    <span className="chunk-pos">{c.pos}</span>
                    <span className="chunk-meta">{c.tokens} tok · <span className="mono-meta">{c.id}</span></span>
                  </header>
                  <p className="chunk-text">{c.text}</p>
                </article>
              ))}
              <button className="chunks-loadmore">Load more chunks →</button>
            </>
          )}

          {tab === "lineage" && (
            <>
              <section className="detail-section">
                <h3>Ingest timeline</h3>
                <ol className="lineage-timeline">
                  <li>
                    <span className="lt-dot" />
                    <div>
                      <div className="lt-title">Uploaded</div>
                      <div className="lt-meta mono-meta">{MOCK_LINEAGE.uploadedAt} · {MOCK_LINEAGE.uploadedBy}</div>
                    </div>
                  </li>
                  <li>
                    <span className="lt-dot" />
                    <div>
                      <div className="lt-title">Initial ingest</div>
                      <div className="lt-meta mono-meta">{MOCK_LINEAGE.uploadedAt} · {MOCK_LINEAGE.ingestDurationMs}ms · pipeline {MOCK_LINEAGE.pipelineVersion}</div>
                    </div>
                  </li>
                  <li>
                    <span className="lt-dot" />
                    <div>
                      <div className="lt-title">Re-ingest #{MOCK_LINEAGE.reingestCount}</div>
                      <div className="lt-meta mono-meta">{MOCK_LINEAGE.lastReingest} · triggered by tag.approved(rman)</div>
                    </div>
                  </li>
                </ol>
              </section>

              <section className="detail-section">
                <h3>Provenance</h3>
                <dl className="detail-kv">
                  <dt>Pipeline version</dt><dd className="mono-meta">{MOCK_LINEAGE.pipelineVersion}</dd>
                  <dt>Workspace</dt><dd className="mono-meta">{MOCK_LINEAGE.workspace}</dd>
                  <dt>Re-ingest count</dt><dd className="mono-meta">{MOCK_LINEAGE.reingestCount}</dd>
                  <dt>Last duration</dt><dd className="mono-meta">{MOCK_LINEAGE.ingestDurationMs} ms</dd>
                  <dt>Bytes</dt><dd className="mono-meta">{MOCK_LINEAGE.bytes.toLocaleString()}</dd>
                  <dt>SHA-256</dt><dd className="mono-meta">{MOCK_LINEAGE.sha256}</dd>
                </dl>
              </section>
            </>
          )}

          {tab === "audit" && (
            <>
              <section className="detail-section">
                <div className="detail-section-head">
                  <h3>Audit trail</h3>
                  <span className="audit-source-pill">
                    <Icon name="lock" size={10} />
                    SQLite · table <code>mutations</code>
                  </span>
                </div>
                <p className="audit-sub">
                  Every steward action on this document, sourced live from the
                  FastAPI backend. Visible in the interface — not buried in
                  server logs.
                </p>

                {audit.loading && (
                  <div className="audit-loading">
                    <Icon name="loader-2" size={14} /> Loading audit trail…
                  </div>
                )}

                {audit.error && (
                  <div className="audit-error">
                    <Icon name="alert-triangle" size={13} />
                    <span>Backend unreachable — {audit.error}</span>
                  </div>
                )}

                {!audit.loading && !audit.error && (
                  <ol className="audit-timeline">
                    {/* Seed events (from doc data, not from mutation log) so
                        the timeline isn't empty before any operator action. */}
                    {doc.review && doc.review.requested_at && (
                      <li className="audit-entry is-seed">
                        <span className="audit-dot" />
                        <div className="audit-body">
                          <div className="audit-line1">
                            <span className="audit-action">Requested for review</span>
                            <span className="audit-actor">by <b>{doc.review.requested_by || "unknown"}</b></span>
                          </div>
                          {doc.review.justification && (
                            <div className="audit-justif">{doc.review.justification}</div>
                          )}
                          <div className="audit-when">{doc.review.requested_at} · seed</div>
                        </div>
                      </li>
                    )}
                    {audit.entries && audit.entries.map(m => (
                      <li key={m.id} className={"audit-entry " + (m.payload && m.payload.review && m.payload.review.state === "rejected" ? "is-reject" : "is-mutation")}>
                        <span className="audit-dot" />
                        <div className="audit-body">
                          <div className="audit-line1">
                            <span className="audit-action">{_formatAuditAction(m)}</span>
                            {m.payload && m.payload.review && m.payload.review.reviewed_by && (
                              <span className="audit-actor">by <b>{m.payload.review.reviewed_by}</b></span>
                            )}
                          </div>
                          {m.payload && m.payload.review && m.payload.review.reason && (
                            <div className="audit-justif">Reason: {m.payload.review.reason}</div>
                          )}
                          <div className="audit-when">
                            {m.ts.replace("T", " ").slice(0, 19)} UTC · mutation #{m.id}
                          </div>
                          <details className="audit-payload">
                            <summary>raw payload</summary>
                            <pre>{JSON.stringify(m.payload, null, 2)}</pre>
                          </details>
                        </div>
                      </li>
                    ))}
                    {audit.entries && audit.entries.length === 0 && !doc.review && (
                      <li className="audit-empty">No steward action recorded for this document yet.</li>
                    )}
                  </ol>
                )}
              </section>
            </>
          )}
        </div>

        <footer className="detail-footer">
          <button className="btn" onClick={() => { onOpenRetag(doc); onClose(); }}>
            <Icon name="plus" size={13} /> Retag
          </button>
          <button className="btn" onClick={() => onViewRaw && onViewRaw()}>
            <Icon name="eye" size={13} /> View raw
          </button>
          <button className="btn primary" onClick={reprocess} disabled={doc.status === "processing"}>
            <Icon name="refresh" size={13} /> Re-process
          </button>
        </footer>
      </aside>
    </>
  );
}

// Steward's window into the actual ingested content. Without this, the
// Approve / Reject buttons in the pending review queue are blind faith.
function RawDocModal({ doc, onClose }) {
  const ref = React.useRef(null);
  React.useEffect(() => {
    const onKey = (e) => { if (e.key === "Escape") onClose(); };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [onClose]);

  const text = _rawTextFor(doc);
  const sizeKb = (text.length / 1024).toFixed(1);

  const download = () => {
    const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    const safeName = (doc.source || "document").replace(/[^a-z0-9.-]+/gi, "_");
    a.href = url;
    a.download = `${safeName}.extracted.txt`;
    document.body.appendChild(a);
    a.click();
    setTimeout(() => { URL.revokeObjectURL(url); a.remove(); }, 0);
  };

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div
        ref={ref}
        className="modal raw-doc-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="raw-doc-title"
        onClick={e => e.stopPropagation()}
      >
        <div className="modal-h">
          <div className="raw-doc-h">
            <h3 id="raw-doc-title">
              <Icon name="eye" size={14} />
              Extracted text — what the indexer ingested
            </h3>
            <div className="raw-doc-meta">
              <code className="mono-meta">{doc.source}</code>
              <span className="sep">·</span>
              <span>{doc.chunks ?? "?"} chunks indexed</span>
              <span className="sep">·</span>
              <span>{sizeKb} KB extracted</span>
              {doc.review && doc.review.state === "pending-review" && (
                <>
                  <span className="sep">·</span>
                  <span className="raw-doc-pending">awaiting steward review</span>
                </>
              )}
            </div>
          </div>
          <button className="modal-x" onClick={onClose} aria-label="Close raw view">
            <Icon name="x" size={14} />
          </button>
        </div>

        <div className="raw-doc-body">
          <pre className="raw-doc-text">{text}</pre>
        </div>

        <div className="modal-footer raw-doc-footer">
          <span className="raw-doc-disclaimer">
            <Icon name="info-circle" size={11} />
            This is the post-extraction text used for retrieval, not the
            original binary. Use it to validate before approving.
          </span>
          <button className="ghost-btn" onClick={download}>
            <Icon name="cloud-upload" size={12} /> Download .txt
          </button>
          <button className="btn" onClick={onClose}>Close</button>
        </div>
      </div>
    </div>
  );
}
