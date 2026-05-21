// Tags / Thesaurus governance — aligned with the screen-tags spec.
const { useState, useMemo } = React;
// Layout: header (with palier RBAC indicator) → Pending requests section →
// filters → category rail + card grid + side detail panel.

const CURRENT_USER = { name: "claire.benoit", palier: 3, role: "admin / steward" };

// UI-facing label for the palier ladder. `palier` is the back-end / API
// term (see README §5); the UI talks Reader / Contributor / Steward so the
// BNP audience isn't forced to learn an internal vocabulary.
const PALIER_ROLE_LABEL = { 1: "Reader", 2: "Contributor", 3: "Steward" };
const _roleLabel = (p) => PALIER_ROLE_LABEL[p] || `Palier ${p}`;

// Thesaurus export — JSON snapshot fit for diffing / re-import / governance review.
function exportThesaurusJson(tags, categories) {
  const payload = {
    workspace: "cib",
    exported_at: new Date().toISOString(),
    exported_by: CURRENT_USER.name,
    categories,
    tags: tags.map(t => ({
      tag: t.tag, tier: t.tier, category: t.category, status: t.status,
      def: t.def, aliases: t.aliases, deprecates: t.deprecates,
      sources_count: t.sources_count, chunks_count: t.chunks_count,
      query_freq_30d: t.query_freq_30d,
      created: t.created, last_edit: t.last_edit
    }))
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  const stamp = new Date().toISOString().slice(0, 10);
  a.href = url;
  a.download = `twin-rag-thesaurus-${stamp}.json`;
  document.body.appendChild(a);
  a.click();
  setTimeout(() => { URL.revokeObjectURL(url); a.remove(); }, 0);
}

function StatusBadge({ status, size = "sm" }) {
  const map = {
    active:               { label: "Active",      cls: "status-active" },
    "pending-promotion":  { label: "Pending",     cls: "status-pending" },
    "pending-review":     { label: "Pending",     cls: "status-pending" },
    deprecated:           { label: "Deprecated",  cls: "status-deprecated" },
    rejected:             { label: "Rejected",    cls: "status-rejected" }
  };
  const m = map[status] || { label: status, cls: "status-active" };
  return <span className={`status-badge ${m.cls} ${size}`}>{m.label}</span>;
}

function TagsEmptyZero({ canSuggest, onRequest }) {
  return (
    <div className="tags-empty zero">
      <div className="tags-empty-illus" aria-hidden="true">
        <svg width="120" height="80" viewBox="0 0 120 80" fill="none">
          <rect x="6" y="18" width="44" height="18" rx="3" stroke="currentColor" strokeWidth="1" strokeDasharray="3 3" opacity="0.45" />
          <rect x="54" y="32" width="56" height="18" rx="3" stroke="currentColor" strokeWidth="1" strokeDasharray="3 3" opacity="0.35" />
          <rect x="20" y="48" width="38" height="18" rx="3" stroke="currentColor" strokeWidth="1" strokeDasharray="3 3" opacity="0.25" />
        </svg>
      </div>
      <div className="tags-empty-title">No tags in this workspace yet</div>
      <p className="tags-empty-body">
        The thesaurus is empty. Start by requesting your first tag — a steward will review and promote
        it to a Tier 1 / 2 / 3 slot. Every tagged source then becomes filterable in Retrieval.
      </p>
      <div className="tags-empty-actions">
        {canSuggest ? (
          <button className="primary-btn" onClick={onRequest}>
            <Icon name="plus" size={12} /> Request the first tag
          </button>
        ) : (
          <span className="tags-empty-hint">
            Reader role — tag requests not allowed. Ask a Contributor or Steward.
          </span>
        )}
      </div>
      <ul className="tags-empty-tips">
        <li><Icon name="info-circle" size={11} /> Tier 1 (Trunk) — gov-validated, applies cross-workspace</li>
        <li><Icon name="info-circle" size={11} /> Tier 2 (Branch) — dept-scoped, steward-approved</li>
        <li><Icon name="info-circle" size={11} /> Tier 3 (Leaf) — user-proposed, lightweight review</li>
      </ul>
    </div>
  );
}

function TagsEmptyFiltered({ q, selectedCat, selectedStatus, categories, suggestions, canSuggest, onClear, onPickTag, onRequest }) {
  const catLabel = selectedCat !== "all" && categories.find(c => c.id === selectedCat);
  const active = [
    q.trim() && { key: "q", label: `search: "${q.trim()}"` },
    catLabel && { key: "cat", label: `category: ${catLabel.label}` },
    selectedStatus !== "all" && { key: "status", label: `status: ${selectedStatus}` }
  ].filter(Boolean);
  return (
    <div className="tags-empty filtered">
      <div className="tags-empty-ico"><Icon name="search" size={20} color="var(--color-text-tertiary)" /></div>
      <div className="tags-empty-title">No tags match the current filter</div>
      {active.length > 0 && (
        <div className="tags-empty-chips">
          {active.map(a => <span key={a.key} className="tags-empty-chip">{a.label}</span>)}
        </div>
      )}
      <div className="tags-empty-actions">
        <button className="primary-btn" onClick={onClear}>Clear filters</button>
        {canSuggest && q.trim() && (
          <button className="ghost-btn" onClick={onRequest}>
            <Icon name="plus" size={12} /> Request <code>{q.trim().toLowerCase().replace(/\s+/g, "-")}</code> as new tag
          </button>
        )}
      </div>
      {suggestions.length > 0 && (
        <div className="tags-empty-suggest">
          <div className="tags-empty-suggest-h">Try one of these instead</div>
          <div className="tags-empty-suggest-row">
            {suggestions.map(s => (
              <button key={s.tag} className="tags-empty-suggest-chip" onClick={() => { onClear(); onPickTag(s.tag); }}>
                <code>{s.tag}</code>
                <span className="tags-empty-suggest-meta">{s.sources_count} docs</span>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

window.TagsTab = function TagsTab({ onPushToast }) {
  const tags = window.MOCK_TAGS_FULL;
  const categories = window.MOCK_TAG_CATEGORIES;

  const [selectedCat, setSelectedCat] = window.useUrlParam("cat", "all");
  const [selectedStatus, setSelectedStatus] = window.useUrlParam("status", "all", {
    validate: v => ["all","active","pending","deprecated","rejected"].includes(v)
  });
  const [q, setQ] = window.useUrlParam("q", "");
  const [selectedTag, setSelectedTag] = window.useUrlParam("tag", "rman");
  const [pendingOpen, setPendingOpen] = useState(true);
  const [modal, setModal] = useState(null); // {kind, tag?}
  // Cross-tab handoff: Retrieval's "Request new tag" link navigates here
  // with ?req=<name>. Auto-open the request modal with the name seeded.
  useEffect(() => {
    const consume = () => {
      const p = new URLSearchParams(window.location.search);
      const req = p.get("req");
      if (req) {
        setModal({ kind: "request", seedName: req });
        p.delete("req");
        const qs = p.toString();
        window.history.replaceState(null, "", window.location.pathname + (qs ? "?" + qs : ""));
      }
    };
    consume();
    window.addEventListener("popstate", consume);
    return () => window.removeEventListener("popstate", consume);
  }, []);
  // Detail-panel collapsed by default under 1500px viewport — the 200/1fr/380
  // grid otherwise crushes the card grid to a single column with a wide
  // blank zone (audit #41). User can toggle freely once open.
  const [panelOpen, setPanelOpen] = useState(() =>
    typeof window !== "undefined" ? window.innerWidth >= 1500 : true
  );

  const requested = tags.filter(t => t.tier === "requested");

  const counts = useMemo(() => {
    const c = { all: tags.filter(t => t.tier !== "requested").length };
    categories.forEach(cat => { c[cat.id] = tags.filter(t => t.category === cat.id && t.tier !== "requested").length; });
    return c;
  }, [tags, categories]);

  const statusOf = (t) => t.tier === "requested" ? "pending-review" : t.status;

  const filtered = tags.filter(t => {
    if (t.tier === "requested") return false; // pending requests live in their own section
    if (selectedCat !== "all" && t.category !== selectedCat) return false;
    if (selectedStatus !== "all" && t.status !== selectedStatus) return false;
    if (q.trim()) {
      const n = q.trim().toLowerCase();
      const hay = (t.tag + " " + t.def + " " + t.aliases.join(" ")).toLowerCase();
      if (!hay.includes(n)) return false;
    }
    return true;
  });

  const detail = tags.find(t => t.tag === selectedTag) || tags[0];
  const canEdit = CURRENT_USER.palier >= 3;
  const canSuggest = CURRENT_USER.palier >= 2;

  return (
    <div className="tags-screen">
      <div className="tags-header">
        <div>
          <h1>Tags</h1>
          <div className="tags-sub">
            <span>Thesaurus governance · {counts.all} active tags · {requested.length} pending requests · workspace <code>cib</code></span>
            <span className="dot-sep">·</span>
            <span className="palier-pill" tabIndex={0} title="Steward — full access: approve/reject tag requests, edit definitions, deprecate, delete with migration, purge expired activity events.">
              {_roleLabel(CURRENT_USER.palier)}
            </span>
          </div>
        </div>
        <div className="tags-header-actions">
          {canSuggest && (
            <button
              className="ghost-btn"
              onClick={() => exportThesaurusJson(tags, categories)}
              title="Download full thesaurus as JSON"
            >
              <Icon name="external-link" size={12} /> Export thesaurus
            </button>
          )}
          <button className="primary-btn" onClick={() => setModal({ kind: "request" })}>
            <Icon name="plus" size={12} /> Request new tag
          </button>
        </div>
      </div>

      {requested.length > 0 && canSuggest && (
        <div className={"pending-section " + (pendingOpen ? "is-open" : "")}>
          <button className="pending-h" onClick={() => setPendingOpen(o => !o)}>
            <Icon name="alert-triangle" size={14} color="var(--twin-amber-vivid)" />
            <span className="pending-title">Pending requests</span>
            <span className="pending-counts"><b>{requested.length}</b> awaiting review</span>
            <Icon name="chevron-down" size={14} color="var(--color-text-tertiary)" style={{ transform: pendingOpen ? "none" : "rotate(-90deg)", transition: "transform .15s" }} />
          </button>
          {pendingOpen && (
            <div className="pending-grid">
              {requested.map(t => (
                <div key={t.tag} className="pending-card requested">
                  <div className="pending-card-h">
                    <code className="pending-tagname">{t.tag}</code>
                    <StatusBadge status="pending-review" />
                  </div>
                  <div className="pending-justif">{t.justification}</div>
                  <div className="pending-meta">
                    Proposed by <b>{t.requested_by}</b> · {t.requested_at} · category <code>{t.category}</code>
                  </div>
                  {canEdit ? (
                    <div className="pending-actions">
                      <button className="primary-btn small" onClick={() => { window.twinCompleteTask && window.twinCompleteTask("tag"); onPushToast && onPushToast({ id: "approve-" + t.tag + "-" + Date.now(), title: "Tag", tagname: t.tag, titleSuffix: "approved", sub: "Added to thesaurus · auto-emit tag.approved", undo: true }); }}>Approve</button>
                      <button className="ghost-btn small" onClick={() => setModal({ kind: "edit-approve", tag: t })}>Edit & approve</button>
                      <button className="ghost-btn small danger" onClick={() => setModal({ kind: "reject", tag: t })}>Reject</button>
                    </div>
                  ) : (
                    <div className="pending-actions">
                      <span className="muted">Awaiting steward review</span>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      <div className="tags-filters">
        <div className="tags-search">
          <Icon name="search" size={13} color="var(--color-text-tertiary)" />
          <input
            type="text"
            value={q}
            onChange={e => setQ(e.target.value)}
            placeholder="Search by name, definition or synonym…"
          />
          {q && <button className="x" onClick={() => setQ("")}><Icon name="x" size={11} /></button>}
        </div>
        <select className="mini-select" value={selectedStatus} onChange={e => setSelectedStatus(e.target.value)}>
          <option value="all">All statuses</option>
          <option value="active">Active</option>
          <option value="pending-promotion">Pending</option>
          <option value="deprecated">Deprecated</option>
          <option value="rejected">Rejected</option>
        </select>
      </div>

      <div className={`tags-body${panelOpen ? "" : " is-panel-collapsed"}`}>
        <aside className="tags-rail">
          <button className={"rail-item " + (selectedCat === "all" ? "is-active" : "")} onClick={() => setSelectedCat("all")}>
            <span className="rail-dot" style={{ background: "var(--color-text-tertiary)" }} />
            <span className="rail-label">All domains</span>
            <span className="rail-count">{counts.all}</span>
          </button>
          {categories.map(c => (
            <button key={c.id} className={"rail-item " + (selectedCat === c.id ? "is-active" : "")} onClick={() => setSelectedCat(c.id)}>
              <span className="rail-dot" style={{ background: c.color }} />
              <span className="rail-label">{c.label}</span>
              <span className="rail-count">{counts[c.id] || 0}</span>
            </button>
          ))}
        </aside>

        <main className="tags-grid-wrap">
          {filtered.length > 0 ? (
            <div className="tags-grid">
              {filtered.map(t => {
                const cat = categories.find(c => c.id === t.category);
                return (
                  <button
                    key={t.tag}
                    className={"tag-card " + (selectedTag === t.tag ? "is-selected" : "")}
                    onClick={() => setSelectedTag(t.tag)}
                  >
                    <div className="tag-card-h">
                      <code className="tag-card-name">{t.tag}</code>
                      {cat && (
                        <span className="domain-badge" style={{ borderColor: cat.color, color: cat.color }}>
                          {cat.label}
                        </span>
                      )}
                    </div>
                    <div className="tag-card-def">{t.def}</div>
                    {t.aliases.length > 0 && (
                      <div className="tag-card-aliases">
                        <span className="al-label">syn:</span>
                        {t.aliases.map(a => <code key={a}>{a}</code>)}
                      </div>
                    )}
                    <div className="tag-card-footer">
                      <span><b>{t.sources_count}</b> docs</span>
                      <span className="dot-sep">·</span>
                      <span>{t.query_freq_30d}/30d</span>
                      <span className="spacer" />
                      <StatusBadge status={t.status} />
                    </div>
                  </button>
                );
              })}
            </div>
          ) : counts.all === 0 ? (
            <TagsEmptyZero canSuggest={canSuggest} onRequest={() => setModal({ kind: "request" })} />
          ) : (
            <TagsEmptyFiltered
              q={q}
              selectedCat={selectedCat}
              selectedStatus={selectedStatus}
              categories={categories}
              suggestions={tags.filter(t => t.tier !== "requested" && t.status === "active").slice(0, 4)}
              canSuggest={canSuggest}
              onClear={() => { setSelectedCat("all"); setSelectedStatus("all"); setQ(""); }}
              onPickTag={(name) => setSelectedTag(name)}
              onRequest={() => setModal({ kind: "request" })}
            />
          )}
        </main>

        {panelOpen ? (
          <TagDetailPanel
            t={detail}
            allTags={tags}
            onSelect={setSelectedTag}
            onAction={setModal}
            canEdit={canEdit}
            canSuggest={canSuggest}
            onClose={() => setPanelOpen(false)}
          />
        ) : (
          detail && (
            <button
              className="tag-detail-rail"
              onClick={() => setPanelOpen(true)}
              title="Show tag details"
              aria-label={`Show details for tag ${detail.tag}`}
            >
              <Icon name="chevron-up" size={11} style={{ transform: "rotate(-90deg)" }} />
              <span>Details</span>
            </button>
          )
        )}
      </div>

      {modal && (
        <TagActionModal
          action={modal}
          allTags={tags}
          onClose={() => setModal(null)}
          onCommit={(msg) => {
            setModal(null);
            onPushToast && onPushToast(msg);
          }}
        />
      )}
    </div>
  );
};

function TagDetailPanel({ t, allTags, onSelect, onAction, canEdit, canSuggest, onClose }) {
  const [moreOpen, setMoreOpen] = React.useState(false);
  const moreRef = React.useRef(null);
  React.useEffect(() => {
    if (!moreOpen) return;
    const onDown = (e) => { if (moreRef.current && !moreRef.current.contains(e.target)) setMoreOpen(false); };
    const onKey = (e) => { if (e.key === "Escape") setMoreOpen(false); };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [moreOpen]);
  if (!t) return null;
  const cat = window.MOCK_TAG_CATEGORIES.find(c => c.id === t.category);
  return (
    <aside className="tag-detail">
      <div className="detail-head">
        <div className="detail-kind" style={{ color: cat ? cat.color : "var(--color-text-secondary)" }}>
          <span className="rail-dot" style={{ background: cat ? cat.color : "var(--color-text-tertiary)" }} />
          {cat ? cat.label : "Uncategorized"}
          {onClose && (
            <button
              className="tag-detail-close"
              onClick={onClose}
              aria-label="Collapse details panel"
              title="Collapse panel"
            >
              <Icon name="x" size={12} />
            </button>
          )}
        </div>
        <div className="tag-detail-h">
          <code className="tag-detail-name">{t.tag}</code>
          <StatusBadge status={t.status} size="md" />
        </div>
        {t.aliases.length > 0 && (
          <div className="tag-detail-aliases">
            <span className="al-label">Synonyms:</span>
            {t.aliases.map(a => <code key={a}>{a}</code>)}
          </div>
        )}
        <div className="detail-summary">{t.def}</div>
      </div>

      <div className="detail-section">
        <div className="detail-section-h">Usage</div>
        <div className="usage-grid">
          <div className="usage-cell">
            <div className="usage-num">{t.sources_count}</div>
            <div className="usage-lbl">Docs</div>
          </div>
          <div className="usage-cell">
            <div className="usage-num">{t.chunks_count.toLocaleString()}</div>
            <div className="usage-lbl">Chunks</div>
          </div>
          <div className="usage-cell">
            <div className="usage-num">{t.query_freq_30d}</div>
            <div className="usage-lbl">Queries / 30d</div>
          </div>
        </div>
      </div>

      {t.examples.length > 0 && (
        <div className="detail-section">
          <div className="detail-section-h">Last tagged docs</div>
          <div className="example-list">
            {t.examples.map(e => (
              <a key={e} className="example-row" href={`?tab=documents&q=${encodeURIComponent(e)}`} onClick={ev => {
                ev.preventDefault();
                const p = new URLSearchParams();
                p.set("tab", "documents");
                p.set("q", e);
                window.history.pushState(null, "", window.location.pathname + "?" + p.toString());
                window.dispatchEvent(new PopStateEvent("popstate"));
              }}>
                <Icon name={e.includes("/") ? "brand-confluence" : "file-text"} size={12} color="var(--color-text-tertiary)" />
                <span>{e}</span>
                <Icon name="arrow-right" size={11} color="var(--color-text-tertiary)" />
              </a>
            ))}
            {t.sources_count > t.examples.length && (
              <a className="example-more" href={`?tab=documents&tag=${encodeURIComponent(t.tag)}`} onClick={ev => {
                ev.preventDefault();
                const p = new URLSearchParams();
                p.set("tab", "documents");
                p.set("tag", t.tag);
                window.history.pushState(null, "", window.location.pathname + "?" + p.toString());
                window.dispatchEvent(new PopStateEvent("popstate"));
              }}>View all {t.sources_count} docs in Documents →</a>
            )}
          </div>
        </div>
      )}

      {t.related.length > 0 && (
        <div className="detail-section">
          <div className="detail-section-h">Co-occurring tags</div>
          <div className="related-list">
            {t.related.map(r => {
              const rt = allTags.find(x => x.tag === r.tag);
              if (!rt) return null;
              return (
                <button key={r.tag} className="related-chip" onClick={() => onSelect(r.tag)}>
                  <code>{r.tag}</code>
                  <span className="related-strength">{(r.strength * 100).toFixed(0)}%</span>
                </button>
              );
            })}
          </div>
        </div>
      )}

      <div className="detail-section">
        <div className="detail-section-h">History</div>
        <div className="history-list">
          <div className="hist-item">
            <span className="hist-when">{t.last_edit.at}</span>
            <span className="hist-what">{t.last_edit.action}</span>
            <span className="hist-who">by {t.last_edit.by}</span>
          </div>
          <div className="hist-item">
            <span className="hist-when">{t.created.at}</span>
            <span className="hist-what">created</span>
            <span className="hist-who">by {t.created.by}</span>
          </div>
        </div>
      </div>

      <div className="detail-actions wrap">
        {!canSuggest && (
          <span className="muted-italic">Reader role — read-only. Upgrade to Contributor to suggest edits.</span>
        )}
        {canSuggest && !canEdit && (
          <button className="ghost-btn small" onClick={() => onAction({ kind: "suggest", tag: t })}>Suggest edit</button>
        )}
        {canEdit && (
          <>
            <button className="ghost-btn small" onClick={() => onAction({ kind: "edit", tag: t })}>Edit</button>
            <button className="ghost-btn small" onClick={() => onAction({ kind: "synonyms", tag: t })}>Manage synonyms</button>
            <div className="tag-actions-more" ref={moreRef}>
              <button
                className={"ghost-btn small" + (moreOpen ? " is-open" : "")}
                onClick={() => setMoreOpen(o => !o)}
                aria-expanded={moreOpen}
                aria-haspopup="menu"
                aria-label="More actions"
                title="More actions"
              >
                More <Icon name="chevron-down" size={9} />
              </button>
              {moreOpen && (
                <div className="tag-actions-more-popover" role="menu">
                  <button
                    className="tag-actions-more-item"
                    role="menuitem"
                    onClick={() => { setMoreOpen(false); onAction({ kind: "deprecate", tag: t }); }}
                  >Deprecate</button>
                  <button
                    className="tag-actions-more-item danger"
                    role="menuitem"
                    onClick={() => { setMoreOpen(false); onAction({ kind: "delete", tag: t }); }}
                  >Delete…</button>
                </div>
              )}
            </div>
          </>
        )}
      </div>
    </aside>
  );
}

function TagActionModal({ action, allTags, onClose, onCommit }) {
  const modalRef = React.useRef(null);
  window.useModalA11y && window.useModalA11y({ open: true, onClose, ref: modalRef });
  const tag = action.tag;
  const [name, setName] = useState(tag ? tag.tag : (action.seedName || ""));
  const [migrateTo, setMigrateTo] = useState("");
  const [migrateStrategy, setMigrateStrategy] = useState("migrate");
  const [newSyn, setNewSyn] = useState("");
  const [reason, setReason] = useState("");

  const titleMap = {
    edit:           "Edit tag",
    suggest:        "Suggest tag edit",
    synonyms:       "Manage synonyms",
    deprecate:      "Deprecate tag",
    delete:         "Delete tag",
    reject:         "Reject request",
    "edit-approve": "Edit & approve request",
    request:        "Request new tag"
  };

  const commit = () => {
    // Any committed change in the Tag governance modal counts as "applying a tag"
    // for the onboarding checklist — it materially affects the thesaurus.
    window.twinCompleteTask && window.twinCompleteTask("tag");
    const stamp = Date.now();
    if (action.kind === "edit") {
      onCommit({ id: "edit-" + stamp, title: "Tag", tagname: tag.tag, titleSuffix: "definition updated", sub: "tag.edited emitted to Activity", undo: true });
    } else if (action.kind === "suggest") {
      onCommit({ id: "sug-" + stamp, title: "Tag", tagname: tag.tag, titleSuffix: "edit suggested", sub: "Awaiting steward review", undo: false });
    } else if (action.kind === "synonyms") {
      onCommit({ id: "syn-" + stamp, title: "Tag", tagname: tag.tag, titleSuffix: "synonyms updated", sub: "Query rewriting refreshed at gateway", undo: true });
    } else if (action.kind === "deprecate") {
      onCommit({ id: "dep-" + stamp, title: "Tag", tagname: tag.tag, titleSuffix: "deprecated", sub: `${tag.sources_count} docs flagged · tag.deprecated emitted`, undo: true });
    } else if (action.kind === "delete") {
      const verb = migrateStrategy === "migrate" ? `migrated to ${migrateTo}` : "deleted (docs untagged)";
      onCommit({ id: "del-" + stamp, title: "Tag", tagname: tag.tag, titleSuffix: verb, sub: `${tag.sources_count} docs updated · tag.deleted emitted`, undo: false });
    } else if (action.kind === "reject") {
      onCommit({ id: "rej-" + stamp, title: "Tag", tagname: tag.tag, titleSuffix: "rejected", sub: reason || "tag.rejected emitted · author notified", undo: false });
    } else if (action.kind === "edit-approve") {
      onCommit({ id: "ea-" + stamp, title: "Tag", tagname: tag.tag, titleSuffix: "approved (edited)", sub: "Added to thesaurus · tag.approved emitted", undo: true });
    } else if (action.kind === "request") {
      onCommit({ id: "req-" + stamp, title: "Tag", tagname: name, titleSuffix: "requested for review", sub: "Queued for steward approval · tag.request_new emitted", undo: false });
    }
  };

  const eligible = allTags.filter(x => tag && x.tag !== tag.tag && x.tier !== "requested" && x.status === "active");

  return (
    <div className="modal-bg" onClick={onClose}>
      <div
        ref={modalRef}
        className="modal tag-action-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="tagaction-title"
        tabIndex={-1}
        onClick={e => e.stopPropagation()}
      >
        <div className="modal-h">
          <h3 id="tagaction-title">{titleMap[action.kind]}</h3>
          {tag && action.kind !== "request" && (
            <div className="modal-h-sub">
              <code>{tag.tag}</code>
              <StatusBadge status={tag.status} />
              <span className="dot-sep">·</span>
              <span>{tag.sources_count} docs · {tag.chunks_count.toLocaleString()} chunks</span>
            </div>
          )}
          <button className="modal-x" onClick={onClose} aria-label="Close dialog"><Icon name="x" size={14} /></button>
        </div>

        <div className="modal-body">
          {(action.kind === "edit" || action.kind === "suggest" || action.kind === "edit-approve") && (
            <>
              <label className="field-label">Name (canonical)</label>
              <input className="text-input" value={name} onChange={e => setName(e.target.value)} disabled={action.kind !== "edit-approve" && action.kind !== "edit"} />
              <label className="field-label">Short definition</label>
              <textarea className="text-input" rows="3" defaultValue={tag.def} />
              <label className="field-label">Long description (optional)</label>
              <textarea className="text-input" rows="3" placeholder="For complex tags — surfaced in autocomplete tooltip." />
              <div className="impact-box">
                <Icon name="info-circle" size={13} color="var(--twin-accent)" />
                <span>
                  {action.kind === "suggest" && <>Your edit will be queued for steward review. Existing chunks remain untouched until approval.</>}
                  {action.kind === "edit" && <>Definition changes are non-destructive — only the autocomplete and detail panel update. Synonyms and structure are managed separately.</>}
                  {action.kind === "edit-approve" && <>You can tweak the proposed definition before accepting. The tag will enter the thesaurus as <b>active</b>.</>}
                </span>
              </div>
            </>
          )}

          {action.kind === "synonyms" && (
            <>
              <label className="field-label">Current synonyms</label>
              <div className="alias-chips">
                {tag.aliases.length === 0 && <span className="muted">No synonyms.</span>}
                {tag.aliases.map(a => (
                  <span key={a} className="alias-chip">
                    <code>{a}</code>
                    <button><Icon name="x" size={10} /></button>
                  </span>
                ))}
              </div>
              <label className="field-label">Add synonym</label>
              <input className="text-input" value={newSyn} onChange={e => setNewSyn(e.target.value)} placeholder="e.g. recovery-manager" />
              <div className="impact-box">
                <Icon name="info-circle" size={13} color="var(--twin-accent)" />
                <span>Synonyms are matched by query rewriting at the gateway. They do not duplicate the index; the canonical tag is preserved on chunks.</span>
              </div>
            </>
          )}

          {action.kind === "deprecate" && (
            <>
              <div className="impact-box warning">
                <Icon name="alert-triangle" size={13} color="var(--twin-amber-vivid)" />
                <span>
                  Deprecating <code>{tag.tag}</code> excludes its {tag.sources_count} docs from default retrieval. Existing tags on chunks are preserved; queries need <code>include_deprecated: true</code> to surface them.
                </span>
              </div>
              <label className="field-label">Reason (optional)</label>
              <textarea className="text-input" rows="3" placeholder="e.g. Superseded by iso20022 — see HLA §4.2" />
            </>
          )}

          {action.kind === "delete" && (
            <>
              <div className="impact-box danger">
                <Icon name="alert-triangle" size={13} color="var(--twin-red-vivid)" />
                <span>
                  <b>{tag.tag}</b> is used on <b>{tag.sources_count} docs</b> ({tag.chunks_count.toLocaleString()} chunks). Deletion cannot be undone — choose a migration strategy:
                </span>
              </div>
              <div className="strategy-radios">
                <label className={"strategy " + (migrateStrategy === "migrate" ? "is-active" : "")}>
                  <input type="radio" checked={migrateStrategy === "migrate"} onChange={() => setMigrateStrategy("migrate")} />
                  <div>
                    <div className="strategy-h">Migrate to another tag</div>
                    <div className="strategy-sub">Re-tag all {tag.sources_count} docs with a replacement.</div>
                    {migrateStrategy === "migrate" && (
                      <select className="text-input mt8" value={migrateTo} onChange={e => setMigrateTo(e.target.value)}>
                        <option value="">— select replacement —</option>
                        {eligible.map(x => <option key={x.tag} value={x.tag}>{x.tag} ({x.sources_count} docs)</option>)}
                      </select>
                    )}
                  </div>
                </label>
                <label className={"strategy " + (migrateStrategy === "untag" ? "is-active" : "")}>
                  <input type="radio" checked={migrateStrategy === "untag"} onChange={() => setMigrateStrategy("untag")} />
                  <div>
                    <div className="strategy-h">Untag and delete</div>
                    <div className="strategy-sub">Docs lose the tag and become untagged on this axis.</div>
                  </div>
                </label>
              </div>
            </>
          )}

          {action.kind === "reject" && (
            <>
              <label className="field-label">Reason</label>
              <textarea className="text-input" rows="3" value={reason} onChange={e => setReason(e.target.value)} placeholder="The author of the request will receive this message. Be specific." />
              <div className="impact-box">
                <Icon name="info-circle" size={13} color="var(--twin-accent)" />
                <span>A <code>tag.rejected</code> event is emitted to Activity with this reason. The requester is notified by email.</span>
              </div>
            </>
          )}

          {action.kind === "request" && (
            <>
              <label className="field-label">Proposed name <span className="hint">lowercase, no spaces</span></label>
              <input className="text-input" value={name} onChange={e => setName(e.target.value.toLowerCase().replace(/\s+/g, "-"))} placeholder="e.g. argocd" />
              <label className="field-label">Definition <span className="hint">200 chars max</span></label>
              <textarea className="text-input" rows="3" maxLength="200" placeholder="What should this tag mean? When should it be applied?" />
              <label className="field-label">Domain</label>
              <select className="text-input">
                {window.MOCK_TAG_CATEGORIES.map(c => <option key={c.id} value={c.id}>{c.label}</option>)}
                <option value="other">Other (specify in justification)</option>
              </select>
              <label className="field-label">Synonyms <span className="hint">optional</span></label>
              <input className="text-input" placeholder="comma-separated, e.g. recovery-manager, backup-tool" />
              <label className="field-label">Justification</label>
              <textarea className="text-input" rows="3" placeholder="Why is the existing taxonomy insufficient? Cite an example use." />
              <div className="impact-box">
                <Icon name="info-circle" size={13} color="var(--twin-accent)" />
                <span>Visibility is auto-set to <code>private</code> (inherited from workspace). Requests reach a steward within 2 business days. An accepted tag enters as <b>active</b>.</span>
              </div>
            </>
          )}
        </div>

        <div className="modal-footer">
          <button className="ghost-btn" onClick={onClose}>Cancel</button>
          <button
            className={"primary-btn " + (action.kind === "delete" || action.kind === "reject" ? "danger" : "")}
            onClick={commit}
          >
            {action.kind === "edit"          && "Save"}
            {action.kind === "suggest"       && "Submit suggestion"}
            {action.kind === "synonyms"      && "Save synonyms"}
            {action.kind === "deprecate"     && "Deprecate"}
            {action.kind === "delete"        && (migrateStrategy === "migrate" ? "Migrate and delete" : "Untag and delete")}
            {action.kind === "reject"        && "Reject request"}
            {action.kind === "edit-approve"  && "Approve with edits"}
            {action.kind === "request"       && "Submit request"}
          </button>
        </div>
      </div>
    </div>
  );
}
