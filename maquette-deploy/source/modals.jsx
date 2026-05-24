// Add source + Retag modals + Toast
const { useState, useEffect, useMemo } = React;

window.AddSourceModal = function AddSourceModal({ open, onClose, onAddToast }) {
  const modalRef = React.useRef(null);
  window.useModalA11y && window.useModalA11y({ open, onClose, ref: modalRef });
  const [files, setFiles] = useState([
    { name: "oracle-config-guide.pdf", size: 2.3, state: "uploaded" },
    { name: "vmware-runbook-2026.pdf", size: 8.7, state: "uploading", progress: 62, uploaded: 5.4 },
    { name: "huge-archive.zip", size: 68, state: "error", error: "Exceeds 50 MB · unsupported type" }
  ]);
  const [urls, setUrls] = useState([
    { url: "confluence.bnp/cib/runbooks", type: "confluence" },
    { url: "sharepoint.bnp/cib/incidents", type: "sharepoint" }
  ]);
  const [urlInput, setUrlInput] = useState("");
  const [tags, setTags] = useState([]);
  const [tagInput, setTagInput] = useState("");
  const [drag, setDrag] = useState(false);
  const [showTooltip, setShowTooltip] = useState(false);

  // Animate the uploading file's progress
  useEffect(() => {
    if (!open) return;
    const tick = setInterval(() => {
      setFiles(fs => fs.map(f => {
        if (f.state !== "uploading") return f;
        const np = Math.min(100, (f.progress || 0) + 4);
        if (np >= 100) return { ...f, state: "uploaded", progress: 100 };
        return { ...f, progress: np, uploaded: (np / 100 * f.size) };
      }));
    }, 320);
    return () => clearInterval(tick);
  }, [open]);

  if (!open) return null;

  const detect = (url) => {
    if (/confluence/i.test(url)) return "confluence";
    if (/sharepoint/i.test(url)) return "sharepoint";
    return "url";
  };
  const addUrls = (val) => {
    const parts = val.split(/[\s,]+/).map(s => s.trim()).filter(Boolean);
    setUrls([...urls, ...parts.map(u => ({ url: u, type: detect(u) }))]);
    setUrlInput("");
  };
  const removeUrl = (u) => setUrls(urls.filter(x => x.url !== u));
  const removeFile = (n) => setFiles(files.filter(f => f.name !== n));
  const addTag = (t) => { if (t && !tags.includes(t)) setTags([...tags, t]); setTagInput(""); };
  const removeTag = (t) => setTags(tags.filter(x => x !== t));

  const tagSugg = window.MOCK_THESAURUS
    .filter(t => !tags.includes(t.tag))
    .filter(t => !tagInput || t.tag.includes(tagInput.toLowerCase()))
    .slice(0, 4);

  const ready = files.filter(f => f.state === "uploaded").length + urls.length;
  const uploading = files.filter(f => f.state === "uploading").length;
  const errors = files.filter(f => f.state === "error").length;

  const submit = () => {
    if (ready === 0) return;
    onClose();
    window.twinCompleteTask && window.twinCompleteTask("addSource");
    // Applying tags at upload time also counts as the "Apply your first tag" task.
    if (tags.length > 0) window.twinCompleteTask && window.twinCompleteTask("tag");
    onAddToast({
      kind: "propagating",
      title: `Adding ${ready} sources…`,
      sub: `1 of ${ready} queued for ingestion`,
      autoDone: { title: `Sources queued for ingestion`, sub: `${ready} added · ${tags.length ? "tags: " + tags.join(", ") : "no initial tags"}`, undo: false }
    });
  };

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div
        ref={modalRef}
        className="modal"
        style={{ width: 480 }}
        role="dialog"
        aria-modal="true"
        aria-labelledby="addsource-title"
        tabIndex={-1}
        onClick={e => e.stopPropagation()}
      >
        <div className="modal-header">
          <div>
            <h2 id="addsource-title">Add source</h2>
          </div>
          <button className="icon-btn" onClick={onClose} aria-label="Close dialog"><Icon name="x" size={18} /></button>
        </div>

        <div className="modal-body">
          <div
            className={drag ? "dropzone drag" : "dropzone"}
            onDragOver={e => { e.preventDefault(); setDrag(true); }}
            onDragLeave={() => setDrag(false)}
            onDrop={e => {
              e.preventDefault(); setDrag(false);
              const dropped = Array.from(e.dataTransfer.files || []).map(f => ({
                name: f.name, size: +(f.size / (1024 * 1024)).toFixed(1), state: "uploading", progress: 0, uploaded: 0
              }));
              setFiles([...files, ...dropped]);
            }}
          >
            <Icon name="cloud-upload" size={28} color="var(--color-text-secondary)" />
            <div className="title">Drop files here or click to browse</div>
            <div className="sub">
              PDF, DOCX, MD, TXT and 35+ formats
              <span
                className="info"
                onMouseEnter={() => setShowTooltip(true)}
                onMouseLeave={() => setShowTooltip(false)}
                style={{ position: "relative" }}
              >
                <Icon name="info-circle" size={13} />
                {showTooltip && (
                  <div className="tooltip" style={{ left: "50%", transform: "translateX(-50%)", top: "calc(100% + 8px)", textAlign: "left" }}>
                    {window.MOCK_FORMAT_CATEGORIES.map(c => (
                      <div key={c.cat} className={c.future ? "tooltip-row future" : "tooltip-row"}>
                        <span className="cat">{c.cat}</span>
                        <span className="fmts">{c.fmts}</span>
                      </div>
                    ))}
                  </div>
                )}
              </span>
              <span style={{ color: "var(--color-text-tertiary)" }}>· max 50 MB per file</span>
            </div>
          </div>

          {files.length > 0 && (
            <div>
              <div className="section-label">
                <span>Files <span style={{ color: "var(--color-text-tertiary)", fontWeight: 400, fontSize: 11 }}>({files.length} added)</span></span>
              </div>
              <div className="file-list" style={{ marginTop: 6 }}>
                {files.map(f => (
                  <div key={f.name} className={f.state === "error" ? "file-row error" : "file-row"}>
                    <Icon name="file-text" size={15} className="file-icon" />
                    <div className="info">
                      <div className="row1">
                        <span className="name">{f.name}</span>
                        <span className="size">{f.size} MB</span>
                      </div>
                      {f.state === "uploading" && (
                        <div className="row2">
                          <div className="bar"><div className="bar-fill" style={{ width: `${f.progress}%` }} /></div>
                          <span style={{ fontFamily: "var(--font-mono)" }}>{f.progress}% · {f.uploaded.toFixed(1)} / {f.size} MB</span>
                        </div>
                      )}
                      {f.state === "error" && (
                        <div className="row2"><Icon name="alert-triangle" size={12} /> {f.error}</div>
                      )}
                    </div>
                    {f.state === "uploaded" && <Icon name="circle-check" size={16} className="ok" />}
                    <button className="x-btn" onClick={() => removeFile(f.name)}><Icon name="x" size={14} /></button>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="or-divider"><span>OR</span></div>

          <div>
            <div className="section-label"><span>Linked sources</span></div>
            <div className="linked-box" style={{ marginTop: 6 }}>
              {urls.map(u => (
                <span key={u.url} className="url-chip">
                  <span className="ico"><SourceIcon type={u.type} size={12} /></span>
                  <span>{u.url}</span>
                  <button className="x-btn" onClick={() => removeUrl(u.url)}><Icon name="x" size={10} /></button>
                </span>
              ))}
              <input
                className="url-input"
                value={urlInput}
                onChange={e => setUrlInput(e.target.value)}
                onKeyDown={e => { if (e.key === "Enter") addUrls(urlInput); }}
                placeholder={urls.length ? "Paste another URL — Enter to add" : "Paste Confluence or SharePoint URL — Enter to add"}
              />
            </div>
            <div className="helper-note" style={{ marginTop: 4 }}>
              Auto-detects Confluence, SharePoint, or generic URL · paste multiple separated by space or comma
            </div>
          </div>

          <div>
            <div className="section-label">
              <span>Apply tags to all</span>
              <span className="optional">optional</span>
            </div>
            <div className="chip-input" style={{ marginTop: 6, background: "var(--color-background-primary)" }}>
              {tags.map(t => <TagChip key={t} tag={t} removable onRemove={removeTag} />)}
              <input
                value={tagInput}
                onChange={e => setTagInput(e.target.value)}
                onKeyDown={e => { if (e.key === "Enter" && tagSugg[0]) addTag(tagSugg[0].tag); }}
                placeholder={tags.length ? "" : "Search tags from thesaurus…"}
                style={{ fontSize: 12 }}
              />
            </div>
            {tagInput && tagSugg.length > 0 && (
              <div className="autocomplete" style={{ marginTop: 4 }}>
                {tagSugg.map((s, i) => (
                  <div key={s.tag} className={`autocomplete-row${i === 0 ? " focus" : ""}`} onMouseDown={() => addTag(s.tag)}>
                    <div className="row1"><span style={{ fontSize: 12 }}>{s.tag}</span><span className="badge">{s.category}</span></div>
                    <div className="def">{s.def}</div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        <div className="modal-footer">
          <div className="status">
            {uploading > 0 && `Uploading ${uploading} of ${files.length} files…`}
            {uploading === 0 && errors > 0 && `${errors} error${errors > 1 ? "s" : ""} · ${ready} ready`}
            {uploading === 0 && errors === 0 && ready > 0 && `${ready} ready to ingest`}
          </div>
          <div className="actions">
            <button className="btn" onClick={onClose}>Cancel</button>
            <button className="btn primary" disabled={ready === 0} onClick={submit}>
              Add {ready} source{ready === 1 ? "" : "s"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};


window.RetagModal = function RetagModal({ open, doc, docs, onClose, onAddToast }) {
  const modalRef = React.useRef(null);
  const bulk = Array.isArray(docs) && docs.length > 0;
  const targets = bulk ? docs : (doc ? [doc] : []);
  const isReady = open && targets.length > 0;
  window.useModalA11y && window.useModalA11y({ open: isReady, onClose, ref: modalRef });
  // For bulk: shared tags = present on all; partial tags = present on some.
  const sharedTags = bulk
    ? targets.reduce((acc, d, i) => i === 0 ? [...d.tags] : acc.filter(t => d.tags.includes(t)), [])
    : (targets[0] ? targets[0].tags : []);
  const partialTags = bulk
    ? Array.from(new Set(targets.flatMap(d => d.tags))).filter(t => !sharedTags.includes(t))
    : [];
  const [current, setCurrent] = useState([]);
  const [pendingAdd, setPendingAdd] = useState([]);
  const [pendingRemove, setPendingRemove] = useState([]);
  const [input, setInput] = useState("");
  const [focusIdx, setFocusIdx] = useState(0);

  useEffect(() => {
    if (isReady) {
      setCurrent(sharedTags);
      setPendingAdd([]);
      setPendingRemove([]);
      setInput("");
    }
    // eslint-disable-next-line
  }, [doc, docs && docs.map(d => d.id).join(",")]);

  const sugg = useMemo(() => {
    const all = window.MOCK_THESAURUS.filter(t => !current.includes(t.tag) && !pendingAdd.includes(t.tag));
    if (!input) return all.slice(0, 4);
    const v = input.toLowerCase();
    return all.filter(t => t.tag.includes(v) || t.def.toLowerCase().includes(v)).slice(0, 5);
  }, [input, current, pendingAdd]);

  if (!isReady) return null;
  const primary = targets[0];

  const addTag = (t) => { setPendingAdd([...pendingAdd, t]); setInput(""); };
  const undoAdd = (t) => setPendingAdd(pendingAdd.filter(x => x !== t));
  const removeCurrent = (t) => {
    if (pendingRemove.includes(t)) setPendingRemove(pendingRemove.filter(x => x !== t));
    else setPendingRemove([...pendingRemove, t]);
  };

  const isRemoving = pendingRemove.length > 0 && pendingAdd.length === 0;
  // Mock preview impact numbers — scale with selection size for bulk.
  const totalChunks = targets.reduce((s, d) => s + (d.chunks || 0), 0);
  const previewChunks = bulk
    ? Math.round(totalChunks * (pendingAdd.length > 0 ? 1 : 0.4)) || (targets.length * 100)
    : (pendingAdd.length > 0 ? 418 : (pendingRemove.length > 0 ? 132 : 0));
  const previewDocs = bulk ? targets.length : (pendingAdd.length > 0 ? 3 : 2);

  const totalChanges = pendingAdd.length + pendingRemove.length;
  const applyLabel = totalChanges === 0
    ? "Apply tag"
    : isRemoving
      ? `Remove ${pendingRemove.length} tag${pendingRemove.length > 1 ? "s" : ""}`
      : totalChanges === 1 ? "Apply tag" : `Apply ${totalChanges} changes`;

  const submit = () => {
    if (totalChanges === 0) return;
    window.twinCompleteTask && window.twinCompleteTask("tag");
    const sample = pendingAdd[0] || pendingRemove[0];
    const action = pendingAdd.length > 0 ? "applied" : "removed";
    const doneSub = bulk
      ? `${targets.length} sources · ${previewChunks.toLocaleString()} chunks updated`
      : primary.source;
    onClose();
    onAddToast({
      kind: "propagating",
      title: bulk ? `Propagating across ${targets.length} sources` : `Propagating tag`,
      tagname: sample,
      sub: `${previewChunks.toLocaleString()} chunks · ~${Math.max(2, Math.round(targets.length * 0.8))} seconds`,
      autoDone: {
        title: `Tag`,
        tagname: sample,
        titleSuffix: bulk ? `${action} to ${targets.length} sources` : action,
        sub: doneSub,
        undo: true,
        undoDocId: primary.id
      }
    });
  };

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div
        ref={modalRef}
        className="modal"
        style={{ width: 500 }}
        role="dialog"
        aria-modal="true"
        aria-labelledby="retag-title"
        tabIndex={-1}
        onClick={e => e.stopPropagation()}
      >
        <div className="modal-header">
          <div style={{ flex: 1 }}>
            <h2 id="retag-title">{bulk ? `Retag ${targets.length} sources` : "Retag document"}</h2>
            <div className="ctx">
              {bulk ? (
                <>
                  <Icon name="tags" size={13} />
                  <span style={{ fontSize: 12, color: "var(--color-text-secondary)" }}>
                    Bulk · changes propagate to every selected source
                  </span>
                  <span className="sep">·</span>
                  <span className="vis-chip">
                    <Icon name="lock" size={11} />
                    {targets[0].workspace} · {totalChunks.toLocaleString()} chunks total
                  </span>
                </>
              ) : (
                <>
                  <SourceIcon type={primary.type} size={13} />
                  <span style={{ fontFamily: primary.type !== "file" ? "var(--font-mono)" : "inherit", overflow: "hidden", textOverflow: "ellipsis", maxWidth: 320, whiteSpace: "nowrap" }}>{primary.source}</span>
                  <span className="sep">·</span>
                  <span className="vis-chip">
                    <Icon name={primary.visibility === "private" ? "lock" : "world"} size={11} />
                    {primary.workspace} · {primary.visibility}
                  </span>
                </>
              )}
            </div>
            {bulk && (
              <div className="bulk-target-strip">
                {targets.slice(0, 4).map(d => (
                  <span key={d.id} className="bulk-target-chip" title={d.source}>
                    <SourceIcon type={d.type} size={11} />
                    <span className={d.type !== "file" ? "mono" : ""}>{d.source}</span>
                  </span>
                ))}
                {targets.length > 4 && (
                  <span className="bulk-target-more">+{targets.length - 4} more</span>
                )}
              </div>
            )}
          </div>
          <button className="icon-btn" onClick={onClose} aria-label="Close dialog"><Icon name="x" size={18} /></button>
        </div>

        <div className="modal-body">
          {(current.length > 0 || pendingRemove.length > 0) && (
            <div>
              <div className="section-label"><span>{bulk ? "Tags on all selected" : "Currently applied"}</span></div>
              <div className="tag-chips" style={{ marginTop: 6 }}>
                {current.map(t => (
                  <span key={t} style={{ opacity: pendingRemove.includes(t) ? 0.45 : 1, textDecoration: pendingRemove.includes(t) ? "line-through" : "none" }}>
                    <TagChip tag={t} removable onRemove={removeCurrent} />
                  </span>
                ))}
                {current.length === 0 && <span className="muted" style={{ fontSize: 11 }}>None shared across all selected sources.</span>}
              </div>
            </div>
          )}
          {bulk && partialTags.length > 0 && (
            <div>
              <div className="section-label"><span>On some selected <span className="hint">— removing these only affects sources where present</span></span></div>
              <div className="tag-chips" style={{ marginTop: 6, opacity: 0.7 }}>
                {partialTags.map(t => <TagChip key={t} tag={t} />)}
              </div>
            </div>
          )}

          <div>
            <div className="section-label"><span>Add tags</span></div>
            {pendingAdd.length > 0 && (
              <div className="tag-chips" style={{ marginTop: 6, marginBottom: 6 }}>
                {pendingAdd.map(t => (
                  <span key={t}><TagChip tag={t} removable onRemove={undoAdd} /></span>
                ))}
              </div>
            )}
            <input
              type="text"
              value={input}
              onChange={e => { setInput(e.target.value); setFocusIdx(0); }}
              onKeyDown={e => {
                if (e.key === "ArrowDown") { e.preventDefault(); setFocusIdx(Math.min(sugg.length - 1, focusIdx + 1)); }
                else if (e.key === "ArrowUp") { e.preventDefault(); setFocusIdx(Math.max(0, focusIdx - 1)); }
                else if (e.key === "Enter" && sugg[focusIdx]) addTag(sugg[focusIdx].tag);
              }}
              placeholder="Start typing — autocomplete from thesaurus"
              style={{
                width: "100%",
                padding: "8px 10px",
                fontSize: 13,
                fontFamily: "var(--font-mono)",
                border: "0.5px solid var(--color-border-tertiary)",
                borderRadius: "var(--border-radius-md)",
                background: "var(--color-background-primary)",
                marginTop: 6
              }}
            />
            <div className="autocomplete">
              <div className="autocomplete-header">
                {sugg.length > 0 ? `${sugg.length} match${sugg.length > 1 ? "es" : ""} in thesaurus` : "No matches"}
              </div>
              {sugg.map((s, i) => (
                <div
                  key={s.tag}
                  className={`autocomplete-row${i === focusIdx ? " focus" : ""}`}
                  onMouseEnter={() => setFocusIdx(i)}
                  onClick={() => addTag(s.tag)}
                >
                  <div className="row1">
                    <span>{s.tag}</span>
                    <span className="badge">{s.category}</span>
                  </div>
                  <div className="def">{s.def}</div>
                </div>
              ))}
              <div className="autocomplete-footer">
                <Icon name="info-circle" size={12} /> No match? Request a new tag in the Tags tab.
              </div>
            </div>
          </div>

          {totalChanges > 0 && (
            <div className={isRemoving ? "preview-impact removing" : "preview-impact"}>
              <div className="head"><Icon name="eye" size={14} /> Preview impact</div>
              <div className="stats">
                <div className="stat">
                  <div className="num">{previewChunks}</div>
                  <div className="lbl">chunks</div>
                </div>
                <div className="stat">
                  <div className="num">{previewDocs}</div>
                  <div className="lbl">docs share these</div>
                </div>
              </div>
              <div className="foot">
                <Icon name="arrow-right" size={12} />
                {isRemoving ? (
                  <>
                    Removing <span className="mini-chip">{pendingRemove[0]}</span> will untag {previewChunks} chunks · 87 of which will no longer be covered by any tag
                  </>
                ) : (
                  <>
                    Adding <span className="mini-chip">{pendingAdd[0]}</span> · 1 of the {previewDocs} docs is untagged
                  </>
                )}
              </div>
            </div>
          )}
        </div>

        <div className="modal-footer">
          <div className="status"></div>
          <div className="actions">
            <button className="btn" onClick={onClose}>Cancel</button>
            <button className="btn primary" disabled={totalChanges === 0} onClick={submit}>{applyLabel}</button>
          </div>
        </div>
      </div>
    </div>
  );
};


// Toast viewport — stack of toasts with auto propagating → done transition + Undo
// A11y: sr-only live regions announce new toasts once; viewport itself is aria-live=off
// to avoid double-announcement on propagating→done text mutation.
const TOAST_MAX_VISIBLE = 3;

window.ToastViewport = function ToastViewport({ toasts, onUndo, onDismiss }) {
  const visible = toasts.slice(-TOAST_MAX_VISIBLE);
  const hidden = toasts.length - visible.length;

  const announcedRef = React.useRef(new Set());
  const [politeMsg, setPoliteMsg] = React.useState("");
  const [assertiveMsg, setAssertiveMsg] = React.useState("");

  React.useEffect(() => {
    toasts.forEach(t => {
      const key = `${t.id}:${t.kind}`;
      if (announcedRef.current.has(key)) return;
      announcedRef.current.add(key);
      const parts = [t.title, t.tagname, t.titleSuffix].filter(Boolean).join(" ");
      const msg = t.sub ? `${parts}. ${t.sub}` : parts;
      if (t.kind === "error") setAssertiveMsg(msg + " · " + Date.now()); // suffix forces re-announce on identical text
      else setPoliteMsg(msg + " · " + Date.now());
    });
    // Garbage-collect announced IDs no longer in toast list
    const live = new Set(toasts.flatMap(t => [`${t.id}:propagating`, `${t.id}:done`, `${t.id}:error`]));
    announcedRef.current.forEach(k => { if (!live.has(k)) announcedRef.current.delete(k); });
  }, [toasts]);

  return (
    <React.Fragment>
      <div className="sr-only" role="status" aria-live="polite" aria-atomic="true">
        {politeMsg.replace(/ · \d+$/, "")}
      </div>
      <div className="sr-only" role="alert" aria-live="assertive" aria-atomic="true">
        {assertiveMsg.replace(/ · \d+$/, "")}
      </div>

      <div className="toast-viewport" aria-label="Notifications">
        {hidden > 0 && (
          <button
            className="toast-stack-more"
            onClick={() => toasts.slice(0, hidden).forEach(onDismiss)}
            aria-label={`${hidden} older notifications, click to dismiss`}
            title="Dismiss older"
          >
            +{hidden} more
          </button>
        )}
        {visible.map(t => <ToastCard key={t.id} toast={t} onUndo={onUndo} onDismiss={onDismiss} />)}
      </div>
    </React.Fragment>
  );
};

function ToastCard({ toast, onUndo, onDismiss }) {
  const kind = toast.kind;
  return (
    <div className={kind === "error" ? "toast error" : "toast"}>
      <span className={`icon ${kind}`}>
        {kind === "propagating" && <Icon name="loader-2" size={18} />}
        {kind === "done" && <Icon name="circle-check" size={18} />}
        {kind === "error" && <Icon name="alert-triangle" size={18} />}
      </span>
      <div className="body">
        <div className="title">
          {toast.title}
          {toast.tagname && <span className="tagname">{toast.tagname}</span>}
          {toast.titleSuffix && <span>{toast.titleSuffix}</span>}
        </div>
        {toast.sub && <div className="sub">{toast.sub}</div>}
      </div>
      {kind === "done" && toast.undo && (
        <button className="undo" onClick={() => onUndo(toast)}>Undo</button>
      )}
      {/* Explicit dismiss × on every toast — user feedback 2026-05-21
          (was only present for kind=error). Propagating toasts now also
          dismissable in case the simulated op never resolves. */}
      <button
        className="toast-dismiss"
        onClick={() => onDismiss(toast)}
        aria-label="Dismiss notification"
        title="Dismiss"
      >
        <Icon name="x" size={11} />
      </button>
      {kind === "done" && toast.undo && <span className="undo-progress" />}
    </div>
  );
}
