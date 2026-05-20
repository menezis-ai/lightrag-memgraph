// Settings tab — profile, tokens, workspace config, providers, members, danger zone.
// RBAC: most read-only for palier ≤2; palier 3 (steward/admin) can edit providers + members.

const { useState: _useStateS, useMemo: _useMemoS } = React;

// UI-facing label for the palier ladder. `palier` is the back-end / API
// term (see README §5); the UI talks Reader / Contributor / Steward so the
// BNP audience isn't forced to learn an internal vocabulary.
const PALIER_ROLE_LABEL = { 1: "Reader", 2: "Contributor", 3: "Steward" };
const _roleLabel = (p) => PALIER_ROLE_LABEL[p] || `Palier ${p}`;

const SETTINGS_SECTIONS = [
  { id: "profile",     label: "Profile",     icon: "circle-dot",     palier: 1 },
  { id: "tokens",      label: "API tokens",  icon: "lock",           palier: 2 },
  { id: "workspace",   label: "Workspace",   icon: "folder",         palier: 1 },
  { id: "providers",   label: "Providers",   icon: "settings",       palier: 1 },
  { id: "members",     label: "Members",     icon: "tags",           palier: 1 },
  { id: "danger",      label: "Danger zone", icon: "alert-triangle", palier: 1 }
];

window.SettingsTab = function SettingsTab({ workspace, kbName, onPushToast }) {
  const user = window.MOCK_CURRENT_USER;
  const canEdit = user.palier >= 3;
  const visible = SETTINGS_SECTIONS.filter(s => user.palier >= s.palier);
  const [sec, setSec] = window.useUrlParam("sec", "profile", {
    validate: v => visible.some(s => s.id === v)
  });
  const active = visible.find(s => s.id === sec) || visible[0];

  return (
    <div className="settings">
      <aside className="settings-rail">
        <div className="settings-rail-h">Settings</div>
        <ul className="settings-rail-list">
          {visible.map(s => (
            <li key={s.id}>
              <button
                className={"settings-rail-item " + (active.id === s.id ? "is-active" : "")}
                onClick={() => setSec(s.id)}
              >
                <Icon name={s.icon} size={13} />
                <span>{s.label}</span>
              </button>
            </li>
          ))}
        </ul>
        <div className="settings-rail-foot">
          <div className="settings-rail-user">
            <div className="settings-rail-avatar" aria-hidden="true">{user.name.split(" ").map(w => w[0]).join("").slice(0, 2)}</div>
            <div className="settings-rail-user-meta">
              <div className="settings-rail-user-name">{user.name}</div>
              <div className="settings-rail-user-palier">{_roleLabel(user.palier)}</div>
            </div>
          </div>
        </div>
      </aside>

      <main className="settings-main">
        {sec === "profile"   && <ProfileSection user={user} onPushToast={onPushToast} />}
        {sec === "tokens"    && <TokensSection canEdit={user.palier >= 2} onPushToast={onPushToast} />}
        {sec === "workspace" && <WorkspaceSection workspace={workspace} kbName={kbName} canEdit={canEdit} onPushToast={onPushToast} />}
        {sec === "providers" && <ProvidersSection canEdit={canEdit} onPushToast={onPushToast} />}
        {sec === "members"   && <MembersSection canEdit={canEdit} onPushToast={onPushToast} />}
        {sec === "danger"    && <DangerSection canEdit={canEdit} workspace={workspace} onPushToast={onPushToast} />}
      </main>
    </div>
  );
};

// ─── Section: Profile ────────────────────────────────────────────────────
function ProfileSection({ user, onPushToast }) {
  return (
    <SettingsBody
      title="Profile"
      sub="Account info inherited from your Keycloak session. Update name/email in the corporate IDP."
    >
      <div className="settings-card">
        <div className="profile-head">
          <div className="profile-avatar">{user.name.split(" ").map(w => w[0]).join("").slice(0, 2)}</div>
          <div>
            <h3 className="profile-name">{user.name}</h3>
            <div className="profile-email">{user.email}</div>
            <div className="profile-badges">
              <span className="palier-pill" title="Determines which actions you can perform.">
                {_roleLabel(user.palier)}
              </span>
            </div>
          </div>
        </div>
        <dl className="settings-kv">
          <dt>Identity provider</dt><dd className="mono-meta">{user.sso}</dd>
          <dt>Session expires</dt><dd className="mono-meta">{user.session_expires}</dd>
        </dl>
      </div>

      <div className="settings-card">
        <div className="settings-card-h">
          <h3>Scopes</h3>
          <span className="settings-card-sub">Permissions attached to your bearer token at gateway level.</span>
        </div>
        <div className="scope-chips">
          {user.scopes.map(s => (
            <code key={s} className="scope-chip">{s}</code>
          ))}
        </div>
      </div>

      <div className="settings-card">
        <div className="settings-card-h">
          <h3>Session</h3>
        </div>
        <div className="settings-row">
          <button className="ghost-btn" onClick={() => onPushToast({ id: "logout-" + Date.now(), title: "Signed out", sub: "Bearer revoked at gateway · redirect to IDP", undo: false })}>
            <Icon name="arrow-right" size={12} /> Sign out
          </button>
          <span className="muted-sm">Local cache (threads, tweaks) is preserved in this browser.</span>
        </div>
      </div>

      <div className="settings-card">
        <div className="settings-card-h">
          <h3>Tutorial</h3>
          <span className="settings-card-sub">Replay the welcome tour and the 6-step checklist.</span>
        </div>
        <div className="settings-row">
          <button
            className="ghost-btn"
            onClick={() => {
              window.applyOnboardingPreset && window.applyOnboardingPreset("welcome");
              onPushToast && onPushToast({ id: "tut-" + Date.now(), title: "Tutorial restarted", sub: "Welcome modal will appear · 0 of 6 steps complete", undo: false });
            }}
          >
            <Icon name="refresh" size={12} /> Restart tutorial
          </button>
          <span className="muted-sm">Resets progress for this browser only — your data is untouched.</span>
        </div>
      </div>
    </SettingsBody>
  );
}

// ─── Section: API tokens ─────────────────────────────────────────────────
function TokensSection({ canEdit, onPushToast }) {
  const [tokens, setTokens] = _useStateS(window.MOCK_API_TOKENS);
  const [newOpen, setNewOpen] = _useStateS(false);
  const [newName, setNewName] = _useStateS("");
  const [newScopes, setNewScopes] = _useStateS(new Set(["read:documents", "read:query"]));
  const [justCreated, setJustCreated] = _useStateS(null);

  const ALL_SCOPES = ["read:documents", "write:documents", "read:query", "read:activity", "admin:tags"];

  const toggleScope = (s) => {
    const next = new Set(newScopes);
    if (next.has(s)) next.delete(s); else next.add(s);
    setNewScopes(next);
  };

  const generate = () => {
    if (!newName.trim()) return;
    const id = "tok_" + Math.random().toString(16).slice(2, 6);
    const secret = "tw_pat_" + Math.random().toString(36).slice(2, 22);
    const tok = {
      id, name: newName.trim(), scopes: [...newScopes],
      last_used: "—", created: new Date().toISOString(), prefix: secret.slice(0, 10)
    };
    setTokens([tok, ...tokens]);
    setJustCreated({ ...tok, secret });
    setNewName(""); setNewOpen(false);
    onPushToast && onPushToast({ id: "tok-" + Date.now(), title: "Token", titleSuffix: "created", sub: `${tok.name} · ${tok.scopes.length} scope${tok.scopes.length > 1 ? "s" : ""}`, undo: false });
  };

  const revoke = (id) => {
    const t = tokens.find(x => x.id === id);
    setTokens(tokens.filter(x => x.id !== id));
    onPushToast && onPushToast({ id: "tok-rev-" + Date.now(), title: "Token", titleSuffix: "revoked", sub: t ? t.name : id, undo: false });
  };

  return (
    <SettingsBody
      title="API tokens"
      sub="Long-lived personal access tokens. Use these from CI, scripts, and integrations. OIDC session bearer for browsers is managed by Keycloak."
    >
      {justCreated && (
        <div className="token-reveal">
          <div className="token-reveal-h">
            <Icon name="circle-check" size={14} color="var(--twin-green-700, #2F7A40)" />
            <b>Token created · copy now</b>
          </div>
          <p className="token-reveal-warn">
            This secret is shown <b>only once</b>. Store it in your secret manager before navigating away.
          </p>
          <div className="token-reveal-secret">
            <code>{justCreated.secret}</code>
            <button
              className="ghost-btn small"
              onClick={() => {
                navigator.clipboard && navigator.clipboard.writeText(justCreated.secret);
                onPushToast && onPushToast({ id: "tok-cp-" + Date.now(), title: "Copied", sub: "Token in clipboard · 15min until clearance", undo: false });
              }}
            ><Icon name="external-link" size={11} /> Copy</button>
          </div>
          <button className="link-btn small" onClick={() => setJustCreated(null)}>Dismiss</button>
        </div>
      )}

      <div className="settings-card">
        <div className="settings-card-h">
          <h3>Active tokens · {tokens.length}</h3>
          {canEdit ? (
            <button className="primary-btn small" onClick={() => setNewOpen(o => !o)}>
              <Icon name={newOpen ? "x" : "plus"} size={11} /> {newOpen ? "Cancel" : "Generate token"}
            </button>
          ) : (
            <span className="muted-sm">Contributor or Steward to create tokens</span>
          )}
        </div>

        {newOpen && (
          <div className="token-new">
            <label className="field-label">Label <span className="hint">— what's this token for?</span></label>
            <input
              type="text"
              className="text-input"
              autoFocus
              placeholder="e.g. ci · embedding-refresh"
              value={newName}
              onChange={e => setNewName(e.target.value)}
            />
            <label className="field-label" style={{ marginTop: 10 }}>Scopes</label>
            <div className="scope-chips selectable">
              {ALL_SCOPES.map(s => (
                <button
                  key={s}
                  className={"scope-chip-toggle " + (newScopes.has(s) ? "is-on" : "")}
                  onClick={() => toggleScope(s)}
                >{s}</button>
              ))}
            </div>
            <div className="settings-row" style={{ marginTop: 12, justifyContent: "flex-end" }}>
              <button className="primary-btn" onClick={generate} disabled={!newName.trim() || newScopes.size === 0}>
                Generate
              </button>
            </div>
          </div>
        )}

        <ul className="token-list">
          {tokens.length === 0 && <li className="muted-sm" style={{ padding: 14 }}>No tokens yet.</li>}
          {tokens.map(t => (
            <li key={t.id} className="token-row">
              <div className="token-row-main">
                <div className="token-name">{t.name}</div>
                <code className="token-prefix">{t.prefix}…</code>
                <div className="token-scopes">
                  {t.scopes.map(s => <code key={s} className="scope-chip tiny">{s}</code>)}
                </div>
              </div>
              <div className="token-row-meta">
                <span>last used <b>{t.last_used}</b></span>
                <span className="dot-sep">·</span>
                <span>created {t.created.slice(0, 10)}</span>
              </div>
              {canEdit && (
                <button className="ghost-btn small danger" onClick={() => revoke(t.id)}>
                  <Icon name="trash" size={11} /> Revoke
                </button>
              )}
            </li>
          ))}
        </ul>
      </div>
    </SettingsBody>
  );
}

// ─── Section: Workspace ──────────────────────────────────────────────────
function WorkspaceSection({ workspace, kbName, canEdit, onPushToast }) {
  const retention = window.MOCK_RETENTION || [];
  const [defaultTags, setDefaultTags] = _useStateS(["cib", "production"]);
  const [tagInput, setTagInput] = _useStateS("");

  const addDefault = () => {
    const v = tagInput.trim().toLowerCase();
    if (!v || defaultTags.includes(v)) return;
    setDefaultTags([...defaultTags, v]);
    setTagInput("");
    onPushToast && onPushToast({ id: "wsd-" + Date.now(), title: "Default tag", tagname: v, titleSuffix: "added", sub: "Applied to new ingestions", undo: true });
  };
  const removeDefault = (t) => {
    setDefaultTags(defaultTags.filter(x => x !== t));
  };

  return (
    <SettingsBody
      title="Workspace"
      sub={`Configuration for workspace ${workspace}. Some values are set at Helm install time and cannot be changed at runtime.`}
    >
      <div className="settings-card">
        <div className="settings-card-h">
          <h3>Identity</h3>
          <span className="env-pill"><Icon name="lock" size={10} /> env-controlled</span>
        </div>
        <dl className="settings-kv">
          <dt>Workspace ID</dt><dd className="mono-meta">{workspace}</dd>
          <dt>Display name</dt><dd className="mono-meta">{kbName}</dd>
          <dt>Visibility</dt><dd className="mono-meta">private <span className="muted-sm">(TWIN_INSTANCE_VISIBILITY)</span></dd>
          <dt>Region</dt><dd className="mono-meta">eu-west-3 · dc-paris</dd>
        </dl>
      </div>

      <div className="settings-card">
        <div className="settings-card-h">
          <h3>Default tags on ingestion</h3>
          <span className="settings-card-sub">Applied automatically to every new source unless overridden in Add source modal.</span>
        </div>
        <div className="tag-chips">
          {defaultTags.map(t => <TagChip key={t} tag={t} removable={canEdit} onRemove={removeDefault} />)}
          {canEdit && (
            <span className="default-tag-add">
              <input
                value={tagInput}
                onChange={e => setTagInput(e.target.value.toLowerCase())}
                onKeyDown={e => { if (e.key === "Enter") addDefault(); }}
                placeholder="Add tag…"
                style={{ fontFamily: "var(--font-mono)", fontSize: 11, padding: "3px 8px", border: "0.5px solid var(--color-border-tertiary)", borderRadius: 999, background: "var(--color-background-primary)", width: 120 }}
              />
            </span>
          )}
        </div>
        {!canEdit && <div className="muted-sm" style={{ marginTop: 8 }}>Steward only to edit.</div>}
      </div>

      <div className="settings-card">
        <div className="settings-card-h">
          <h3>Retention policy</h3>
          <span className="env-pill"><Icon name="lock" size={10} /> env-controlled</span>
        </div>
        <table className="retention-table">
          <thead><tr><th>Area</th><th>TTL</th><th>Note</th></tr></thead>
          <tbody>
            {retention.map(r => (
              <tr key={r.area}>
                <td>{r.area}</td>
                <td><code className="mono-meta">{r.ttl}</code></td>
                <td className="muted-sm">{r.note}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <div className="muted-sm" style={{ marginTop: 8 }}>
          Aligned with policy <code>twin-cib-retention-v2.1</code>. Override requires a Tier-1 governance ticket.
        </div>
      </div>
    </SettingsBody>
  );
}

// ─── Section: Providers ──────────────────────────────────────────────────
function ProvidersSection({ canEdit, onPushToast }) {
  const p = window.MOCK_PROVIDERS;
  return (
    <SettingsBody
      title="Providers"
      sub="LLM, embedder, and reranker configuration. Credentials live in the secret store and are referenced by alias."
    >
      <ProviderCard
        kind="LLM" data={p.llm} canEdit={canEdit}
        extra={[
          { label: "Rate limit", value: `${p.llm.rate_limit_rpm} rpm` },
          { label: "Monthly quota", value: `$${p.llm.monthly_quota_usd.toLocaleString()}` },
          { label: "Spent this month", value: `$${p.llm.monthly_spend_usd.toFixed(2)}`, bar: p.llm.monthly_spend_usd / p.llm.monthly_quota_usd }
        ]}
        onTest={() => onPushToast && onPushToast({ id: "p-test-" + Date.now(), kind: "propagating", title: "Testing LLM connection…", sub: "POST /v1/chat/completions · 3 tokens · sample probe", autoDone: { title: "LLM", titleSuffix: "responsive", sub: `${p.llm.provider} · ${p.llm.model} · 412ms`, undo: false } })}
      />
      <ProviderCard
        kind="Embedder" data={p.embedder} canEdit={canEdit}
        extra={[
          { label: "Vector dimensions", value: `${p.embedder.dims.toLocaleString()}` },
          { label: "Rate limit", value: `${p.embedder.rate_limit_rpm} rpm` }
        ]}
        onTest={() => onPushToast && onPushToast({ id: "p-emb-" + Date.now(), kind: "propagating", title: "Testing embedder…", sub: "Embed 'twin rag healthcheck' · 1 vector", autoDone: { title: "Embedder", titleSuffix: "responsive", sub: `${p.embedder.model} · 168ms · ${p.embedder.dims}d`, undo: false } })}
      />
      <ProviderCard
        kind="Reranker" data={p.reranker} canEdit={canEdit}
        extra={[
          { label: "Enabled", value: p.reranker.enabled ? "yes" : "no" }
        ]}
        onTest={() => onPushToast && onPushToast({ id: "p-rr-" + Date.now(), kind: "propagating", title: "Testing reranker…", sub: "10 candidates · single batch", autoDone: { title: "Reranker", titleSuffix: "responsive", sub: `${p.reranker.model} · 84ms`, undo: false } })}
      />
    </SettingsBody>
  );
}

function ProviderCard({ kind, data, extra, canEdit, onTest }) {
  return (
    <div className="settings-card provider-card">
      <div className="settings-card-h">
        <h3>{kind}</h3>
        <span className="provider-pill">{data.provider}</span>
      </div>
      <dl className="settings-kv">
        <dt>Model</dt><dd className="mono-meta">{data.model}</dd>
        <dt>Base URL</dt><dd className="mono-meta">{data.base_url}</dd>
        {data.key_ref && <><dt>API key</dt><dd className="mono-meta">{data.key_ref}</dd></>}
      </dl>
      <div className="provider-extras">
        {extra.map((e, i) => (
          <div key={i} className="provider-extra">
            <div className="provider-extra-label">{e.label}</div>
            <div className="provider-extra-value">{e.value}</div>
            {e.bar !== undefined && (
              <div className="provider-bar"><span style={{ width: `${Math.min(100, e.bar * 100)}%` }} /></div>
            )}
          </div>
        ))}
      </div>
      <div className="settings-row">
        <button className="ghost-btn" onClick={onTest}><Icon name="refresh" size={11} /> Test connection</button>
        {canEdit ? (
          <button className="ghost-btn">Configure</button>
        ) : (
          <span className="muted-sm">Steward only to edit</span>
        )}
      </div>
    </div>
  );
}

// ─── Section: Members ────────────────────────────────────────────────────
function MembersSection({ canEdit, onPushToast }) {
  const [members, setMembers] = _useStateS(window.MOCK_MEMBERS);
  const [inviteOpen, setInviteOpen] = _useStateS(false);
  const [inviteEmail, setInviteEmail] = _useStateS("");
  const [invitePalier, setInvitePalier] = _useStateS(1);

  const invite = () => {
    if (!inviteEmail.includes("@")) return;
    window.twinCompleteTask && window.twinCompleteTask("invite");
    const m = {
      email: inviteEmail.trim(),
      name: inviteEmail.split("@")[0].split(".").map(w => w[0].toUpperCase() + w.slice(1)).join(" "),
      palier: invitePalier, role: invitePalier === 1 ? "Reader" : invitePalier === 2 ? "Contributor" : "Steward",
      joined: new Date().toISOString().slice(0, 10), last_seen: "—", status: "invited"
    };
    setMembers([...members, m]);
    setInviteEmail(""); setInvitePalier(1); setInviteOpen(false);
    onPushToast && onPushToast({ id: "inv-" + Date.now(), title: "Invitation sent", sub: `${m.email} · ${_roleLabel(m.palier)}`, undo: true });
  };

  const setPalier = (email, palier) => {
    setMembers(members.map(m => m.email === email ? { ...m, palier, role: palier === 1 ? "Reader" : palier === 2 ? "Contributor" : "Steward" } : m));
    onPushToast && onPushToast({ id: "p-" + Date.now(), title: "Role updated", sub: `${email} → ${_roleLabel(palier)}`, undo: true });
  };

  const remove = (email) => {
    setMembers(members.filter(m => m.email !== email));
    onPushToast && onPushToast({ id: "rm-" + Date.now(), title: "Member removed", sub: email, undo: false });
  };

  const counts = { 3: members.filter(m => m.palier === 3).length, 2: members.filter(m => m.palier === 2).length, 1: members.filter(m => m.palier === 1).length };

  return (
    <SettingsBody
      title="Members"
      sub="Workspace access list. Role governs what each member can do across Tags, Documents, Activity, and Settings."
    >
      <div className="settings-card">
        <div className="settings-card-h">
          <h3>{members.length} members</h3>
          <span className="settings-card-sub">
            <b>{counts[3]}</b> stewards · <b>{counts[2]}</b> contributors · <b>{counts[1]}</b> readers
          </span>
          {canEdit && (
            <button className="primary-btn small" onClick={() => setInviteOpen(o => !o)}>
              <Icon name={inviteOpen ? "x" : "plus"} size={11} /> {inviteOpen ? "Cancel" : "Invite"}
            </button>
          )}
        </div>

        {inviteOpen && (
          <div className="invite-row">
            <input
              type="email"
              className="text-input"
              autoFocus
              placeholder="email@bnpparibas.com"
              value={inviteEmail}
              onChange={e => setInviteEmail(e.target.value)}
              style={{ flex: 1 }}
            />
            <select className="mini-select" value={invitePalier} onChange={e => setInvitePalier(parseInt(e.target.value))}>
              <option value="1">Reader</option>
              <option value="2">Contributor</option>
              <option value="3">Steward</option>
            </select>
            <button className="primary-btn" onClick={invite} disabled={!inviteEmail.includes("@")}>Send</button>
          </div>
        )}

        <table className="members-table">
          <thead>
            <tr>
              <th>Member</th>
              <th>Role</th>
              <th>Joined</th>
              <th>Last seen</th>
              <th>Status</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {members.map(m => (
              <tr key={m.email}>
                <td>
                  <div className="member-cell">
                    <div className="member-avatar">{m.name.split(" ").map(w => w[0]).join("").slice(0, 2)}</div>
                    <div>
                      <div className="member-name">{m.name}</div>
                      <div className="member-email mono-meta">{m.email}</div>
                    </div>
                  </div>
                </td>
                <td>
                  {canEdit && m.email !== window.MOCK_CURRENT_USER.email ? (
                    <select
                      className="mini-select"
                      value={m.palier}
                      onChange={e => setPalier(m.email, parseInt(e.target.value))}
                    >
                      <option value="1">Reader</option>
                      <option value="2">Contributor</option>
                      <option value="3">Steward</option>
                    </select>
                  ) : (
                    <span className="palier-pill small">{_roleLabel(m.palier)}</span>
                  )}
                </td>
                <td className="muted-sm">{m.joined}</td>
                <td className="muted-sm">{m.last_seen}</td>
                <td>
                  <span className={"member-status " + m.status}>
                    {m.status === "active" && <><span className="status-dot" /> active</>}
                    {m.status === "invited" && <>invited</>}
                  </span>
                </td>
                <td className="member-actions">
                  {canEdit && m.email !== window.MOCK_CURRENT_USER.email && (
                    <button className="ghost-btn small danger" onClick={() => remove(m.email)}>
                      <Icon name="trash" size={10} />
                    </button>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </SettingsBody>
  );
}

// ─── Section: Danger zone ────────────────────────────────────────────────
function DangerSection({ canEdit, workspace, onPushToast }) {
  const [leaveOpen, setLeaveOpen] = _useStateS(false);
  const [confirmText, setConfirmText] = _useStateS("");

  return (
    <SettingsBody
      title="Danger zone"
      sub="Destructive actions. None are reversible from the UI — recovery requires a support ticket and steward sign-off."
    >
      <div className="settings-card danger">
        <div className="settings-card-h">
          <h3>Leave workspace</h3>
        </div>
        <p className="settings-body-p">
          You'll lose access to <code className="mono-meta">{workspace}</code>. A steward can re-invite you later.
          Your contributions (tag suggestions, queries, uploads) stay attributed in the activity log.
        </p>
        <div className="settings-row">
          {!leaveOpen ? (
            <button className="ghost-btn danger" onClick={() => setLeaveOpen(true)}>
              <Icon name="arrow-right" size={12} /> Leave workspace
            </button>
          ) : (
            <>
              <input
                type="text"
                className="text-input"
                placeholder={`Type "${workspace}" to confirm`}
                value={confirmText}
                onChange={e => setConfirmText(e.target.value)}
                autoFocus
                style={{ maxWidth: 280 }}
              />
              <button
                className="primary-btn danger"
                disabled={confirmText !== workspace}
                onClick={() => {
                  setLeaveOpen(false); setConfirmText("");
                  onPushToast && onPushToast({ id: "leave-" + Date.now(), title: "Workspace left", sub: `You no longer have access to ${workspace}`, undo: false });
                }}
              >Confirm</button>
              <button className="ghost-btn" onClick={() => { setLeaveOpen(false); setConfirmText(""); }}>Cancel</button>
            </>
          )}
        </div>
      </div>

      <div className="settings-card danger">
        <div className="settings-card-h">
          <h3>Delete workspace</h3>
          {!canEdit && <span className="env-pill"><Icon name="lock" size={10} /> palier 3 only</span>}
        </div>
        <p className="settings-body-p">
          Permanently deletes <code className="mono-meta">{workspace}</code> with all sources, tags, chunks, threads, and activity.
          The Memgraph + vector indices are zeroed and the Helm release is left orphaned (cluster-ops manual cleanup).
        </p>
        <div className="settings-row">
          <button
            className="ghost-btn danger"
            disabled={!canEdit}
            onClick={() => onPushToast && onPushToast({ id: "del-ws-" + Date.now(), kind: "error", title: "Workspace deletion not available in demo", sub: "Backend endpoint /workspaces/{id} DELETE stubbed" })}
          >
            <Icon name="trash" size={12} /> Delete workspace
          </button>
          <span className="muted-sm">Requires two-step confirmation in prod (UI + email).</span>
        </div>
      </div>
    </SettingsBody>
  );
}

// ─── Shared body shell ───────────────────────────────────────────────────
function SettingsBody({ title, sub, children }) {
  return (
    <div className="settings-body">
      <header className="settings-body-h">
        <h1>{title}</h1>
        {sub && <p className="settings-body-sub">{sub}</p>}
      </header>
      <div className="settings-body-content">
        {children}
      </div>
    </div>
  );
}
