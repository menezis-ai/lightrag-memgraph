# Twin maquette UI conventions

Updated 2026-05-21 — landed with issue [#49](http://192.168.1.61:3000/julien/twindb-lightrag-memgraph/issues/49) (post cross-tab audit).

Scope: the SPA in `maquette-deploy/source/` served through
`maquette-deploy/`. Independent from the React port at
`lightrag_webui_twin/`.

## Empty states

**Always use** `<EmptyState>` from `icons.jsx` (`window.EmptyState`):

```jsx
<EmptyState
  icon="search"
  title="No results"
  sub="Try a different filter or clear the active one."
  actions={<>
    <button className="ghost-btn small" onClick={clear}>Clear filters</button>
  </>}
/>
```

The proto's older empty-state shapes — `.empty-state` (Retrieval),
`.empty-pane` / `.empty-workspace` (Documents), `.tags-empty.zero` /
`.tags-empty.filtered` (Tags) — keep working in parallel so existing
call-sites don't break. **Don't add new ones**; replace with
`<EmptyState>` when you touch the surrounding code.

## Buttons

| Use | Class | Notes |
|---|---|---|
| Primary CTA (one per screen) | `btn primary` *or* `primary-btn` | Aliased visually. Either form is valid. |
| Secondary action | `ghost-btn` | Preferred. `btn` (no modifier) is legacy but still valid. |
| Inline text-link | `link-btn` | Looks like a link, behaves like a button (a11y win). |
| Small / tight density | `<class> small` | Modifier works on every button class above. |
| Destructive | `<class> danger` | Use ghost form unless the destructive op is the headline action of the modal. |

## Colors

- **Always** use design tokens (`var(--twin-accent)`,
  `var(--twin-amber-vivid)`, `var(--color-status-failed)`, etc.).
- **Never** hardcode hex literals — they don't theme-flip and they
  silently drift from the rest of the palette.

## Tokens reference (excerpt)

| Token | Use |
|---|---|
| `--twin-accent` | Primary brand color (Twin blue). |
| `--twin-accent-soft-bg` / `--twin-accent-soft-text` / `--twin-accent-soft-border` | Low-saturation pill / chip backgrounds. |
| `--twin-amber-vivid` / `--twin-amber-700` / `--twin-amber-50` | Warning chrome (pending review queue). |
| `--twin-green-700` / `--twin-green-50` | Success state (status pill, source ready). |
| `--twin-red-vivid` / `--twin-red-border` | Error / destructive. |
| `--color-text-primary` / `--color-text-secondary` / `--color-text-tertiary` | 3-tier text hierarchy. |
| `--color-background-primary` / `--color-background-secondary` / `--color-background-tertiary` | 3-tier surface hierarchy. |
| `--color-border-default` / `--color-border-secondary` / `--color-border-tertiary` | 3-tier border hierarchy. |

## Operator override pattern

When the proto's `styles.css` is wrong/missing, **don't fork the
source bundle** — add the fix in `maquette-deploy/operator-overrides.css`,
which is appended to `styles.css` at Docker build time. Tag each block
with the issue number so the rationale stays attached:

```css
/* ── #NN: brief title ──────────────────────────────────────────────
 * Why this exists, in 3-5 lines.
 */
```

If the override is too logic-heavy for CSS, drop a whole-file overlay
in `maquette-deploy/patches/site-overlay/` (applied after the source
bundle copy in the Dockerfile). Document the overlay in
`maquette-deploy/README.md`.
