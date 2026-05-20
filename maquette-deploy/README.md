# Maquette deploy

Reproducible deploy of the **Sweden bundle** prototype (the static
HTML+JSX maquette designed by the BNP / Sigilum team, currently held
outside this repo at `/Users/julien/Downloads/design_twinrag_backend/`)
behind Caddy with auto-HTTPS via Let's Encrypt.

Live target: <https://maquette.sigilum.fr/>
Host: OVH VPS `37.59.104.111` (also runs Dokploy + Traefik).

## Why this folder

The Sweden bundle isn't versioned (designer-side, not in this repo) but
the **deploy recipe** (`Dockerfile` + `Caddyfile`) is — so we can
re-deploy from a fresh machine without re-deriving the config each time.

## Layout expected at build time

```
maquette-deploy/
├── Dockerfile          ← here, versioned
├── Caddyfile           ← here, versioned
├── README.md           ← here, versioned
└── site/               ← copy of Sweden bundle (NOT versioned)
    ├── Twin RAG WebUI.html
    ├── *.jsx
    ├── styles.css
    ├── data.js
    └── ...
```

## Build + deploy (from a Mac with SSH access to OVH)

```bash
# 1. Stage the bundle alongside the deploy artifacts
rsync -a --exclude '.DS_Store' --exclude 'README.md' \
  ~/Downloads/design_twinrag_backend/ \
  /tmp/twin-maquette-deploy/site/
cp maquette-deploy/Dockerfile maquette-deploy/Caddyfile /tmp/twin-maquette-deploy/

# 2. Ship + build on the OVH host
tar czf /tmp/twin-maquette-deploy.tgz -C /tmp/twin-maquette-deploy .
scp /tmp/twin-maquette-deploy.tgz erwin:/tmp/
ssh erwin '
  rm -rf ~/twin-maquette && mkdir -p ~/twin-maquette
  tar xzf /tmp/twin-maquette-deploy.tgz -C ~/twin-maquette
  cd ~/twin-maquette && docker build -t twin-maquette:demo .
'
```

## First-time deploy (swarm service + Traefik)

```bash
ssh erwin '
  docker service create \
    --name twin-maquette \
    --network dokploy-network \
    --env SITE_ADDR=:80 \
    --label "traefik.enable=true" \
    --label "traefik.http.routers.twin-maquette.rule=Host(\`maquette.sigilum.fr\`)" \
    --label "traefik.http.routers.twin-maquette.entrypoints=websecure" \
    --label "traefik.http.routers.twin-maquette.tls=true" \
    --label "traefik.http.routers.twin-maquette.tls.certresolver=letsencrypt" \
    --label "traefik.http.services.twin-maquette.loadbalancer.server.port=80" \
    --label "traefik.http.routers.twin-maquette-http.rule=Host(\`maquette.sigilum.fr\`)" \
    --label "traefik.http.routers.twin-maquette-http.entrypoints=web" \
    --label "traefik.http.routers.twin-maquette-http.middlewares=twin-maquette-https" \
    --label "traefik.http.middlewares.twin-maquette-https.redirectscheme.scheme=https" \
    twin-maquette:demo
'
```

## Update an existing deploy

```bash
ssh erwin 'docker service update --force --image twin-maquette:demo twin-maquette'
```

## DNS

Cloudflare → `maquette.sigilum.fr` `A` → `37.59.104.111` · proxy **off**
(grey cloud — required so Caddy/Traefik can pass the ACME HTTP-01
challenge for Let's Encrypt).

## Patch overlays

Operator-side deviations from the designer's Sweden bundle are versioned
in this folder so the source bundle stays pristine. Two kinds:

### `operator-overrides.css`
Appended to the bundle's `styles.css` at build time. Covers:
- **Widescreen fill fix** — `.docs` and `.retrieval` get `flex: 1; min-width: 0`
  so they span the full viewport on wide displays (proto leaves a ~50%
  dead stripe). Equivalent fix is also in the React port at
  `lightrag_webui_twin/src/styles/overrides.css` (PR #27 on stable/0.5.x).
- **Base font bump** — `body { font-size: 14px }` (was 13px). Operator
  feedback on 4K displays.
- **QW1 Topbar 3-col flex** — proto pins `.tabs` with `position: absolute;
  left: 50%` while `.brand` and `.topbar-right` both grab `flex: 1`. At
  1366px with a long kb name + a fully-populated right cluster (sys
  indicator + workspace pill + bell + theme) the absolute tabs overlap
  the surrounding zones. Override drops the absolute positioning and
  switches to a 3-col flow where tabs is the flex-grow middle slot.
- **QW2 Query mode info button** — adds `.field-label-info / .info-btn /
  .query-mode-tooltip` so the Retrieval params panel can render an
  anchored popover next to the "Query mode" select (see
  `patches/site-overlay/retrieval.jsx`).

### `patches/site-overlay/`
Whole-file overlays that ship on top of the bundle's same-named files
(`COPY patches/site-overlay/ /srv/` runs after `COPY site/ /srv/`). Used
when a feature is too logic-heavy for a CSS-only delta. Current overlays:

- **`documents.jsx`** — adds the **steward review queue** at the top of
  the Documents tab: cards for docs flagged `review.state == "pending-review"`,
  with **Approve / Edit & approve / Reject** actions (palier-3 only). Reject
  opens an inline modal with a required reason. Mirrors the tag-governance
  flow in `tags.jsx`. Pending docs are excluded from the main grid + the
  status pill counters so the visible totals stay coherent.
- **`activity.jsx`** — adds the `doc-review` kind to the activity feed's
  `KIND_META` (icon: `circle-check`, color: accent). Doc-review events
  (approve/reject) emitted by the steward queue surface in the Activity
  tab alongside tag mutations, retrievals, etc.
- **`data.js`** — adds two pending-review documents (`d13`, `d14`) for
  demo purposes + two seeded `doc-review` activity events (approve +
  reject) at the top of `MOCK_ACTIVITY` so the audit trail is visible
  on first load. UX rewording sweep (QW5) also lands here: a few mock
  strings that surfaced `palier 1/2/3` switch to the UI-facing
  `Reader / Contributor / Steward` vocabulary the BNP audience reads.
- **`retrieval.jsx`** *(QW2)* — adds the `QueryModeInfo` popover next
  to the "Query mode" label so operators don't have to learn what
  `naive / local / global / hybrid / mix / bypass` mean from context.
- **`tags.jsx`** *(QW4 + QW5)* — adds the missing `Rejected` option to
  the status filter and switches `palier 1/2/3` wording to
  `Reader / Contributor / Steward` (incl. the role pill, pending-review
  captions, request modal copy, and read-only hints). The internal
  `palier` integer is preserved as the back-end / API contract; the
  rename only touches what an operator reads on screen.
- **`api.jsx`** *(QW5)* — `Scopes: ...(palier 2+)` becomes
  `Scopes: ...(Contributor or Steward)`.
- **`system-status.jsx`** *(QW5)* — the LLM-quota banner CTA now says
  `Steward only` instead of `palier 3`.

When the designer ships a new Sweden bundle, re-`cp` the same overlay
files from `~/Downloads/design_twinrag_backend/` after applying the
relevant patches manually, OR re-derive them from a fresh bundle.
