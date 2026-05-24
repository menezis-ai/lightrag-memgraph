# Maquette deploy

Reproducible deploy of the Twin RAG WebUI maquette behind Caddy with
auto-HTTPS via Let's Encrypt + a FastAPI/SQLite persistence backend.

Live target: <https://maquette.sigilum.fr/>
Host: OVH VPS `37.59.104.111` (also runs Dokploy + Traefik).

Two containers (web + api):
- `twin-maquette:demo` — Caddy serving the static SPA from `source/`,
  reverse-proxying `/api/*` to the api sibling.
- `twin-maquette-api:demo` — FastAPI + SQLite persistence layer in
  `backend/` (real DB file, mounted from a Docker volume → state
  survives container rebuilds).

The architecture was previously a sql.js WASM client-side blob (PR #57),
reverted because BNP infra won't accept a CDN-fetched WASM and a real
backend with auditable code + Docker-mounted SQLite is the
production-shaped answer.

## Layout

```
maquette-deploy/
├── Dockerfile               ← caddy image build
├── Caddyfile                ← reverse_proxy /api/* → backend
├── operator-overrides.css   ← appended to source/styles.css at build
├── stack.yml                ← docker stack deploy spec (web + api)
├── source/                  ← full SPA (HTML + JSX + CSS + data.js + db.jsx)
├── backend/                 ← FastAPI + SQLite + seed JSON
├── README.md                ← this
├── STYLES.md                ← UI conventions decided post-audit (#49)
└── DEMO_MANU.md             ← demo script for the Manu 2026-05-22 slot
```

Everything is versioned in this repo. No external dependency on a
working copy elsewhere on the machine.

## Build + deploy (from a Mac with SSH access to OVH)

```bash
# 1. Stage exactly what the image needs
rm -rf /tmp/twin-maquette-deploy
mkdir -p /tmp/twin-maquette-deploy
cp -r maquette-deploy/source maquette-deploy/backend /tmp/twin-maquette-deploy/
cp maquette-deploy/Dockerfile maquette-deploy/Caddyfile \
   maquette-deploy/operator-overrides.css maquette-deploy/stack.yml \
   /tmp/twin-maquette-deploy/

# 2. Ship + build both images on the OVH host
COPYFILE_DISABLE=1 tar czf /tmp/twin-maquette-deploy.tgz \
   -C /tmp/twin-maquette-deploy .
scp /tmp/twin-maquette-deploy.tgz erwin:/tmp/
ssh erwin '
  rm -rf ~/twin-maquette && mkdir -p ~/twin-maquette
  tar xzf /tmp/twin-maquette-deploy.tgz -C ~/twin-maquette
  cd ~/twin-maquette
  docker build -t twin-maquette:demo .
  docker build -t twin-maquette-api:demo backend/
'
```

`COPYFILE_DISABLE=1` keeps macOS AppleDouble (`._*`) metadata out of
the tarball — `seed-data/._docs.json` would otherwise leak into the
image and break the FastAPI seed loop (it'd try to parse the binary
fork as JSON).

## First-time deploy (Docker Stack + Traefik)

The two services (web + api) are orchestrated via `stack.yml` so a
single command stands them up on the existing `dokploy-network`. The
api stays internal (no Traefik labels); only the web container is
public.

```bash
ssh erwin '
  # If a legacy single `twin-maquette` service exists (pre-stack era),
  # remove it first so the stack can create its own twin-maquette_web.
  docker service rm twin-maquette 2>/dev/null || true

  cd ~/twin-maquette
  docker stack deploy -c stack.yml twin-maquette
'
```

## Update an existing deploy

```bash
# Rebuild on OVH then redeploy the stack (services pick up new images
# via image-resolve heuristics; --force-recreate via stack is not a
# thing, so explicit per-service update is the right verb).
ssh erwin '
  cd ~/twin-maquette
  docker build -t twin-maquette:demo .
  docker build -t twin-maquette-api:demo backend/
  docker service update --force --image twin-maquette:demo twin-maquette_web
  docker service update --force --image twin-maquette-api:demo twin-maquette_api
'
```

## Inspect the SQLite snapshot

```bash
ssh erwin '
  docker exec $(docker ps -q -f name=twin-maquette_api) \
    sqlite3 /data/twin-demo.sqlite \
    "SELECT kind, COUNT(*) FROM entities GROUP BY kind;"
'
```

## DNS

Cloudflare → `maquette.sigilum.fr` `A` → `37.59.104.111` · proxy **off**
(grey cloud — required so Caddy/Traefik can pass the ACME HTTP-01
challenge for Let's Encrypt).

## `operator-overrides.css`

Appended to `source/styles.css` at image build time. Keeps the proto's
own stylesheet untouched so a future re-import of the bundle stays a
clean overlay. Tagged blocks by feature / issue number — `grep -nE
'^/\* ── ' operator-overrides.css` to navigate.

## History note

Until 2026-05-22 the SPA lived outside this repo at
`~/Downloads/design_twinrag_backend/` and was applied via a
`patches/site-overlay/` directory (whole-file overrides COPYed after
the bundle). That was a SPOF — the source bundle was hand-managed,
unversioned, machine-local. The cleanup commit flattened everything
into `source/` and dropped the overlay system. See git log for the
per-feature additions that previously had their own overlay entry.
