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

## Known patches kept inside the Sweden bundle's source

Back-port of the **widescreen fill fix** appended to `styles.css`:
`.docs` and `.retrieval` get `flex: 1; min-width: 0` so the Documents
and Retrieval tabs span the full viewport width instead of leaving a
~50% dead stripe on the right. Equivalent fix is also in the React port
at `lightrag_webui_twin/src/styles/overrides.css` (PR #27).
