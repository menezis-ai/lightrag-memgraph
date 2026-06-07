# twin-real deploy

Real LightRAG + Memgraph + React port behind Traefik + Basic Auth,
running as a `docker stack` (swarm mode).

**Distinct from `maquette-deploy/`** — that one is the standalone MSW
demo (no backend). This one is the full Couche 3 stack: real LLM
calls, persisted Memgraph state, full `/twin/api/*` overlay.

## Prerequisites

| | |
|---|---|
| DNS | An A record pointing the chosen hostname to the host running the stack. |
| SSH | An alias to the host (used below as `<host>`). |
| Docker swarm | Already initialized on the host. |
| OpenAI key | Stored in a local `0600` env file, never echoed. |
| Basic Auth | A htpasswd line `user:$apr1$...` generated locally (see "Generate auth" below). |

## Generate auth

The Traefik label expects a `$apr1$` htpasswd line. Generate with:

```bash
htpasswd -nbB twin '<chosen-password>' | head -1
# user:$2y$05$...
```

Or with Python (no apr-utils dependency):

```python
import bcrypt, sys
pwd = sys.argv[1].encode()
print(f"twin:{bcrypt.hashpw(pwd, bcrypt.gensalt(rounds=5)).decode()}")
```

Save the resulting line as `TWIN_BASIC_AUTH=` inside your local env file
(see next section). Keep the cleartext password out of the chat — share
it with operators via your usual off-band channel.

## Local env file (NOT committed)

Create `~/.twin-real.env` on your workstation:

```env
OPENAI_API_KEY=sk-proj-...           # rotated; see incident
TWIN_BASIC_AUTH=twin:$$2y$$05$$...   # apr1/bcrypt line, $ doubled
IMAGE_TAG=2026-06-01a                # bump on each deploy
WORKSPACE=cib                        # default workspace
```

Permission: `chmod 600 ~/.twin-real.env`.

## Build + deploy

From the repo root (replace `<host>` with your SSH alias):

```bash
# 1. Build the image locally (multi-stage: bun web + python runtime).
#    --platform linux/amd64 is required when building from an Apple
#    Silicon Mac for an x86_64 host.
docker build --platform linux/amd64 \
  -f deploy/ovh-twin/Dockerfile \
  -t twin-real-app:2026-06-01a .

# 2. Save the image to a tarball
docker save twin-real-app:2026-06-01a | gzip > /tmp/twin-real-app.tar.gz

# 3. Ship to the host
scp /tmp/twin-real-app.tar.gz <host>:/tmp/
scp deploy/ovh-twin/stack.yml <host>:~/twin-real/stack.yml
scp ~/.twin-real.env <host>:~/twin-real/twin-real.env

# 4. On the host: load + deploy
ssh <host> '
  cd ~/twin-real
  docker load < /tmp/twin-real-app.tar.gz
  set -a; source twin-real.env; set +a
  docker stack deploy -c stack.yml twin-real
  rm /tmp/twin-real-app.tar.gz
'

# 5. Watch the rollout
ssh <host> 'docker stack ps twin-real --no-trunc | head -10'
```

Then browse to whichever hostname your Traefik router is configured for
in `stack.yml` (Basic Auth prompt → enter the password you set above).

## Rollback

```bash
ssh <host> 'docker stack rm twin-real'
```

The Memgraph volume (`twin-real-memgraph`) and the LightRAG storage
volume (`twin-real-rag`) persist across `rm`/`deploy` cycles — they're
only wiped with an explicit `docker volume rm`. Useful for testing
upgrades without losing the workspace.

## Operational notes

- The first ingestion will create the Memgraph indexes (KV / Vector /
  DocStatus / Graph) for the chosen workspace. ~10s for an empty DB.
- LLM calls are charged to the key you provided. Default model is
  `gpt-4o-mini` (cheap, ~$0.15/M input tokens at the time of writing).
  Override via the `LLM_MODEL` env var in `~/.twin-real.env` if you
  want a different model.
- The bundled WebUI uses `debugUser` (palier 3 / Steward) for every
  authenticated visitor — Basic Auth is the only access gate. Do NOT
  share the URL publicly without first wiring JWT/IdP middleware
  (Couche 3 §3.3 in `WEBUI-WIRING-PLAN.md`).
- Tag categories bootstrap from the 6-entry internal seed on first
  boot. Operators can later upload a JSON taxonomy via the WebUI
  "Import categories" button (Tags page) — replaces in place.
