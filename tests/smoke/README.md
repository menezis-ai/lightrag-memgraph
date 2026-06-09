# Twin Runtime Smoke

This folder contains a procedural smoke-test contract for restricted BNP-style
containers. The JSON file is the checklist; `run_smoke.py` executes it and
writes a machine-readable report that can be reviewed by CI or an AI assistant.

Required environment:

```bash
export TWIN_SMOKE_BASE_URL="https://your-runtime-host"
export ARTIFACTORY_USERNAME="..."
export ARTIFACTORY_PASSWORD="..."
python tests/smoke/run_smoke.py tests/smoke/bnp-runtime-smoke.json
```

The default manifest assumes the WebUI is mounted at `/webui`, local JWT auth is
enabled, and the runtime uses HTTPS. It validates:

- `/webui` is served.
- `/auth-status` reports login required before authentication.
- `/twin/api/documents` rejects anonymous access.
- `/login` returns a token and a secure `twin_local_token` cookie.
- Native LightRAG routes and Twin overlay routes are reachable after login.
- `/logout` clears the session.

Outputs:

- `/tmp/twin-smoke-report.json`: structured pass/fail report.
- `/tmp/twin-smoke-http.log`: compact HTTP trace without secrets.

For a local HTTP-only fixture, set `auth.attach_bearer_after_login` to `true` in
a copied manifest. Secure cookies are intentionally not sent by browsers or
`CookieJar` over plain HTTP.
