# Twin KMS 1.0.0 Release Candidate Checklist

Date: 2026-06-21

Scope: close the Twin KMS 1.0.0 development train and prepare a production
release candidate from a clean checkout/image.

## Required Gates

All gates below must be green before tagging the release candidate.

| Gate | Command / evidence | Status |
| --- | --- | --- |
| Backend tests from clean checkout | `uv run pytest -q` | Pending final clean checkout run |
| Frontend lint | `cd lightrag_webui_twin && npm run lint` | Pending final clean checkout run |
| Frontend typecheck | `cd lightrag_webui_twin && npm run typecheck` | Pending final clean checkout run |
| Frontend unit tests | `cd lightrag_webui_twin && npm run test:run` | Pending final clean checkout run |
| Frontend production build | `cd lightrag_webui_twin && npm run build` | Pending final clean checkout run |
| Frontend production audit | `cd lightrag_webui_twin && npm audit --omit=dev` | Pending final clean checkout run |
| Playwright MSW E2E | `cd lightrag_webui_twin && npx playwright test` | Pending final clean checkout run |
| Real-backend E2E | Forgejo real backend workflow / staging evidence | Pending staging run |
| Docker image build | Build release image from clean checkout | Pending image build |
| Runtime smoke test | `python tests/smoke/run_smoke.py tests/smoke/runtime-smoke.json` against built image | Pending image smoke |
| Python dependency audit | `uvx pip-audit -r requirements/constraints-prod.txt --no-deps --disable-pip --ignore-vuln CVE-2026-30762 --ignore-vuln CVE-2026-39413 --ignore-vuln PYSEC-2026-1325` | Pending final clean checkout run |
| SonarQube quality gate | `/opt/homebrew/bin/sonar-scanner` using `sonar-project.properties` and `SONAR_TOKEN` | Pending scanner run |

## Risk Acceptance

`lightrag-hku==1.4.9.11` remains pinned for this release train.

The known advisories `CVE-2026-30762`, `CVE-2026-39413`, and
`PYSEC-2026-1325` are temporarily accepted with compensating controls, owner,
rationale, and review date in:

```text
docs/security/lightrag-1.4.9.11-risk-acceptance.md
```

Review date: 2026-07-21.

## SonarQube

The scanner configuration is versioned in `sonar-project.properties`.

The scan expects:

- SonarQube server: `http://192.168.1.49:9000`
- project key: `twindb-lightrag-memgraph`
- project version: `1.0.0`
- Python coverage report: `coverage.xml`
- WebUI LCOV report: `lightrag_webui_twin/coverage/lcov.info`
- token provided out of band via `SONAR_TOKEN`

Suggested preparation:

```bash
coverage run -m pytest tests/ --ignore=tests/test_bench.py
coverage xml
cd lightrag_webui_twin
npm run test:run -- --coverage
cd ..
/opt/homebrew/bin/sonar-scanner
```

Do not commit scanner credentials or generated scanner output.

## Version Closure

Once all gates are green:

1. Attach or link CI, audit, SonarQube, Docker build, and smoke-test evidence.
2. Tag the release candidate.
3. Keep the LightRAG risk acceptance active only until the planned upgrade work
   replaces it.
