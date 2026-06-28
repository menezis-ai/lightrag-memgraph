# Repository Guidelines

## Project Structure & Module Organization
Core runtime code lives in `src/twindb_lightrag_memgraph/`. It contains storage adapters, server overlay, intelligence features, and packaged WebUI assets under `webui_dist/`.  
Frontend source is in `lightrag_webui_twin/src/` with tests and UI fixtures in `lightrag_webui_twin/test` and `lightrag_webui_twin/src/test`.  
Python tests are in `tests/` (plus domain suites like `tests/test_server/` and `tests/test_intelligence/`), and restricted runtime checks are in `tests/smoke/`.  
Supporting material is under `docs/`, `scripts/`, `deploy/`, and `ENV_VARIABLES.txt` (authoritative env reference).  
Operational files include `Dockerfile`, `docker-compose.yml`, `pyproject.toml`, and workflow metadata.

## Build, Test, and Development Commands
Backend setup:
- `pip install -c requirements/constraints-dev.txt -e ".[server,intelligence,test]"` for full local development.
- `pip install -e ".[test]"` / `-e ".[test-server]"` / `-e ".[test-intelligence]"` for scoped test installs.
- `pip install -e ".[all]"` for complete surface coverage.

Backend test commands:
- `pytest tests/ --ignore=tests/test_bench.py -v`
- `MEMGRAPH_URI=bolt://localhost:7687 pytest tests/ --ignore=tests/test_bench.py -v`
- `MEMGRAPH_URI=bolt://localhost:7687 pytest tests/test_bench.py -v -s`
- `coverage run -m pytest tests/ --ignore=tests/test_bench.py && coverage xml`

WebUI commands (from `lightrag_webui_twin/`):
- `npm ci`
- `npm run lint`
- `npm run typecheck`
- `npm run test:run`
- `npm run build`
- `npm run test:e2e`
- `npm run test:e2e:real` for backend-coupled end-to-end checks.

Local runtime:
- `python twin_main.py`
- `python -m twindb_lightrag_memgraph.lightrag_server`

## Coding Style & Naming Conventions
Python follows practical PEP 8: 4-space indentation, snake_case for functions/modules, PascalCase for classes, and UPPER_SNAKE for constants. Prefer explicit names and add/extend tests with behavior changes.  
TypeScript/React uses strict settings from `lightrag_webui_twin/tsconfig.app.json` (`strict`, `noUnusedLocals`, `noUnusedParameters`) and ESLint in `lightrag_webui_twin/eslint.config.js`. Prefer `PascalCase.tsx` for components, `camelCase` for hooks/utilities, and colocated tests.

## Testing Guidelines
Use `tests/conftest.py` behavior: `@pytest.mark.integration` tests auto-skip unless `MEMGRAPH_URI` is set, so local offline runs stay fast.  
Test naming convention is `test_*.py` with function/class names describing expected behavior.  
For smoke checks, run `python tests/smoke/run_smoke.py tests/smoke/runtime-smoke.json`.

## Commit & Pull Request Guidelines
No enforced commit-message schema is defined in repository files; follow the project intent model used in docs: commit messages should state *what changed*, *why*, and the issue/decision context they resolve.  
Use concise subjects (e.g., `server/folder: enforce X-Twin-Folder checks`) and include any breaking behavior or security impact in the body.  
PRs should include: change summary, commands run, config/environment impact, affected files/paths, and links to related issue or incident notes.

## Security & Configuration Tips
Use environment variables for all runtime behavior and read `ENV_VARIABLES.txt` before adding new knobs. Keep secrets out of tests and tracked files.  
Avoid checking in generated lockfiles or deployment artifacts not required by the project.
