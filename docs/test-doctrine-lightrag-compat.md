# Test doctrine — LightRAG compat = explicit regression discipline

> Every Twin extension must prove, with a test, that it does **not** alter LightRAG's base behavior when absent or feature-flag-disabled.

## Why this doctrine exists

The product thesis is unambiguous:

> `register()` monkey-patches LightRAG **without modifying its source**.

That is also the **BNP audit defense**:

- Eric (RAG 1.5, 2026-05-28): the system must stay "chunks + vecteurs only — passé sous silence en audit".
- Louis (ISAB, 2026-05-28): the angle vendeur is "extension du patch LightRAG déjà en prod, pas un nouveau produit".

The moment a Twin extension silently changes LightRAG semantics, both the product framing and the audit defense collapse. Good intentions and code review are not enough — the constraint has to be enforced by tests.

## The rule

For **every** Twin extension — including but not limited to:

- storage backends (`kv_impl.py`, `vector_impl.py`, `docstatus_impl.py`)
- native shims (`server/native_shims.py`)
- classification hooks (`_classification_hook.py`)
- query overlays (`server/twin_query_routes.py`)
- IdP middleware (`server/idp_jwt.py`)
- folder binding (`server/folder.py`)
- buffered graph proxy (`_buffered_graph.py`)

…the test suite must include **at least one assertion** that the LightRAG-native path behaves identically when the extension is absent or its feature flag is off.

## Existing examples of the right shape

These tests are the templates to follow for new extensions:

| Test                                            | What it proves                                                                  |
| ----------------------------------------------- | ------------------------------------------------------------------------------- |
| `tests/test_register.py`                        | `register()` is idempotent — re-registration does not double-patch the dicts.   |
| `tests/test_server/test_route_parity.py`        | Native LightRAG routes and Twin shims coexist without unintended shadowing.     |
| `tests/test_use_database.py`                    | The `USE DATABASE` Memgraph quirk does not break the Neo4j path.                |
| `tests/test_consume_and_drop.py`                | `await result.consume()` discipline holds across error paths.                   |

## Two nets, not one

### Coarse net — CI matrix

`.forgejo/workflows/ci.yml` runs:

- **unit-tests**: LightRAG `1.4.9.11` / `1.4.11` / `1.4.12` × Python 3.10–3.13
- **integration-tests**: same LightRAG matrix × Memgraph `3.9.0` / `3.10.1`

This catches LightRAG version drift, but it does **not** prove that a given Twin extension preserves base behavior — it only proves that the patched system works on the matrix.

### Fine net — per-extension regression tests

This is what this doctrine mandates. Every extension PR adds a test of the form:

```python
# pseudo-shape
def test_extension_off_matches_lightrag_native():
    result_native = lightrag_call_without_extension(...)
    result_patched_off = lightrag_call_with_extension_disabled(...)
    assert result_native == result_patched_off
```

If the extension cannot be cleanly disabled (no flag, no opt-in), that is itself a smell to address before merge.

## Specific enforcement rules

- **New Twin extension PR with no LightRAG-compat regression test → reject before review.**
- **Bumping the LightRAG version in the CI matrix** → also add a smoke that runs the patched and unpatched paths side by side on the new version.
- **Wrapping a LightRAG call to add behavior X** → require a test that runs the wrapper with X disabled and asserts identical output to the bare call.
- **Runtime-level compat checks** against a deployed instance → reuse `tests/smoke/run_smoke.py` (stdlib-only manifest runner, designed for BNP-style restricted containers — see `tests/smoke/README.md`).

## What is not covered by this doctrine

- **UI fixtures (MSW only)** are not compat-relevant — they exercise the WebUI contract, not the LightRAG runtime. See `docs/test-doctrine-graph.md` for the graph contract rules instead.
- **Pure storage backends** ship their own integration tests in `tests/test_kv.py` etc. — these are necessary but not sufficient; a compat test that proves "LightRAG with Memgraph storage behaves like LightRAG with the reference storage on this query" is the missing piece.

## Related

- `CLAUDE.md` § "Repository nature" — the "no source modification" thesis.
- `CLAUDE.md` § "CI matrix" — the coarse net described above.
- `docs/test-doctrine-graph.md` — the sibling doctrine on graph contract testing.
- `tests/smoke/README.md` — the stdlib-only runtime smoke runner.
