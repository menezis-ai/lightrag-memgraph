# 👁️ CASSANDRE'S BLACK BOOK

> Case files for the Menezis Empire. Not a log — a record of skeletons.
> Codebase: `twindb-lightrag-memgraph` (Memgraph KV/Vector/DocStatus backends for LightRAG).

---

## 2026-06-15 - [The Silent Drop]

**Evidence:** `src/twindb_lightrag_memgraph/vector_impl.py::drop()` ended with a
blanket `except Exception: pass  # Index may not exist` wrapping
`DROP VECTOR INDEX`. The `DETACH DELETE` above it was NOT guarded, but the index
drop swallowed *everything*. Regardless of what failed, the method then returned
`{"status": "success", "message": "Vector namespace ... dropped"}`.

**Verdict:** A false-success generator. The catch-all buried connection resets,
auth failures, and permission denials. Worst case: nodes deleted but the vector
index left behind — the exact stale state the `REMOVE`-before-`DELETE` dance in
that same method exists to prevent — while the caller is told the drop
succeeded. Silent failure + lying status = an operator who never knows the
cluster is in a half-dropped state until the next ingest corrupts. Every OTHER
handler in this repo logs and narrows by error message; this was the lone
outlier.

**Sentence:** Narrowed the handler to swallow ONLY the idempotent
"index does not exist" case (matching the convention already used in
`query()`), log it at debug, and re-raise anything else. Added two regression
tests in `tests/test_consume_and_drop.py`:
`test_drop_propagates_drop_index_failure` (proven to FAIL against the old code —
"DID NOT RAISE") and `test_drop_swallows_missing_index` (idempotency preserved).
Doctrine: **Operation AEGIS** — fail loud, never report success on failure.

---

## 2026-06-15 - [Dependency Drift / lightrag-hku] — FALSE POSITIVE for the audit, REAL for CI

**Evidence:** `pyproject.toml` pins `lightrag-hku>=1.4.9,<2.0.0`. The container
resolved `lightrag-hku==1.5.3`, which added abstract methods
(`get_doc_by_content_hash`, `get_doc_by_file_basename`) to the DocStatus base
class and reshaped the query/registration APIs. On a clean checkout the suite is
already **48 failed / 151 passed / 6 errors** BEFORE any of my changes.

**Verdict:** Not a security vuln, and NOT caused by today's fix — do not waste
time blaming local edits for these reds again. But the loose upper pin lets a
single minor release of a core dependency break the entire backend silently
between deploys (a supply-chain-adjacent stability risk). A "works on the pinned
version" defense is invalid the moment `pip install` floats the minor.

**Sentence:** Recommended (NOT yet done — out of scope for the drop fix): tighten
the pin to a tested minor band (e.g. `>=1.4.9,<1.5`) or add a compat shim for the
new abstract methods. Recorded so future patrols measure their blast radius
against the **48/6** baseline, not zero.

---
