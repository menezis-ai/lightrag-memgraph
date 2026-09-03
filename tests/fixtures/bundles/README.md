# Frozen KB bundles

`twin-kb-bundle-v1/` is a **real** export produced by
`scripts/portability_freeze_fixture.py` against a live Memgraph, then committed
verbatim. It is the only bundle in the suite that the current commit did not
just write.

## Why it exists

Every other portability test exports with the current exporter and re-imports
with the current importer. Both ends move together, so renaming a field,
dropping a store or changing the canonical serialisation keeps the suite green
— while making every bundle already delivered to an operator unimportable. The
frozen bundle is the only artefact that can catch that, and
`tests/test_portability/test_golden_bundle.py` holds the code to it: integrity
and canonical form under the current reader, the published manifest schema, the
hash derivations, the v1 store roster, and — on Memgraph — a full dry-run →
apply → validate of these exact bytes.

## When a golden test fails

The question is never "how do I refresh the fixture". It is:

1. **Is the change compatible with bundles already in the field?** If yes, the
   importer needs a compatibility path, and the fixture stays as it is — it is
   the regression test for that path.
2. **If not, it is a format break.** Bump the bundle `format_version`, freeze a
   new fixture into its own directory beside this one, and keep this one for as
   long as v1 bundles are claimed to be importable.

Regenerating the fixture to turn a red test green destroys the only evidence
that the format did not move. `scripts/portability_freeze_fixture.py` refuses to
overwrite an existing fixture without `--force` for that reason.

## Regenerating (deliberately)

```bash
MEMGRAPH_URI=bolt://127.0.0.1:7687 python scripts/portability_freeze_fixture.py --force
```

The script seeds a scratch workspace (`golden_source`) covering every
exportable store — the two `optional` ones included — exports it, refuses to
freeze anything that is not `verified` or that was produced off the supported
LightRAG minor, and wipes the scratch workspace afterwards. The bundle ids,
timestamps and hashes it produces are new on every run; that is expected, and
it is the diff you review.
