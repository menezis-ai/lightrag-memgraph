"""The frozen v1 bundle: current code held to bytes it did not just write.

Every other portability test round-trips through the exporter *and* the importer
of the same commit, so both ends move together: renaming a field or dropping a
store keeps the suite green while making every bundle already handed to an
operator unimportable.  These tests read
``tests/fixtures/bundles/twin-kb-bundle-v1`` — a real export, committed, never
regenerated to make a test pass — so a v1 compatibility break has to announce
itself here.

When one of these fails, the question is never "how do I refresh the fixture".
It is: is this change compatible with bundles already in the field?  If yes,
the importer needs a compatibility path.  If no, the bundle format version is
bumped and the new format gets its own fixture directory beside this one.
Regeneration: ``scripts/portability_freeze_fixture.py`` (see the README there).
"""

from __future__ import annotations

import json
from typing import Any

import jsonschema
import pytest

from twindb_lightrag_memgraph.portability.bundle import BundleReader, inspect_bundle
from twindb_lightrag_memgraph.portability.canonical import jcs_sha256
from twindb_lightrag_memgraph.portability.jsonl import iter_jsonl
from twindb_lightrag_memgraph.portability.manifest import state_hash_of
from twindb_lightrag_memgraph.portability.stores import (
    STORES,
    exportable_stores,
    store_by_file,
)
from twindb_lightrag_memgraph.portability.validate import normalized_source_state_hash

from tests._repo_only import require_repo_path

from ._golden import (
    GOLDEN_RUNTIME_FOLDER,
    GOLDEN_SOURCE_FOLDERS,
    GOLDEN_V1_DIR,
    golden_embedding,
    png_chunks,
    wipe_workspace,
)

ALL_GOLDEN_FOLDERS = (*GOLDEN_SOURCE_FOLDERS, GOLDEN_RUNTIME_FOLDER)


@pytest.fixture
def golden_manifest() -> dict[str, Any]:
    return json.loads((GOLDEN_V1_DIR / "manifest.json").read_text(encoding="utf-8"))


def test_golden_bundle_is_intact_under_the_current_reader():
    """The frozen bytes still pass integrity, and every line is still canonical."""
    inspection = inspect_bundle(GOLDEN_V1_DIR)
    assert inspection.ok, inspection.problems

    with BundleReader(GOLDEN_V1_DIR) as reader:
        assert reader.manifest is not None
        jsonl = [e for e in reader.manifest.files if e.path.endswith(".jsonl")]
        assert jsonl, "the fixture must contain JSONL files"
        for entry in jsonl:
            # iter_jsonl re-serialises each line and refuses any that is not
            # byte-identical, so this pins jcs_dumps/jcs_loads against bytes
            # written by an earlier commit.
            records = list(iter_jsonl(reader.path_of(entry.path), entry.sha256))
            assert len(records) == entry.records, entry.path


def test_golden_manifest_matches_the_published_schema(golden_manifest):
    """The published contract must still describe a bundle already in the field."""
    # docs/ is excluded from the BNP export, so this one assertion skips there
    # while the rest of the golden checks still run against the shipped fixture.
    path = require_repo_path(
        "docs/templates/kb-bundle.manifest.schema.json", module_level=False
    )
    schema = json.loads(path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(golden_manifest)


def test_golden_manifest_hashes_recompute_from_the_frozen_content(golden_manifest):
    """manifest_hash and state_hash still derive from the frozen bytes."""
    body = {k: v for k, v in golden_manifest.items() if k != "manifest_hash"}
    assert jcs_sha256(body) == golden_manifest["manifest_hash"]

    recomputed = state_hash_of(
        {f["path"]: f["sha256"] for f in golden_manifest["files"]}
    )
    assert recomputed == golden_manifest["state_hash"]


def test_golden_bundle_pins_the_v1_store_roster(golden_manifest):
    """Adding, renaming or dropping a store is a v1 compatibility event."""
    frozen = {entry["store"] for entry in golden_manifest["files"]}
    current = {
        spec.name
        for spec in exportable_stores(include_activity=True, include_procedures=True)
    }
    assert frozen == current, (
        f"store roster drifted — only in the fixture: {sorted(frozen - current)}; "
        f"only in the current registry: {sorted(current - frozen)}"
    )

    for entry in golden_manifest["files"]:
        if entry["path"].endswith(".jsonl"):
            spec = store_by_file(entry["path"])
            assert spec is not None, f"no store owns {entry['path']} any more"
            assert spec.name == entry["store"]

    # A fixture with an empty store proves nothing about that store's records.
    # Only JSONL entries carry records: a store may also own binary files (the
    # procedure schematics), which are legitimately recordless.
    records_by_store: dict[str, int] = dict.fromkeys(current, 0)
    for entry in golden_manifest["files"]:
        if entry["path"].endswith(".jsonl"):
            records_by_store[entry["store"]] += entry["records"]
    empty = sorted(name for name, count in records_by_store.items() if count == 0)
    assert not empty, f"the fixture must exercise every store; empty: {empty}"

    never = {spec.name for spec in STORES if spec.portability == "never"}
    assert not (frozen & never), "a never-export store reached the bundle"


def test_golden_procedure_schematic_is_a_real_image(golden_manifest):
    """The frozen schematic must be a decodable PNG, not opaque bytes.

    The exporter copies a schematic through without looking inside it, so a
    file that merely starts with the PNG signature would round-trip happily
    while proving nothing about images. Every chunk CRC is verified, which is
    stricter than asking a lenient decoder whether it can open the file.
    """
    assets = [
        entry for entry in golden_manifest["files"] if entry["path"].endswith(".png")
    ]
    assert assets, "the fixture must carry a procedure schematic"
    for entry in assets:
        data = (GOLDEN_V1_DIR / entry["path"]).read_bytes()
        assert len(data) == entry["bytes"]
        tags = [tag for tag, _ in png_chunks(data)]
        assert tags[0] == "IHDR" and tags[-1] == "IEND" and "IDAT" in tags


def test_identity_folder_mapping_reserialises_every_record_unchanged():
    """Normalising the frozen records must be a no-op under an identity map.

    This walks every record of every store through the current
    ``_normalized_record`` + ``jcs_dumps`` and requires the result to hash back
    to the frozen ``state_hash``. The second assertion is what keeps the first
    honest: a real remap must change the hash, so an identity match cannot be
    passing because the normalisation stopped doing anything.
    """
    identity = {folder: folder for folder in ALL_GOLDEN_FOLDERS}
    with BundleReader(GOLDEN_V1_DIR) as reader:
        assert reader.manifest is not None
        frozen_state_hash = reader.manifest.state_hash
        assert normalized_source_state_hash(reader, identity) == frozen_state_hash
        remapped = normalized_source_state_hash(
            reader, {GOLDEN_SOURCE_FOLDERS[0]: "gt1"}
        )
    assert remapped != frozen_state_hash


@pytest.mark.integration
async def test_golden_bundle_still_imports_into_an_empty_workspace(
    monkeypatch, tmp_path
):
    """The end of the contract: a bundle from an earlier commit still restores.

    Offline the fixture only proves the *reader* still understands it. This
    drives the operator path — dry-run, apply, validate — so a record shape the
    current stores can no longer consume fails here rather than at a customer.
    """
    import twindb_lightrag_memgraph
    from twindb_lightrag_memgraph import _pool
    from twindb_lightrag_memgraph.portability.importer import apply_import
    from twindb_lightrag_memgraph.portability.plan import create_dry_run, write_report
    from twindb_lightrag_memgraph.portability.validate import validate_import
    from twindb_lightrag_memgraph.server import folder_store

    twindb_lightrag_memgraph.register()
    target_ws = "golden_target"

    async def _run(query: str) -> None:
        async with _pool.get_session() as session:
            result = await session.run(query)
            await result.consume()

    async def _wipe() -> None:
        # Shared with the freezer, and deliberately not hand-rolled here: it
        # REMOVEs the vector label before DETACH DELETE and re-raises anything
        # but "index does not exist", so a failed teardown cannot leave stale
        # vector-index state for the tests that run after this one.
        await wipe_workspace(_run, workspace=target_ws, folders=ALL_GOLDEN_FOLDERS)

    monkeypatch.setenv("TWIN_PORTABILITY_DIR", str(tmp_path / "portability"))
    monkeypatch.setenv("TWIN_FOLDERS_RUNTIME_FILE", str(tmp_path / "folders.json"))
    monkeypatch.setenv("TWIN_PROCEDURE_STORE_FILE", str(tmp_path / "procedures.json"))
    monkeypatch.setenv("TWIN_MIP_MAX_CLASSIFICATION", "C2")
    monkeypatch.setenv("TWIN_DEFAULT_FOLDER", GOLDEN_SOURCE_FOLDERS[0])
    monkeypatch.setenv(
        "TWIN_FOLDERS_JSON",
        json.dumps(
            [
                {
                    "id": GOLDEN_SOURCE_FOLDERS[0],
                    "label": "Golden one",
                    "kind": "primary",
                },
                {"id": GOLDEN_SOURCE_FOLDERS[1], "label": "Golden two", "kind": "team"},
            ]
        ),
    )
    folder_store.reset_runtime_store()
    await _wipe()
    try:
        embedding = golden_embedding()
        # Restoring into folder ids the target already declares is exactly the
        # case the planner refuses to assume: an identity mapping has to be
        # approved explicitly. The runtime-only folder collides with nothing and
        # is recreated by the runtime_folders store.
        folder_map = {folder: folder for folder in GOLDEN_SOURCE_FOLDERS}
        report = await create_dry_run(
            GOLDEN_V1_DIR,
            workspace=target_ws,
            folder_map=folder_map,
            embedding_func=embedding,
        )
        assert report["blocking"] == [], report["blocking"]

        report_path = write_report(tmp_path / "report.json", report)
        applied = await apply_import(
            GOLDEN_V1_DIR,
            report_path=report_path,
            checkpoint_path=tmp_path / "checkpoint.json",
            embedding_func=embedding,
        )
        assert applied["ok"] is True and applied["resumed"] is False

        validation = await validate_import(
            GOLDEN_V1_DIR,
            workspace=target_ws,
            folder_map=folder_map,
            embedding_func=embedding,
        )
        assert validation["ok"] is True, validation["problems"]
        assert validation["expected_state_hash"] == validation["actual_state_hash"]
    finally:
        await _wipe()
        folder_store.reset_runtime_store()
