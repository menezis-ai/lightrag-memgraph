"""Procedure-PDF ingestion profile (PROCEDURE-PROFILE-PLAN.md, PR 1).

Unit tests on ``_procedure`` (tier gates, deterministic template detection,
schematic-page location, dual-pass orchestration, failure degradation), the
``_procedure_store`` bundle store, and seam-contract tests on the registry
routing (procedure → parked bundle, NOT enqueued; standard path untouched).
The vision LLM and the pypdfium2 render are always monkeypatched — no
network, no native render. Text extraction runs against the synthetic
template fixture (``tests/procedure_pdf_fixture.py``) through the real
pypdf, so detection is exercised on a genuine PDF text layer.
"""

import base64
import json
import sys

import pytest

from tests.procedure_pdf_fixture import (
    PROCEDURE_PAGES,
    PROCEDURE_SCHEMATIC_PAGES,
    build_plain_pdf,
    build_procedure_pdf,
    build_textonly_procedure_pdf,
)
from twindb_lightrag_memgraph import _conversion, _procedure, _procedure_store, _vision
from twindb_lightrag_memgraph._constants import doc_type_context
from twindb_lightrag_memgraph.patches import registry

PROCEDURE_ENV_VARS = (
    "TWIN_PROCEDURE",
    "TWIN_PROCEDURE_STORE_FILE",
    "TWIN_PROCEDURE_RENDER_SCALE",
    "TWIN_PROCEDURE_MAX_SCHEMATICS",
    "TWIN_PROCEDURE_MAX_BYTES",
    "TWIN_VISION",
    "TWIN_VISION_BASE_URL",
    "TWIN_VISION_MODEL",
)


@pytest.fixture(autouse=True)
def _clean_procedure_env(monkeypatch, tmp_path):
    for var in PROCEDURE_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv(
        "TWIN_PROCEDURE_STORE_FILE", str(tmp_path / "bundles" / "store.json")
    )
    _procedure.reset_caches()
    _vision.reset_caches()
    yield
    _procedure.reset_caches()
    _vision.reset_caches()


@pytest.fixture
def profile_ready(monkeypatch):
    """Force every availability probe green (no optional dep needed)."""
    monkeypatch.setattr(_procedure, "_pdfium_importable", lambda: True)
    monkeypatch.setattr(_procedure, "_pypdf_importable", lambda: True)
    monkeypatch.setattr(_vision, "is_enabled", lambda: True)


def _head_text() -> str:
    return "\n".join("\n".join(page) for page in PROCEDURE_PAGES[:2])


def _write(tmp_path, name, data):
    path = tmp_path / name
    path.write_bytes(data)
    return path


def _scripted_vision(monkeypatch, fail_stage=None):
    """Monkeypatch render + vision chat with scripted JSON replies."""
    calls = []

    def chat(messages):
        system = messages[0]["content"]
        if system == _procedure.BLIND_SYSTEM_PROMPT:
            stage = "blind"
        elif system == _procedure.INFORMED_SYSTEM_PROMPT:
            stage = "informed"
        else:
            stage = "comparator"
        calls.append((stage, messages))
        if fail_stage == stage:
            raise RuntimeError("endpoint down")
        if stage == "comparator":
            return json.dumps({"coherent": True, "divergences": [], "summary": "ok"})
        return json.dumps(
            {"title": f"{stage} title", "description": f"{stage} desc", "tasks": []}
        )

    monkeypatch.setattr(_vision, "vision_chat_sync", chat)
    monkeypatch.setattr(
        _procedure, "_render_page_png_sync", lambda _p, _i: b"png-bytes"
    )
    return calls


# ---------------------------------------------------------------------------
# Tier gates
# ---------------------------------------------------------------------------


def test_mode_off_disables_even_when_ready(profile_ready, monkeypatch):
    monkeypatch.setenv("TWIN_PROCEDURE", "off")
    assert _procedure.is_enabled() is False


def test_mode_auto_requires_deps_and_vision(monkeypatch):
    monkeypatch.setattr(_procedure, "_pdfium_importable", lambda: False)
    monkeypatch.setattr(_procedure, "_pypdf_importable", lambda: True)
    monkeypatch.setattr(_vision, "is_enabled", lambda: True)
    assert _procedure.is_enabled() is False
    monkeypatch.setattr(_procedure, "_pdfium_importable", lambda: True)
    assert _procedure.is_enabled() is True
    monkeypatch.setattr(_vision, "is_enabled", lambda: False)
    assert _procedure.is_enabled() is False


def test_mode_on_unusable_warns_once(monkeypatch, caplog):
    monkeypatch.setenv("TWIN_PROCEDURE", "on")
    monkeypatch.setattr(_procedure, "_pdfium_importable", lambda: False)
    monkeypatch.setattr(_procedure, "_pypdf_importable", lambda: False)
    monkeypatch.setattr(_vision, "is_enabled", lambda: False)
    with caplog.at_level("WARNING"):
        assert _procedure.is_enabled() is False
        assert _procedure.is_enabled() is False
    warnings = [r for r in caplog.records if "[procedure] extra" in r.getMessage()]
    assert len(warnings) == 1


def test_numeric_envs_fall_back_on_garbage(monkeypatch):
    monkeypatch.setenv("TWIN_PROCEDURE_RENDER_SCALE", "garbage")
    monkeypatch.setenv("TWIN_PROCEDURE_MAX_SCHEMATICS", "-3")
    monkeypatch.setenv("TWIN_PROCEDURE_MAX_BYTES", "zero")
    assert _procedure.render_scale() == _procedure.DEFAULT_RENDER_SCALE
    assert _procedure.max_schematics() == _procedure.DEFAULT_MAX_SCHEMATICS
    assert _procedure.max_procedure_bytes() == _procedure.DEFAULT_MAX_BYTES
    monkeypatch.setenv("TWIN_PROCEDURE_RENDER_SCALE", "40")  # out of range
    assert _procedure.render_scale() == _procedure.DEFAULT_RENDER_SCALE


# ---------------------------------------------------------------------------
# Deterministic detection + schematic location
# ---------------------------------------------------------------------------


def test_detects_template_markers():
    assert _procedure.detect_procedure(_head_text()) is True


def test_reference_alone_is_not_enough():
    assert _procedure.detect_procedure("see ITG0162 for details") is False


def test_itgc_reference_with_full_signature_detects():
    text = "ITGC0094-PRO-CIM procedure\nLevel 2\n4- Operational procedures"
    assert _procedure.detect_procedure(text) is True


def test_partial_signature_is_rejected():
    """Detection is the CONJUNCTION of the template markers — a document
    merely quoting an ITG code plus one stray marker must not match."""
    assert _procedure.detect_procedure("ITG0162 incident\nLevel 2 only") is False
    assert (
        _procedure.detect_procedure("ITG0162 ref\n4- Operational procedures") is False
    )
    assert _procedure.detect_procedure("ITG0162 quoted\nIT GROUP footer") is False


def test_plain_text_not_detected():
    assert _procedure.detect_procedure("Quarterly report\nLevel 2 support") is False
    assert _procedure.detect_procedure("") is False


def test_find_schematic_pages():
    pages = ["cover", "Schematic: Qualify", "text", "schematic: Close", ""]
    assert _procedure.find_schematic_pages(pages) == [1, 3]


def test_should_consider_gates(profile_ready, monkeypatch, tmp_path):
    pdf = _write(tmp_path, "doc.pdf", b"%PDF-1.4 fake")
    assert _procedure.should_consider(pdf) is True
    # forced-standard bypass
    with doc_type_context("standard"):
        assert _procedure.should_consider(pdf) is False
    # non-pdf extension
    other = _write(tmp_path, "doc.docx", b"bytes")
    assert _procedure.should_consider(other) is False
    # size cap
    monkeypatch.setenv("TWIN_PROCEDURE_MAX_BYTES", "4")
    assert _procedure.should_consider(pdf) is False
    # missing file
    monkeypatch.delenv("TWIN_PROCEDURE_MAX_BYTES")
    assert _procedure.should_consider(tmp_path / "ghost.pdf") is False


def test_should_consider_always_claims_forced_documents(
    profile_ready, monkeypatch, tmp_path
):
    """A forced document is claimed even when the cheap gates would reject
    it — the refusal must be an explicit failed bundle, never a silent
    standard enqueue."""
    monkeypatch.setenv("TWIN_PROCEDURE_MAX_BYTES", "4")
    oversized = _write(tmp_path, "big.pdf", b"%PDF-1.4 way too big")
    other = _write(tmp_path, "doc.docx", b"bytes")
    with doc_type_context("procedure"):
        assert _procedure.should_consider(oversized) is True
        assert _procedure.should_consider(other) is True


async def test_route_check_guards_parked_file_failing_gates(
    profile_ready, monkeypatch, tmp_path
):
    """A file the store already claimed keeps routing to the profile even
    when the cheap auto gates reject it on a later scan (finding: forced
    oversized/non-detectable doc rescanned without its header)."""
    pdf = _write(tmp_path, "doc.pdf", b"%PDF-1.4 parked")
    monkeypatch.setenv("TWIN_PROCEDURE_MAX_BYTES", "4")
    assert _procedure.should_consider(pdf) is False
    assert await _procedure.aroute_check(pdf) is False  # nothing parked yet

    _park_minimal(original_path=str(pdf), content_hash=None)
    assert await _procedure.aroute_check(pdf) is True
    # Explicit operator override stays honored.
    with doc_type_context("standard"):
        assert await _procedure.aroute_check(pdf) is False


# ---------------------------------------------------------------------------
# Fixture PDF ↔ real pypdf roundtrip
# ---------------------------------------------------------------------------


def test_fixture_roundtrip_through_pypdf(tmp_path):
    pytest.importorskip("pypdf")
    path = _write(tmp_path, "itg0162.pdf", build_procedure_pdf())
    pages = _procedure._extract_pages_text_sync(path)
    assert pages is not None and len(pages) == len(PROCEDURE_PAGES)
    assert _procedure.detect_procedure("\n".join(pages[:2])) is True
    assert _procedure.find_schematic_pages(pages) == list(PROCEDURE_SCHEMATIC_PAGES)


def test_plain_fixture_not_detected(tmp_path):
    pytest.importorskip("pypdf")
    path = _write(tmp_path, "report.pdf", build_plain_pdf())
    pages = _procedure._extract_pages_text_sync(path)
    assert pages is not None
    assert _procedure.detect_procedure("\n".join(pages[:2])) is False


def test_render_page_png_real(tmp_path):
    """Real pypdfium2 render of the synthetic fixture (skipped without dep)."""
    pytest.importorskip("pypdfium2")
    pytest.importorskip("PIL")
    path = _write(tmp_path, "itg0162.pdf", build_procedure_pdf())
    png = _procedure._render_page_png_sync(path, PROCEDURE_SCHEMATIC_PAGES[0])
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(png) > 500


# ---------------------------------------------------------------------------
# Bundle store
# ---------------------------------------------------------------------------


def _park_minimal(state="pending", **overrides):
    fields = dict(
        file_name="doc.pdf",
        original_path="/inputs/doc.pdf",
        track_id="tid",
        state=state,
        reason="ok",
        source="detected",
        folder="default",
        content_hash="hash-1",
        full_text="text",
        schematics=[],
        classification=None,
    )
    fields.update(overrides)
    return _procedure_store.create_bundle(**fields)


def test_store_crud_roundtrip():
    bundle_id = _park_minimal()
    bundle = _procedure_store.get_bundle(bundle_id)
    assert bundle["state"] == "pending"
    assert bundle["file_name"] == "doc.pdf"

    updated = _procedure_store.update_bundle(bundle_id, state="rejected")
    assert updated["state"] == "rejected"
    assert _procedure_store.get_bundle(bundle_id)["state"] == "rejected"

    assert _procedure_store.list_bundles(state="rejected")[0]["id"] == bundle_id
    assert _procedure_store.list_bundles(state="pending") == []

    assert _procedure_store.delete_bundle(bundle_id) is True
    assert _procedure_store.get_bundle(bundle_id) is None
    assert _procedure_store.delete_bundle(bundle_id) is False


def test_store_rejects_invalid_states():
    with pytest.raises(ValueError):
        _park_minimal(state="weird")
    bundle_id = _park_minimal()
    with pytest.raises(ValueError):
        _procedure_store.update_bundle(bundle_id, state="weird")


def test_store_update_unknown_id_returns_none():
    assert _procedure_store.update_bundle("nope", state="rejected") is None


def test_store_corrupt_file_is_quarantined_and_degrades_atomically(
    tmp_path, monkeypatch
):
    store_file = tmp_path / "corrupt.json"
    store_file.write_text("{not json", encoding="utf-8")
    monkeypatch.setenv("TWIN_PROCEDURE_STORE_FILE", str(store_file))

    # First read quarantines AND raises — never "restart empty".
    with pytest.raises(_procedure_store.StoreDegradedError):
        _procedure_store.list_bundles()

    # The corrupt bytes were moved aside, not destroyed.
    quarantined = list(tmp_path.glob("corrupt.json.corrupt-*"))
    assert len(quarantined) == 1
    assert quarantined[0].read_text(encoding="utf-8") == "{not json"

    # Mutations refuse under the same lock while the marker exists: no
    # second empty store can be written next to the quarantined truth.
    with pytest.raises(_procedure_store.StoreDegradedError):
        _park_minimal()
    assert not store_file.exists()

    # Explicit recovery: the operator removes the quarantine files.
    quarantined[0].unlink()
    assert _procedure_store.is_degraded() is False
    bundle_id = _park_minimal()
    assert _procedure_store.get_bundle(bundle_id) is not None
    assert json.loads(store_file.read_text(encoding="utf-8"))["version"] == 1


def test_store_reserve_is_get_or_create():
    first, created1 = _procedure_store.reserve_bundle(
        content_hash="abc",
        file_name="a.pdf",
        original_path="/inputs/a.pdf",
        track_id="t1",
        source="detected",
        folder="f1",
        operator_classification=None,
    )
    assert created1 is True
    assert first["state"] == "processing"

    # Same content, other path/folder -> the existing bundle is returned and
    # the new path is recorded as a duplicate request, NOT a new run.
    second, created2 = _procedure_store.reserve_bundle(
        content_hash="abc",
        file_name="b.pdf",
        original_path="/inputs/b.pdf",
        track_id="t2",
        source="detected",
        folder="f2",
        operator_classification=None,
    )
    assert created2 is False
    assert second["id"] == first["id"]
    stored = _procedure_store.get_bundle(first["id"])
    request = stored["duplicate_requests"][0]
    assert request["path"] == "/inputs/b.pdf"
    assert request["folder"] == "f2"
    assert request["track_id"] == "t2"

    # Same (path, folder) again -> no duplicate entry.
    _procedure_store.reserve_bundle(
        content_hash="abc",
        file_name="b.pdf",
        original_path="/inputs/b.pdf",
        track_id="t3",
        source="detected",
        folder="f2",
        operator_classification=None,
    )
    stored = _procedure_store.get_bundle(first["id"])
    assert len(stored["duplicate_requests"]) == 1

    with pytest.raises(ValueError):
        _procedure_store.reserve_bundle(
            content_hash="",
            file_name="c.pdf",
            original_path="/inputs/c.pdf",
            track_id=None,
            source="detected",
            folder=None,
            operator_classification=None,
        )


def test_store_find_bundles_by_path_matches_duplicates_too():
    bundle_id = _park_minimal(original_path="/inputs/a.pdf")
    _procedure_store.record_request(
        bundle_id,
        path="/inputs/b.pdf",
        folder="f2",
        track_id=None,
        operator_classification=None,
        file_name="b.pdf",
    )

    assert _procedure_store.find_bundles_by_path("/inputs/a.pdf")[0]["id"] == bundle_id
    assert _procedure_store.find_bundles_by_path("/inputs/b.pdf")[0]["id"] == bundle_id
    assert _procedure_store.find_bundles_by_path("/inputs/zz.pdf") == []
    assert _procedure_store.find_bundles_by_path("") == []


def test_store_claimed_paths_cache_follows_writes():
    assert _procedure_store.claimed_paths() == frozenset()
    bundle_id = _park_minimal(original_path="/inputs/a.pdf")
    assert "/inputs/a.pdf" in _procedure_store.claimed_paths()
    # Cached read (same mtime/size) then invalidation on the next write.
    assert "/inputs/a.pdf" in _procedure_store.claimed_paths()
    _procedure_store.delete_bundle(bundle_id)
    assert _procedure_store.claimed_paths() == frozenset()


def test_store_degraded_lifecycle(tmp_path, monkeypatch):
    store_file = tmp_path / "store.json"
    monkeypatch.setenv("TWIN_PROCEDURE_STORE_FILE", str(store_file))
    assert _procedure_store.is_degraded() is False

    store_file.write_text("{not json", encoding="utf-8")
    with pytest.raises(_procedure_store.StoreDegradedError):
        _procedure_store.list_bundles()  # quarantines and raises
    assert _procedure_store.is_degraded() is True
    # Reads keep raising while the marker exists (fresh empty file or not).
    with pytest.raises(_procedure_store.StoreDegradedError):
        _procedure_store.list_bundles()
    with pytest.raises(_procedure_store.StoreDegradedError):
        _procedure_store.claimed_paths()

    # Explicit recovery: the operator removes the quarantine files.
    for quarantined in tmp_path.glob("store.json.corrupt-*"):
        quarantined.unlink()
    assert _procedure_store.is_degraded() is False
    assert _procedure_store.list_bundles() == []


def test_store_record_request_raises_on_missing_bundle():
    with pytest.raises(LookupError):
        _procedure_store.record_request(
            "ghost",
            path="/inputs/a.pdf",
            folder="f1",
            track_id=None,
            operator_classification=None,
            file_name="a.pdf",
        )


def test_store_known_key_merges_stricter_classification():
    """A C2 re-request behind a C1 on the same (path, folder) key must raise
    the recorded classification — primary request included."""
    first, _ = _procedure_store.reserve_bundle(
        content_hash="merge",
        file_name="a.pdf",
        original_path="/inputs/a.pdf",
        track_id="t1",
        source="detected",
        folder="f1",
        operator_classification="C1",
    )
    # Same key as the PRIMARY request, stricter class -> bundle raised.
    _procedure_store.reserve_bundle(
        content_hash="merge",
        file_name="a.pdf",
        original_path="/inputs/a.pdf",
        track_id="t2",
        source="detected",
        folder="f1",
        operator_classification="C2",
    )
    stored = _procedure_store.get_bundle(first["id"])
    assert stored["operator_classification"] == "C2"
    assert not stored.get("duplicate_requests")

    # Duplicate-request key: C1 recorded, then C2 merges in place.
    _procedure_store.reserve_bundle(
        content_hash="merge",
        file_name="b.pdf",
        original_path="/inputs/b.pdf",
        track_id="t3",
        source="detected",
        folder="f2",
        operator_classification="C1",
    )
    _procedure_store.reserve_bundle(
        content_hash="merge",
        file_name="b.pdf",
        original_path="/inputs/b.pdf",
        track_id="t4",
        source="detected",
        folder="f2",
        operator_classification="C2",
    )
    stored = _procedure_store.get_bundle(first["id"])
    assert len(stored["duplicate_requests"]) == 1
    assert stored["duplicate_requests"][0]["operator_classification"] == "C2"
    # And a weaker class never downgrades.
    _procedure_store.reserve_bundle(
        content_hash="merge",
        file_name="b.pdf",
        original_path="/inputs/b.pdf",
        track_id="t5",
        source="detected",
        folder="f2",
        operator_classification="C1",
    )
    stored = _procedure_store.get_bundle(first["id"])
    assert stored["duplicate_requests"][0]["operator_classification"] == "C2"


def test_store_scan_reservation_records_no_folder():
    """Anything discovered via /documents/scan enters the claim index but
    never carries the scan's folder (no silent membership grant)."""
    novel, created = _procedure_store.reserve_bundle(
        content_hash="novel-scan",
        file_name="novel.pdf",
        original_path="/inputs/novel.pdf",
        track_id="t0",
        source="detected",
        folder="f_scan",
        operator_classification=None,
        via_scan=True,
    )
    assert created is True
    assert novel["folder"] is None

    first, _ = _procedure_store.reserve_bundle(
        content_hash="scan",
        file_name="a.pdf",
        original_path="/inputs/a.pdf",
        track_id="t1",
        source="detected",
        folder="f1",
        operator_classification=None,
    )
    _procedure_store.reserve_bundle(
        content_hash="scan",
        file_name="copy.pdf",
        original_path="/inputs/copy.pdf",
        track_id="t2",
        source="detected",
        folder="f_scan",
        operator_classification=None,
        via_scan=True,
    )
    stored = _procedure_store.get_bundle(first["id"])
    assert stored["duplicate_requests"][0]["path"] == "/inputs/copy.pdf"
    assert stored["duplicate_requests"][0]["folder"] is None
    # The path is guarded against later rescans.
    assert _procedure_store.find_bundles_by_path("/inputs/copy.pdf")


def _mp_reserve(store_file):
    """Worker for the multiprocess reservation race (spawn-imported)."""
    import os as _os

    _os.environ["TWIN_PROCEDURE_STORE_FILE"] = store_file
    from twindb_lightrag_memgraph import _procedure_store as store

    bundle, created = store.reserve_bundle(
        content_hash="mp-hash",
        file_name="doc.pdf",
        original_path="/inputs/doc.pdf",
        track_id=None,
        source="detected",
        folder="default",
        operator_classification=None,
    )
    return bundle["id"], created


def test_store_reserve_is_atomic_across_processes(tmp_path):
    """Real multi-process race: N workers reserving the same content hash
    must yield exactly ONE created reservation (flock get-or-create)."""
    import multiprocessing

    store_file = str(tmp_path / "mp-store.json")
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(4) as pool:
        results = pool.map(_mp_reserve, [store_file] * 8)

    assert len({bundle_id for bundle_id, _ in results}) == 1
    assert sum(1 for _, created in results if created) == 1


# ---------------------------------------------------------------------------
# Orchestration (real pypdf text, scripted render + vision)
# ---------------------------------------------------------------------------


async def test_detected_procedure_parks_pending_bundle(monkeypatch, tmp_path):
    pytest.importorskip("pypdf")
    calls = _scripted_vision(monkeypatch)
    path = _write(tmp_path, "itg0162.pdf", build_procedure_pdf())

    outcome = await _procedure.aprocess_procedure(path, "tid-1")

    assert outcome is not None and outcome.state == "pending"
    bundle = _procedure_store.get_bundle(outcome.bundle_id)
    assert bundle["state"] == "pending"
    assert bundle["source"] == "detected"
    assert bundle["track_id"] == "tid-1"
    assert "Categorize and enrich" in bundle["full_text"]
    assert len(bundle["schematics"]) == len(PROCEDURE_SCHEMATIC_PAGES)
    for entry, page_index in zip(
        bundle["schematics"], PROCEDURE_SCHEMATIC_PAGES, strict=True
    ):
        assert entry["page"] == page_index + 1
        assert base64.b64decode(entry["png_base64"]) == b"png-bytes"
        assert entry["blind"]["title"] == "blind title"
        assert entry["informed"]["title"] == "informed title"
        assert entry["divergence"]["coherent"] is True
        assert entry["error"] is None

    # Context asymmetry contract: the blind pass never sees the document
    # text; the informed pass and the comparator always do.
    by_stage = {}
    for stage, messages in calls:
        by_stage.setdefault(stage, []).append(json.dumps(messages))
    marker = "Categorize and enrich"
    assert all(marker not in m for m in by_stage["blind"])
    assert all(marker in m for m in by_stage["informed"])
    assert all(marker in m for m in by_stage["comparator"])


async def test_plain_pdf_returns_none_without_bundle(monkeypatch, tmp_path):
    pytest.importorskip("pypdf")
    _scripted_vision(monkeypatch)
    path = _write(tmp_path, "report.pdf", build_plain_pdf())
    assert await _procedure.aprocess_procedure(path, "tid-2") is None
    assert _procedure_store.list_bundles() == []


async def test_forced_procedure_parks_undetected_pdf(monkeypatch, tmp_path):
    pytest.importorskip("pypdf")
    _scripted_vision(monkeypatch)
    path = _write(tmp_path, "report.pdf", build_plain_pdf())
    with doc_type_context("procedure"):
        outcome = await _procedure.aprocess_procedure(path, "tid-3")
    assert outcome is not None
    assert _procedure_store.get_bundle(outcome.bundle_id)["source"] == "forced"


async def test_vision_failure_parks_failed_bundle_with_partials(monkeypatch, tmp_path):
    pytest.importorskip("pypdf")
    _scripted_vision(monkeypatch, fail_stage="informed")
    path = _write(tmp_path, "itg0162.pdf", build_procedure_pdf())

    outcome = await _procedure.aprocess_procedure(path, "tid-4")

    assert outcome is not None and outcome.state == "failed"
    bundle = _procedure_store.get_bundle(outcome.bundle_id)
    assert bundle["state"] == "failed"
    assert "endpoint down" in bundle["reason"]
    # Partial results are preserved for the review/retry UI: the informed
    # pass failed, but the render, the blind pass AND the comparator (which
    # depends only on the blind pass) results are all kept.
    entry = bundle["schematics"][0]
    assert entry["png_base64"] is not None
    assert entry["blind"] is not None
    assert entry["informed"] is None
    assert entry["divergence"] is not None
    assert entry["error"] is not None


async def test_task_entries_validated_to_the_eight_fields(monkeypatch, tmp_path):
    """tasks=[{}] parses as JSON but violates the prompt contract."""
    pytest.importorskip("pypdf")

    def sloppy_chat(_messages):
        return json.dumps({"title": "t", "description": "d", "tasks": [{}]})

    monkeypatch.setattr(_vision, "vision_chat_sync", sloppy_chat)
    monkeypatch.setattr(_procedure, "_render_page_png_sync", lambda _p, _i: b"png")
    path = _write(tmp_path, "itg0162.pdf", build_procedure_pdf())

    outcome = await _procedure.aprocess_procedure(path, "tid-21")

    assert outcome is not None and outcome.state == "failed"
    assert "eight string fields" in outcome.reason


async def test_invalid_llm_reply_shape_parks_failed(monkeypatch, tmp_path):
    """A reply that parses as JSON but lacks the contract fields is refused."""
    pytest.importorskip("pypdf")

    def bad_chat(_messages):
        return json.dumps({"unexpected": "shape"})

    monkeypatch.setattr(_vision, "vision_chat_sync", bad_chat)
    monkeypatch.setattr(_procedure, "_render_page_png_sync", lambda _p, _i: b"png")
    path = _write(tmp_path, "itg0162.pdf", build_procedure_pdf())

    outcome = await _procedure.aprocess_procedure(path, "tid-12")

    assert outcome is not None and outcome.state == "failed"
    assert "expected shape" in outcome.reason


async def test_text_extraction_failure(monkeypatch, tmp_path):
    _scripted_vision(monkeypatch)
    monkeypatch.setattr(
        _procedure, "_extract_pages_text_sync", lambda _p, _limit=None: None
    )
    path = _write(tmp_path, "doc.pdf", b"%PDF-1.4 broken")

    # Auto mode: cannot detect -> standard path, no bundle.
    assert await _procedure.aprocess_procedure(path, "tid-5") is None
    assert _procedure_store.list_bundles() == []

    # Forced mode: the operator asked for the profile -> visible failure.
    with doc_type_context("procedure"):
        outcome = await _procedure.aprocess_procedure(path, "tid-6")
    assert outcome is not None and outcome.state == "failed"
    assert "text-extraction-failed" in outcome.reason


async def test_no_schematic_is_never_approvable(monkeypatch, tmp_path):
    """A procedure whose schematics were not located is the silent-loss case
    this profile exists to prevent — it parks as ``failed``, not pending."""
    pytest.importorskip("pypdf")
    calls = _scripted_vision(monkeypatch)
    path = _write(tmp_path, "itg9999.pdf", build_textonly_procedure_pdf())

    outcome = await _procedure.aprocess_procedure(path, "tid-7")

    assert outcome is not None and outcome.state == "failed"
    assert "no-schematic-found" in outcome.reason
    assert _procedure_store.get_bundle(outcome.bundle_id)["schematics"] == []
    assert calls == []  # no vision spend without a schematic


async def test_schematic_truncation_is_never_silently_approvable(monkeypatch, tmp_path):
    """The cap bounds the vision spend, but a truncated bundle is incomplete
    — it must surface as failed with the true total, not pending/ok."""
    pytest.importorskip("pypdf")
    calls = _scripted_vision(monkeypatch)
    monkeypatch.setenv("TWIN_PROCEDURE_MAX_SCHEMATICS", "1")
    path = _write(tmp_path, "itg0162.pdf", build_procedure_pdf())

    outcome = await _procedure.aprocess_procedure(path, "tid-8")

    assert outcome.state == "failed"
    assert "schematics-truncated" in outcome.reason
    bundle = _procedure_store.get_bundle(outcome.bundle_id)
    assert len(bundle["schematics"]) == 1
    assert bundle["schematics_total"] == len(PROCEDURE_SCHEMATIC_PAGES)
    assert len([c for c in calls if c[0] == "blind"]) == 1


async def test_event_sink_receives_parked_event(monkeypatch, tmp_path):
    pytest.importorskip("pypdf")
    _scripted_vision(monkeypatch)
    events = []
    _procedure.set_event_sink(lambda kind, payload: events.append((kind, payload)))
    path = _write(tmp_path, "itg0162.pdf", build_procedure_pdf())

    outcome = await _procedure.aprocess_procedure(path, "tid-9")

    assert events and events[0][0] == "procedure-parked"
    assert events[0][1]["bundle_id"] == outcome.bundle_id


async def test_failing_event_sink_does_not_break_parking(monkeypatch, tmp_path):
    pytest.importorskip("pypdf")
    _scripted_vision(monkeypatch)

    def broken_sink(_kind, _payload):
        raise RuntimeError("sink down")

    _procedure.set_event_sink(broken_sink)
    path = _write(tmp_path, "itg0162.pdf", build_procedure_pdf())
    outcome = await _procedure.aprocess_procedure(path, "tid-10")
    assert outcome is not None and outcome.state == "pending"


async def test_detection_probe_error_degrades_to_standard_path(monkeypatch, tmp_path):
    """Selection-phase errors (auto mode) mean "cannot claim" -> standard."""

    def boom(_p, _limit=None):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(_procedure, "_extract_pages_text_sync", boom)
    path = _write(tmp_path, "doc.pdf", b"%PDF-1.4")
    assert await _procedure.aprocess_procedure(path, "tid-11") is None


async def test_processing_error_fails_closed_with_bundle(monkeypatch, tmp_path):
    """Once the profile claims the document, an unexpected error must park a
    failed bundle — never fall through to the standard enqueue."""
    pytest.importorskip("pypdf")
    _scripted_vision(monkeypatch)

    def boom(_p):
        raise RuntimeError("hash exploded")

    monkeypatch.setattr(_procedure, "_content_hash_sync", boom)
    path = _write(tmp_path, "itg0162.pdf", build_procedure_pdf())

    outcome = await _procedure.aprocess_procedure(path, "tid-13")

    assert outcome is not None and outcome.state == "failed"
    assert "hash exploded" in outcome.reason
    bundle = _procedure_store.get_bundle(outcome.bundle_id)
    assert bundle["state"] == "failed"


async def test_store_failure_yields_error_outcome(monkeypatch, tmp_path):
    """Store fully down -> "error" outcome (the seam refuses the enqueue)."""
    pytest.importorskip("pypdf")
    _scripted_vision(monkeypatch)

    def refuse(**_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(_procedure._procedure_store, "reserve_bundle", refuse)
    monkeypatch.setattr(_procedure._procedure_store, "create_bundle", refuse)
    path = _write(tmp_path, "itg0162.pdf", build_procedure_pdf())

    outcome = await _procedure.aprocess_procedure(path, "tid-14")

    assert outcome is not None
    assert outcome.state == "error"
    assert outcome.bundle_id is None


async def test_rescan_reuses_active_bundle(monkeypatch, tmp_path):
    """Idempotence: a rescan of the still-parked original must not re-burn
    render + LLM calls into a duplicate bundle."""
    pytest.importorskip("pypdf")
    calls = _scripted_vision(monkeypatch)
    path = _write(tmp_path, "itg0162.pdf", build_procedure_pdf())

    first = await _procedure.aprocess_procedure(path, "tid-15")
    spent = len(calls)
    second = await _procedure.aprocess_procedure(path, "tid-16")

    assert second is not None
    assert second.bundle_id == first.bundle_id
    assert second.reason.startswith("already-parked")
    assert len(calls) == spent  # zero additional vision spend
    assert len(_procedure_store.list_bundles()) == 1


async def test_forced_then_rescan_without_header_reuses_bundle(monkeypatch, tmp_path):
    """The finding-1 scenario: a forced, auto-undetectable PDF is parked;
    a later scan WITHOUT the header must reuse the bundle — never slip
    through auto-detection into the standard (unapproved) enqueue."""
    pytest.importorskip("pypdf")
    _scripted_vision(monkeypatch)
    path = _write(tmp_path, "report.pdf", build_plain_pdf())

    with doc_type_context("procedure"):
        first = await _procedure.aprocess_procedure(path, "tid-22")
    assert first is not None

    second = await _procedure.aprocess_procedure(path, "tid-23")  # no header
    assert second is not None, "parked document escaped to the standard path"
    assert second.bundle_id == first.bundle_id
    assert len(_procedure_store.list_bundles()) == 1


async def test_rejected_is_terminal_until_retry(monkeypatch, tmp_path):
    """A rescan of a rejected document must NOT re-run vision or resurrect
    the bundle — relaunch is exclusively the PR 2 retry action."""
    pytest.importorskip("pypdf")
    calls = _scripted_vision(monkeypatch)
    path = _write(tmp_path, "itg0162.pdf", build_procedure_pdf())

    first = await _procedure.aprocess_procedure(path, "tid-24")
    _procedure_store.update_bundle(first.bundle_id, state="rejected")
    spent = len(calls)

    second = await _procedure.aprocess_procedure(path, "tid-25")

    assert second is not None and second.state == "rejected"
    assert second.bundle_id == first.bundle_id
    assert len(calls) == spent
    assert len(_procedure_store.list_bundles()) == 1


async def test_same_content_other_path_is_a_duplicate_request(monkeypatch, tmp_path):
    """Same bytes uploaded under a new name: recorded on the existing
    bundle as a duplicate request, zero re-spend, no second bundle."""
    pytest.importorskip("pypdf")
    calls = _scripted_vision(monkeypatch)
    pdf_bytes = build_procedure_pdf()
    path_a = _write(tmp_path, "itg0162.pdf", pdf_bytes)
    path_b = _write(tmp_path, "itg0162-copy.pdf", pdf_bytes)

    first = await _procedure.aprocess_procedure(path_a, "tid-26")
    spent = len(calls)
    second = await _procedure.aprocess_procedure(path_b, "tid-27")

    assert second.bundle_id == first.bundle_id
    assert len(calls) == spent
    bundle = _procedure_store.get_bundle(first.bundle_id)
    assert str(path_b) in [r["path"] for r in bundle["duplicate_requests"]]
    # And the duplicate path is now guarded against rescans too.
    third = await _procedure.aprocess_procedure(path_b, "tid-28")
    assert third.bundle_id == first.bundle_id


async def test_pre_guard_reuse_records_other_folder_request(monkeypatch, tmp_path):
    """A same-path reuse from another folder must not lose the new request's
    folder / track / operator classification (PR 2 membership + gate)."""
    pytest.importorskip("pypdf")
    from twindb_lightrag_memgraph._constants import (
        operator_classification_context,
        storage_folder_context,
    )

    _scripted_vision(monkeypatch)
    path = _write(tmp_path, "itg0162.pdf", build_procedure_pdf())

    with storage_folder_context("folder_a"):
        first = await _procedure.aprocess_procedure(path, "tid-33")
    with (
        storage_folder_context("folder_b"),
        operator_classification_context("C2"),
    ):
        second = await _procedure.aprocess_procedure(path, "tid-34")

    assert second.bundle_id == first.bundle_id
    bundle = _procedure_store.get_bundle(first.bundle_id)
    assert bundle["folder"] == "folder_a"
    request = bundle["duplicate_requests"][0]
    assert request["folder"] == "folder_b"
    assert request["operator_classification"] == "C2"
    assert request["track_id"] == "tid-34"


async def test_scan_reuse_never_records_membership_request(monkeypatch, tmp_path):
    """A rescan from another folder is NOT an ingestion request: reusing
    the bundle must not record the scan's captured folder (that would
    silently grant a future membership in whatever folder the global scan
    happened to run under)."""
    pytest.importorskip("pypdf")
    from twindb_lightrag_memgraph._constants import storage_folder_context

    _scripted_vision(monkeypatch)
    path = _write(tmp_path, "itg0162.pdf", build_procedure_pdf())

    with storage_folder_context("folder_a"):
        first = await _procedure.aprocess_procedure(path, "tid-38")
    with storage_folder_context("folder_b"):
        second = await _procedure.aprocess_procedure(path, "tid-39", from_scan=True)

    assert second.bundle_id == first.bundle_id
    bundle = _procedure_store.get_bundle(first.bundle_id)
    assert not bundle.get("duplicate_requests")


async def test_reuse_fails_closed_when_request_cannot_be_recorded(
    monkeypatch, tmp_path
):
    """An operator upload whose duplicate request cannot be persisted must
    NOT be announced as accepted — folder/classification would be lost."""
    pytest.importorskip("pypdf")
    from twindb_lightrag_memgraph._constants import storage_folder_context

    _scripted_vision(monkeypatch)
    path = _write(tmp_path, "itg0162.pdf", build_procedure_pdf())
    with storage_folder_context("folder_a"):
        first = await _procedure.aprocess_procedure(path, "tid-40")

    def refuse(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(_procedure._procedure_store, "record_request", refuse)
    with storage_folder_context("folder_b"):
        second = await _procedure.aprocess_procedure(path, "tid-41")

    assert second is not None and second.state == "error"
    assert "duplicate-request-persist-failed" in second.reason
    assert second.bundle_id == first.bundle_id


async def test_settle_lost_reservation_is_an_error(monkeypatch, tmp_path):
    """A reservation that vanished mid-flight must surface as error, never
    be announced pending (nothing was persisted)."""
    pytest.importorskip("pypdf")
    _scripted_vision(monkeypatch)
    monkeypatch.setattr(
        _procedure._procedure_store, "update_bundle", lambda *_a, **_k: None
    )
    path = _write(tmp_path, "itg0162.pdf", build_procedure_pdf())

    outcome = await _procedure.aprocess_procedure(path, "tid-42")

    assert outcome is not None and outcome.state == "error"
    assert "settle-lost" in outcome.reason


async def test_aprocess_refuses_when_store_unreadable(monkeypatch, tmp_path):
    """Guard IO errors fail CLOSED: error outcome, never auto-detection."""

    def boom():
        raise OSError("io error")

    monkeypatch.setattr(_procedure._procedure_store, "is_degraded", boom)
    path = _write(tmp_path, "doc.pdf", b"%PDF-1.4")

    outcome = await _procedure.aprocess_procedure(path, "tid-35")

    assert outcome is not None and outcome.state == "error"
    assert "store-unreadable" in outcome.reason


async def test_route_check_fails_closed_on_store_error(
    profile_ready, monkeypatch, tmp_path
):
    def boom():
        raise OSError("io error")

    monkeypatch.setattr(_procedure._procedure_store, "claimed_paths", boom)
    monkeypatch.setenv("TWIN_PROCEDURE_MAX_BYTES", "4")
    pdf = _write(tmp_path, "doc.pdf", b"%PDF-1.4 over the tiny cap")
    assert await _procedure.aroute_check(pdf) is True


async def test_forced_oversized_parks_failed_not_standard(monkeypatch, tmp_path):
    """Finding 4: an explicitly declared procedure above the size cap must
    produce an explicit failed bundle, never be indexed without approval."""
    _scripted_vision(monkeypatch)
    monkeypatch.setenv("TWIN_PROCEDURE_MAX_BYTES", "4")
    path = _write(tmp_path, "big.pdf", b"%PDF-1.4 far too big for the cap")

    with doc_type_context("procedure"):
        outcome = await _procedure.aprocess_procedure(path, "tid-29")

    assert outcome is not None and outcome.state == "failed"
    assert "file-too-large" in outcome.reason


async def test_forced_non_pdf_parks_failed_not_standard(monkeypatch, tmp_path):
    _scripted_vision(monkeypatch)
    path = _write(tmp_path, "doc.docx", b"not a pdf")

    with doc_type_context("procedure"):
        outcome = await _procedure.aprocess_procedure(path, "tid-30")

    assert outcome is not None and outcome.state == "failed"
    assert "unsupported-extension" in outcome.reason


async def test_operator_classification_persisted_in_bundle(monkeypatch, tmp_path):
    """The X-Twin-Classification choice dies with the request context — the
    bundle must carry it for the PR 2 approve-time MIP gate."""
    pytest.importorskip("pypdf")
    from twindb_lightrag_memgraph._constants import operator_classification_context

    _scripted_vision(monkeypatch)
    path = _write(tmp_path, "itg0162.pdf", build_procedure_pdf())

    with operator_classification_context("C2"):
        outcome = await _procedure.aprocess_procedure(path, "tid-17")

    bundle = _procedure_store.get_bundle(outcome.bundle_id)
    assert bundle["operator_classification"] == "C2"
    assert bundle["content_hash"]


async def test_standard_pdf_only_pays_detection_pages(monkeypatch, tmp_path):
    """A non-procedure PDF must never pay a full-document text extraction."""
    pytest.importorskip("pypdf")
    real = _procedure._extract_pages_text_sync
    limits = []

    def spy(path, limit=None):
        limits.append(limit)
        return real(path, limit)

    monkeypatch.setattr(_procedure, "_extract_pages_text_sync", spy)
    path = _write(tmp_path, "report.pdf", build_plain_pdf())

    assert await _procedure.aprocess_procedure(path, "tid-18") is None
    assert limits == [_procedure.DETECTION_PAGES]


# ---------------------------------------------------------------------------
# Seam-contract tests against the real document_routes module
# ---------------------------------------------------------------------------


@pytest.fixture
def dr_module(monkeypatch):
    """Real ``document_routes`` module with full state restore (family of
    the ``test_conversion.py`` fixture — the module is process-shared)."""
    monkeypatch.setattr(sys, "argv", ["pytest"])
    dr = pytest.importorskip(
        "lightrag.api.routers.document_routes",
        reason="native document routes unavailable on this LightRAG",
    )
    saved = {"pipeline_enqueue_file": dr.pipeline_enqueue_file}
    sentinel = "_twindb_convert_enqueue_patched"
    had_sentinel = hasattr(dr, sentinel)
    saved_sentinel = getattr(dr, sentinel, None)
    had_generate = hasattr(dr, "generate_track_id")
    saved_generate = getattr(dr, "generate_track_id", None)
    yield dr
    dr.pipeline_enqueue_file = saved["pipeline_enqueue_file"]
    if had_sentinel:
        setattr(dr, sentinel, saved_sentinel)
    elif hasattr(dr, sentinel):
        delattr(dr, sentinel)
    if had_generate:
        dr.generate_track_id = saved_generate
    elif hasattr(dr, "generate_track_id"):
        delattr(dr, "generate_track_id")


def _install_enqueue_patch(dr, fake_orig):
    dr.pipeline_enqueue_file = fake_orig
    dr._twindb_convert_enqueue_patched = False
    registry._patch_pipeline_enqueue_conversion()
    assert dr.pipeline_enqueue_file is not fake_orig
    return dr.pipeline_enqueue_file


async def test_seam_parks_procedure_and_skips_enqueue(dr_module, monkeypatch, tmp_path):
    async def fake_orig(rag, file_path, *args, **kwargs):
        raise AssertionError("native path must not run for a parked procedure")

    wrapped = _install_enqueue_patch(dr_module, fake_orig)
    monkeypatch.setattr(_procedure, "should_consider", lambda _p: True)
    seen = {}

    async def fake_process(path, track_id, *, from_scan=False):
        seen["args"] = (path, track_id, from_scan)
        return _procedure.ProcedureOutcome("bundle-1", "pending", "ok")

    monkeypatch.setattr(_procedure, "aprocess_procedure", fake_process)

    path = tmp_path / "itg0162.pdf"
    result = await wrapped(object(), path, "tid-9", from_scan=True)

    assert result == (True, "tid-9")
    assert seen["args"] == (path, "tid-9", True)


async def test_seam_generates_track_id_when_missing(dr_module, monkeypatch, tmp_path):
    async def fake_orig(rag, file_path, *args, **kwargs):
        raise AssertionError("native path must not run")

    wrapped = _install_enqueue_patch(dr_module, fake_orig)
    monkeypatch.setattr(_procedure, "should_consider", lambda _p: True)
    monkeypatch.setattr(
        dr_module, "generate_track_id", lambda prefix: f"{prefix}-gen", raising=False
    )
    captured = {}

    async def fake_process(path, track_id, *, from_scan=False):
        captured["track_id"] = track_id
        return _procedure.ProcedureOutcome("bundle-2", "pending", "ok")

    monkeypatch.setattr(_procedure, "aprocess_procedure", fake_process)

    result = await wrapped(object(), tmp_path / "doc.pdf")
    assert result == (True, "unknown-gen")
    assert captured["track_id"] == "unknown-gen"


async def test_seam_error_outcome_reports_error_document(
    dr_module, monkeypatch, tmp_path
):
    """ "error" outcome (parking impossible) -> explicit FAILED error-doc,
    never a silent fall-through to the standard enqueue (fail closed)."""

    async def fake_orig(rag, file_path, *args, **kwargs):
        raise AssertionError("native path must not run — the gate fails closed")

    wrapped = _install_enqueue_patch(dr_module, fake_orig)
    monkeypatch.setattr(_procedure, "should_consider", lambda _p: True)

    async def fake_process(path, track_id, *, from_scan=False):
        return _procedure.ProcedureOutcome(None, "error", "store down")

    monkeypatch.setattr(_procedure, "aprocess_procedure", fake_process)

    class _Rag:
        def __init__(self):
            self.error_calls = []

        async def apipeline_enqueue_error_documents(self, error_files, track_id):
            self.error_calls.append((error_files, track_id))

    rag = _Rag()
    path = tmp_path / "itg0162.pdf"
    path.write_bytes(b"%PDF-1.4")
    result = await wrapped(rag, path, "tid-20")

    assert result == (False, "tid-20")
    assert len(rag.error_calls) == 1
    assert rag.error_calls[0][0][0]["error_description"] == (
        "Procedure ingestion error"
    )
    assert rag.error_calls[0][0][0]["original_error"] == "store down"


async def test_seam_settle_write_error_reports_real_cause(
    dr_module, profile_ready, monkeypatch, tmp_path
):
    """A failed reservation write must surface its real cause in the FAILED
    error-document, never the successful processing reason ("ok")."""
    pytest.importorskip("pypdf")
    _scripted_vision(monkeypatch)

    async def fake_orig(rag, file_path, *args, **kwargs):
        raise AssertionError("native path must not run — the gate fails closed")

    wrapped = _install_enqueue_patch(dr_module, fake_orig)

    def fail_settle(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(_procedure._procedure_store, "update_bundle", fail_settle)

    class _Rag:
        def __init__(self):
            self.error_calls = []

        async def apipeline_enqueue_error_documents(self, error_files, track_id):
            self.error_calls.append((error_files, track_id))

    rag = _Rag()
    path = _write(tmp_path, "itg0162.pdf", build_procedure_pdf())
    result = await wrapped(rag, path, "tid-43")

    assert result == (False, "tid-43")
    original_error = rag.error_calls[0][0][0]["original_error"]
    assert "settle-persist-failed: OSError: disk full" in original_error


async def test_seam_never_enqueues_claimed_file_when_store_corrupt(
    dr_module, profile_ready, monkeypatch, tmp_path
):
    """Required scenario: a forced, auto-undetectable PDF is parked, the
    store is then corrupted, and a rescan arrives WITHOUT the header and
    failing the cheap gates — the native path must NEVER run; the file
    surfaces as an explicit FAILED error-document until the operator
    recovers the quarantined store."""
    pytest.importorskip("pypdf")
    _scripted_vision(monkeypatch)

    async def fake_orig(rag, file_path, *args, **kwargs):
        raise AssertionError("native path ran: the approval gate failed open")

    wrapped = _install_enqueue_patch(dr_module, fake_orig)
    path = _write(tmp_path, "report.pdf", build_plain_pdf())

    with doc_type_context("procedure"):
        first = await _procedure.aprocess_procedure(path, "tid-36")
    assert first is not None

    # Corrupt the store, and make the rescan fail the cheap auto gates too.
    _procedure_store.store_path().write_text("{not json", encoding="utf-8")
    monkeypatch.setenv("TWIN_PROCEDURE_MAX_BYTES", "4")

    class _Rag:
        def __init__(self):
            self.error_calls = []

        async def apipeline_enqueue_error_documents(self, error_files, track_id):
            self.error_calls.append((error_files, track_id))

    rag = _Rag()
    result = await wrapped(rag, path, "tid-37")

    assert result == (False, "tid-37")
    assert len(rag.error_calls) == 1
    assert _procedure_store.is_degraded() is True


async def test_seam_continues_standard_when_not_a_procedure(
    dr_module, monkeypatch, tmp_path
):
    """``aprocess_procedure`` returning None -> untouched standard routing."""
    seen = {}

    async def fake_orig(rag, file_path, *args, **kwargs):
        seen["call"] = (rag, file_path, args, kwargs)
        return True, "native-track"

    wrapped = _install_enqueue_patch(dr_module, fake_orig)
    monkeypatch.setattr(_procedure, "should_consider", lambda _p: True)

    async def fake_process(path, track_id, *, from_scan=False):
        return None

    monkeypatch.setattr(_procedure, "aprocess_procedure", fake_process)
    monkeypatch.setattr(_conversion, "should_convert", lambda _p: False)
    monkeypatch.setattr(_vision, "should_process", lambda _p: False)

    rag = object()
    path = tmp_path / "report.pdf"
    result = await wrapped(rag, path, "tid-1", from_scan=True)
    assert result == (True, "native-track")
    assert seen["call"] == (rag, path, ("tid-1",), {"from_scan": True})


async def test_seam_off_is_bit_identical(dr_module, monkeypatch, tmp_path):
    """LightRAG-compat contract: profile off -> delegation is verbatim."""
    monkeypatch.setenv("TWIN_PROCEDURE", "off")
    seen = {}

    async def fake_orig(rag, file_path, *args, **kwargs):
        seen["call"] = (rag, file_path, args, kwargs)
        return True, "native-track"

    wrapped = _install_enqueue_patch(dr_module, fake_orig)
    monkeypatch.setattr(_conversion, "should_convert", lambda _p: False)
    monkeypatch.setattr(_vision, "should_process", lambda _p: False)

    rag = object()
    path = tmp_path / "itg0162.pdf"
    result = await wrapped(rag, path, "tid-1", from_scan=True)
    assert result == (True, "native-track")
    assert seen["call"] == (rag, path, ("tid-1",), {"from_scan": True})


# ---------------------------------------------------------------------------
# Middleware: X-Twin-Doc-Type header binding
# ---------------------------------------------------------------------------


async def test_background_task_reapplies_doc_type():
    """The doc-type context must survive the BackgroundTasks boundary where
    LightRAG actually runs ``pipeline_enqueue_file`` — otherwise a forced
    X-Twin-Doc-Type dies with the request and never reaches the seam."""
    from starlette.background import BackgroundTasks

    from twindb_lightrag_memgraph import _patch_background_tasks_folder_context
    from twindb_lightrag_memgraph._constants import get_active_doc_type

    seen: list[str | None] = []

    async def task():
        seen.append(get_active_doc_type())

    _patch_background_tasks_folder_context()
    background = BackgroundTasks()
    with doc_type_context("procedure"):
        background.add_task(task)

    assert get_active_doc_type() is None
    await background()
    assert seen == ["procedure"]


async def test_upload_middleware_binds_doc_type_header(monkeypatch):
    monkeypatch.setenv("TWIN_DEFAULT_FOLDER", "default")
    monkeypatch.setenv(
        "TWIN_FOLDERS_JSON",
        '[{"id":"default","label":"Default","kind":"kb"}]',
    )

    fastapi = pytest.importorskip("fastapi")
    from httpx import ASGITransport, AsyncClient

    from twindb_lightrag_memgraph import _install_storage_folder_capture
    from twindb_lightrag_memgraph._constants import get_active_doc_type

    app = fastapi.FastAPI()
    _install_storage_folder_capture(app)

    @app.post("/documents/upload")
    async def upload_probe():
        return {"doc_type": get_active_doc_type()}

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        forced = await client.post(
            "/documents/upload",
            headers={"X-Twin-Folder": "default", "X-Twin-Doc-Type": "Procedure"},
        )
        standard = await client.post(
            "/documents/upload",
            headers={"X-Twin-Folder": "default", "X-Twin-Doc-Type": "standard"},
        )
        absent = await client.post(
            "/documents/upload", headers={"X-Twin-Folder": "default"}
        )
        rejected = await client.post(
            "/documents/upload",
            headers={"X-Twin-Folder": "default", "X-Twin-Doc-Type": "diagram"},
        )

    assert forced.json() == {"doc_type": "procedure"}
    assert standard.json() == {"doc_type": "standard"}
    # LightRAG-compat: no header -> nothing bound, auto-detection decides.
    assert absent.json() == {"doc_type": None}
    assert rejected.status_code == 400
    assert "accepts only" in rejected.json()["detail"]
