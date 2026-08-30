"""Procedure-PDF ingestion profile (docs/adr/007-procedure-pdf-profile.md).

Unit tests on ``_procedure`` (tier gates, deterministic template detection,
schematic-page location, dual-pass orchestration, failure degradation), the
``_procedure_store`` bundle store, and seam-contract tests on the registry
routing (procedure → parked bundle, NOT enqueued; standard path untouched).
The vision LLM and the pypdfium2 render are always monkeypatched — no
network, no native render. Text extraction runs against the synthetic
template fixture (``tests/procedure_pdf_fixture.py``) through the real
pypdf, so detection is exercised on a genuine PDF text layer.
"""

import asyncio
import base64
import builtins
import functools
import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

from tests.procedure_pdf_fixture import (
    PROCEDURE_PAGES,
    PROCEDURE_SCHEMATIC_PAGES,
    build_plain_pdf,
    build_procedure_pdf,
    build_textonly_procedure_pdf,
)
from twindb_lightrag_memgraph import (
    _conversion,
    _procedure,
    _procedure_store,
    _vision,
    classification,
)
from twindb_lightrag_memgraph._constants import (
    doc_type_context,
    operator_classification_context,
    storage_folder_context,
)
from twindb_lightrag_memgraph.patches import registry

PROCEDURE_ENV_VARS = (
    "TWIN_PROCEDURE",
    "TWIN_PROCEDURE_STORE_FILE",
    "TWIN_PROCEDURE_RENDER_SCALE",
    "TWIN_PROCEDURE_MAX_SCHEMATICS",
    "TWIN_PROCEDURE_MAX_BYTES",
    "TWIN_PROCEDURE_MAX_TOKENS",
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

    def chat(messages, **_kwargs):
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


async def test_unparseable_vision_reply_is_retried_once(monkeypatch):
    replies = iter(["not-json", json.dumps({"ok": True})])
    calls = 0

    def chat(_messages, **_kwargs):
        nonlocal calls
        calls += 1
        return next(replies)

    monkeypatch.setattr(_vision, "vision_chat_sync", chat)

    result = await _procedure._vision_json_call([], "informed-pass")

    assert result == {"ok": True}
    assert calls == 2


async def test_unparseable_vision_reply_still_fails_after_retry(monkeypatch):
    calls = 0

    def chat(_messages, **_kwargs):
        nonlocal calls
        calls += 1
        return "not-json"

    monkeypatch.setattr(_vision, "vision_chat_sync", chat)

    with pytest.raises(ValueError, match="informed-pass: unparseable JSON reply"):
        await _procedure._vision_json_call([], "informed-pass")

    assert calls == 2


async def test_procedure_passes_cap_the_completion_length(monkeypatch):
    """Every procedure pass must send an explicit max_tokens.

    Without a cap the provider default decides how much completion it emits,
    and a JSON object cut mid-structure is unparseable *deterministically* —
    so the retry above re-fails identically instead of recovering. That is the
    signature the live gate hit on 2026-07-25.
    """
    seen = []

    def chat(_messages, **kwargs):
        seen.append(kwargs)
        return json.dumps({"ok": True})

    monkeypatch.setattr(_vision, "vision_chat_sync", chat)

    await _procedure._vision_json_call([], "informed-pass")

    assert seen == [{"max_tokens": _procedure.DEFAULT_PROCEDURE_MAX_TOKENS}]
    # Sized for the worst realistic schematic: hitting the cap loses the whole
    # object, so it must never be the binding constraint on a legitimate reply.
    assert _procedure.DEFAULT_PROCEDURE_MAX_TOKENS >= 8192


async def test_unparseable_reply_is_carried_into_the_error(monkeypatch):
    """The raw reply must survive into the message, bounded.

    "unparseable JSON reply" with no sample is undiagnosable from a CI log —
    the previous occurrence was lost exactly that way. The excerpt is capped so
    a runaway reply cannot flood the log or the parked bundle's reason field.
    """
    truncated = '{"title": "Qualify incident", "tasks": [{"id": "T2.1", ' + "x" * 5000

    monkeypatch.setattr(_vision, "vision_chat_sync", lambda _m, **_k: truncated)

    with pytest.raises(ValueError) as excinfo:
        await _procedure._vision_json_call([], "informed-pass")

    message = str(excinfo.value)
    assert f"({len(truncated)} chars)" in message, message
    assert '{"title": "Qualify incident"' in message
    assert message.endswith("…")
    assert len(message) < len(truncated) / 2, "excerpt is not bounded"


async def test_empty_reply_error_reports_zero_length(monkeypatch):
    """An empty completion is a distinct symptom from a malformed one."""
    monkeypatch.setattr(_vision, "vision_chat_sync", lambda _m, **_k: "")

    with pytest.raises(ValueError, match=r"\(0 chars\)"):
        await _procedure._vision_json_call([], "blind-pass")


async def test_unparseable_reply_excerpt_boundary_and_newlines_are_exact(
    monkeypatch, caplog
):
    raw = ("line one\n" + "x" * _procedure._REPLY_EXCERPT_CHARS)[
        : _procedure._REPLY_EXCERPT_CHARS
    ]
    monkeypatch.setattr(_vision, "vision_chat_sync", lambda _m, **_k: raw)
    monkeypatch.setattr(_vision, "vision_timeout_seconds", lambda: 12.5)
    timeouts = []

    real_wait_for = asyncio.wait_for

    async def wait_for(awaitable, *, timeout):
        timeouts.append(timeout)
        return await real_wait_for(awaitable, timeout=timeout)

    monkeypatch.setattr(asyncio, "wait_for", wait_for)

    expected = (
        f"informed-pass: unparseable JSON reply ({len(raw)} chars): "
        f"{raw.replace(chr(10), ' ')}"
    )
    with caplog.at_level("WARNING"), pytest.raises(ValueError) as excinfo:
        await _procedure._vision_json_call([], "informed-pass")

    assert str(excinfo.value) == expected
    assert timeouts == [12.5, 12.5]
    assert [record.getMessage() for record in caplog.records] == [
        "twindb procedure: informed-pass got an unparseable reply "
        f"({len(raw)} chars): {raw.replace(chr(10), ' ')}"
    ]


async def test_procedure_max_tokens_is_env_overridable(monkeypatch):
    """Operators tune the cap without a code change; a silly value is refused."""
    monkeypatch.setenv("TWIN_PROCEDURE_MAX_TOKENS", "32768")
    assert _procedure.procedure_max_tokens() == 32768

    # Below the floor a cap can only truncate a legitimate reply.
    monkeypatch.setenv("TWIN_PROCEDURE_MAX_TOKENS", "16")
    assert _procedure.procedure_max_tokens() == _procedure.DEFAULT_PROCEDURE_MAX_TOKENS

    monkeypatch.setenv("TWIN_PROCEDURE_MAX_TOKENS", "not-a-number")
    assert _procedure.procedure_max_tokens() == _procedure.DEFAULT_PROCEDURE_MAX_TOKENS


async def test_junk_prefixed_fenced_reply_is_recovered(monkeypatch):
    """The exact shape the live gate returned on 2026-07-25.

    A junk prefix before the fence used to defeat the parser: the greedy
    fallback anchored on that first brace and produced an invalid span while
    the real object sat intact a few characters later.
    """
    reply = (
        '{";}```json\n{\n  "title": "Qualify and resolve the incident",\n'
        '  "description": "T1.1 then T2.1 then T3.1.",\n  "tasks": []\n}\n```'
    )
    monkeypatch.setattr(_vision, "vision_chat_sync", lambda _m, **_k: reply)

    result = await _procedure._vision_json_call([], "informed-pass")

    assert result["title"] == "Qualify and resolve the incident"
    assert result["tasks"] == []


def test_pass_payload_validator_checks_every_task_field():
    task = {field: f"value-{field}" for field in _procedure._TASK_FIELDS}
    payload = {"title": "Flow", "description": "Steps", "tasks": [task]}
    assert _procedure._validate_pass_payload(payload, "blind-pass") == payload

    for field in _procedure._TASK_FIELDS:
        invalid = dict(task)
        invalid[field] = None
        with pytest.raises(
            ValueError,
            match=(
                r"blind-pass: task #0 does not carry the eight string fields "
                r"of the contract"
            ),
        ):
            _procedure._validate_pass_payload(
                {"title": "Flow", "description": "Steps", "tasks": [invalid]},
                "blind-pass",
            )


@pytest.mark.parametrize(
    "payload",
    [
        {"title": 1, "description": "Steps", "tasks": []},
        {"title": "Flow", "description": 1, "tasks": []},
        {"title": "Flow", "description": "  ", "tasks": []},
        {"title": "Flow", "description": "Steps", "tasks": {}},
    ],
)
def test_pass_payload_validator_rejects_each_top_level_shape(payload):
    with pytest.raises(
        ValueError,
        match=r"informed-pass: reply does not match the expected shape",
    ):
        _procedure._validate_pass_payload(payload, "informed-pass")


def test_pass_payload_shape_diagnostic_is_bounded_and_exact():
    payload = {f"k{i}": i for i in range(13)}
    with pytest.raises(ValueError) as excinfo:
        _procedure._validate_pass_payload(payload, "informed-pass")
    got = ", ".join(f"k{i}:int" for i in range(12))
    assert str(excinfo.value) == (
        "informed-pass: reply does not match the expected shape "
        f"(got 13 key(s): {got})"
    )

    with pytest.raises(ValueError) as excinfo:
        _procedure._validate_pass_payload({}, "blind-pass")
    assert str(excinfo.value) == (
        "blind-pass: reply does not match the expected shape " "(got 0 key(s): none)"
    )


@pytest.mark.parametrize(
    "payload",
    [
        {"coherent": 1, "divergences": [], "summary": "ok"},
        {"coherent": True, "divergences": "none", "summary": "ok"},
        {"coherent": True, "divergences": [1], "summary": "ok"},
        {"coherent": True, "divergences": [], "summary": None},
    ],
)
def test_comparator_validator_rejects_each_invalid_shape(payload):
    with pytest.raises(ValueError) as excinfo:
        _procedure._validate_comparator_payload(payload)
    assert str(excinfo.value) == ("comparator: reply does not match the expected shape")


def test_comparator_validator_returns_only_contract_fields():
    payload = {
        "coherent": False,
        "divergences": ["missing T2.1"],
        "summary": "gap",
        "ignored": "noise",
    }
    assert _procedure._validate_comparator_payload(payload) == {
        "coherent": False,
        "divergences": ["missing T2.1"],
        "summary": "gap",
    }


# ---------------------------------------------------------------------------
# Tier gates
# ---------------------------------------------------------------------------


def test_event_sink_success_and_failure_are_exact(monkeypatch, caplog):
    calls = []
    _procedure.set_event_sink(lambda kind, payload: calls.append((kind, payload)))
    _procedure._emit("procedure-created", {"bundle_id": "b-1"})
    assert calls == [("procedure-created", {"bundle_id": "b-1"})]

    def fail(kind, payload):
        assert kind == "procedure-failed"
        assert payload == {"bundle_id": "b-2"}
        raise RuntimeError("ledger down")

    _procedure.set_event_sink(fail)
    with caplog.at_level("WARNING"):
        _procedure._emit("procedure-failed", {"bundle_id": "b-2"})
    assert [record.getMessage() for record in caplog.records] == [
        "twindb procedure: event sink failed for procedure-failed "
        "(RuntimeError: ledger down)"
    ]


async def test_direct_park_persists_and_emits_the_exact_contract(
    monkeypatch, tmp_path, caplog
):
    path = tmp_path / "procedure.pdf"
    created = []
    events = []

    def create_bundle(**kwargs):
        created.append(kwargs)
        return "bundle-1"

    monkeypatch.setattr(_procedure_store, "create_bundle", create_bundle)
    _procedure.set_event_sink(lambda kind, payload: events.append((kind, payload)))
    schematic = {"page": 2, "informed": {"description": "flow"}}

    with (
        storage_folder_context("folder_1"),
        operator_classification_context("C2"),
        caplog.at_level("INFO"),
    ):
        outcome = await _procedure._park(
            path,
            "track-1",
            state="pending",
            reason="ok",
            source="detected",
            content_hash="sha256",
            full_text="full text",
            schematics=[schematic],
            schematics_total=3,
            classification={"class_id": "C1"},
        )

    assert created == [
        {
            "file_name": "procedure.pdf",
            "original_path": str(path),
            "track_id": "track-1",
            "state": "pending",
            "reason": "ok",
            "source": "detected",
            "folder": "folder_1",
            "content_hash": "sha256",
            "full_text": "full text",
            "schematics": [schematic],
            "schematics_total": 3,
            "classification": {"class_id": "C1"},
            "operator_classification": "C2",
        }
    ]
    assert events == [
        (
            "procedure-parked",
            {
                "bundle_id": "bundle-1",
                "file_name": "procedure.pdf",
                "state": "pending",
                "reason": "ok",
                "source": "detected",
                "schematics": 1,
            },
        )
    ]
    assert outcome == _procedure.ProcedureOutcome("bundle-1", "pending", "ok")
    assert [record.getMessage() for record in caplog.records] == [
        "twindb procedure: procedure.pdf parked for approval "
        "(bundle bundle-1, state=pending, 1 schematic(s), ok)"
    ]


async def test_settle_persists_and_emits_the_exact_contract(
    monkeypatch, tmp_path, caplog
):
    path = tmp_path / "procedure.pdf"
    updated = []
    events = []
    schematics = [{"page": 4}]

    def update_bundle(bundle_id, **kwargs):
        updated.append((bundle_id, kwargs))
        return {"id": bundle_id, **kwargs}

    monkeypatch.setattr(_procedure_store, "update_bundle", update_bundle)
    _procedure.set_event_sink(lambda kind, payload: events.append((kind, payload)))

    with caplog.at_level("INFO"):
        outcome = await _procedure._settle(
            "bundle-2",
            path,
            "forced",
            state="failed",
            reason="vision down",
            full_text="text",
            schematics=schematics,
        )

    assert updated == [
        (
            "bundle-2",
            {
                "state": "failed",
                "reason": "vision down",
                "full_text": "text",
                "schematics": schematics,
            },
        )
    ]
    assert events == [
        (
            "procedure-failed",
            {
                "bundle_id": "bundle-2",
                "file_name": "procedure.pdf",
                "state": "failed",
                "reason": "vision down",
                "source": "forced",
                "schematics": 1,
            },
        )
    ]
    assert outcome == _procedure.ProcedureOutcome("bundle-2", "failed", "vision down")
    assert [record.getMessage() for record in caplog.records] == [
        "twindb procedure: procedure.pdf parked for approval "
        "(bundle bundle-2, state=failed, 1 schematic(s), vision down)"
    ]


async def test_settle_persist_failure_and_lost_reservation_are_exact(
    monkeypatch, tmp_path, caplog
):
    path = tmp_path / "procedure.pdf"

    def fail_update(bundle_id, **kwargs):
        assert bundle_id == "bundle-3"
        assert kwargs == {"state": "pending", "reason": "ok"}
        raise OSError("disk full")

    monkeypatch.setattr(_procedure_store, "update_bundle", fail_update)
    with caplog.at_level("ERROR"):
        outcome = await _procedure._settle(
            "bundle-3", path, "detected", state="pending", reason="ok"
        )
    persist_reason = "settle-persist-failed: OSError: disk full — refusing the enqueue"
    assert outcome == _procedure.ProcedureOutcome("bundle-3", "error", persist_reason)
    assert [record.getMessage() for record in caplog.records] == [
        "twindb procedure: procedure.pdf — could not settle bundle bundle-3: "
        f"{persist_reason}"
    ]

    caplog.clear()
    monkeypatch.setattr(_procedure_store, "update_bundle", lambda *_a, **_k: None)
    with caplog.at_level("ERROR"):
        outcome = await _procedure._settle(
            "bundle-4", path, "forced", state="failed", reason="vision down"
        )
    vanished = (
        "settle-lost: bundle bundle-4 disappeared before its results could be "
        "persisted — refusing the enqueue"
    )
    assert outcome == _procedure.ProcedureOutcome("bundle-4", "error", vanished)
    assert [record.getMessage() for record in caplog.records] == [
        f"twindb procedure: procedure.pdf — {vanished}"
    ]


async def test_failed_direct_park_emits_failed_event(monkeypatch, tmp_path):
    path = tmp_path / "procedure.pdf"
    events = []
    monkeypatch.setattr(
        _procedure_store, "create_bundle", lambda **_kwargs: "bundle-failed"
    )
    _procedure.set_event_sink(lambda kind, payload: events.append((kind, payload)))

    outcome = await _procedure._park(
        path,
        None,
        state="failed",
        reason="bad input",
        source="forced",
        content_hash=None,
        full_text="",
        schematics=[],
        schematics_total=0,
        classification=None,
    )

    assert outcome == _procedure.ProcedureOutcome(
        "bundle-failed", "failed", "bad input"
    )
    assert events == [
        (
            "procedure-failed",
            {
                "bundle_id": "bundle-failed",
                "file_name": "procedure.pdf",
                "state": "failed",
                "reason": "bad input",
                "source": "forced",
                "schematics": 0,
            },
        )
    ]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(None, None), (" unexpected ", None), (" ON ", True), ("no", False)],
)
def test_procedure_mode_resolution_is_strict(raw, expected, monkeypatch):
    if raw is not None:
        monkeypatch.setenv("TWIN_PROCEDURE", raw)
    assert _procedure._resolve_mode() is expected


@pytest.mark.parametrize(
    ("module_name", "probe_name", "cache_name"),
    [
        ("pypdfium2", "_pdfium_importable", "_pdfium_available"),
        ("pypdf", "_pypdf_importable", "_pypdf_available"),
    ],
)
def test_dependency_probes_cache_success_and_failure(
    module_name, probe_name, cache_name, monkeypatch
):
    probe = getattr(_procedure, probe_name)
    monkeypatch.setitem(sys.modules, module_name, SimpleNamespace())
    assert probe() is True
    monkeypatch.delitem(sys.modules, module_name)

    real_import = builtins.__import__

    def fail_import(name, *args, **kwargs):
        if name == module_name:
            raise ImportError("removed after probe")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_import)
    assert probe() is True

    setattr(_procedure, cache_name, None)
    assert probe() is False
    assert probe() is False


def test_reset_caches_restores_all_procedure_singletons():
    _procedure._pdfium_available = True
    _procedure._pypdf_available = True
    _procedure._forced_on_warned = True
    _procedure._event_sink = object()
    _procedure._settings_provider = object()

    _procedure.reset_caches()

    assert _procedure._pdfium_available is None
    assert _procedure._pypdf_available is None
    assert _procedure._forced_on_warned is False
    assert _procedure._event_sink is None
    assert _procedure._settings_provider is None


def test_mode_off_disables_even_when_ready(profile_ready, monkeypatch):
    monkeypatch.setenv("TWIN_PROCEDURE", "off")
    assert _procedure.is_enabled() is False


class TestRenderPixelCap:
    """Audit 2026-08-06, R-08b: the MediaBox geometry is attacker-controlled;
    the procedure render must share _pdf_vision's MAX_RENDER_PIXELS cap."""

    def test_normal_page_keeps_configured_scale(self, monkeypatch):
        monkeypatch.setenv("TWIN_PROCEDURE_RENDER_SCALE", "2.0")
        # A4-ish page: 612x792pt at scale 2 → ~0.97 Mpx, far under the cap.
        assert _procedure._capped_render_scale(612, 792) == 2.0

    def test_giant_page_is_scaled_down_under_the_cap(self, monkeypatch):
        monkeypatch.setenv("TWIN_PROCEDURE_RENDER_SCALE", "8.0")
        # The audit's crafted geometry: 3000x3000pt would render 6000x6000px
        # = 36 Mpx (0.38 GB RSS for ONE page) without the cap.
        scale = _procedure._capped_render_scale(3000, 3000)
        import math

        pixels = math.ceil(3000 * scale) * math.ceil(3000 * scale)
        assert pixels <= 16_000_000  # _pdf_vision.MAX_RENDER_PIXELS
        assert scale < 8.0

    def test_unrenderable_geometry_refuses_cleanly(self, monkeypatch):
        monkeypatch.setenv("TWIN_PROCEDURE_RENDER_SCALE", "2.0")
        with pytest.raises(ValueError, match="invalid procedure page geometry"):
            _procedure._capped_render_scale(0, 792)
        with pytest.raises(ValueError, match="invalid procedure page geometry"):
            _procedure._capped_render_scale(float("inf"), 792)


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
    assert [record.getMessage() for record in caplog.records] == [
        "twindb procedure: TWIN_PROCEDURE=on but the profile is not usable "
        "(pypdfium2: False, pypdf: False, vision tier: False) — install the "
        "[procedure] extra and configure the vision endpoint; every document "
        "follows the standard path"
    ]


async def test_runtime_activation_contracts_are_exact(
    profile_ready, monkeypatch, caplog
):
    monkeypatch.setenv("TWIN_PROCEDURE", "on")

    async def invalid_provider():
        return {"procedure_enabled": "yes"}

    _procedure.set_settings_provider(invalid_provider)
    assert await _procedure.is_effectively_enabled() is True

    async def disabled_provider():
        return {"procedure_enabled": False}

    _procedure.set_settings_provider(disabled_provider)
    assert await _procedure.is_effectively_enabled() is False

    async def failed_provider():
        raise RuntimeError("store down")

    _procedure.set_settings_provider(failed_provider)
    with caplog.at_level("WARNING"):
        assert await _procedure.is_effectively_enabled() is False
    assert [record.getMessage() for record in caplog.records] == [
        "twindb procedure: settings provider failed (RuntimeError: store down) "
        "— new procedure ingestion disabled"
    ]


async def test_forced_selection_block_reasons_are_exact(profile_ready, monkeypatch):
    disabled = (
        "procedure-disabled: an administrator must enable procedure "
        "ingestion in Settings > Vision"
    )
    unavailable = (
        "procedure-unavailable: PDF extraction or Vision prerequisites "
        "are not configured; check Settings > Vision"
    )

    monkeypatch.setenv("TWIN_PROCEDURE", "off")
    assert await _procedure._forced_selection_block_reason() == disabled
    monkeypatch.setenv("TWIN_PROCEDURE", "on")
    assert await _procedure._forced_selection_block_reason() is None

    async def enabled_provider():
        return {"procedure_enabled": True}

    _procedure.set_settings_provider(enabled_provider)
    monkeypatch.setattr(_procedure, "is_available", lambda: False)
    assert await _procedure._forced_selection_block_reason() == unavailable

    monkeypatch.setattr(_procedure, "is_available", lambda: True)

    async def disabled_provider():
        return {"procedure_enabled": False}

    _procedure.set_settings_provider(disabled_provider)
    assert await _procedure._forced_selection_block_reason() == disabled


def test_advisory_classification_success_and_failure_are_exact(
    monkeypatch, tmp_path, caplog
):
    path = tmp_path / "procedure.pdf"
    monkeypatch.setattr(
        classification,
        "detect_classification",
        lambda received: SimpleNamespace(
            class_id="C2", class_name="Confidential", reason="label"
        ),
    )
    assert _procedure._advisory_classification(path) == {
        "class_id": "C2",
        "class_name": "Confidential",
        "reason": "label",
    }

    monkeypatch.setattr(
        classification,
        "detect_classification",
        lambda received: SimpleNamespace(class_id="C1", class_name="Internal"),
    )
    assert _procedure._advisory_classification(path) == {
        "class_id": "C1",
        "class_name": "Internal",
        "reason": None,
    }

    def fail(received):
        assert received == path
        raise RuntimeError("detector down")

    monkeypatch.setattr(classification, "detect_classification", fail)
    with caplog.at_level("WARNING"):
        assert _procedure._advisory_classification(path) is None
    assert [record.getMessage() for record in caplog.records] == [
        "twindb procedure: advisory classification failed for procedure.pdf "
        "(RuntimeError: detector down)"
    ]


async def test_store_guard_failure_modes_are_exact(monkeypatch, tmp_path, caplog):
    path = tmp_path / "procedure.pdf"
    degraded_reason = (
        "store-degraded: the bundle claim index was quarantined — refusing "
        "every enqueue until the .corrupt-* files next to the store are "
        "explicitly recovered and removed"
    )

    monkeypatch.setattr(_procedure_store, "is_degraded", lambda: True)
    with caplog.at_level("ERROR"):
        outcome = await _procedure._guard_store_and_find_existing(path)
    assert outcome == _procedure.ProcedureOutcome(None, "error", degraded_reason)
    assert [record.getMessage() for record in caplog.records] == [
        f"twindb procedure: procedure.pdf — {degraded_reason}"
    ]

    caplog.clear()
    monkeypatch.setattr(_procedure_store, "is_degraded", lambda: False)

    def unreadable(received):
        assert received == path
        raise RuntimeError("disk offline")

    monkeypatch.setattr(_procedure, "_find_existing_for_path_sync", unreadable)
    with caplog.at_level("ERROR"):
        outcome = await _procedure._guard_store_and_find_existing(path)
    reason = (
        "store-unreadable: RuntimeError: disk offline — refusing the enqueue "
        "(the claim index cannot be consulted)"
    )
    assert outcome == _procedure.ProcedureOutcome(None, "error", reason)
    assert [record.getMessage() for record in caplog.records] == [
        f"twindb procedure: procedure.pdf — {reason}"
    ]

    caplog.clear()

    def quarantined():
        raise _procedure_store.StoreDegradedError("quarantined")

    monkeypatch.setattr(_procedure_store, "is_degraded", quarantined)
    with caplog.at_level("ERROR"):
        outcome = await _procedure._guard_store_and_find_existing(path)
    assert outcome == _procedure.ProcedureOutcome(None, "error", degraded_reason)
    assert [record.getMessage() for record in caplog.records] == [
        f"twindb procedure: procedure.pdf — {degraded_reason}"
    ]


def test_existing_path_guard_matches_hash_and_hashless_bundles_exactly(
    monkeypatch, tmp_path
):
    path = tmp_path / "procedure.pdf"
    mismatch = {"id": "newer", "content_hash": "other"}
    match = {"id": "older", "content_hash": "current"}
    monkeypatch.setattr(
        _procedure_store,
        "find_bundles_by_path",
        lambda received: [mismatch, match] if received == str(path) else [],
    )
    monkeypatch.setattr(_procedure, "_content_hash_sync", lambda received: "current")
    assert _procedure._find_existing_for_path_sync(path) is match

    hashless = {"id": "failed", "content_hash": None}
    monkeypatch.setattr(
        _procedure_store, "find_bundles_by_path", lambda _received: [hashless, match]
    )
    assert _procedure._find_existing_for_path_sync(path) is hashless

    def unreadable(_received):
        raise OSError("file vanished")

    monkeypatch.setattr(_procedure, "_content_hash_sync", unreadable)
    assert _procedure._find_existing_for_path_sync(path) is hashless
    monkeypatch.setattr(
        _procedure_store, "find_bundles_by_path", lambda _received: [match]
    )
    assert _procedure._find_existing_for_path_sync(path) is None
    monkeypatch.setattr(
        _procedure_store,
        "find_bundles_by_path",
        lambda _received: [{"id": "invalid-empty", "content_hash": ""}],
    )
    assert _procedure._find_existing_for_path_sync(path) is None


async def test_duplicate_request_and_reservation_capture_exact_context(
    monkeypatch, tmp_path
):
    path = tmp_path / "procedure.pdf"
    requests = []
    reservations = []
    monkeypatch.setattr(
        _procedure_store,
        "record_request",
        lambda bundle_id, **kwargs: requests.append((bundle_id, kwargs)),
    )
    monkeypatch.setattr(
        _procedure_store,
        "reserve_bundle",
        lambda **kwargs: reservations.append(kwargs) or ({"id": "bundle-8"}, True),
    )

    with (
        storage_folder_context("folder_b"),
        operator_classification_context("C2"),
    ):
        await _procedure._record_duplicate_request({"id": "bundle-7"}, path, "track-7")
        result = _procedure._reserve_sync(path, "track-8", "forced", "sha256", True)

    assert requests == [
        (
            "bundle-7",
            {
                "path": str(path),
                "folder": "folder_b",
                "track_id": "track-7",
                "operator_classification": "C2",
                "file_name": "procedure.pdf",
            },
        )
    ]
    assert reservations == [
        {
            "content_hash": "sha256",
            "file_name": "procedure.pdf",
            "original_path": str(path),
            "track_id": "track-8",
            "source": "forced",
            "folder": "folder_b",
            "operator_classification": "C2",
            "via_scan": True,
        }
    ]
    assert result == ({"id": "bundle-8"}, True)


def test_reuse_outcome_and_log_are_exact(tmp_path, caplog):
    path = tmp_path / "procedure.pdf"
    bundle = {"id": "bundle-9", "state": "rejected", "reason": "operator"}
    with caplog.at_level("INFO"):
        assert _procedure._reuse_outcome(bundle, path) == _procedure.ProcedureOutcome(
            "bundle-9", "rejected", "already-parked: operator"
        )
    assert [record.getMessage() for record in caplog.records] == [
        "twindb procedure: procedure.pdf already claimed by bundle bundle-9 "
        "(state=rejected) — reusing, no reprocessing"
    ]
    assert _procedure._reuse_outcome(
        {"id": "bundle-10", "state": "pending"}, path
    ) == _procedure.ProcedureOutcome("bundle-10", "pending", "already-parked: ")


async def test_reuse_request_persist_failure_is_exact(monkeypatch, tmp_path, caplog):
    path = tmp_path / "procedure.pdf"
    existing = {"id": "bundle-11", "state": "pending"}

    async def fail_record(bundle, received, track_id):
        assert bundle is existing
        assert received == path
        assert track_id == "track-11"
        raise OSError("disk full")

    monkeypatch.setattr(_procedure, "_record_duplicate_request", fail_record)
    with caplog.at_level("ERROR"):
        outcome = await _procedure._reuse_existing_request(
            existing, path, "track-11", from_scan=False
        )
    reason = (
        "duplicate-request-persist-failed: OSError: disk full — refusing the " "enqueue"
    )
    assert outcome == _procedure.ProcedureOutcome("bundle-11", "error", reason)
    assert [record.getMessage() for record in caplog.records] == [
        f"twindb procedure: procedure.pdf — {reason}"
    ]


def test_forced_validation_is_inclusive_at_size_cap(monkeypatch, tmp_path):
    pdf = _write(tmp_path, "procedure.pdf", b"1234")
    other = _write(tmp_path, "procedure.docx", b"1234")
    monkeypatch.setenv("TWIN_PROCEDURE_MAX_BYTES", "4")

    assert _procedure._forced_validation_problem(pdf) is None
    assert _procedure._forced_validation_problem(other) == (
        "unsupported-extension: the procedure profile handles PDF only — "
        "reroute as a standard document"
    )


async def test_auto_detection_probe_contract_is_exact(monkeypatch, tmp_path, caplog):
    path = tmp_path / "procedure.pdf"
    calls = []

    def extract(received, limit):
        calls.append((received, limit))
        return ["ITG0162", "Level 2\n4- Operational procedures"]

    monkeypatch.setattr(_procedure, "_extract_pages_text_sync", extract)
    assert await _procedure._auto_detected_procedure(path) is True
    assert calls == [(path, _procedure.DETECTION_PAGES)]

    def fail(_received, _limit):
        raise RuntimeError("reader down")

    monkeypatch.setattr(_procedure, "_extract_pages_text_sync", fail)
    with caplog.at_level("WARNING"):
        assert await _procedure._auto_detected_procedure(path) is False
    assert [record.getMessage() for record in caplog.records] == [
        "twindb procedure: detection probe failed for procedure.pdf "
        "(RuntimeError: reader down) — standard path"
    ]


async def test_admin_toggle_can_enable_new_procedure_routing(
    profile_ready, monkeypatch, tmp_path
):
    monkeypatch.setenv("TWIN_PROCEDURE", "off")

    async def provider():
        return {"procedure_enabled": True}

    _procedure.set_settings_provider(provider)
    path = _write(tmp_path, "runtime-enabled.pdf", b"%PDF-1.4")

    assert await _procedure.aroute_check(path) is True


async def test_admin_toggle_off_keeps_existing_bundle_claimed(
    profile_ready, monkeypatch, tmp_path
):
    async def provider():
        return {"procedure_enabled": False}

    _procedure.set_settings_provider(provider)
    path = _write(tmp_path, "already-parked.pdf", b"%PDF-1.4")
    monkeypatch.setattr(
        _procedure_store, "claimed_paths", lambda: frozenset({str(path)})
    )

    assert await _procedure.aroute_check(path) is True


async def test_forced_procedure_fails_closed_when_admin_toggle_is_off(
    profile_ready, tmp_path
):
    async def provider():
        return {"procedure_enabled": False}

    _procedure.set_settings_provider(provider)
    path = _write(tmp_path, "forced-while-disabled.pdf", b"%PDF-1.4")

    with doc_type_context("procedure"):
        assert await _procedure.aroute_check(path) is True
        outcome = await _procedure.aprocess_procedure(path, "track-disabled")

    assert outcome is not None
    assert outcome.state == "failed"
    assert "administrator must enable" in outcome.reason
    assert _procedure_store.find_bundles_by_path(str(path))[0]["state"] == "failed"


async def test_forced_procedure_fails_closed_when_legacy_env_is_off(
    profile_ready, monkeypatch, tmp_path
):
    monkeypatch.setenv("TWIN_PROCEDURE", "off")
    path = _write(tmp_path, "forced-while-env-disabled.pdf", b"%PDF-1.4")

    with doc_type_context("procedure"):
        outcome = await _procedure.aprocess_procedure(path, "track-env-disabled")

    assert outcome is not None
    assert outcome.state == "failed"
    assert "administrator must enable" in outcome.reason


async def test_forced_procedure_reports_unavailable_prerequisites(
    monkeypatch, tmp_path
):
    async def provider():
        return {"procedure_enabled": True}

    _procedure.set_settings_provider(provider)
    monkeypatch.setattr(_procedure, "is_available", lambda: False)
    path = _write(tmp_path, "forced-without-prerequisites.pdf", b"%PDF-1.4")

    with doc_type_context("procedure"):
        outcome = await _procedure.aprocess_procedure(path, "track-unavailable")

    assert outcome is not None
    assert outcome.state == "failed"
    assert "prerequisites are not configured" in outcome.reason


def test_numeric_envs_fall_back_on_garbage(monkeypatch):
    monkeypatch.setenv("TWIN_PROCEDURE_RENDER_SCALE", "garbage")
    monkeypatch.setenv("TWIN_PROCEDURE_MAX_SCHEMATICS", "-3")
    monkeypatch.setenv("TWIN_PROCEDURE_MAX_BYTES", "zero")
    assert _procedure.render_scale() == _procedure.DEFAULT_RENDER_SCALE
    assert _procedure.max_schematics() == _procedure.DEFAULT_MAX_SCHEMATICS
    assert _procedure.max_procedure_bytes() == _procedure.DEFAULT_MAX_BYTES
    monkeypatch.setenv("TWIN_PROCEDURE_RENDER_SCALE", "40")  # out of range
    assert _procedure.render_scale() == _procedure.DEFAULT_RENDER_SCALE


@pytest.mark.parametrize("value", ["0.5", "8"])
def test_render_scale_accepts_inclusive_boundaries(value, monkeypatch):
    monkeypatch.setenv("TWIN_PROCEDURE_RENDER_SCALE", value)
    assert _procedure.render_scale() == float(value)


def test_numeric_envs_accept_documented_minimums(monkeypatch):
    monkeypatch.setenv("TWIN_PROCEDURE_MAX_BYTES", "1")
    monkeypatch.setenv("TWIN_PROCEDURE_MAX_SCHEMATICS", "1")
    monkeypatch.setenv("TWIN_PROCEDURE_MAX_TOKENS", "2048")
    assert _procedure.max_procedure_bytes() == 1
    assert _procedure.max_schematics() == 1
    assert _procedure.procedure_max_tokens() == 2048


def test_numeric_envs_reject_zero_and_just_above_render_ceiling(monkeypatch):
    monkeypatch.setenv("TWIN_PROCEDURE_MAX_BYTES", "0")
    monkeypatch.setenv("TWIN_PROCEDURE_MAX_SCHEMATICS", "0")
    monkeypatch.setenv("TWIN_PROCEDURE_RENDER_SCALE", "9")
    assert _procedure.max_procedure_bytes() == _procedure.DEFAULT_MAX_BYTES
    assert _procedure.max_schematics() == _procedure.DEFAULT_MAX_SCHEMATICS
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


def test_should_consider_disabled_and_exact_size_boundary(
    profile_ready, monkeypatch, tmp_path, caplog
):
    path = _write(tmp_path, "boundary.pdf", b"1234")
    monkeypatch.setenv("TWIN_PROCEDURE", "off")
    assert _procedure.should_consider(path) is False

    monkeypatch.setenv("TWIN_PROCEDURE", "on")
    monkeypatch.setenv("TWIN_PROCEDURE_MAX_BYTES", "4")
    assert _procedure.should_consider(path) is True

    path.write_bytes(b"12345")
    with caplog.at_level("WARNING"):
        assert _procedure.should_consider(path) is False
    assert [record.getMessage() for record in caplog.records] == [
        "twindb procedure: boundary.pdf exceeds TWIN_PROCEDURE_MAX_BYTES "
        "(5 bytes) — standard path"
    ]


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


def test_text_extraction_limit_and_empty_pages_are_exact(monkeypatch, tmp_path):
    path = tmp_path / "document.pdf"
    opened = []

    class Page:
        def __init__(self, text):
            self.text = text

        def extract_text(self):
            return self.text

    def reader(received):
        opened.append(received)
        return SimpleNamespace(pages=[Page("page one"), Page(None), Page("page three")])

    monkeypatch.setitem(sys.modules, "pypdf", SimpleNamespace(PdfReader=reader))

    assert _procedure._extract_pages_text_sync(path, limit=2) == ["page one", ""]
    assert opened == [str(path)]


def test_text_extraction_failure_is_contained_and_exact(monkeypatch, tmp_path, caplog):
    path = tmp_path / "broken.pdf"

    def reader(received):
        assert received == str(path)
        raise ValueError("bad xref")

    monkeypatch.setitem(sys.modules, "pypdf", SimpleNamespace(PdfReader=reader))

    with caplog.at_level("WARNING"):
        assert _procedure._extract_pages_text_sync(path) is None

    assert [record.getMessage() for record in caplog.records] == [
        "twindb procedure: text extraction failed for broken.pdf "
        "(ValueError: bad xref)"
    ]


def test_content_hash_is_sha256_of_all_file_bytes(tmp_path):
    path = _write(tmp_path, "payload.pdf", b"abc" * (1024 * 1024))

    assert (
        _procedure._content_hash_sync(path)
        == hashlib.sha256(path.read_bytes()).hexdigest()
    )


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


def test_render_page_png_closes_document_and_uses_configured_scale(
    monkeypatch, tmp_path
):
    path = tmp_path / "document.pdf"
    saved = []
    closed = []

    class Image:
        def save(self, buffer, format):
            assert format == "PNG"
            buffer.write(b"rendered-png")
            saved.append(True)

    class Bitmap:
        def to_pil(self):
            return Image()

    class Page:
        def get_size(self):
            # A4-ish geometry — comfortably under the R-08b pixel cap at
            # scale 3.5, so the configured scale survives unchanged.
            return (612.0, 792.0)

        def render(self, scale):
            assert scale == 3.5
            return Bitmap()

    class Document:
        def __init__(self, received):
            assert received == str(path)

        def __getitem__(self, page_index):
            assert page_index == 4
            return Page()

        def close(self):
            closed.append(True)

    monkeypatch.setitem(sys.modules, "pypdfium2", SimpleNamespace(PdfDocument=Document))
    monkeypatch.setattr(_procedure, "render_scale", lambda: 3.5)

    assert _procedure._render_page_png_sync(path, 4) == b"rendered-png"
    assert saved == [True]
    assert closed == [True]


def test_png_data_url_is_exact():
    assert _procedure._png_data_url(b"png") == (
        f"data:image/png;base64,{base64.b64encode(b'png').decode('ascii')}"
    )


# ---------------------------------------------------------------------------
# Bundle store
# ---------------------------------------------------------------------------


def test_store_path_and_timestamp_contracts_are_exact(monkeypatch, tmp_path):
    monkeypatch.delenv("TWIN_PROCEDURE_STORE_FILE", raising=False)
    monkeypatch.delenv("WORKING_DIR", raising=False)
    assert _procedure_store.store_path() == _procedure_store.Path(
        "twin_procedure_bundles.json"
    )
    monkeypatch.setenv("WORKING_DIR", str(tmp_path / "working"))
    assert _procedure_store.store_path() == (
        tmp_path / "working" / "twin_procedure_bundles.json"
    )
    explicit = tmp_path / "explicit.json"
    monkeypatch.setenv("TWIN_PROCEDURE_STORE_FILE", f"  {explicit}  ")
    assert _procedure_store.store_path() == explicit
    assert _procedure_store._now_iso().endswith("+00:00")


def test_store_bundle_paths_and_known_request_keys_are_exact():
    bundle = {
        "original_path": "/inputs/a.pdf",
        "folder": "f1",
        "duplicate_requests": [
            "invalid",
            {"path": "/inputs/b.pdf", "folder": "f2"},
            {"path": "", "folder": None},
            {},
        ],
    }
    assert _procedure_store._bundle_paths(bundle) == {
        "/inputs/a.pdf",
        "/inputs/b.pdf",
    }
    assert _procedure_store._known_request_keys(bundle) == {
        ("/inputs/a.pdf", "f1"),
        ("/inputs/b.pdf", "f2"),
        ("", None),
    }
    assert _procedure_store._bundle_paths({}) == set()
    assert _procedure_store._known_request_keys({}) == {("", None)}


def test_append_request_full_contract_and_known_keys_are_exact(monkeypatch):
    monkeypatch.setattr(_procedure_store, "_now_iso", lambda: "2026-08-02T12:00Z")
    bundle = {
        "original_path": "/inputs/a.pdf",
        "folder": "f1",
        "operator_classification": "C1",
        "duplicate_requests": ["invalid"],
    }
    assert (
        _procedure_store._append_request(
            bundle,
            path="/inputs/a.pdf",
            folder="f1",
            track_id="ignored",
            operator_classification="C1",
            file_name="a.pdf",
        )
        is False
    )
    missing_paths = {
        "folder": "primary-folder",
        "duplicate_requests": [{"folder": "duplicate-folder"}],
    }
    assert (
        _procedure_store._append_request(
            missing_paths,
            path="",
            folder="primary-folder",
            track_id=None,
            operator_classification=None,
            file_name="",
        )
        is False
    )
    assert (
        _procedure_store._append_request(
            missing_paths,
            path="",
            folder="duplicate-folder",
            track_id=None,
            operator_classification=None,
            file_name="",
        )
        is False
    )
    assert (
        _procedure_store._append_request(
            bundle,
            path="/inputs/a.pdf",
            folder="f1",
            track_id="upgrade",
            operator_classification="C2",
            file_name="a.pdf",
        )
        is True
    )
    assert bundle["operator_classification"] == "C2"
    assert (
        _procedure_store._append_request(
            bundle,
            path="/inputs/a.pdf",
            folder="f1",
            track_id="no-downgrade",
            operator_classification="C1",
            file_name="a.pdf",
        )
        is False
    )
    assert bundle["operator_classification"] == "C2"
    assert (
        _procedure_store._append_request(
            bundle,
            path="/inputs/b.pdf",
            folder="f2",
            track_id="track-b",
            operator_classification="C1",
            file_name="b.pdf",
        )
        is True
    )
    assert bundle["duplicate_requests"] == [
        "invalid",
        {
            "path": "/inputs/b.pdf",
            "folder": "f2",
            "track_id": "track-b",
            "operator_classification": "C1",
            "file_name": "b.pdf",
            "requested_at": "2026-08-02T12:00Z",
        },
    ]
    assert (
        _procedure_store._append_request(
            bundle,
            path="/inputs/b.pdf",
            folder="f2",
            track_id="duplicate",
            operator_classification="C1",
            file_name="renamed.pdf",
        )
        is False
    )


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


def test_create_bundle_persists_the_complete_exact_schema(monkeypatch):
    monkeypatch.setattr(
        _procedure_store.uuid,
        "uuid4",
        lambda: SimpleNamespace(hex="bundlefixed000000000000000000000"),
    )
    monkeypatch.setattr(_procedure_store, "_now_iso", lambda: "2026-08-02T12:00Z")
    bundle_id = _procedure_store.create_bundle(
        file_name="procédure.pdf",
        original_path="/inputs/procédure.pdf",
        track_id="track-1",
        state="pending",
        reason="à valider",
        source="forced",
        folder="f1",
        content_hash="sha256",
        full_text="texte é",
        schematics=[{"page": 2}],
        classification={"class_id": "C2"},
        operator_classification="C1",
    )
    assert bundle_id == "bundlefixed000000000000000000000"
    assert _procedure_store.get_bundle(bundle_id) == {
        "id": bundle_id,
        "file_name": "procédure.pdf",
        "original_path": "/inputs/procédure.pdf",
        "track_id": "track-1",
        "state": "pending",
        "reason": "à valider",
        "source": "forced",
        "folder": "f1",
        "content_hash": "sha256",
        "full_text": "texte é",
        "schematics": [{"page": 2}],
        "schematics_total": 0,
        "classification": {"class_id": "C2"},
        "operator_classification": "C1",
        "created_at": "2026-08-02T12:00Z",
        "updated_at": "2026-08-02T12:00Z",
    }
    raw = _procedure_store.store_path().read_text(encoding="utf-8")
    assert '"version": 1' in raw
    assert "procédure.pdf" in raw
    assert "\\u00e9" not in raw

    with pytest.raises(ValueError) as excinfo:
        _park_minimal(state="invalid")
    assert str(excinfo.value) == "invalid bundle state: 'invalid'"


def test_reserve_bundle_new_schema_and_existing_selection_are_exact(monkeypatch):
    monkeypatch.setattr(
        _procedure_store.uuid,
        "uuid4",
        lambda: SimpleNamespace(hex="reservationfixed0000000000000000"),
    )
    timestamps = iter(["2026-08-02T12:00Z", "2026-08-02T12:01Z", "2026-08-02T12:02Z"])
    monkeypatch.setattr(_procedure_store, "_now_iso", lambda: next(timestamps))
    created_bundle, created = _procedure_store.reserve_bundle(
        content_hash="sha-reserve",
        file_name="a.pdf",
        original_path="/inputs/a.pdf",
        track_id="track-a",
        source="detected",
        folder="f1",
        operator_classification="C1",
    )
    assert created is True
    assert created_bundle == {
        "id": "reservationfixed0000000000000000",
        "file_name": "a.pdf",
        "original_path": "/inputs/a.pdf",
        "track_id": "track-a",
        "state": "processing",
        "reason": "processing",
        "source": "detected",
        "folder": "f1",
        "content_hash": "sha-reserve",
        "full_text": "",
        "schematics": [],
        "schematics_total": 0,
        "classification": None,
        "operator_classification": "C1",
        "created_at": "2026-08-02T12:00Z",
        "updated_at": "2026-08-02T12:00Z",
    }

    existing, created = _procedure_store.reserve_bundle(
        content_hash="sha-reserve",
        file_name="b.pdf",
        original_path="/inputs/b.pdf",
        track_id="track-b",
        source="forced",
        folder="f2",
        operator_classification="C2",
    )
    assert created is False
    assert existing["id"] == created_bundle["id"]
    assert existing["updated_at"] == "2026-08-02T12:02Z"
    assert existing["duplicate_requests"] == [
        {
            "path": "/inputs/b.pdf",
            "folder": "f2",
            "track_id": "track-b",
            "operator_classification": "C2",
            "file_name": "b.pdf",
            "requested_at": "2026-08-02T12:01Z",
        }
    ]


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


def test_store_list_and_path_search_sort_filter_contracts_are_exact(monkeypatch):
    bundles = {
        "old": {
            "id": "old",
            "state": "pending",
            "created_at": "2026-01-01",
            "original_path": "/inputs/a.pdf",
        },
        "new": {
            "id": "new",
            "state": "failed",
            "created_at": "2026-08-02",
            "original_path": "/inputs/other.pdf",
            "duplicate_requests": [{"path": "/inputs/a.pdf"}],
        },
        "undated": {
            "id": "undated",
            "state": "pending",
            "original_path": "/inputs/a.pdf",
        },
    }
    monkeypatch.setattr(_procedure_store, "_load", lambda path: bundles)
    assert [bundle["id"] for bundle in _procedure_store.list_bundles()] == [
        "new",
        "old",
        "undated",
    ]
    assert [
        bundle["id"] for bundle in _procedure_store.list_bundles(state="pending")
    ] == ["old", "undated"]
    assert [
        bundle["id"]
        for bundle in _procedure_store.find_bundles_by_path("/inputs/a.pdf")
    ] == ["new", "old", "undated"]
    assert _procedure_store.find_bundles_by_path("") == []


def test_store_record_request_changed_and_unchanged_are_exact(monkeypatch):
    bundle_id = _park_minimal(
        original_path="/inputs/a.pdf",
        folder="f1",
        operator_classification="C1",
    )
    monkeypatch.setattr(_procedure_store, "_now_iso", lambda: "2026-08-02T12:00Z")
    assert (
        _procedure_store.record_request(
            bundle_id,
            path="/inputs/b.pdf",
            folder="f2",
            track_id="track-b",
            operator_classification="C2",
            file_name="b.pdf",
        )
        is True
    )
    stored = _procedure_store.get_bundle(bundle_id)
    assert stored["updated_at"] == "2026-08-02T12:00Z"
    assert stored["duplicate_requests"] == [
        {
            "path": "/inputs/b.pdf",
            "folder": "f2",
            "track_id": "track-b",
            "operator_classification": "C2",
            "file_name": "b.pdf",
            "requested_at": "2026-08-02T12:00Z",
        }
    ]
    assert (
        _procedure_store.record_request(
            bundle_id,
            path="/inputs/b.pdf",
            folder="f2",
            track_id="ignored",
            operator_classification="C1",
            file_name="ignored.pdf",
        )
        is False
    )


def test_store_update_and_transition_contracts_are_exact(monkeypatch):
    bundle_id = _park_minimal()
    timestamps = iter(["update-time", "transition-time"])
    monkeypatch.setattr(_procedure_store, "_now_iso", lambda: next(timestamps))

    updated = _procedure_store.update_bundle(
        bundle_id, state="failed", reason="vision failed"
    )
    assert updated["state"] == "failed"
    assert updated["reason"] == "vision failed"
    assert updated["updated_at"] == "update-time"

    assert (
        _procedure_store.transition_bundle(
            bundle_id, ("pending",), state="approved", reason="wrong source"
        )
        is None
    )
    transitioned = _procedure_store.transition_bundle(
        bundle_id, ("failed",), state="processing", reason="retry"
    )
    assert transitioned["state"] == "processing"
    assert transitioned["reason"] == "retry"
    assert transitioned["updated_at"] == "transition-time"
    assert _procedure_store.transition_bundle("ghost", ("failed",)) is None

    for function, args in (
        (_procedure_store.update_bundle, (bundle_id,)),
        (_procedure_store.transition_bundle, (bundle_id, ("processing",))),
    ):
        with pytest.raises(ValueError) as excinfo:
            function(*args, state="invalid")
        assert str(excinfo.value) == "invalid bundle state: 'invalid'"


def test_store_recovery_removes_sorted_markers_and_logs_exactly(
    tmp_path, monkeypatch, caplog
):
    store_file = tmp_path / "store.json"
    monkeypatch.setenv("TWIN_PROCEDURE_STORE_FILE", str(store_file))
    for name in ("store.json.corrupt-z", "store.json.corrupt-a"):
        (tmp_path / name).write_text("forensics", encoding="utf-8")

    with caplog.at_level("WARNING"):
        assert _procedure_store.recover_store() == [
            "store.json.corrupt-a",
            "store.json.corrupt-z",
        ]
    assert _procedure_store.quarantine_files() == []
    assert [record.getMessage() for record in caplog.records] == [
        "twindb procedure: degraded-store recovery — removed quarantine "
        "file(s) store.json.corrupt-a, store.json.corrupt-z; the profile "
        "resumes normal operation"
    ]
    caplog.clear()
    assert _procedure_store.recover_store() == []
    assert caplog.records == []


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


def test_store_write_atomic_contract_and_cleanup_are_exact(monkeypatch, tmp_path):
    path = tmp_path / "parent" / "nested" / "store.json"
    calls = []
    real_mkstemp = _procedure_store.tempfile.mkstemp
    real_fdopen = _procedure_store.os.fdopen
    real_replace = _procedure_store.os.replace
    real_unlink = _procedure_store.os.unlink

    def mkstemp(*, prefix, suffix, dir):
        calls.append(("mkstemp", prefix, suffix, dir))
        return real_mkstemp(prefix=prefix, suffix=suffix, dir=dir)

    def fdopen(fd, mode, *, encoding):
        calls.append(("fdopen", mode, encoding))
        return real_fdopen(fd, mode, encoding=encoding)

    def replace(source, destination):
        calls.append(("replace", _procedure_store.Path(source).parent, destination))
        return real_replace(source, destination)

    monkeypatch.setattr(_procedure_store.tempfile, "mkstemp", mkstemp)
    monkeypatch.setattr(_procedure_store.os, "fdopen", fdopen)
    monkeypatch.setattr(_procedure_store.os, "replace", replace)

    _procedure_store._write(path, {"b1": {"label": "procédure"}})

    assert calls[0] == ("mkstemp", "store.json.", ".tmp", path.parent)
    assert calls[1] == ("fdopen", "w", "utf-8")
    assert calls[2] == ("replace", path.parent, path)
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "version": 1,
        "bundles": {"b1": {"label": "procédure"}},
    }
    assert "\\u00e9" not in path.read_text(encoding="utf-8")
    assert list(path.parent.glob("*.tmp")) == []

    unlinked = []

    def unlink(target):
        unlinked.append(target)
        return real_unlink(target)

    def fail_dump(*_args, **_kwargs):
        raise RuntimeError("serialization failed")

    monkeypatch.setattr(_procedure_store.os, "unlink", unlink)
    monkeypatch.setattr(_procedure_store.json, "dump", fail_dump)
    with pytest.raises(RuntimeError, match="serialization failed"):
        _procedure_store._write(tmp_path / "failed.json", {})
    assert len(unlinked) == 1
    assert not _procedure_store.Path(unlinked[0]).exists()


def test_quarantine_success_failure_and_logs_are_exact(monkeypatch, tmp_path, caplog):
    path = _write(tmp_path, "store.json", b"bad json")
    monkeypatch.setattr(
        _procedure_store.uuid,
        "uuid4",
        lambda: SimpleNamespace(hex="123456789abcdef"),
    )
    with caplog.at_level("ERROR"):
        _procedure_store._quarantine(path)
    target = tmp_path / "store.json.corrupt-12345678"
    assert target.read_bytes() == b"bad json"
    assert [record.getMessage() for record in caplog.records] == [
        f"twindb procedure: bundle store {path} is not valid JSON — "
        "quarantined as store.json.corrupt-12345678. The claim index is LOST: "
        "the procedure profile now refuses every enqueue until the .corrupt-* "
        "files are explicitly recovered and removed"
    ]

    caplog.clear()
    path.write_text("still bad", encoding="utf-8")

    def fail_replace(source, destination):
        assert source == path
        assert destination == target
        raise OSError("permission denied")

    monkeypatch.setattr(_procedure_store.os, "replace", fail_replace)
    with caplog.at_level("WARNING"), pytest.raises(OSError, match="permission denied"):
        _procedure_store._quarantine(path)
    assert [record.getMessage() for record in caplog.records] == [
        f"twindb procedure: bundle store {path} is corrupt and could not be "
        "quarantined (permission denied) — refusing to overwrite it"
    ]


def test_load_failure_shapes_and_messages_are_exact(monkeypatch, tmp_path, caplog):
    path = tmp_path / "store.json"
    marker = tmp_path / "store.json.corrupt-old"
    marker.write_text("bad", encoding="utf-8")
    with pytest.raises(_procedure_store.StoreDegradedError) as excinfo:
        _procedure_store._load(path)
    assert str(excinfo.value) == (
        f"bundle store {path} is degraded (quarantine marker present)"
    )
    marker.unlink()

    path.write_text("{bad json", encoding="utf-8")
    quarantined = []
    monkeypatch.setattr(
        _procedure_store, "_quarantine", lambda received: quarantined.append(received)
    )
    with pytest.raises(_procedure_store.StoreDegradedError) as excinfo:
        _procedure_store._load(path)
    assert str(excinfo.value) == f"bundle store {path} was corrupt — quarantined"
    assert quarantined == [path]

    path.write_text("[]", encoding="utf-8")
    quarantined.clear()
    with pytest.raises(_procedure_store.StoreDegradedError) as excinfo:
        _procedure_store._load(path)
    assert str(excinfo.value) == f"bundle store {path} was corrupt — quarantined"
    assert quarantined == [path]

    real_read_text = _procedure_store.Path.read_text

    def unreadable(received, *, encoding):
        if received == path:
            assert encoding == "utf-8"
            raise OSError("disk offline")
        return real_read_text(received, encoding=encoding)

    monkeypatch.setattr(_procedure_store.Path, "read_text", unreadable)
    with caplog.at_level("WARNING"), pytest.raises(OSError, match="disk offline"):
        _procedure_store._load(path)
    assert [record.getMessage() for record in caplog.records] == [
        f"twindb procedure: cannot read bundle store {path} (disk offline)"
    ]


def test_degraded_marker_directory_errors_fail_closed(monkeypatch, tmp_path):
    path = tmp_path / "store.json"

    def missing(_self, _pattern):
        raise FileNotFoundError

    monkeypatch.setattr(_procedure_store.Path, "glob", missing)
    assert _procedure_store._degraded_marker_exists(path) is False

    def unreadable(_self, _pattern):
        raise OSError("directory offline")

    monkeypatch.setattr(_procedure_store.Path, "glob", unreadable)
    assert _procedure_store._degraded_marker_exists(path) is True


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


def test_store_claimed_paths_cache_key_value_and_degraded_error_are_exact(monkeypatch):
    _procedure_store._paths_cache = None
    _park_minimal(original_path="/inputs/cached.pdf")
    store_file = _procedure_store.store_path()
    stat = store_file.stat()

    assert _procedure_store.claimed_paths() == frozenset({"/inputs/cached.pdf"})
    expected_key = (str(store_file), stat.st_mtime_ns, stat.st_size)
    assert _procedure_store._paths_cache == (
        expected_key,
        frozenset({"/inputs/cached.pdf"}),
    )

    def should_not_reload(_path):
        raise AssertionError("cache miss")

    monkeypatch.setattr(_procedure_store, "_load", should_not_reload)
    assert _procedure_store.claimed_paths() == frozenset({"/inputs/cached.pdf"})

    monkeypatch.setattr(_procedure_store, "_degraded_marker_exists", lambda path: True)
    with pytest.raises(_procedure_store.StoreDegradedError) as excinfo:
        _procedure_store.claimed_paths()
    assert str(excinfo.value) == (
        f"bundle store {store_file} is degraded (quarantine marker present)"
    )


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
    with pytest.raises(LookupError) as excinfo:
        _procedure_store.record_request(
            "ghost",
            path="/inputs/a.pdf",
            folder="f1",
            track_id=None,
            operator_classification=None,
            file_name="a.pdf",
        )
    assert str(excinfo.value) == "bundle ghost no longer exists"


def test_store_reserve_reuses_newest_matching_bundle(monkeypatch):
    bundles = {
        "old": {
            "id": "old",
            "content_hash": "same",
            "created_at": "2026-01-01",
        },
        "new": {
            "id": "new",
            "content_hash": "same",
            "created_at": "2026-08-02",
        },
        "undated": {"id": "undated", "content_hash": "same"},
    }
    monkeypatch.setattr(_procedure_store, "_load", lambda path: bundles)
    appended = []

    def append(existing, **kwargs):
        appended.append((existing, kwargs))
        return False

    monkeypatch.setattr(_procedure_store, "_append_request", append)
    monkeypatch.setattr(
        _procedure_store,
        "_write",
        lambda *_args: (_ for _ in ()).throw(AssertionError("unexpected write")),
    )

    existing, created = _procedure_store.reserve_bundle(
        content_hash="same",
        file_name="copy.pdf",
        original_path="/inputs/copy.pdf",
        track_id="track-copy",
        source="detected",
        folder="f2",
        operator_classification="C2",
    )
    assert existing is bundles["new"]
    assert created is False
    assert appended == [
        (
            bundles["new"],
            {
                "path": "/inputs/copy.pdf",
                "folder": "f2",
                "track_id": "track-copy",
                "operator_classification": "C2",
                "file_name": "copy.pdf",
            },
        )
    ]

    with pytest.raises(ValueError) as excinfo:
        _procedure_store.reserve_bundle(
            content_hash="",
            file_name="empty.pdf",
            original_path="/inputs/empty.pdf",
            track_id=None,
            source="detected",
            folder=None,
            operator_classification=None,
        )
    assert str(excinfo.value) == "reserve_bundle requires a content_hash"


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


def test_approval_markdown_helpers_keep_exact_minimal_contract():
    bundle = {
        "folder": "f1",
        "operator_classification": "C1",
        "duplicate_requests": [
            "invalid",
            {"folder": "f1", "operator_classification": None},
            {"folder": "f2", "operator_classification": "C3"},
            {"folder": None, "operator_classification": "C2"},
        ],
    }
    assert _procedure.bundle_folders(bundle) == ["f1", "f2"]
    assert _procedure.strictest_operator_classification(bundle) == "C3"
    assert (
        _procedure.strictest_operator_classification(
            {"operator_classification": "C2", "duplicate_requests": []}
        )
        == "C2"
    )

    task = {
        "id": "T2.1",
        "title": "Qualify incident",
        "responsible": "Incident manager",
        "actors": "Support",
        "inputs": "Alert",
        "outputs": "Ticket",
        "conditions": "Severity > 1",
        "links": "INC",
    }
    assert _procedure._task_markdown(task) == (
        "- T2.1 — Qualify incident (responsible: Incident manager; "
        "actors: Support; inputs: Alert; outputs: Ticket; "
        "conditions: Severity > 1; links: INC)"
    )
    assert _procedure._task_markdown({}) == "- ?"
    assert _procedure._task_markdown({"id": "T1", "title": "BOX"}) == "- T1 — BOX"

    entry = {
        "page": 3,
        "informed": {
            "title": "  Incident flow  ",
            "description": "  Canonical steps.  ",
            "tasks": [task, "invalid"],
        },
    }
    assert _procedure._schematic_markdown(entry) == [
        "## Schematic (page 3): Incident flow",
        "Canonical steps.",
        _procedure._task_markdown(task),
    ]
    assert _procedure._schematic_markdown(
        {"page": 8, "informed": {"title": "", "description": "", "tasks": []}}
    ) == ["## Schematic (page 8)", ""]
    assert _procedure.compose_approved_markdown({}) == ""

    bundle.update(
        {
            "full_text": "  Original procedure text.  ",
            "schematics": [entry, {"informed": None}, "invalid"],
        }
    )
    assert _procedure.compose_approved_markdown(bundle) == (
        "Original procedure text.\n\n---\n\n"
        "# Process schematics (vision descriptions)\n\n"
        "## Schematic (page 3): Incident flow\n\n"
        "Canonical steps.\n\n" + _procedure._task_markdown(task)
    )


# ---------------------------------------------------------------------------
# Orchestration (real pypdf text, scripted render + vision)
# ---------------------------------------------------------------------------


async def test_run_profile_without_text_returns_exact_failed_fields(
    monkeypatch, tmp_path
):
    path = tmp_path / "procedure.pdf"
    monkeypatch.setattr(_procedure, "_extract_pages_text_sync", lambda received: None)
    monkeypatch.setattr(
        _procedure,
        "_advisory_classification",
        lambda received: {"class_id": "C2"},
    )

    assert await _procedure._run_profile(path) == {
        "state": "failed",
        "reason": "text-extraction-failed: cannot read the PDF text layer",
        "classification": {"class_id": "C2"},
    }


async def test_run_profile_without_schematic_is_never_pending(monkeypatch, tmp_path):
    path = tmp_path / "procedure.pdf"
    monkeypatch.setattr(
        _procedure, "_extract_pages_text_sync", lambda received: ["page 1", "page 2"]
    )
    monkeypatch.setattr(_procedure, "_advisory_classification", lambda received: None)

    assert await _procedure._run_profile(path) == {
        "state": "failed",
        "reason": (
            "no-schematic-found: the template's Schematic pages were not "
            "located — retry, or reroute as a standard document"
        ),
        "full_text": "page 1\n\npage 2",
        "schematics": [],
        "schematics_total": 0,
        "classification": None,
    }


async def test_run_profile_truncation_keeps_selected_entries_and_exact_warning(
    monkeypatch, tmp_path, caplog
):
    path = tmp_path / "procedure.pdf"
    monkeypatch.setattr(
        _procedure,
        "_extract_pages_text_sync",
        lambda received: ["Schematic: one", "Schematic: two"],
    )
    monkeypatch.setattr(_procedure, "max_schematics", lambda: 1)
    monkeypatch.setattr(_procedure, "_advisory_classification", lambda received: None)
    processed = []

    async def process(received, page_index, full_text):
        processed.append((received, page_index, full_text))
        return {"page": page_index + 1}, None

    monkeypatch.setattr(_procedure, "_process_schematic", process)

    with caplog.at_level("WARNING"):
        result = await _procedure._run_profile(path)

    assert processed == [(path, 0, "Schematic: one\n\nSchematic: two")]
    assert result == {
        "state": "failed",
        "reason": (
            "schematics-truncated: 2 schematic pages found, cap 1 — raise "
            "TWIN_PROCEDURE_MAX_SCHEMATICS and retry"
        ),
        "full_text": "Schematic: one\n\nSchematic: two",
        "schematics": [{"page": 1}],
        "schematics_total": 2,
        "classification": None,
    }
    assert [record.getMessage() for record in caplog.records] == [
        "twindb procedure: procedure.pdf has 2 schematic pages, cap 1 "
        "(TWIN_PROCEDURE_MAX_SCHEMATICS)"
    ]


async def test_run_profile_exact_cap_is_not_truncated_and_joins_failures(
    monkeypatch, tmp_path
):
    path = tmp_path / "procedure.pdf"
    monkeypatch.setattr(
        _procedure,
        "_extract_pages_text_sync",
        lambda received: ["Schematic: one", "Schematic: two"],
    )
    monkeypatch.setattr(_procedure, "max_schematics", lambda: 2)
    monkeypatch.setattr(_procedure, "_advisory_classification", lambda received: None)

    async def process(received, page_index, full_text):
        assert received == path
        assert full_text == "Schematic: one\n\nSchematic: two"
        return {"page": page_index + 1}, f"page {page_index + 1}: failed"

    monkeypatch.setattr(_procedure, "_process_schematic", process)

    assert await _procedure._run_profile(path) == {
        "state": "failed",
        "reason": "page 1: failed; page 2: failed",
        "full_text": "Schematic: one\n\nSchematic: two",
        "schematics": [{"page": 1}, {"page": 2}],
        "schematics_total": 2,
        "classification": None,
    }


async def test_run_profile_success_reason_is_exact(monkeypatch, tmp_path):
    path = tmp_path / "procedure.pdf"
    monkeypatch.setattr(
        _procedure, "_extract_pages_text_sync", lambda received: ["Schematic: one"]
    )
    monkeypatch.setattr(_procedure, "max_schematics", lambda: 1)
    monkeypatch.setattr(_procedure, "_advisory_classification", lambda received: None)
    monkeypatch.setattr(
        _procedure,
        "_process_schematic",
        lambda received, page_index, full_text: _async(({"page": 1}, None)),
    )

    assert await _procedure._run_profile(path) == {
        "state": "pending",
        "reason": "ok",
        "full_text": "Schematic: one",
        "schematics": [{"page": 1}],
        "schematics_total": 1,
        "classification": None,
    }


async def test_selected_processing_reservation_and_settlement_are_exact(
    monkeypatch, tmp_path
):
    path = _write(tmp_path, "procedure.pdf", b"pdf")
    monkeypatch.setattr(_procedure, "_content_hash_sync", lambda received: "hash")
    reserve_calls = []

    def reserve(received, track_id, source, content_hash, from_scan):
        reserve_calls.append((received, track_id, source, content_hash, from_scan))
        return {"id": "bundle-3"}, True

    fields = {
        "state": "pending",
        "reason": "ok",
        "full_text": "text",
        "schematics": [],
    }
    settle_calls = []

    async def settle(bundle_id, received, source, **kwargs):
        settle_calls.append((bundle_id, received, source, kwargs))
        return _procedure.ProcedureOutcome(bundle_id, kwargs["state"], kwargs["reason"])

    profile_calls = []

    async def run_profile(received):
        profile_calls.append(received)
        return fields

    monkeypatch.setattr(_procedure, "_reserve_sync", reserve)
    monkeypatch.setattr(_procedure, "_run_profile", run_profile)
    monkeypatch.setattr(_procedure, "_settle", settle)

    outcome = await _procedure._aprocess_selected(path, "track-3", "detected", True)

    assert reserve_calls == [(path, "track-3", "detected", "hash", True)]
    assert profile_calls == [path]
    assert settle_calls == [("bundle-3", path, "detected", fields)]
    assert outcome == _procedure.ProcedureOutcome("bundle-3", "pending", "ok")


async def test_selected_processing_failure_settles_exactly(
    monkeypatch, tmp_path, caplog
):
    path = _write(tmp_path, "procedure.pdf", b"pdf")
    monkeypatch.setattr(_procedure, "_content_hash_sync", lambda received: "hash")
    monkeypatch.setattr(
        _procedure,
        "_reserve_sync",
        lambda received, track_id, source, content_hash, from_scan: (
            {"id": "bundle-failure"},
            True,
        ),
    )

    async def fail_profile(received):
        assert received == path
        raise RuntimeError("vision exploded")

    settled = []

    async def settle(bundle_id, received, source, **kwargs):
        settled.append((bundle_id, received, source, kwargs))
        return _procedure.ProcedureOutcome(bundle_id, kwargs["state"], kwargs["reason"])

    monkeypatch.setattr(_procedure, "_run_profile", fail_profile)
    monkeypatch.setattr(_procedure, "_settle", settle)

    with caplog.at_level("ERROR"):
        outcome = await _procedure._aprocess_selected(
            path, "track-failure", "forced", False
        )

    reason = "procedure-error: RuntimeError: vision exploded"
    assert settled == [
        (
            "bundle-failure",
            path,
            "forced",
            {"state": "failed", "reason": reason},
        )
    ]
    assert outcome == _procedure.ProcedureOutcome("bundle-failure", "failed", reason)
    assert [record.getMessage() for record in caplog.records] == [
        "twindb procedure: procedure.pdf — unexpected processing error"
    ]


async def test_fail_closed_park_contract_and_store_failure_are_exact(
    monkeypatch, tmp_path, caplog
):
    path = tmp_path / "procedure.pdf"
    calls = []

    async def park(received, track_id, **kwargs):
        calls.append((received, track_id, kwargs))
        return _procedure.ProcedureOutcome("bundle-closed", "failed", kwargs["reason"])

    monkeypatch.setattr(_procedure, "_park", park)
    outcome = await _procedure._fail_closed_park(
        path, "track-closed", "forced", "unsupported"
    )
    assert calls == [
        (
            path,
            "track-closed",
            {
                "state": "failed",
                "reason": "unsupported",
                "source": "forced",
                "content_hash": None,
                "full_text": "",
                "schematics": [],
                "schematics_total": 0,
                "classification": None,
            },
        )
    ]
    assert outcome == _procedure.ProcedureOutcome(
        "bundle-closed", "failed", "unsupported"
    )

    async def fail_park(*_args, **_kwargs):
        raise OSError("store offline")

    monkeypatch.setattr(_procedure, "_park", fail_park)
    with caplog.at_level("ERROR"):
        outcome = await _procedure._fail_closed_park(
            path, "track-closed", "detected", "processing failed"
        )
    assert outcome == _procedure.ProcedureOutcome(None, "error", "processing failed")
    assert [record.getMessage() for record in caplog.records] == [
        "twindb procedure: procedure.pdf — could not even park a failed "
        "bundle ; refusing the enqueue"
    ]


async def test_aprocess_selection_and_unexpected_failure_arguments_are_exact(
    monkeypatch, tmp_path, caplog
):
    path = _write(tmp_path, "procedure.pdf", b"pdf")
    monkeypatch.setattr(
        _procedure, "_guard_store_and_find_existing", lambda received: _async(None)
    )
    monkeypatch.setattr(
        _procedure, "_forced_selection_block_reason", lambda: _async(None)
    )
    monkeypatch.setattr(_procedure, "_forced_validation_problem", lambda received: None)
    selected = []

    async def process_selected(received, track_id, source, from_scan):
        selected.append((received, track_id, source, from_scan))
        raise RuntimeError("pipeline down")

    parked = []

    async def fail_closed(received, track_id, source, reason):
        parked.append((received, track_id, source, reason))
        return _procedure.ProcedureOutcome("bundle-error", "failed", reason)

    monkeypatch.setattr(_procedure, "_aprocess_selected", process_selected)
    monkeypatch.setattr(_procedure, "_fail_closed_park", fail_closed)

    with doc_type_context("procedure"), caplog.at_level("ERROR"):
        outcome = await _procedure.aprocess_procedure(
            path, "track-process", from_scan=True
        )

    reason = "procedure-error: RuntimeError: pipeline down"
    assert selected == [(path, "track-process", "forced", True)]
    assert parked == [(path, "track-process", "forced", reason)]
    assert outcome == _procedure.ProcedureOutcome("bundle-error", "failed", reason)
    assert [record.getMessage() for record in caplog.records] == [
        "twindb procedure: procedure.pdf — unexpected processing error"
    ]

    selected.clear()
    parked.clear()
    caplog.clear()
    monkeypatch.setattr(
        _procedure, "_auto_detected_procedure", lambda received: _async(True)
    )

    async def successful(received, track_id, source, from_scan):
        selected.append((received, track_id, source, from_scan))
        return _procedure.ProcedureOutcome("bundle-detected", "pending", "ok")

    monkeypatch.setattr(_procedure, "_aprocess_selected", successful)
    outcome = await _procedure.aprocess_procedure(path, "track-detected")
    assert selected == [(path, "track-detected", "detected", False)]
    assert outcome == _procedure.ProcedureOutcome("bundle-detected", "pending", "ok")


async def test_aprocess_forced_block_and_validation_park_arguments_are_exact(
    monkeypatch, tmp_path
):
    path = _write(tmp_path, "procedure.pdf", b"pdf")
    monkeypatch.setattr(
        _procedure, "_guard_store_and_find_existing", lambda received: _async(None)
    )
    parked = []

    async def fail_closed(received, track_id, source, reason):
        parked.append((received, track_id, source, reason))
        return _procedure.ProcedureOutcome("bundle-forced", "failed", reason)

    monkeypatch.setattr(_procedure, "_fail_closed_park", fail_closed)
    monkeypatch.setattr(
        _procedure,
        "_forced_selection_block_reason",
        lambda: _async("admin-disabled"),
    )

    with doc_type_context("procedure"):
        outcome = await _procedure.aprocess_procedure(path, "track-blocked")
    assert parked == [(path, "track-blocked", "forced", "admin-disabled")]
    assert outcome == _procedure.ProcedureOutcome(
        "bundle-forced", "failed", "admin-disabled"
    )

    parked.clear()
    monkeypatch.setattr(
        _procedure, "_forced_selection_block_reason", lambda: _async(None)
    )
    monkeypatch.setattr(
        _procedure, "_forced_validation_problem", lambda received: "too-large"
    )
    with doc_type_context("procedure"):
        outcome = await _procedure.aprocess_procedure(path, "track-invalid")
    assert parked == [(path, "track-invalid", "forced", "too-large")]
    assert outcome == _procedure.ProcedureOutcome(
        "bundle-forced", "failed", "too-large"
    )


async def _async(value):
    return value


async def test_retry_bundle_success_and_failure_funnels_are_exact(
    monkeypatch, tmp_path, caplog
):
    path = _write(tmp_path, "procedure.pdf", b"pdf")
    transitions = []

    def transition(bundle_id, expected_states, **kwargs):
        transitions.append((bundle_id, expected_states, kwargs))
        return {
            "id": bundle_id,
            "original_path": str(path),
            "source": "forced",
        }

    fields = {"state": "pending", "reason": "ok", "schematics": []}
    settle_calls = []
    profile_calls = []

    async def settle(bundle_id, received, source, **kwargs):
        settle_calls.append((bundle_id, received, source, kwargs))
        return _procedure.ProcedureOutcome(bundle_id, kwargs["state"], kwargs["reason"])

    async def run_profile(received):
        profile_calls.append(received)
        return fields

    monkeypatch.setattr(_procedure_store, "transition_bundle", transition)
    monkeypatch.setattr(_procedure, "_run_profile", run_profile)
    monkeypatch.setattr(_procedure, "_settle", settle)

    outcome = await _procedure.aretry_bundle("bundle-4")

    assert transitions == [
        (
            "bundle-4",
            _procedure.RETRYABLE_STATES,
            {"state": "processing", "reason": "processing (retry)"},
        )
    ]
    assert settle_calls == [("bundle-4", path, "forced", fields)]
    assert profile_calls == [path]
    assert outcome == _procedure.ProcedureOutcome("bundle-4", "pending", "ok")

    def fail_transition(*_args, **_kwargs):
        raise RuntimeError("store down")

    monkeypatch.setattr(_procedure_store, "transition_bundle", fail_transition)
    with caplog.at_level("ERROR"):
        outcome = await _procedure.aretry_bundle("bundle-5")
    assert outcome == _procedure.ProcedureOutcome(
        "bundle-5", "error", "retry-error: RuntimeError: store down"
    )
    assert [record.getMessage() for record in caplog.records] == [
        "twindb procedure: retry of bundle-5 — retry-error: RuntimeError: store down"
    ]


async def test_retry_defaults_and_profile_failure_are_exact(
    monkeypatch, tmp_path, caplog
):
    path = _write(tmp_path, "procedure.pdf", b"pdf")
    monkeypatch.setattr(
        _procedure_store,
        "transition_bundle",
        lambda *_args, **_kwargs: {"original_path": str(path)},
    )

    async def fail_profile(received):
        assert received == path
        raise RuntimeError("vision down")

    settled = []

    async def settle(bundle_id, received, source, **kwargs):
        settled.append((bundle_id, received, source, kwargs))
        return _procedure.ProcedureOutcome(bundle_id, kwargs["state"], kwargs["reason"])

    monkeypatch.setattr(_procedure, "_run_profile", fail_profile)
    monkeypatch.setattr(_procedure, "_settle", settle)
    with caplog.at_level("ERROR"):
        outcome = await _procedure.aretry_bundle("bundle-defaults")

    reason = "procedure-error: RuntimeError: vision down"
    assert settled == [
        (
            "bundle-defaults",
            path,
            "detected",
            {"state": "failed", "reason": reason},
        )
    ]
    assert outcome == _procedure.ProcedureOutcome("bundle-defaults", "failed", reason)
    assert [record.getMessage() for record in caplog.records] == [
        "twindb procedure: procedure.pdf — unexpected retry error"
    ]


async def test_retry_missing_fields_settles_on_empty_path_and_detected_source(
    monkeypatch,
):
    monkeypatch.setattr(
        _procedure_store, "transition_bundle", lambda *_args, **_kwargs: {}
    )
    settled = []

    async def settle(bundle_id, path, source, **kwargs):
        settled.append((bundle_id, path, source, kwargs))
        return _procedure.ProcedureOutcome(bundle_id, kwargs["state"], kwargs["reason"])

    monkeypatch.setattr(_procedure, "_settle", settle)

    outcome = await _procedure.aretry_bundle("bundle-empty")

    reason = (
        "original-missing: the source file left the input directory — "
        "re-upload the document"
    )
    assert settled == [
        (
            "bundle-empty",
            _procedure.Path(""),
            "detected",
            {"state": "failed", "reason": reason},
        )
    ]
    assert outcome == _procedure.ProcedureOutcome("bundle-empty", "failed", reason)


async def test_retry_missing_original_settles_exact_reason(monkeypatch, tmp_path):
    missing = tmp_path / "missing.pdf"
    monkeypatch.setattr(
        _procedure_store,
        "transition_bundle",
        lambda *_args, **_kwargs: {
            "original_path": str(missing),
            "source": "detected",
        },
    )
    calls = []

    async def settle(bundle_id, path, source, **kwargs):
        calls.append((bundle_id, path, source, kwargs))
        return _procedure.ProcedureOutcome(bundle_id, kwargs["state"], kwargs["reason"])

    monkeypatch.setattr(_procedure, "_settle", settle)

    outcome = await _procedure.aretry_bundle("bundle-6")

    reason = (
        "original-missing: the source file left the input directory — "
        "re-upload the document"
    )
    assert calls == [
        ("bundle-6", missing, "detected", {"state": "failed", "reason": reason})
    ]
    assert outcome == _procedure.ProcedureOutcome("bundle-6", "failed", reason)


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


async def test_timeout_mid_dual_pass_preserves_independent_results(
    monkeypatch, tmp_path
):
    path = tmp_path / "procedure.pdf"
    stages = []

    async def call(_messages, stage, _validate=None):
        stages.append(stage)
        if stage == "informed-pass":
            raise asyncio.TimeoutError
        if stage == "blind-pass":
            return {"title": "blind", "description": "observed", "tasks": []}
        return {"coherent": False, "divergences": ["missing T2.1"], "summary": "gap"}

    monkeypatch.setattr(_procedure, "_vision_json_call", call)
    monkeypatch.setattr(
        _procedure, "_render_page_png_sync", lambda _path, _page: b"png"
    )

    entry, error = await _procedure._process_schematic(path, 2, "full text")

    assert stages == ["blind-pass", "informed-pass", "comparator"]
    assert entry["blind"] == {
        "title": "blind",
        "description": "observed",
        "tasks": [],
    }
    assert entry["informed"] is None
    assert entry["divergence"] == {
        "coherent": False,
        "divergences": ["missing T2.1"],
        "summary": "gap",
    }
    assert error == "page 3: schematic-timeout after 60s"
    assert entry["error"] == error


async def test_process_schematic_preserves_dual_pass_protocol_exactly(
    monkeypatch, tmp_path
):
    path = tmp_path / "procedure.pdf"
    calls = []
    renders = []
    blind = {"title": "blind é", "description": "observed 🚀", "tasks": []}
    informed = {"title": "informed", "description": "canonical", "tasks": []}
    divergence = {"coherent": True, "divergences": [], "summary": "aligned"}

    async def call(messages, stage, validate=None):
        calls.append((messages, stage, validate))
        return {
            "blind-pass": blind,
            "informed-pass": informed,
            "comparator": divergence,
        }[stage]

    monkeypatch.setattr(_procedure, "_vision_json_call", call)

    def render(received, page):
        renders.append((received, page))
        return b"png"

    monkeypatch.setattr(_procedure, "_render_page_png_sync", render)

    entry, error = await _procedure._process_schematic(path, 1, "full text 🚀")

    data_url = f"data:image/png;base64,{base64.b64encode(b'png').decode('ascii')}"
    image_part = {"type": "image_url", "image_url": {"url": data_url}}
    assert entry == {
        "page": 2,
        "png_base64": base64.b64encode(b"png").decode("ascii"),
        "blind": blind,
        "informed": informed,
        "divergence": divergence,
        "error": None,
    }
    assert error is None
    assert renders == [(path, 1)]
    assert [(messages, stage) for messages, stage, _validate in calls] == [
        (
            [
                {"role": "system", "content": _procedure.BLIND_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this process schematic."},
                        image_part,
                    ],
                },
            ],
            "blind-pass",
        ),
        (
            [
                {"role": "system", "content": _procedure.INFORMED_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Full document text:\n\nfull text 🚀\n\n"
                                "Describe this process schematic."
                            ),
                        },
                        image_part,
                    ],
                },
            ],
            "informed-pass",
        ),
        (
            [
                {"role": "system", "content": _procedure.COMPARATOR_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Full document text:\n\nfull text 🚀\n\n"
                        "Blind schematic description (JSON):\n\n"
                        + json.dumps(blind, ensure_ascii=False)
                    ),
                },
            ],
            "comparator",
        ),
    ]
    assert calls[0][2].keywords == {"stage": "blind-pass"}
    assert calls[1][2].keywords == {"stage": "informed-pass"}
    assert calls[2][2] is _procedure._validate_comparator_payload


async def test_blind_pass_failure_does_not_abort_independent_informed_pass(
    monkeypatch, tmp_path
):
    path = tmp_path / "procedure.pdf"
    stages = []

    async def call(_messages, stage, _validate=None):
        stages.append(stage)
        if stage == "blind-pass":
            raise ValueError("blind malformed")
        return {"title": "informed", "description": "kept", "tasks": []}

    monkeypatch.setattr(_procedure, "_vision_json_call", call)
    monkeypatch.setattr(
        _procedure, "_render_page_png_sync", lambda _path, _page: b"png"
    )

    entry, error = await _procedure._process_schematic(path, 0, "full text")

    assert stages == ["blind-pass", "informed-pass"]
    assert entry["blind"] is None
    assert entry["informed"] == {
        "title": "informed",
        "description": "kept",
        "tasks": [],
    }
    assert entry["divergence"] is None
    assert error == "page 1: ValueError: blind malformed"


async def test_comparator_failure_is_preserved_as_exact_partial_error(
    monkeypatch, tmp_path
):
    path = tmp_path / "procedure.pdf"
    valid = {"title": "flow", "description": "steps", "tasks": []}

    async def call(_messages, stage, _validate=None):
        if stage == "comparator":
            raise RuntimeError("audit unavailable")
        return valid

    monkeypatch.setattr(_procedure, "_vision_json_call", call)
    monkeypatch.setattr(
        _procedure, "_render_page_png_sync", lambda _path, _page: b"png"
    )

    entry, error = await _procedure._process_schematic(path, 4, "full text")

    assert entry["blind"] == valid
    assert entry["informed"] == valid
    assert entry["divergence"] is None
    assert error == "page 5: RuntimeError: audit unavailable"


async def test_task_entries_validated_to_the_eight_fields(monkeypatch, tmp_path):
    """tasks=[{}] parses as JSON but violates the prompt contract."""
    pytest.importorskip("pypdf")

    def sloppy_chat(_messages, **_kwargs):
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

    def bad_chat(_messages, **_kwargs):
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
    profile_ready, monkeypatch, tmp_path, caplog
):
    def boom():
        raise OSError("io error")

    monkeypatch.setattr(_procedure._procedure_store, "claimed_paths", boom)
    monkeypatch.setenv("TWIN_PROCEDURE_MAX_BYTES", "4")
    pdf = _write(tmp_path, "doc.pdf", b"%PDF-1.4 over the tiny cap")
    with caplog.at_level("ERROR"):
        assert await _procedure.aroute_check(pdf) is True
    assert [record.getMessage() for record in caplog.records] == [
        "twindb procedure: rescan guard cannot read the store for doc.pdf "
        "— failing CLOSED, the file routes to the profile"
    ]


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

    async def fake_route(_path):
        return True

    monkeypatch.setattr(_procedure, "aroute_check", fake_route)
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

    async def fake_route(_path):
        return True

    monkeypatch.setattr(_procedure, "aroute_check", fake_route)
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

    async def fake_route(_path):
        return True

    monkeypatch.setattr(_procedure, "aroute_check", fake_route)

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

    async def fake_route(_path):
        return True

    monkeypatch.setattr(_procedure, "aroute_check", fake_route)

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


async def test_shape_failure_is_retried_like_a_parse_failure(monkeypatch):
    """A corrupted key is transient endpoint noise, not a permanent verdict.

    Measured 2026-07-25 against the real model: an otherwise perfect object
    came back with `{@title` instead of `title`. Validation used to run AFTER
    the retry loop, so one mangled character lost the whole schematic for good.
    """
    replies = iter(
        [
            json.dumps({"{@title": "T", "description": "D", "tasks": []}),
            json.dumps({"title": "T", "description": "D", "tasks": []}),
        ]
    )
    calls = 0

    def chat(_messages, **_kwargs):
        nonlocal calls
        calls += 1
        return next(replies)

    monkeypatch.setattr(_vision, "vision_chat_sync", chat)

    result = await _procedure._vision_json_call(
        [],
        "informed-pass",
        functools.partial(_procedure._validate_pass_payload, stage="informed-pass"),
    )

    assert result["title"] == "T"
    assert calls == 2, "a shape failure must consume the retry"


async def test_shape_failure_still_fails_after_the_retry(monkeypatch):
    """Retrying is not hiding: a persistently wrong shape still fails loudly."""
    monkeypatch.setattr(
        _vision,
        "vision_chat_sync",
        lambda _m, **_k: json.dumps({"{@title": "T", "description": "D", "tasks": []}),
    )

    with pytest.raises(ValueError, match="does not match the expected shape"):
        await _procedure._vision_json_call(
            [],
            "informed-pass",
            functools.partial(_procedure._validate_pass_payload, stage="informed-pass"),
        )
