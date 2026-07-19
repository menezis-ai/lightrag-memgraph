"""Vision image-ingestion tier (MARKITDOWN-INGESTION-PLAN.md, PR 2).

Unit tests on ``_vision`` (config gates, OCR pre-filter, drop classes,
markdown composition, tolerant JSON parse, failure degradation) plus
seam-contract tests on the registry routing (image → vision pipeline,
refusal → explicit error-document, non-image path untouched). The vision
LLM and RapidOCR are always monkeypatched — no network, no model files.
"""

import sys

import pytest

from twindb_lightrag_memgraph import _conversion, _vision
from twindb_lightrag_memgraph.patches import registry

VISION_ENV_VARS = (
    "TWIN_VISION",
    "TWIN_VISION_BASE_URL",
    "TWIN_VISION_API_KEY",
    "TWIN_VISION_MODEL",
    "TWIN_VISION_FORMATS",
    "TWIN_VISION_MAX_BYTES",
    "TWIN_VISION_TIMEOUT",
    "TWIN_VISION_MIN_OCR_CHARS",
    "TWIN_VISION_DROP_CLASSES",
)


@pytest.fixture(autouse=True)
def _clean_vision_env(monkeypatch):
    for var in VISION_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    _vision.reset_caches()
    yield
    _vision.reset_caches()


@pytest.fixture
def vision_configured(monkeypatch):
    monkeypatch.setenv("TWIN_VISION_BASE_URL", "http://vllm.internal/v1")
    monkeypatch.setenv("TWIN_VISION_MODEL", "gemma-4-31b-it")
    monkeypatch.setattr(_vision, "_openai_importable", lambda: True)


def _write_png(tmp_path, name="img.png", size=64):
    path = tmp_path / name
    path.write_bytes(b"\x89PNG" + b"\x00" * size)
    return path


# ---------------------------------------------------------------------------
# Config gates
# ---------------------------------------------------------------------------


def test_disabled_without_endpoint_config(monkeypatch):
    monkeypatch.setattr(_vision, "_openai_importable", lambda: True)
    assert _vision.is_enabled() is False  # auto + no endpoint


def test_enabled_with_endpoint_config(vision_configured):
    assert _vision.is_enabled() is True


def test_mode_off_wins_over_config(vision_configured, monkeypatch):
    monkeypatch.setenv("TWIN_VISION", "off")
    assert _vision.is_enabled() is False


def test_mode_on_without_config_warns_and_disables(monkeypatch, caplog):
    monkeypatch.setenv("TWIN_VISION", "on")
    monkeypatch.setattr(_vision, "_openai_importable", lambda: True)
    with caplog.at_level("WARNING"):
        assert _vision.is_enabled() is False
    assert "TWIN_VISION=on but the tier is not usable" in caplog.text


def test_should_process_gates(vision_configured, monkeypatch, tmp_path):
    image = _write_png(tmp_path)
    text = tmp_path / "notes.txt"
    text.write_text("x")

    assert _vision.should_process(image) is True
    assert _vision.should_process(text) is False
    assert _vision.should_process(tmp_path / "ghost.png") is False

    monkeypatch.setenv("TWIN_VISION_MAX_BYTES", "8")
    assert _vision.should_process(image) is False


def test_extra_supported_extensions_default():
    assert _vision.extra_supported_extensions() == (".jpeg", ".jpg", ".png")


# ---------------------------------------------------------------------------
# Pipeline behavior (OCR + LLM monkeypatched)
# ---------------------------------------------------------------------------


async def test_prefilter_refuses_low_text_image(
    vision_configured, monkeypatch, tmp_path
):
    monkeypatch.setattr(_vision, "_ocr_text_sync", lambda _p: "hi")

    def no_llm(_p):
        raise AssertionError("vision LLM must not be called under the pre-filter")

    monkeypatch.setattr(_vision, "_call_vision_llm_sync", no_llm)
    outcome = await _vision.aprocess_image(_write_png(tmp_path))
    assert outcome.markdown is None
    assert "vision-prefilter" in outcome.reason


async def test_prefilter_disabled_with_zero_threshold(
    vision_configured, monkeypatch, tmp_path
):
    monkeypatch.setenv("TWIN_VISION_MIN_OCR_CHARS", "0")
    monkeypatch.setattr(_vision, "_ocr_text_sync", lambda _p: "")
    monkeypatch.setattr(
        _vision,
        "_call_vision_llm_sync",
        lambda _p: '{"image_classification": "diagram", "content": "An architecture diagram."}',
    )
    outcome = await _vision.aprocess_image(_write_png(tmp_path))
    assert outcome.markdown is not None
    assert outcome.classification == "diagram"


async def test_missing_rapidocr_bypasses_prefilter(
    vision_configured, monkeypatch, tmp_path
):
    monkeypatch.setattr(_vision, "_ocr_text_sync", lambda _p: None)
    monkeypatch.setattr(
        _vision,
        "_call_vision_llm_sync",
        lambda _p: '{"image_classification": "photo", "content": "A datacenter rack."}',
    )
    outcome = await _vision.aprocess_image(_write_png(tmp_path))
    assert outcome.markdown is not None


async def test_drop_classes_refuse_noise_images(
    vision_configured, monkeypatch, tmp_path
):
    monkeypatch.setattr(_vision, "_ocr_text_sync", lambda _p: "BNP PARIBAS " * 3)
    monkeypatch.setattr(
        _vision,
        "_call_vision_llm_sync",
        lambda _p: '{"image_classification": "Logo", "content": "BNP logo."}',
    )
    outcome = await _vision.aprocess_image(_write_png(tmp_path))
    assert outcome.markdown is None
    assert "image-dropped" in outcome.reason
    assert outcome.classification == "Logo"


async def test_markdown_composition_includes_ocr_section(
    vision_configured, monkeypatch, tmp_path
):
    monkeypatch.setattr(
        _vision, "_ocr_text_sync", lambda _p: "Total HT 1 200 EUR TVA 240 EUR"
    )
    monkeypatch.setattr(
        _vision,
        "_call_vision_llm_sync",
        lambda _p: '{"image_classification": "table", "content": "Invoice totals table."}',
    )
    path = _write_png(tmp_path, name="facture.png")
    outcome = await _vision.aprocess_image(path)
    assert outcome.markdown is not None
    assert outcome.markdown.startswith("# facture.png")
    assert "_Image type: table_" in outcome.markdown
    assert "Invoice totals table." in outcome.markdown
    assert "## Extracted text (OCR)" in outcome.markdown
    assert "Total HT 1 200 EUR" in outcome.markdown


async def test_fenced_json_reply_is_parsed(vision_configured, monkeypatch, tmp_path):
    monkeypatch.setattr(_vision, "_ocr_text_sync", lambda _p: "x" * 40)
    monkeypatch.setattr(
        _vision,
        "_call_vision_llm_sync",
        lambda _p: 'Here it is:\n```json\n{"image_classification": "screenshot", "content": "A settings page."}\n```',
    )
    outcome = await _vision.aprocess_image(_write_png(tmp_path))
    assert outcome.markdown is not None
    assert outcome.classification == "screenshot"


async def test_unparseable_reply_refuses(vision_configured, monkeypatch, tmp_path):
    monkeypatch.setattr(_vision, "_ocr_text_sync", lambda _p: "x" * 40)
    monkeypatch.setattr(_vision, "_call_vision_llm_sync", lambda _p: "not json at all")
    outcome = await _vision.aprocess_image(_write_png(tmp_path))
    assert outcome.markdown is None
    assert "unparseable" in outcome.reason


async def test_llm_error_refuses_with_reason(vision_configured, monkeypatch, tmp_path):
    monkeypatch.setattr(_vision, "_ocr_text_sync", lambda _p: "x" * 40)

    def boom(_p):
        raise ConnectionError("endpoint down")

    monkeypatch.setattr(_vision, "_call_vision_llm_sync", boom)
    outcome = await _vision.aprocess_image(_write_png(tmp_path))
    assert outcome.markdown is None
    assert "vision-llm-error" in outcome.reason
    assert "endpoint down" in outcome.reason


async def test_pipeline_timeout_refuses(vision_configured, monkeypatch, tmp_path):
    import time

    monkeypatch.setenv("TWIN_VISION_TIMEOUT", "0.05")
    monkeypatch.setattr(_vision, "_ocr_text_sync", lambda _p: time.sleep(1) or "late")
    outcome = await _vision.aprocess_image(_write_png(tmp_path))
    assert outcome.markdown is None
    assert "vision-timeout" in outcome.reason


# ---------------------------------------------------------------------------
# Registry seam routing
# ---------------------------------------------------------------------------


class _FakeRag:
    def __init__(self):
        self.enqueue_calls = []
        self.error_calls = []

    async def apipeline_enqueue_documents(self, content, **kwargs):
        self.enqueue_calls.append((content, kwargs))

    async def apipeline_enqueue_error_documents(self, error_files, track_id):
        self.error_calls.append((error_files, track_id))


@pytest.fixture
def dr_module(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["pytest"])
    dr = pytest.importorskip(
        "lightrag.api.routers.document_routes",
        reason="native document routes unavailable on this LightRAG",
    )
    saved_enqueue = dr.pipeline_enqueue_file
    had_sentinel = getattr(dr, "_twindb_convert_enqueue_patched", None)
    yield dr
    dr.pipeline_enqueue_file = saved_enqueue
    if had_sentinel is None:
        if hasattr(dr, "_twindb_convert_enqueue_patched"):
            delattr(dr, "_twindb_convert_enqueue_patched")
    else:
        dr._twindb_convert_enqueue_patched = had_sentinel


def _install_patch(dr, fake_orig):
    dr.pipeline_enqueue_file = fake_orig
    dr._twindb_convert_enqueue_patched = False
    registry._patch_pipeline_enqueue_conversion()
    return dr.pipeline_enqueue_file


async def test_seam_routes_image_to_vision_and_enqueues(
    dr_module, vision_configured, monkeypatch, tmp_path
):
    async def fake_orig(rag, file_path, *args, **kwargs):
        raise AssertionError("native path must not run for a vision image")

    wrapped = _install_patch(dr_module, fake_orig)
    monkeypatch.setattr(_conversion, "should_convert", lambda _p: False)

    async def fake_process(_p):
        return _vision.VisionOutcome(
            markdown="# schema.png\n\ncontent", reason="ok", classification="diagram"
        )

    monkeypatch.setattr(_vision, "aprocess_image", fake_process)

    rag = _FakeRag()
    path = _write_png(tmp_path, name="schema.png")
    result = await wrapped(rag, path, "tid-img")

    assert result == (True, "tid-img")
    assert rag.enqueue_calls == [
        ("# schema.png\n\ncontent", {"file_paths": "schema.png", "track_id": "tid-img"})
    ]
    assert rag.error_calls == []


async def test_seam_reports_error_document_on_vision_refusal(
    dr_module, vision_configured, monkeypatch, tmp_path
):
    async def fake_orig(rag, file_path, *args, **kwargs):
        raise AssertionError("native path must not run for a vision image")

    wrapped = _install_patch(dr_module, fake_orig)
    monkeypatch.setattr(_conversion, "should_convert", lambda _p: False)

    async def fake_process(_p):
        return _vision.VisionOutcome(
            markdown=None, reason="image-dropped: classification 'logo'"
        )

    monkeypatch.setattr(_vision, "aprocess_image", fake_process)

    rag = _FakeRag()
    path = _write_png(tmp_path, name="logo.png")
    success, track_id = await wrapped(rag, path, "tid-logo")

    assert success is False
    assert track_id == "tid-logo"
    assert rag.enqueue_calls == []
    error_files, reported_track = rag.error_calls[0]
    assert reported_track == "tid-logo"
    assert error_files[0]["file_path"] == "logo.png"
    assert "image-dropped" in error_files[0]["original_error"]


async def test_seam_leaves_non_image_non_convert_untouched(
    dr_module, vision_configured, monkeypatch, tmp_path
):
    seen = {}

    async def fake_orig(rag, file_path, *args, **kwargs):
        seen["call"] = (rag, file_path, args, kwargs)
        return True, "native"

    wrapped = _install_patch(dr_module, fake_orig)
    monkeypatch.setattr(_conversion, "should_convert", lambda _p: False)

    rag = object()
    path = tmp_path / "notes.txt"
    path.write_text("plain")
    result = await wrapped(rag, path, "tid-n")

    assert result == (True, "native")
    assert seen["call"] == (rag, path, ("tid-n",), {})
