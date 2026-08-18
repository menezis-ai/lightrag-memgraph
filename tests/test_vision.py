"""Vision image-ingestion tier (MARKITDOWN-INGESTION-PLAN.md, PR 2).

Unit tests on ``_vision`` (config gates, OCR pre-filter, drop classes,
markdown composition, tolerant JSON parse, failure degradation) plus
seam-contract tests on the registry routing (image → vision pipeline,
refusal → explicit error-document, non-image path untouched). The vision
LLM and RapidOCR are always monkeypatched — no network, no model files.
"""

import base64
import builtins
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

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
    "TWIN_VISION_EXTRA_BODY",
)


@pytest.fixture(autouse=True)
def _clean_vision_env(monkeypatch):
    for var in VISION_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    # This suite pins the RAW enqueue contract on every LightRAG line; the
    # preconverted-parse seam has its own suite (test_preconverted_parse.py).
    monkeypatch.setenv("TWIN_PRECONVERTED_PARSE", "off")
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


def test_mode_and_endpoint_configuration_are_strict(monkeypatch):
    assert _vision._resolve_mode() is None
    assert _vision._endpoint_configured() is False

    monkeypatch.setenv("TWIN_VISION", "  On  ")
    assert _vision._resolve_mode() is True
    monkeypatch.setenv("TWIN_VISION", "NO")
    assert _vision._resolve_mode() is False
    monkeypatch.setenv("TWIN_VISION", "unexpected")
    assert _vision._resolve_mode() is None

    monkeypatch.setenv("TWIN_VISION_BASE_URL", "http://vision.test/v1")
    assert _vision._endpoint_configured() is False
    monkeypatch.delenv("TWIN_VISION_BASE_URL")
    monkeypatch.setenv("TWIN_VISION_MODEL", "model")
    assert _vision._endpoint_configured() is False
    monkeypatch.setenv("TWIN_VISION_BASE_URL", "http://vision.test/v1")
    assert _vision._endpoint_configured() is True


def test_openai_import_probe_caches_import_failure(monkeypatch):
    real_import = builtins.__import__
    attempts = 0

    def fail_openai(name, *args, **kwargs):
        nonlocal attempts
        if name == "openai":
            attempts += 1
            raise ImportError("not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_openai)

    assert _vision._openai_importable() is False
    assert _vision._openai_importable() is False
    assert attempts == 1


def test_openai_import_probe_caches_success(monkeypatch):
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace())

    assert _vision._openai_importable() is True

    monkeypatch.delitem(sys.modules, "openai")
    real_import = builtins.__import__

    def fail_openai(name, *args, **kwargs):
        if name == "openai":
            raise ImportError("removed after the successful probe")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_openai)
    assert _vision._openai_importable() is True


def test_reset_caches_restores_every_lazy_singleton():
    _vision._openai_available = True
    _vision._ocr_engine = object()
    _vision._ocr_missing_warned = True
    _vision._client = object()
    _vision._settings_provider = object()

    _vision.reset_caches()

    assert _vision._openai_available is None
    assert _vision._ocr_engine is None
    assert _vision._ocr_missing_warned is False
    assert _vision._client is None
    assert _vision._settings_provider is None


def test_vision_environment_parsers_preserve_boundary_contracts(monkeypatch):
    assert _vision.vision_formats() == frozenset({"png", "jpg", "jpeg"})
    assert _vision.drop_classes() == frozenset({"invalid", "logo", "signature"})
    assert _vision.min_ocr_chars() == _vision.DEFAULT_MIN_OCR_CHARS
    assert _vision.max_image_bytes() == _vision.DEFAULT_MAX_BYTES
    assert _vision.vision_timeout_seconds() == _vision.DEFAULT_TIMEOUT_SECONDS

    monkeypatch.setenv("TWIN_VISION_FORMATS", " .PNG, JpEg, .gif, ,")
    monkeypatch.setenv("TWIN_VISION_DROP_CLASSES", " Logo, SIGNATURE, ")
    monkeypatch.setenv("TWIN_VISION_MIN_OCR_CHARS", "0")
    monkeypatch.setenv("TWIN_VISION_MAX_BYTES", "1")
    monkeypatch.setenv("TWIN_VISION_TIMEOUT", "0.25")
    assert _vision.vision_formats() == frozenset({"png", "jpeg", "gif"})
    assert _vision.drop_classes() == frozenset({"logo", "signature"})
    assert _vision.min_ocr_chars() == 0
    assert _vision.max_image_bytes() == 1
    assert _vision.vision_timeout_seconds() == 0.25
    assert _vision.extra_supported_extensions() == (".gif", ".jpeg", ".png")

    monkeypatch.setenv("TWIN_VISION_FORMATS", ", ,")
    monkeypatch.setenv("TWIN_VISION_MIN_OCR_CHARS", "-1")
    monkeypatch.setenv("TWIN_VISION_MAX_BYTES", "0")
    monkeypatch.setenv("TWIN_VISION_TIMEOUT", "invalid")
    assert _vision.vision_formats() == _vision.DEFAULT_VISION_FORMATS
    assert _vision.min_ocr_chars() == _vision.DEFAULT_MIN_OCR_CHARS
    assert _vision.max_image_bytes() == _vision.DEFAULT_MAX_BYTES
    assert _vision.vision_timeout_seconds() == _vision.DEFAULT_TIMEOUT_SECONDS
    monkeypatch.setenv("TWIN_VISION_TIMEOUT", "0")
    assert _vision.vision_timeout_seconds() == _vision.DEFAULT_TIMEOUT_SECONDS


async def test_runtime_settings_validate_types_and_normalize_classes(monkeypatch):
    monkeypatch.setenv("TWIN_VISION_MIN_OCR_CHARS", "7")
    monkeypatch.setenv("TWIN_VISION_DROP_CLASSES", "logo")

    async def provider():
        return {
            "min_ocr_chars": False,
            "drop_classes": [" Diagram ", "", 42],
        }

    _vision.set_settings_provider(provider)
    assert await _vision._effective_settings() == (
        7,
        frozenset({"diagram", "42"}),
    )

    async def valid_provider():
        return {"min_ocr_chars": 0, "drop_classes": []}

    _vision.set_settings_provider(valid_provider)
    assert await _vision._effective_settings() == (0, frozenset())


async def test_runtime_settings_failure_falls_back_and_is_observable(
    monkeypatch, caplog
):
    monkeypatch.setenv("TWIN_VISION_MIN_OCR_CHARS", "9")
    monkeypatch.setenv("TWIN_VISION_DROP_CLASSES", "signature")

    async def provider():
        raise RuntimeError("store unavailable")

    _vision.set_settings_provider(provider)
    with caplog.at_level("WARNING"):
        settings = await _vision._effective_settings()

    assert settings == (9, frozenset({"signature"}))
    assert [record.getMessage() for record in caplog.records] == [
        "twindb vision: settings provider failed (RuntimeError: store unavailable) "
        "— env defaults"
    ]


def test_client_lazy_init_is_locked_and_transport_bounded(
    vision_configured, monkeypatch
):
    monkeypatch.setenv("TWIN_VISION_TIMEOUT", "0.125")
    barrier = threading.Barrier(8)
    constructed = []

    def fake_openai(**kwargs):
        constructed.append(kwargs)
        time.sleep(0.02)
        return object()

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=fake_openai))

    def get_client(_index):
        barrier.wait(timeout=1)
        return _vision._get_client()

    with ThreadPoolExecutor(max_workers=8) as executor:
        clients = list(executor.map(get_client, range(8)))

    assert all(client is clients[0] for client in clients)
    assert len(constructed) == 1
    assert constructed[0]["base_url"] == "http://vllm.internal/v1"
    assert constructed[0]["api_key"] == "twin-vision"
    assert constructed[0]["timeout"] == 0.125
    assert constructed[0]["max_retries"] == 0


@pytest.mark.parametrize(
    ("configured_key", "expected_key"),
    [("tenant-secret", "tenant-secret"), ("   ", "twin-vision")],
)
def test_client_api_key_uses_configured_value_or_stable_fallback(
    configured_key, expected_key, vision_configured, monkeypatch
):
    monkeypatch.setenv("TWIN_VISION_API_KEY", configured_key)
    constructed = []

    def fake_openai(**kwargs):
        constructed.append(kwargs)
        return object()

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=fake_openai))

    _vision._get_client()

    assert constructed[0]["api_key"] == expected_key


def test_mode_off_wins_over_config(vision_configured, monkeypatch):
    monkeypatch.setenv("TWIN_VISION", "off")
    assert _vision.is_enabled() is False


def test_mode_on_without_config_warns_and_disables(monkeypatch, caplog):
    monkeypatch.setenv("TWIN_VISION", "on")
    monkeypatch.setattr(_vision, "_openai_importable", lambda: True)
    with caplog.at_level("WARNING"):
        assert _vision.is_enabled() is False
    assert [record.getMessage() for record in caplog.records] == [
        "twindb vision: TWIN_VISION=on but the tier is not usable (openai "
        "importable: True, TWIN_VISION_BASE_URL/TWIN_VISION_MODEL set: False) — "
        "image ingestion disabled"
    ]


def test_mode_on_with_ready_tier_emits_no_warning(
    vision_configured, monkeypatch, caplog
):
    monkeypatch.setenv("TWIN_VISION", "on")

    with caplog.at_level("WARNING"):
        assert _vision.is_enabled() is True

    assert caplog.records == []


def test_should_process_gates(vision_configured, monkeypatch, tmp_path):
    image = _write_png(tmp_path)
    text = tmp_path / "notes.txt"
    text.write_text("x")

    assert _vision.should_process(image) is True

    monkeypatch.setenv("TWIN_VISION", "off")
    assert _vision.should_process(image) is False
    assert _vision.should_process(text) is False
    assert _vision.should_process(tmp_path / "ghost.png") is False

    monkeypatch.delenv("TWIN_VISION")
    monkeypatch.setenv("TWIN_VISION_MAX_BYTES", "8")
    # Size does not relinquish ownership to the native parser. The full
    # processor reports an explicit refusal instead.
    assert _vision.should_process(image) is True


def test_should_process_normalizes_custom_dotted_extension(
    vision_configured, monkeypatch, tmp_path
):
    monkeypatch.setenv("TWIN_VISION_FORMATS", ".GIF")
    image = _write_png(tmp_path, name="DIAGRAM.GIF")
    other = _write_png(tmp_path, name="photo.png")

    assert _vision.should_process(image) is True
    assert _vision.should_process(other) is False


async def test_oversized_image_is_refused_before_ocr(
    vision_configured, monkeypatch, tmp_path, caplog
):
    monkeypatch.setenv("TWIN_VISION_MAX_BYTES", "8")

    def no_ocr(_path):
        raise AssertionError("OCR must not run above the vision size cap")

    monkeypatch.setattr(_vision, "_ocr_text_sync", no_ocr)
    path = _write_png(tmp_path)
    with caplog.at_level("WARNING"):
        outcome = await _vision.aprocess_image(path)

    assert outcome == _vision.VisionOutcome(
        markdown=None,
        reason=(
            "vision-size-limit: image rejected because file size is "
            f"{path.stat().st_size} bytes; configured maximum is 8 bytes"
        ),
    )
    assert [record.getMessage() for record in caplog.records] == [
        "twindb vision: img.png — vision-size-limit: image rejected because file size is "
        f"{path.stat().st_size} bytes; configured maximum is 8 bytes"
    ]


async def test_image_at_exact_size_limit_is_accepted(
    vision_configured, monkeypatch, tmp_path
):
    path = _write_png(tmp_path)
    monkeypatch.setenv("TWIN_VISION_MAX_BYTES", str(path.stat().st_size))
    monkeypatch.setattr(_vision, "_ocr_text_sync", lambda _path: "x" * 20)
    monkeypatch.setattr(
        _vision,
        "_call_vision_llm_sync",
        lambda _path: ('{"image_classification": "diagram", "content": "useful"}'),
    )

    outcome = await _vision.aprocess_image(path)

    assert outcome.reason == "ok"


def test_extra_supported_extensions_default():
    assert _vision.extra_supported_extensions() == (".jpeg", ".jpg", ".png")


def test_ocr_source_normalizes_expected_lines():
    class Engine:
        def __call__(self, source):
            assert source == "image-source"
            return (
                [
                    [[0, 0, 1, 1], "first line", 0.99],
                    [[0, 1, 1, 2], "second line", 0.98],
                ],
                [0.01, 0.02, 0.03],
            )

    _vision._ocr_engine = Engine()

    assert _vision._ocr_source_sync("image-source", "invoice") == (
        "first line second line"
    )


def test_ocr_source_accepts_minimal_two_field_lines():
    class Engine:
        def __call__(self, _source):
            return ([[(0, 0), "minimal line"]], None)

    _vision._ocr_engine = Engine()

    assert _vision._ocr_source_sync("source", "minimal") == "minimal line"


def test_ocr_source_empty_result_is_an_observed_empty_string():
    class Engine:
        def __call__(self, _source):
            return ([], None)

    _vision._ocr_engine = Engine()

    assert _vision._ocr_source_sync("source", "blank") == ""


def test_ocr_source_lazily_constructs_the_engine(monkeypatch):
    constructed = []

    class Engine:
        def __call__(self, source):
            assert source == "source"
            return ([[(0, 0), "lazy text"]], None)

    def rapidocr():
        constructed.append(True)
        return Engine()

    monkeypatch.setitem(
        sys.modules,
        "rapidocr_onnxruntime",
        SimpleNamespace(RapidOCR=rapidocr),
    )

    assert _vision._ocr_source_sync("source", "lazy") == "lazy text"
    assert _vision._ocr_source_sync("source", "lazy") == "lazy text"
    assert constructed == [True]


def test_missing_ocr_dependency_warns_once(monkeypatch, caplog):
    real_import = builtins.__import__

    def fail_rapidocr(name, *args, **kwargs):
        if name == "rapidocr_onnxruntime":
            raise ImportError("not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_rapidocr)

    with caplog.at_level("WARNING"):
        assert _vision._ocr_source_sync("source", "first") is None
        assert _vision._ocr_source_sync("source", "second") is None

    assert _vision._ocr_missing_warned is True
    assert [record.getMessage() for record in caplog.records] == [
        "twindb vision: rapidocr_onnxruntime not installed — OCR pre-filter "
        "bypassed, every image goes to the vision LLM"
    ]


def test_aberrant_ocr_response_bypasses_prefilter_and_warns(caplog):
    class Engine:
        def __call__(self, _source):
            return ([{"text": "not the RapidOCR tuple contract"}], None)

    _vision._ocr_engine = Engine()

    with caplog.at_level("WARNING"):
        text = _vision._ocr_source_sync("image-source", "invoice")

    assert text is None
    assert [record.getMessage() for record in caplog.records] == [
        "twindb vision: OCR returned malformed data for invoice "
        "(unexpected RapidOCR line shape) — pre-filter bypassed"
    ]


@pytest.mark.parametrize(
    "bad_line",
    [
        [(0, 0), object()],
        42,
    ],
)
def test_aberrant_ocr_line_shapes_keep_the_explicit_contract(bad_line, caplog):
    class Engine:
        def __call__(self, _source):
            return ([bad_line], None)

    _vision._ocr_engine = Engine()

    with caplog.at_level("WARNING"):
        text = _vision._ocr_source_sync("image-source", "invoice")

    assert text is None
    assert [record.getMessage() for record in caplog.records] == [
        "twindb vision: OCR returned malformed data for invoice "
        "(unexpected RapidOCR line shape) — pre-filter bypassed"
    ]


def test_ocr_engine_exception_bypasses_prefilter_and_warns(caplog):
    class Engine:
        def __call__(self, _source):
            raise RuntimeError("bad tensor")

    _vision._ocr_engine = Engine()

    with caplog.at_level("WARNING"):
        text = _vision._ocr_source_sync("image-source", "scan")

    assert text is None
    assert [record.getMessage() for record in caplog.records] == [
        "twindb vision: OCR failed for scan (bad tensor) — pre-filter bypassed"
    ]


def test_ocr_text_redecode_only_runs_after_an_empty_fast_path(monkeypatch, tmp_path):
    path = _write_png(tmp_path)
    pixels = object()
    calls = []

    def ocr(source, label):
        calls.append((source, label))
        return "" if len(calls) == 1 else "recovered text"

    monkeypatch.setattr(_vision, "_ocr_source_sync", ocr)

    def decode(received_path):
        assert received_path == path
        return pixels

    monkeypatch.setattr(_vision, "_decode_rgb_pixels", decode)

    assert _vision._ocr_text_sync(path) == "recovered text"
    assert calls == [
        (str(path), path.name),
        (pixels, f"{path.name} (re-decoded)"),
    ]

    calls.clear()
    monkeypatch.setattr(
        _vision, "_ocr_source_sync", lambda source, label: "fast-path text"
    )

    def no_decode(_path):
        raise AssertionError("non-empty OCR must not be decoded twice")

    monkeypatch.setattr(_vision, "_decode_rgb_pixels", no_decode)
    assert _vision._ocr_text_sync(path) == "fast-path text"


def test_decode_rgb_pixels_returns_converted_array(monkeypatch, tmp_path):
    path = tmp_path / "scan.jpg"
    converted = object()
    pixels = object()

    class FakeImage:
        size = (3, 2)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def convert(self, mode):
            assert mode == "RGB"
            return converted

    # Equality is accepted: the budget is an inclusive maximum.
    monkeypatch.setattr(_vision, "OCR_REDECODE_MAX_PIXELS", 6)
    monkeypatch.setitem(
        sys.modules,
        "PIL",
        SimpleNamespace(Image=SimpleNamespace(open=lambda received: FakeImage())),
    )
    monkeypatch.setitem(
        sys.modules,
        "numpy",
        SimpleNamespace(asarray=lambda image: pixels if image is converted else None),
    )

    assert _vision._decode_rgb_pixels(path) is pixels


def test_decode_rgb_pixels_enforces_pixel_budget(monkeypatch, tmp_path, caplog):
    path = tmp_path / "scan.png"

    class FakeImage:
        size = (3, 2)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def convert(self, _mode):
            raise AssertionError("pixels must stay lazy above the budget")

    monkeypatch.setattr(_vision, "OCR_REDECODE_MAX_PIXELS", 5)
    monkeypatch.setitem(
        sys.modules,
        "PIL",
        SimpleNamespace(Image=SimpleNamespace(open=lambda received: FakeImage())),
    )
    monkeypatch.setitem(sys.modules, "numpy", SimpleNamespace())

    with caplog.at_level("WARNING"):
        assert _vision._decode_rgb_pixels(path) is None

    assert [record.getMessage() for record in caplog.records] == [
        "twindb vision: scan.png is 3x2 — skipping OCR re-decode above "
        "the 5 pixel budget"
    ]


def test_decode_rgb_pixels_contains_decoder_failures(monkeypatch, tmp_path, caplog):
    path = tmp_path / "broken.png"

    def fail_open(received):
        assert received == path
        raise OSError("bad header")

    monkeypatch.setitem(
        sys.modules,
        "PIL",
        SimpleNamespace(Image=SimpleNamespace(open=fail_open)),
    )
    monkeypatch.setitem(sys.modules, "numpy", SimpleNamespace())

    with caplog.at_level("DEBUG"):
        assert _vision._decode_rgb_pixels(path) is None

    assert [record.getMessage() for record in caplog.records] == [
        "twindb vision: cannot re-decode broken.png for OCR (OSError: bad header)"
    ]


def test_ocr_png_bytes_decodes_rgb_before_calling_engine(monkeypatch):
    converted = object()
    pixels = object()
    seen = {}

    class FakeImage:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def convert(self, mode):
            assert mode == "RGB"
            return converted

    def open_image(stream):
        assert stream.read() == b"rendered-png"
        return FakeImage()

    monkeypatch.setitem(
        sys.modules,
        "PIL",
        SimpleNamespace(Image=SimpleNamespace(open=open_image)),
    )
    monkeypatch.setitem(
        sys.modules,
        "numpy",
        SimpleNamespace(asarray=lambda image: pixels if image is converted else None),
    )

    def ocr(source, label):
        seen["call"] = (source, label)
        return "diagram text"

    monkeypatch.setattr(_vision, "_ocr_source_sync", ocr)

    assert (
        _vision.ocr_png_bytes_sync(b"rendered-png", label="page 2 visual")
        == "diagram text"
    )
    assert seen["call"] == (pixels, "page 2 visual")


def test_ocr_png_bytes_uses_stable_default_label(monkeypatch):
    pixels = object()

    class FakeImage:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def convert(self, _mode):
            return object()

    monkeypatch.setitem(
        sys.modules,
        "PIL",
        SimpleNamespace(Image=SimpleNamespace(open=lambda _stream: FakeImage())),
    )
    monkeypatch.setitem(
        sys.modules, "numpy", SimpleNamespace(asarray=lambda _image: pixels)
    )
    monkeypatch.setattr(
        _vision,
        "_ocr_source_sync",
        lambda source, label: f"{source is pixels}:{label}",
    )

    assert _vision.ocr_png_bytes_sync(b"png") == "True:PDF visual"


def test_ocr_png_decode_failure_is_contained_and_observable(monkeypatch, caplog):
    def fail_decode(_stream):
        raise ValueError("broken PNG")

    monkeypatch.setitem(
        sys.modules,
        "PIL",
        SimpleNamespace(Image=SimpleNamespace(open=fail_decode)),
    )
    monkeypatch.setitem(sys.modules, "numpy", SimpleNamespace())

    with caplog.at_level("WARNING"):
        assert _vision.ocr_png_bytes_sync(b"broken", label="page 9") is None

    assert [record.getMessage() for record in caplog.records] == [
        "twindb vision: cannot decode page 9 for OCR (broken PNG)"
    ]


@pytest.mark.parametrize(
    ("raw", "expected", "warning"),
    [
        ("", {}, None),
        (
            '{"provider": {"order": ["Cerebras"]}}',
            {"provider": {"order": ["Cerebras"]}},
            None,
        ),
        ("not-json", {}, "is not valid JSON"),
        ('["not", "an", "object"]', {}, "must be a JSON object"),
    ],
)
def test_vision_extra_body_contract(raw, expected, warning, monkeypatch, caplog):
    monkeypatch.setenv("TWIN_VISION_EXTRA_BODY", raw)

    with caplog.at_level("WARNING"):
        assert _vision.vision_extra_body() == expected

    if warning is None:
        assert caplog.text == ""
    elif warning == "is not valid JSON":
        assert len(caplog.records) == 1
        message = caplog.records[0].getMessage()
        assert message.startswith(
            "twindb vision: TWIN_VISION_EXTRA_BODY is not valid JSON ("
        )
        assert message.endswith(") — ignored")
    else:
        assert [record.getMessage() for record in caplog.records] == [
            "twindb vision: TWIN_VISION_EXTRA_BODY must be a JSON object — ignored"
        ]


def test_vision_extra_body_unset_is_empty_and_silent(caplog):
    with caplog.at_level("WARNING"):
        assert _vision.vision_extra_body() == {}

    assert caplog.records == []


def test_vision_extra_body_invalid_json_keeps_parser_diagnostic(monkeypatch, caplog):
    monkeypatch.setenv("TWIN_VISION_EXTRA_BODY", "not-json")

    with caplog.at_level("WARNING"):
        assert _vision.vision_extra_body() == {}

    assert len(caplog.records) == 1
    assert isinstance(caplog.records[0].args[1], json.JSONDecodeError)


def test_vision_chat_sync_sends_strict_json_request(monkeypatch):
    calls = []

    class Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            message = SimpleNamespace(content='{"ok": true}')
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    monkeypatch.setattr(_vision, "_get_client", lambda: client)
    monkeypatch.setenv("TWIN_VISION_MODEL", "vision-model")
    monkeypatch.setenv(
        "TWIN_VISION_EXTRA_BODY", '{"provider": {"order": ["Cerebras"]}}'
    )
    messages = [{"role": "user", "content": "payload"}]

    assert _vision.vision_chat_sync(messages, max_tokens=8192) == '{"ok": true}'
    assert calls == [
        {
            "model": "vision-model",
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": messages,
            "max_tokens": 8192,
            "extra_body": {"provider": {"order": ["Cerebras"]}},
        }
    ]


def test_vision_chat_sync_normalizes_empty_content(monkeypatch):
    class Completions:
        def create(self, **kwargs):
            assert "max_tokens" not in kwargs
            assert "extra_body" not in kwargs
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=None))]
            )

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    monkeypatch.setattr(_vision, "_get_client", lambda: client)
    monkeypatch.setenv("TWIN_VISION_MODEL", "vision-model")

    assert _vision.vision_chat_sync([]) == ""


def test_vision_llm_call_encodes_original_bytes_and_mime(monkeypatch, tmp_path):
    path = tmp_path / "photo.JPEG"
    path.write_bytes(b"original-image-bytes")
    seen = {}

    def chat(messages, **kwargs):
        seen["messages"] = messages
        seen["kwargs"] = kwargs
        return "raw reply"

    monkeypatch.setattr(_vision, "vision_chat_sync", chat)

    assert _vision._call_vision_llm_sync(path) == "raw reply"
    assert seen["kwargs"] == {}
    assert seen["messages"][0] == {
        "role": "system",
        "content": _vision.VISION_SYSTEM_PROMPT,
    }
    user = seen["messages"][1]
    assert user == {
        "role": "user",
        "content": [
            {"type": "text", "text": _vision.VISION_USER_PROMPT},
            {
                "type": "image_url",
                "image_url": {"url": user["content"][1]["image_url"]["url"]},
            },
        ],
    }
    data_url = user["content"][1]["image_url"]["url"]
    prefix, encoded = data_url.split(",", 1)
    assert prefix == "data:image/jpeg;base64"
    assert base64.b64decode(encoded) == b"original-image-bytes"


def test_vision_llm_call_uses_png_fallback_for_unknown_extension(monkeypatch, tmp_path):
    path = tmp_path / "diagram.bmp"
    path.write_bytes(b"bitmap")
    seen = {}

    def chat(messages, **_kwargs):
        seen["messages"] = messages
        return "reply"

    monkeypatch.setattr(_vision, "vision_chat_sync", chat)

    assert _vision._call_vision_llm_sync(path) == "reply"
    url = seen["messages"][1]["content"][1]["image_url"]["url"]
    assert url == f"data:image/png;base64,{base64.b64encode(b'bitmap').decode('ascii')}"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("", None),
        (
            '{"image_classification": "table", "content": "totals"}',
            {"image_classification": "table", "content": "totals"},
        ),
        (
            'prefix {} then {"content": "wider object", "n": 1} trailing',
            {"content": "wider object", "n": 1},
        ),
        ('broken {not-json then {"ok": 1}', {"ok": 1}),
        (
            'prefix {"first": "0123456789"} then {"z": 0}',
            {"first": "0123456789"},
        ),
        ('prefix {"a": 1} then {"b": 2}', {"a": 1}),
        ("prefix {} trailing", {}),
        ('{"content": ', None),
    ],
)
def test_parse_vision_json_tolerates_noise_without_inventing_data(raw, expected):
    assert _vision._parse_vision_json(raw) == expected


def test_validate_vision_payload_enforces_and_normalizes_contract():
    assert _vision.validate_vision_payload(
        {"image_classification": "  Diagram ", "content": " useful text  "},
        stage="pdf-visual",
    ) == ("Diagram", "useful text")
    assert _vision.validate_vision_payload(
        {"image_classification": "  ", "content": "text"}
    ) == ("unknown", "text")

    with pytest.raises(ValueError, match="custom-stage: reply does not match"):
        _vision.validate_vision_payload([], stage="custom-stage")
    with pytest.raises(
        ValueError,
        match="custom-stage: reply must contain string image_classification and content",
    ):
        _vision.validate_vision_payload(
            {"image_classification": "diagram", "content": 42},
            stage="custom-stage",
        )
    with pytest.raises(ValueError, match="vision: reply does not match"):
        _vision.validate_vision_payload([])
    with pytest.raises(
        ValueError,
        match="vision: reply must contain string image_classification and content",
    ):
        _vision.validate_vision_payload(
            {"image_classification": "diagram", "content": 42}
        )


def test_compose_markdown_has_stable_exact_shape(tmp_path):
    path = tmp_path / "invoice.png"

    assert _vision._compose_markdown(path, "table", "  totals  ", None) == (
        "# invoice.png\n\n_Image type: table_\n\ntotals"
    )
    assert _vision._compose_markdown(path, "table", "  totals  ", "  OCR text  ") == (
        "# invoice.png\n\n_Image type: table_\n\ntotals\n\n"
        "## Extracted text (OCR)\n\nOCR text"
    )


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
    assert outcome == _vision.VisionOutcome(
        markdown=None,
        reason=(
            "vision-prefilter: image rejected before vision analysis; "
            "OCR detected 2 text characters, below configured minimum 20"
        ),
    )


async def test_prefilter_disabled_with_zero_threshold(
    vision_configured, monkeypatch, tmp_path
):
    monkeypatch.setenv("TWIN_VISION_MIN_OCR_CHARS", "0")
    monkeypatch.setattr(_vision, "_ocr_text_sync", lambda _p: "")
    monkeypatch.setattr(
        _vision,
        "_call_vision_llm_sync",
        lambda _p: (
            '{"image_classification": "diagram", "content": "An architecture diagram."}'
        ),
    )
    outcome = await _vision.aprocess_image(_write_png(tmp_path))
    assert outcome.markdown is not None
    assert outcome.classification == "diagram"


@pytest.mark.parametrize(
    ("threshold", "ocr_text", "expected_reason"),
    [
        (
            1,
            "",
            "vision-prefilter: image rejected before vision analysis; OCR detected "
            "0 text characters, below configured minimum 1",
        ),
        (2, "xx", "ok"),
    ],
)
async def test_prefilter_threshold_boundaries(
    threshold,
    ocr_text,
    expected_reason,
    vision_configured,
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("TWIN_VISION_MIN_OCR_CHARS", str(threshold))
    monkeypatch.setattr(_vision, "_ocr_text_sync", lambda _path: ocr_text)
    monkeypatch.setattr(
        _vision,
        "_call_vision_llm_sync",
        lambda _path: ('{"image_classification": "diagram", "content": "useful"}'),
    )

    outcome = await _vision.aprocess_image(_write_png(tmp_path))

    assert outcome.reason == expected_reason


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
    assert outcome == _vision.VisionOutcome(
        markdown=None,
        reason=(
            "image-dropped: image rejected by active Vision policy; "
            "classified as 'logo', an excluded class"
        ),
        classification="Logo",
    )


async def test_markdown_composition_includes_ocr_section(
    vision_configured, monkeypatch, tmp_path, caplog
):
    path = _write_png(tmp_path, name="facture.png")
    calls = []

    def ocr(received_path):
        calls.append(("ocr", received_path))
        return "Total HT 1 200 EUR TVA 240 EUR"

    def llm(received_path):
        calls.append(("llm", received_path))
        return '{"image_classification": "table", "content": "Invoice totals table."}'

    monkeypatch.setattr(_vision, "_ocr_text_sync", ocr)
    monkeypatch.setattr(_vision, "_call_vision_llm_sync", llm)
    with caplog.at_level("INFO"):
        outcome = await _vision.aprocess_image(path)

    expected_markdown = (
        "# facture.png\n\n_Image type: table_\n\nInvoice totals table.\n\n"
        "## Extracted text (OCR)\n\nTotal HT 1 200 EUR TVA 240 EUR"
    )
    assert outcome == _vision.VisionOutcome(
        markdown=expected_markdown,
        reason="ok",
        classification="table",
    )
    assert calls == [("ocr", path), ("llm", path)]
    assert [record.getMessage() for record in caplog.records] == [
        f"twindb vision: facture.png → table ({len(expected_markdown)} chars)"
    ]


async def test_fenced_json_reply_is_parsed(vision_configured, monkeypatch, tmp_path):
    monkeypatch.setattr(_vision, "_ocr_text_sync", lambda _p: "x" * 40)
    monkeypatch.setattr(
        _vision,
        "_call_vision_llm_sync",
        lambda _p: (
            'Here it is:\n```json\n{"image_classification": "screenshot", "content": "A settings page."}\n```'
        ),
    )
    outcome = await _vision.aprocess_image(_write_png(tmp_path))
    assert outcome.markdown is not None
    assert outcome.classification == "screenshot"


async def test_unparseable_reply_refuses(vision_configured, monkeypatch, tmp_path):
    monkeypatch.setattr(_vision, "_ocr_text_sync", lambda _p: "x" * 40)
    monkeypatch.setattr(_vision, "_call_vision_llm_sync", lambda _p: "not json at all")
    outcome = await _vision.aprocess_image(_write_png(tmp_path))
    assert outcome == _vision.VisionOutcome(
        markdown=None,
        reason="vision-llm-error: unparseable JSON reply",
    )


@pytest.mark.parametrize(
    "payload",
    [
        {"image_classification": ["diagram"], "content": "valid text"},
        {"image_classification": "diagram", "content": {"text": "no"}},
        {"image_classification": 7, "content": "valid text"},
        {"image_classification": "diagram", "content": 42},
    ],
)
async def test_malformed_payload_refuses_as_llm_error(
    payload, vision_configured, monkeypatch, tmp_path
):
    monkeypatch.setattr(_vision, "_ocr_text_sync", lambda _p: "x" * 40)
    monkeypatch.setattr(
        _vision, "_call_vision_llm_sync", lambda _p: json.dumps(payload)
    )

    outcome = await _vision.aprocess_image(_write_png(tmp_path))

    assert outcome == _vision.VisionOutcome(
        markdown=None,
        reason=(
            "vision-llm-error: vision-image: reply must contain string "
            "image_classification and content"
        ),
    )


async def test_empty_classification_is_normalized_to_unknown(
    vision_configured, monkeypatch, tmp_path
):
    monkeypatch.setattr(_vision, "_ocr_text_sync", lambda _p: "x" * 40)
    monkeypatch.setattr(
        _vision,
        "_call_vision_llm_sync",
        lambda _p: '{"image_classification": "  ", "content": "Useful diagram."}',
    )

    outcome = await _vision.aprocess_image(_write_png(tmp_path))

    assert outcome.classification == "unknown"
    assert outcome.markdown is not None
    assert "_Image type: unknown_" in outcome.markdown


async def test_llm_error_refuses_with_reason(vision_configured, monkeypatch, tmp_path):
    monkeypatch.setattr(_vision, "_ocr_text_sync", lambda _p: "x" * 40)

    def boom(_p):
        raise ConnectionError("endpoint down")

    monkeypatch.setattr(_vision, "_call_vision_llm_sync", boom)
    outcome = await _vision.aprocess_image(_write_png(tmp_path))
    assert outcome == _vision.VisionOutcome(
        markdown=None,
        reason="vision-llm-error: ConnectionError: endpoint down",
    )


async def test_pipeline_timeout_refuses(
    vision_configured, monkeypatch, tmp_path, caplog
):
    import time

    monkeypatch.setenv("TWIN_VISION_TIMEOUT", "0.05")
    monkeypatch.setattr(_vision, "_ocr_text_sync", lambda _p: time.sleep(1) or "late")
    path = _write_png(tmp_path)
    with caplog.at_level("WARNING"):
        outcome = await _vision.aprocess_image(path)
    assert outcome == _vision.VisionOutcome(
        markdown=None,
        reason="vision-timeout: no result within 0s",
    )
    assert (
        f"twindb vision: {path.name} — vision-timeout: no result within 0s"
        in caplog.text
    )


async def test_missing_image_is_a_stable_input_error(tmp_path):
    path = tmp_path / "missing.png"

    outcome = await _vision.aprocess_image(path)

    assert outcome.markdown is None
    assert outcome.classification is None
    assert outcome.reason.startswith("vision-input-error: FileNotFoundError: ")
    assert str(path) in outcome.reason


async def test_unexpected_pipeline_exception_is_contained_and_logged(
    monkeypatch, tmp_path, caplog
):
    async def crash(_path):
        raise RuntimeError("unexpected collapse")

    monkeypatch.setattr(_vision, "_aprocess_inner", crash)
    path = _write_png(tmp_path, name="collapse.png")

    with caplog.at_level("WARNING"):
        outcome = await _vision.aprocess_image(path)

    assert outcome == _vision.VisionOutcome(
        markdown=None,
        reason="vision-error: RuntimeError: unexpected collapse",
    )
    assert (
        "twindb vision: collapse.png — vision-error: RuntimeError: unexpected collapse"
        in caplog.text
    )


@pytest.mark.parametrize("content", ["", "   ", "INVALID", " invalid "])
async def test_empty_or_invalid_content_is_rejected_by_policy(
    content, vision_configured, monkeypatch, tmp_path
):
    monkeypatch.setattr(_vision, "_ocr_text_sync", lambda _p: "x" * 40)
    monkeypatch.setattr(
        _vision,
        "_call_vision_llm_sync",
        lambda _p: json.dumps({"image_classification": "Diagram", "content": content}),
    )

    outcome = await _vision.aprocess_image(_write_png(tmp_path))

    assert outcome == _vision.VisionOutcome(
        markdown=None,
        reason=(
            "image-dropped: image rejected by active Vision policy; "
            "no informational content was detected"
        ),
        classification="Diagram",
    )


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


async def test_seam_never_falls_back_to_native_for_oversized_image(
    dr_module, vision_configured, monkeypatch, tmp_path
):
    async def fake_orig(rag, file_path, *args, **kwargs):
        raise AssertionError("native path must not run for a vision image")

    wrapped = _install_patch(dr_module, fake_orig)
    monkeypatch.setattr(_conversion, "should_convert", lambda _p: False)
    monkeypatch.setenv("TWIN_VISION_MAX_BYTES", "8")

    rag = _FakeRag()
    success, track_id = await wrapped(
        rag,
        _write_png(tmp_path, name="oversized.png"),
        "tid-oversized",
    )

    assert success is False
    assert track_id == "tid-oversized"
    assert rag.enqueue_calls == []
    error_files, reported_track = rag.error_calls[0]
    assert reported_track == "tid-oversized"
    assert "vision-size-limit" in error_files[0]["original_error"]


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
