"""MarkItDown pre-conversion tier (MARKITDOWN-INGESTION-PLAN.md, PR 1).

Three layers:

- pure unit tests on ``_conversion`` (config resolution, gates, failure
  degradation) — no LightRAG import;
- seam-contract tests on the registry patches against the real
  ``document_routes`` module (delegation-verbatim, converted enqueue under
  the original name, error paths, idempotence, whitelist extension) — the
  LightRAG-compat doctrine tests for this extension;
- real-markitdown conversion tests, skipped when the [convert] extra is
  not installed.
"""

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from twindb_lightrag_memgraph import _conversion
from twindb_lightrag_memgraph.patches import registry

# ---------------------------------------------------------------------------
# _conversion unit tests (no LightRAG)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_convert_env(monkeypatch):
    for var in (
        "TWIN_CONVERT",
        "TWIN_CONVERT_FORMATS",
        "TWIN_CONVERT_MAX_BYTES",
        "TWIN_CONVERT_TIMEOUT",
    ):
        monkeypatch.delenv(var, raising=False)
    _conversion.reset_caches()
    yield
    _conversion.reset_caches()


def test_mode_off_disables_even_when_available(monkeypatch):
    monkeypatch.setenv("TWIN_CONVERT", "off")
    monkeypatch.setattr(_conversion, "is_available", lambda: True)
    assert _conversion.is_enabled() is False


def test_mode_auto_follows_availability(monkeypatch):
    monkeypatch.setattr(_conversion, "is_available", lambda: True)
    assert _conversion.is_enabled() is True
    monkeypatch.setattr(_conversion, "is_available", lambda: False)
    assert _conversion.is_enabled() is False


def test_mode_on_without_markitdown_degrades_off(monkeypatch, caplog):
    monkeypatch.setenv("TWIN_CONVERT", "on")
    monkeypatch.setattr(_conversion, "is_available", lambda: False)
    with caplog.at_level("WARNING"):
        assert _conversion.is_enabled() is False
    assert "TWIN_CONVERT=on but markitdown is not importable" in caplog.text


def test_formats_default_and_env_override(monkeypatch):
    assert _conversion.conversion_formats() == _conversion.DEFAULT_CONVERT_FORMATS
    monkeypatch.setenv("TWIN_CONVERT_FORMATS", " PDF, .docx ,html ")
    assert _conversion.conversion_formats() == frozenset({"pdf", "docx", "html"})
    monkeypatch.setenv("TWIN_CONVERT_FORMATS", " , ")
    assert _conversion.conversion_formats() == _conversion.DEFAULT_CONVERT_FORMATS


def test_numeric_envs_fall_back_on_garbage(monkeypatch):
    monkeypatch.setenv("TWIN_CONVERT_MAX_BYTES", "not-a-number")
    monkeypatch.setenv("TWIN_CONVERT_TIMEOUT", "-3")
    assert _conversion.max_convert_bytes() == _conversion.DEFAULT_MAX_BYTES
    assert _conversion.convert_timeout_seconds() == (
        _conversion.DEFAULT_TIMEOUT_SECONDS
    )
    monkeypatch.setenv("TWIN_CONVERT_MAX_BYTES", "1024")
    monkeypatch.setenv("TWIN_CONVERT_TIMEOUT", "5.5")
    assert _conversion.max_convert_bytes() == 1024
    assert _conversion.convert_timeout_seconds() == 5.5


def test_extra_supported_extensions_follow_format_set(monkeypatch):
    monkeypatch.setenv("TWIN_CONVERT_FORMATS", "xls,msg")
    assert _conversion.extra_supported_extensions() == (".msg", ".xls")


def test_should_convert_gates(monkeypatch, tmp_path):
    monkeypatch.setattr(_conversion, "is_available", lambda: True)
    covered = tmp_path / "doc.docx"
    covered.write_bytes(b"x" * 10)
    uncovered = tmp_path / "doc.json"
    uncovered.write_bytes(b"{}")
    missing = tmp_path / "ghost.docx"

    assert _conversion.should_convert(covered) is True
    assert _conversion.should_convert(uncovered) is False
    assert _conversion.should_convert(missing) is False

    monkeypatch.setenv("TWIN_CONVERT_MAX_BYTES", "5")
    assert _conversion.should_convert(covered) is False

    monkeypatch.setenv("TWIN_CONVERT", "off")
    monkeypatch.delenv("TWIN_CONVERT_MAX_BYTES")
    assert _conversion.should_convert(covered) is False


async def test_aconvert_file_degrades_to_none_on_error(monkeypatch, tmp_path):
    path = tmp_path / "doc.docx"
    path.write_bytes(b"x")

    def boom(_path):
        raise RuntimeError("converter exploded")

    monkeypatch.setattr(_conversion, "_convert_sync", boom)
    assert await _conversion.aconvert_file(path) is None

    monkeypatch.setattr(_conversion, "_convert_sync", lambda _p: "   \n ")
    assert await _conversion.aconvert_file(path) is None

    monkeypatch.setattr(_conversion, "_convert_sync", lambda _p: "# Converted")
    assert await _conversion.aconvert_file(path) == "# Converted"


async def test_aconvert_file_times_out(monkeypatch, tmp_path):
    path = tmp_path / "doc.docx"
    path.write_bytes(b"x")
    monkeypatch.setenv("TWIN_CONVERT_TIMEOUT", "0.05")

    import time

    monkeypatch.setattr(
        _conversion, "_convert_sync", lambda _p: time.sleep(1) or "late"
    )
    assert await _conversion.aconvert_file(path) is None


# ---------------------------------------------------------------------------
# Seam-contract tests against the real document_routes module
# ---------------------------------------------------------------------------


class _FakeRag:
    def __init__(self, enqueue_error=None):
        self.enqueue_calls = []
        self.error_calls = []
        self._enqueue_error = enqueue_error

    async def apipeline_enqueue_documents(self, content, **kwargs):
        if self._enqueue_error is not None:
            raise self._enqueue_error
        self.enqueue_calls.append((content, kwargs))

    async def apipeline_enqueue_error_documents(self, error_files, track_id):
        self.error_calls.append((error_files, track_id))


@pytest.fixture
def dr_module(monkeypatch):
    """Real ``document_routes`` module with full state restore.

    The module is process-shared (see ``ensure_fresh_native_document_router``
    in conftest) — every attribute this suite touches is snapshotted and
    restored so the conversion patches never leak into other test files.
    """
    monkeypatch.setattr(sys, "argv", ["pytest"])
    dr = pytest.importorskip(
        "lightrag.api.routers.document_routes",
        reason="native document routes unavailable on this LightRAG",
    )
    saved = {
        "pipeline_enqueue_file": dr.pipeline_enqueue_file,
        "manager_init": dr.DocumentManager.__init__,
        "is_supported_file": dr.DocumentManager.is_supported_file,
    }
    sentinels = (
        "_twindb_convert_enqueue_patched",
        "_twindb_doc_manager_ext_patched",
        "_twindb_doc_manager_supported_patched",
    )
    saved_sentinels = {
        name: getattr(dr, name) for name in sentinels if hasattr(dr, name)
    }
    yield dr
    dr.pipeline_enqueue_file = saved["pipeline_enqueue_file"]
    dr.DocumentManager.__init__ = saved["manager_init"]
    dr.DocumentManager.is_supported_file = saved["is_supported_file"]
    for name in sentinels:
        if name in saved_sentinels:
            setattr(dr, name, saved_sentinels[name])
        elif hasattr(dr, name):
            delattr(dr, name)


def _install_enqueue_patch(dr, fake_orig):
    dr.pipeline_enqueue_file = fake_orig
    dr._twindb_convert_enqueue_patched = False
    registry._patch_pipeline_enqueue_conversion()
    assert dr.pipeline_enqueue_file is not fake_orig
    return dr.pipeline_enqueue_file


async def test_wrapper_delegates_verbatim_when_not_converting(
    dr_module, monkeypatch, tmp_path
):
    """LightRAG-compat contract: no conversion decision => untouched call."""
    seen = {}

    async def fake_orig(rag, file_path, *args, **kwargs):
        seen["call"] = (rag, file_path, args, kwargs)
        return True, "native-track"

    wrapped = _install_enqueue_patch(dr_module, fake_orig)
    monkeypatch.setattr(_conversion, "should_convert", lambda _p: False)

    rag = object()
    path = tmp_path / "doc.docx"
    result = await wrapped(rag, path, "tid-1", from_scan=True)

    assert result == (True, "native-track")
    assert seen["call"] == (rag, path, ("tid-1",), {"from_scan": True})


async def test_wrapper_delegates_when_conversion_fails(
    dr_module, monkeypatch, tmp_path
):
    async def fake_orig(rag, file_path, *args, **kwargs):
        return True, "native-track"

    wrapped = _install_enqueue_patch(dr_module, fake_orig)
    monkeypatch.setattr(_conversion, "should_convert", lambda _p: True)

    async def failed_convert(_p):
        return None

    monkeypatch.setattr(_conversion, "aconvert_file", failed_convert)

    rag = _FakeRag()
    result = await wrapped(rag, tmp_path / "doc.docx", "tid-2")
    assert result == (True, "native-track")
    assert rag.enqueue_calls == []


async def test_wrapper_enqueues_markdown_under_original_name(
    dr_module, monkeypatch, tmp_path
):
    async def fake_orig(rag, file_path, *args, **kwargs):
        raise AssertionError("native path must not run on conversion")

    wrapped = _install_enqueue_patch(dr_module, fake_orig)
    monkeypatch.setattr(_conversion, "should_convert", lambda _p: True)

    async def convert(_p):
        return "# Converted\n\ncontent"

    monkeypatch.setattr(_conversion, "aconvert_file", convert)

    rag = _FakeRag()
    path = tmp_path / "report.pptx"
    path.write_bytes(b"binary")
    result = await wrapped(rag, path, "tid-3")

    assert result == (True, "tid-3")
    assert rag.enqueue_calls == [
        (
            "# Converted\n\ncontent",
            {"file_paths": "report.pptx", "track_id": "tid-3"},
        )
    ]


async def test_wrapper_generates_track_id_and_forwards_from_scan(
    dr_module, monkeypatch, tmp_path
):
    async def fake_orig(rag, file_path, *args, **kwargs):
        raise AssertionError("native path must not run on conversion")

    wrapped = _install_enqueue_patch(dr_module, fake_orig)
    monkeypatch.setattr(_conversion, "should_convert", lambda _p: True)

    async def convert(_p):
        return "# md"

    monkeypatch.setattr(_conversion, "aconvert_file", convert)

    rag = _FakeRag()
    path = tmp_path / "report.xlsx"
    path.write_bytes(b"binary")
    success, track_id = await wrapped(rag, path, from_scan=True)

    assert success is True
    assert isinstance(track_id, str) and track_id
    content, kwargs = rag.enqueue_calls[0]
    assert kwargs["track_id"] == track_id
    assert kwargs["from_scan"] is True


async def test_wrapper_reports_error_document_on_enqueue_failure(
    dr_module, monkeypatch, tmp_path
):
    async def fake_orig(rag, file_path, *args, **kwargs):
        raise AssertionError("native path must not run on conversion")

    wrapped = _install_enqueue_patch(dr_module, fake_orig)
    monkeypatch.setattr(_conversion, "should_convert", lambda _p: True)

    async def convert(_p):
        return "# md"

    monkeypatch.setattr(_conversion, "aconvert_file", convert)

    rag = _FakeRag(enqueue_error=RuntimeError("memgraph down"))
    path = tmp_path / "report.pdf"
    path.write_bytes(b"binary")
    success, track_id = await wrapped(rag, path, "tid-4")

    assert success is False
    assert track_id == "tid-4"
    assert len(rag.error_calls) == 1
    error_files, reported_track = rag.error_calls[0]
    assert reported_track == "tid-4"
    assert error_files[0]["file_path"] == "report.pdf"


async def test_enqueue_patch_is_idempotent(dr_module, monkeypatch):
    async def fake_orig(rag, file_path, *args, **kwargs):
        return True, "t"

    wrapped = _install_enqueue_patch(dr_module, fake_orig)
    registry._patch_pipeline_enqueue_conversion()
    assert dr_module.pipeline_enqueue_file is wrapped


def _whitelist_is_settable(dr) -> bool:
    """1.4.x: plain attribute (patchable). 1.5.x: read-only property derived
    from the parser registry — the patch degrades gracefully there."""
    return not isinstance(
        getattr(dr.DocumentManager, "supported_extensions", None), property
    )


def test_whitelist_patch_extends_document_manager(dr_module, monkeypatch, tmp_path):
    if not _whitelist_is_settable(dr_module):
        pytest.skip("1.5.x derives supported_extensions from the parser registry")
    monkeypatch.setenv("TWIN_CONVERT_FORMATS", "xls,msg")
    monkeypatch.setattr(_conversion, "is_available", lambda: True)
    dr_module._twindb_doc_manager_ext_patched = False
    registry._patch_document_manager_extensions()

    manager = dr_module.DocumentManager(str(tmp_path))
    assert ".xls" in manager.supported_extensions
    assert ".msg" in manager.supported_extensions
    # Native entries untouched, no duplicates.
    assert ".pdf" in manager.supported_extensions
    assert len(set(manager.supported_extensions)) == len(
        tuple(manager.supported_extensions)
    )
    assert manager.is_supported_file("legacy.xls")
    assert manager.is_supported_file("mail.msg")


def test_whitelist_patch_degrades_on_readonly_property(
    dr_module, monkeypatch, tmp_path, caplog
):
    if _whitelist_is_settable(dr_module):
        pytest.skip("1.4.x whitelist is a plain settable attribute")
    monkeypatch.setenv("TWIN_CONVERT_FORMATS", "xls,msg")
    dr_module._twindb_doc_manager_ext_patched = False
    registry._patch_document_manager_extensions()

    with caplog.at_level("WARNING"):
        manager = dr_module.DocumentManager(str(tmp_path))
    assert "could not extend supported_extensions" in caplog.text
    # Native behavior intact — the manager still works.
    assert ".pdf" in manager.supported_extensions


def test_is_supported_file_accepts_tier_extensions_on_both_lines(
    dr_module, monkeypatch, tmp_path
):
    """The ENFORCEMENT check (what the upload route asks) must accept the
    tier extensions on 1.4.x AND 1.5.x. On 1.5.x the whitelist property is
    read-only, so before the ``is_supported_file`` wrapper the runtime
    config advertised e.g. ``png`` while the API still 400-ed the upload
    (review finding on fix/webui-image-upload-whitelist)."""
    from twindb_lightrag_memgraph import _vision

    monkeypatch.setattr(_conversion, "is_available", lambda: True)
    monkeypatch.setattr(_vision, "is_enabled", lambda: True)
    dr_module._twindb_doc_manager_ext_patched = False
    dr_module._twindb_doc_manager_supported_patched = False
    registry._patch_document_manager_extensions()

    manager = dr_module.DocumentManager(str(tmp_path))
    assert manager.is_supported_file("diagram.png")
    assert manager.is_supported_file("photo.JPEG")  # case-insensitive
    assert manager.is_supported_file("legacy.xls")
    assert manager.is_supported_file("report.pdf")  # native set untouched
    assert not manager.is_supported_file("archive.zip")  # no tier owns it

    # Tiers re-checked per call (late-read): switching them off restores
    # the native answer without a process restart.
    monkeypatch.setattr(_vision, "is_enabled", lambda: False)
    monkeypatch.setenv("TWIN_CONVERT", "off")
    assert not manager.is_supported_file("diagram.png")
    assert manager.is_supported_file("report.pdf")


def test_whitelist_patch_is_idempotent(dr_module, monkeypatch, tmp_path):
    monkeypatch.setenv("TWIN_CONVERT_FORMATS", "msg")
    monkeypatch.setattr(_conversion, "is_available", lambda: True)
    dr_module._twindb_doc_manager_ext_patched = False
    registry._patch_document_manager_extensions()
    first_init = dr_module.DocumentManager.__init__
    registry._patch_document_manager_extensions()
    assert dr_module.DocumentManager.__init__ is first_init
    if _whitelist_is_settable(dr_module):
        manager = dr_module.DocumentManager(str(tmp_path))
        assert tuple(manager.supported_extensions).count(".msg") == 1


# ---------------------------------------------------------------------------
# Real markitdown conversions ([convert] extra required)
# ---------------------------------------------------------------------------


def _require_markitdown():
    return pytest.importorskip("markitdown", reason="[convert] extra not installed")


async def test_real_html_conversion_strips_tags(monkeypatch, tmp_path):
    _require_markitdown()
    monkeypatch.setattr(_conversion, "is_available", lambda: True)
    path = tmp_path / "page.html"
    path.write_text(
        "<html><head><script>evil()</script><title>T</title></head>"
        "<body><h1>Heading</h1><p>Body text.</p></body></html>",
        encoding="utf-8",
    )
    markdown = await _conversion.aconvert_file(path)
    assert markdown is not None
    assert "Heading" in markdown
    assert "Body text." in markdown
    assert "<h1>" not in markdown
    assert "evil()" not in markdown


async def test_real_csv_conversion_builds_table(monkeypatch, tmp_path):
    _require_markitdown()
    monkeypatch.setattr(_conversion, "is_available", lambda: True)
    path = tmp_path / "data.csv"
    path.write_text("name,amount\nalpha,10\nbeta,20\n", encoding="utf-8")
    markdown = await _conversion.aconvert_file(path)
    assert markdown is not None
    assert "|" in markdown  # markdown table, not raw CSV
    assert "alpha" in markdown and "beta" in markdown
