"""Qualify the real upstream parser with an in-process HTTP/real Memgraph harness.

Synthetic fixtures are not the original Lexmark files. These tests do not
exercise the BNP pipeline or nginx and make no production-remediation claim.
"""

import asyncio
from contextlib import suppress
from importlib.metadata import version
import json
from pathlib import Path
from types import SimpleNamespace
import threading
import time
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
from packaging.version import Version
import pytest

from tests.test_http_e2e import _build_app
from lightrag_memgraph import _pool
from lightrag_memgraph.kv_impl import MemgraphKVStorage

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        Version(version("lightrag-hku")) < Version("1.5.7"),
        reason="Qualifies the supported 1.5.7 parser API",
    ),
]


def _pdf(duplicate_key=False):
    """Small text PDF with correct xref offsets, optionally a repeated /Type."""
    stream = b"BT /F1 12 Tf 30 100 Td (PDF responsiveness qualification.) Tj ET"
    page_type = b"/Type /Page /Type /Page" if duplicate_key else b"/Type /Page"
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< "
        + page_type
        + b" /Parent 2 0 R /MediaBox [0 0 300 300] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length "
        + str(len(stream)).encode()
        + b" >>\nstream\n"
        + stream
        + b"\nendstream",
    ]
    data = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, obj in enumerate(objects, 1):
        offsets.append(len(data))
        data.extend(f"{number} 0 obj\n".encode() + obj + b"\nendobj\n")
    xref = len(data)
    data.extend(f"xref\n0 {len(offsets)}\n0000000000 65535 f \n".encode())
    for offset in offsets[1:]:
        data.extend(f"{offset:010d} 00000 n \n".encode())
    data.extend(
        f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    )
    return bytes(data)


@pytest.fixture
async def kv_store(monkeypatch):
    monkeypatch.setenv("MEMGRAPH_WORKSPACE", "pdf_" + uuid4().hex)
    store = MemgraphKVStorage("full_docs", {}, None)
    await store.initialize()
    try:
        yield store
    finally:
        await store.drop()
        async with _pool.get_session() as session:
            await (
                await session.run(f"DROP INDEX ON :`{store._label()}`(id)")
            ).consume()
        await _pool.close_driver()


async def _observe_parser(monkeypatch, source: Path, store, *, inline=False):
    from lightrag.parser.legacy import extractors
    from lightrag.parser.legacy.parser import LegacyParser

    loop = asyncio.get_running_loop()
    entered = asyncio.Event()
    release = threading.Event()
    worker_threads = []
    original = extractors.extract_text

    def held_extraction(*args, **kwargs):
        worker_threads.append(threading.get_ident())
        loop.call_soon_threadsafe(entered.set)
        if not release.wait(5):
            raise TimeoutError(
                "Parser qualification watchdog did not release extraction"
            )
        return original(*args, **kwargs)  # Actual pypdf extraction, not a replacement.

    monkeypatch.setattr(extractors, "extract_text", held_extraction)

    async def persist(doc_id, data):
        await store.upsert({doc_id: data})

    ctx = SimpleNamespace(
        doc_id="pdf-document",
        file_path=source.name,
        rag=SimpleNamespace(_persist_parsed_full_docs=persist),
        resolve=lambda engine: SimpleNamespace(source_path=source),
        archive_source=AsyncMock(),
    )
    app = _build_app(ctx.rag)

    @app.post("/documents/pdf")
    async def parse_pdf():
        if inline:
            # Deliberately bad control: the same real extractor on the event loop.
            text = extractors.extract_text(source.read_bytes(), "pdf")
            await persist(ctx.doc_id, {"content": text})
            return {"content": text}
        return (await LegacyParser().parse(ctx)).to_dict()

    # Independent of asyncio: an inline regression cannot prevent this release.
    watchdog = threading.Timer(1.5, release.set)
    watchdog.daemon = True
    observation = {"probe_ms": [], "parse_active": []}
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        started = time.perf_counter()
        watchdog.start()
        task = asyncio.create_task(client.post("/documents/pdf"))
        try:
            await asyncio.wait_for(entered.wait(), 5)
            for path in ("/health", "/ready", "/health", "/ready", "/health"):
                probe_started = time.perf_counter()
                response = await client.get(path)
                elapsed = (time.perf_counter() - probe_started) * 1000
                if "first_probe_ms" not in observation:
                    observation["first_probe_ms"] = (
                        time.perf_counter() - started
                    ) * 1000
                observation["probe_ms"].append(elapsed)
                observation["parse_active"].append(
                    not task.done() and not release.is_set()
                )
                assert response.status_code == 200
                assert response.headers["content-type"].startswith("application/json")
            release.set()
            parsed = await asyncio.wait_for(asyncio.shield(task), 5)
            assert (await client.get("/health")).json() == {"status": "healthy"}
        finally:
            release.set()
            watchdog.cancel()
            watchdog.join(timeout=1)
            if not task.done():
                task.cancel()
            with suppress(asyncio.CancelledError):
                await task
    observation["worker_threads"] = worker_threads
    return observation, parsed


def _assert_responsive(observation):
    # Includes time queued before the event loop could even start the first probe.
    assert observation["first_probe_ms"] < 500, f"event loop stalled: {observation}"
    assert max(observation["probe_ms"]) < 200, observation
    assert all(observation["parse_active"]), (
        "Probes must finish before extraction is released"
    )
    assert observation["worker_threads"]
    assert threading.get_ident() not in observation["worker_threads"]


@pytest.mark.parametrize("kind", ["valid", "duplicate-dictionary", "invalid"])
async def test_upstream_pdf_parser_keeps_probes_responsive(
    monkeypatch, tmp_path, kv_store, caplog, request, kind
):
    source = tmp_path / "synthetic.pdf"
    source.write_bytes(
        b"%PDF-1.4\ninvalid\n%%EOF\n"
        if kind == "invalid"
        else _pdf(kind == "duplicate-dictionary")
    )
    observed, response = await _observe_parser(monkeypatch, source, kv_store)
    request.node.user_properties.append(("pdf_probe_metrics", json.dumps(observed)))
    _assert_responsive(observed)
    assert response.headers["content-type"].startswith("application/json")
    if kind == "invalid":
        assert response.status_code == 500
        assert response.json()["type"].endswith("Error")
        assert await kv_store.get_by_id("pdf-document") is None
    else:
        assert response.status_code == 200
        assert "PDF responsiveness qualification." in response.json()["content"]
        assert (
            "PDF responsiveness qualification."
            in (await kv_store.get_by_id("pdf-document"))["content"]
        )
    if kind == "duplicate-dictionary":
        assert any(
            "multiple definition" in record.getMessage().lower()
            for record in caplog.records
        )


async def test_responsiveness_detector_rejects_inline_pdf_parsing(
    monkeypatch, tmp_path, kv_store, request
):
    source = tmp_path / "inline-control.pdf"
    source.write_bytes(_pdf())
    observed, response = await _observe_parser(
        monkeypatch, source, kv_store, inline=True
    )
    request.node.user_properties.append(("pdf_probe_metrics", json.dumps(observed)))
    assert response.status_code == 200
    with pytest.raises(AssertionError, match="event loop stalled"):
        _assert_responsive(observed)
