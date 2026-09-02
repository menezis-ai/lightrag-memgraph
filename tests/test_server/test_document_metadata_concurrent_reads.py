"""Concurrency contract of ``routes_documents._doc_tags_and_source_links``.

The three reads behind ``GET /documents/{doc_id}/metadata`` overlap, and the two
optional ones (graph tags, source links) are cancelled and reaped whenever the
helper unwinds early: the request-critical document read raising (a 404 for a
doc outside the active folder), or the request itself being cancelled — a
client disconnect — while any of the reads is still pending. Neither path may
leave read-pool connections in flight. Mutation-proven twice: a plain
``asyncio.gather`` passes the happy path and fails the 404 case (the optional
tasks survive), and a guard wrapped around the document await only fails the
external-cancellation case (the source-link task survives).
"""

from __future__ import annotations

import asyncio

import pytest
from fastapi import HTTPException

from twindb_lightrag_memgraph.server import webui_router
from twindb_lightrag_memgraph.server.webui import routes_documents

_DOC = {"id": "doc-1", "metadata": {"folder": "default"}}


class _Probe:
    """Fake reads that report their overlap and whether they were cancelled."""

    def __init__(self, *, doc_fails: bool = False) -> None:
        self.doc_fails = doc_fails
        self.in_flight = 0
        self.peak = 0
        self.cancelled: set[str] = set()
        self.finished: set[str] = set()
        self.doc_done = asyncio.Event()

    async def _read(self, name: str, value, delay: float):
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        try:
            await asyncio.sleep(delay)
            self.finished.add(name)
            return value
        except asyncio.CancelledError:
            self.cancelled.add(name)
            raise
        finally:
            self.in_flight -= 1

    async def doc(self, _doc_id: str):
        # Let the optional reads reach their await before this one settles, so
        # the assertions observe real overlap rather than eager scheduling.
        await asyncio.sleep(0.005)
        if self.doc_fails:
            raise HTTPException(status_code=404, detail="Document doc-1 not found")
        try:
            return await self._read("doc", dict(_DOC), 0.005)
        finally:
            self.doc_done.set()

    async def tags(self, _doc_id: str):
        return await self._read("tags", ["oracle"], 0.05)

    async def links(self, _doc_id: str):
        return await self._read("links", [{"url": "u"}], 0.05)


@pytest.fixture
def probe_factory(monkeypatch):
    def _install(**kwargs) -> _Probe:
        probe = _Probe(**kwargs)
        monkeypatch.setattr(webui_router, "_get_doc_for_active_folder", probe.doc)
        monkeypatch.setattr(webui_router, "_graph_tags_for_doc_or_none", probe.tags)
        monkeypatch.setattr(routes_documents, "_source_links_for_document", probe.links)
        return probe

    return _install


async def test_reads_overlap_and_results_land_in_order(probe_factory):
    probe = probe_factory()
    doc, tags, links = await routes_documents._doc_tags_and_source_links(
        "doc-1", webui_router
    )
    assert (doc, tags, links) == (_DOC, ["oracle"], [{"url": "u"}])
    assert probe.peak == 3, "the optional reads must run alongside the doc read"
    assert probe.finished == {"doc", "tags", "links"}


async def test_document_failure_cancels_and_reaps_the_optional_reads(probe_factory):
    probe = probe_factory(doc_fails=True)
    before = {t for t in asyncio.all_tasks() if not t.done()}
    with pytest.raises(HTTPException) as excinfo:
        await routes_documents._doc_tags_and_source_links("doc-1", webui_router)
    assert excinfo.value.status_code == 404
    # Cancelled BEFORE they could finish (their delay is 10x the doc read's).
    assert probe.cancelled == {"tags", "links"}
    assert probe.finished == set()
    assert probe.in_flight == 0, "a cancelled read must release its slot"
    leaked = {t for t in asyncio.all_tasks() if not t.done()} - before
    assert not leaked, f"optional reads survived the failure: {leaked}"


async def test_route_still_returns_404_through_the_helper(probe_factory):
    probe_factory(doc_fails=True)
    with pytest.raises(HTTPException) as excinfo:
        await routes_documents.get_document_metadata("doc-1")
    assert excinfo.value.status_code == 404


async def test_cancellation_after_the_document_read_reaps_both_optional_reads(
    probe_factory,
):
    """A client disconnect while the optional reads are still in flight must
    cancel BOTH of them — not just the one currently awaited. Review finding on
    PR #486: the first version only guarded the document await, so cancelling
    while awaiting the tags task left the source-link task running on the
    shared read pool."""
    probe = probe_factory()
    before = {t for t in asyncio.all_tasks() if not t.done()}
    outer = asyncio.create_task(
        routes_documents._doc_tags_and_source_links("doc-1", webui_router)
    )
    await probe.doc_done.wait()
    await asyncio.sleep(0)  # let the helper move on to awaiting the tags task
    outer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await outer
    assert probe.finished == {"doc"}
    assert probe.cancelled == {"tags", "links"}, probe.cancelled
    assert probe.in_flight == 0, "a cancelled read must release its slot"
    leaked = {t for t in asyncio.all_tasks() if not t.done()} - before
    assert not leaked, f"optional reads survived the cancellation: {leaked}"
