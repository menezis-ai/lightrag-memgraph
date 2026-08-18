"""Compatibility proofs for the LightRAG 1.4.x content-dedupe enqueue seam."""

from __future__ import annotations

from lightrag.utils import compute_mdhash_id, sanitize_text_for_encoding

from twindb_lightrag_memgraph._constants import get_confirmed_content_doc_ids
from twindb_lightrag_memgraph.patches import registry


def test_content_derived_ids_match_upstream_computation():
    """Our replica of 1.4.x's enqueue id derivation must not drift."""
    bodies = ["  Hello\r\nworld  ", "second doc"]
    expected = {
        compute_mdhash_id(sanitize_text_for_encoding(body), prefix="doc-")
        for body in bodies
    }

    assert registry._content_derived_doc_ids_for_enqueue(bodies, None) == expected
    assert (
        registry._content_derived_doc_ids_for_enqueue(bodies, ["explicit-id"])
        == frozenset()
    )
    assert (
        registry._content_derived_doc_ids_for_enqueue([b"bytes"], None) == frozenset()
    )


async def test_legacy_enqueue_wrapper_binds_computed_ids(monkeypatch):
    """The 1.4.x wrapper must carry its producer evidence into the call."""
    from lightrag import LightRAG
    from lightrag.base import DocProcessingStatus

    bodies = ["first body", "second body"]
    expected = registry._content_derived_doc_ids_for_enqueue(bodies, None)
    observed: dict[str, object] = {}

    async def fake_enqueue(
        self,
        input,
        ids=None,
        file_paths=None,
        track_id=None,
    ):
        observed.update(
            {
                "self": self,
                "input": input,
                "ids": ids,
                "file_paths": file_paths,
                "track_id": track_id,
                "confirmed_ids": get_confirmed_content_doc_ids(),
            }
        )
        return "upstream-result"

    monkeypatch.setattr(LightRAG, "apipeline_enqueue_documents", fake_enqueue)
    monkeypatch.setattr(
        LightRAG,
        "_twin_legacy_content_dedupe_patched",
        False,
        raising=False,
    )
    monkeypatch.setattr(
        DocProcessingStatus,
        "__dataclass_fields__",
        {
            name: field
            for name, field in DocProcessingStatus.__dataclass_fields__.items()
            if name != "content_hash"
        },
    )

    registry._patch_legacy_content_dedupe_context()

    class StubLightRAG(LightRAG):
        def __init__(self) -> None:
            pass

    rag = StubLightRAG()
    result = await rag.apipeline_enqueue_documents(
        bodies,
        None,
        ["first.txt", "second.txt"],
        "track-legacy",
    )

    assert result == "upstream-result"
    assert observed == {
        "self": rag,
        "input": bodies,
        "ids": None,
        "file_paths": ["first.txt", "second.txt"],
        "track_id": "track-legacy",
        "confirmed_ids": expected,
    }
    assert get_confirmed_content_doc_ids() == frozenset()
