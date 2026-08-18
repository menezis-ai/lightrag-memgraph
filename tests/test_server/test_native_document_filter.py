"""Parity guards for the native document shim's local filters."""

from __future__ import annotations

from twindb_lightrag_memgraph.server.native_shims import _filter_docs


def test_combined_document_filters_preserve_order_and_field_aliases(monkeypatch):
    monkeypatch.setenv("TWIN_DEFAULT_FOLDER", "operations")
    docs = [
        {
            "doc_id": "first",
            "file_path": "/runbooks/oracle-primary.pdf",
            "content_summary": "Recovery procedure",
            "tags": ["approved"],
            "folder": "operations",
        },
        {
            "id": "alias-id",
            "source": "/runbooks/oracle-alias.pdf",
            "summary": "Recovery procedure",
            "tags": ["approved"],
            "metadata": None,
        },
        {
            "doc_id": "wrong-tag",
            "file_path": "/runbooks/oracle-draft.pdf",
            "content_summary": "Recovery procedure",
            "tags": ["draft"],
            "folder": "operations",
        },
        {
            "doc_id": "wrong-folder",
            "file_path": "/runbooks/oracle-archive.pdf",
            "content_summary": "Recovery procedure",
            "tags": ["approved"],
            "folder": "archive",
        },
        {
            "doc_id": "last",
            "file_path": "/runbooks/oracle-secondary.pdf",
            "content_summary": "Recovery procedure",
            "tags": ["approved", "database"],
            "metadata": {},
        },
    ]

    result = _filter_docs(
        docs,
        q="ORACLE",
        tag="approved",
        folder="operations",
        source="RUNBOOKS",
    )

    assert [doc.get("doc_id") or doc.get("id") for doc in result] == [
        "first",
        "alias-id",
        "last",
    ]
    assert _filter_docs(
        docs,
        q=None,
        tag=None,
        folder="operations",
        doc_id="alias-id",
    ) == [docs[1]]
