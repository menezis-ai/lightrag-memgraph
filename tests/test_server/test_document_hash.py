from twindb_lightrag_memgraph.server.document_hash import (
    enrich_metadata_with_document_hash,
    lightrag_content_hash_from_doc_id,
)


def test_extracts_lightrag_content_hash_from_doc_id():
    digest = "ABCDEF0123456789ABCDEF0123456789"

    assert lightrag_content_hash_from_doc_id(f"doc-{digest}") == digest.lower()


def test_rejects_non_lightrag_or_non_hash_doc_ids():
    assert lightrag_content_hash_from_doc_id("oracle-restart-procedure.pdf") is None
    assert lightrag_content_hash_from_doc_id("doc-not-a-real-hash") is None


def test_enriches_metadata_with_lightrag_content_hash():
    digest = "abcdef0123456789abcdef0123456789"

    metadata = enrich_metadata_with_document_hash({}, f"doc-{digest}")

    assert metadata["content_hash"] == digest
    assert metadata["content_hash_source"] == "lightrag_doc_id"


def test_keeps_existing_hash_metadata_unchanged():
    metadata = enrich_metadata_with_document_hash(
        {"sha256": "real-sha256"},
        "doc-abcdef0123456789abcdef0123456789",
    )

    assert metadata == {"sha256": "real-sha256"}
