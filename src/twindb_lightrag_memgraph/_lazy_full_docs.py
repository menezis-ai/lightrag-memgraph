"""Lazy full_docs proxy: purge content after processing, reconstruct on demand.

Strategy: write-then-purge.  The LightRAG pipeline writes ``full_docs``
normally.  After ``_insert_done`` fires (doc is PROCESSED), a post-index
hook deletes the raw content.  If the content is ever needed again
(e.g., consistency check, re-processing after status reset), it is
reconstructed by concatenating ``text_chunks`` using the chunk IDs
stored in ``doc_status.chunks_list``.

Controlled by ``MEMGRAPH_PURGE_FULL_DOCS=on`` (off by default).
"""

import logging
import os

from ._constants import MEMGRAPH_PURGE_FULL_DOCS_ENV

logger = logging.getLogger("twindb_lightrag_memgraph")

_reconstruction_patched: set[int] = set()


def is_enabled() -> bool:
    """Return True if lazy full_docs purge is enabled."""
    return os.environ.get(MEMGRAPH_PURGE_FULL_DOCS_ENV, "off").lower() == "on"


async def purge_processed_full_docs(lightrag_instance) -> None:
    """Post-index hook: delete raw content from full_docs for PROCESSED docs.

    Registered via :func:`register_post_index_hook` when
    ``MEMGRAPH_PURGE_FULL_DOCS=on``.
    """
    if not is_enabled():
        return

    # Patch reconstruction on first invocation per instance
    instance_id = id(lightrag_instance)
    if instance_id not in _reconstruction_patched:
        patch_full_docs_with_lazy_reconstruction(lightrag_instance)
        _reconstruction_patched.add(instance_id)

    from lightrag.base import DocStatus

    try:
        processed = await lightrag_instance.doc_status.get_docs_by_status(
            DocStatus.PROCESSED
        )
        if not processed:
            return

        # Only purge entries that still have content in full_docs
        doc_ids_to_purge = []
        for doc_id in processed:
            content = await lightrag_instance.full_docs.get_by_id(doc_id)
            if content is not None:
                doc_ids_to_purge.append(doc_id)

        if doc_ids_to_purge:
            await lightrag_instance.full_docs.delete(doc_ids_to_purge)
            logger.info(
                "Purged full_docs content for %d PROCESSED documents",
                len(doc_ids_to_purge),
            )
    except Exception:
        logger.exception("Failed to purge full_docs after indexation")


async def _reconstruct_from_chunks(doc_id, doc_status_store, text_chunks_store):
    """Reassemble full document content from text_chunks via doc_status.chunks_list."""
    doc_status_data = await doc_status_store.get_by_id(doc_id)
    if not doc_status_data:
        return None

    chunks_list = doc_status_data.get("chunks_list", [])
    if not chunks_list:
        logger.warning("No chunks_list for doc %s, cannot reconstruct", doc_id)
        return None

    chunks_data = await text_chunks_store.get_by_ids(chunks_list)
    parts = [chunk["content"] for chunk in chunks_data if chunk and "content" in chunk]

    if not parts:
        logger.warning("No chunk content found for doc %s", doc_id)
        return None

    file_path = doc_status_data.get("file_path", "unknown_source")
    logger.info(
        "Reconstructed %d chars for doc %s from %d chunks",
        sum(len(p) for p in parts),
        doc_id,
        len(parts),
    )
    return {"content": "\n".join(parts), "file_path": file_path}


def patch_full_docs_with_lazy_reconstruction(lightrag_instance) -> None:
    """Patch ``full_docs.get_by_id`` to reconstruct from text_chunks when missing."""
    full_docs = lightrag_instance.full_docs
    doc_status_store = lightrag_instance.doc_status
    text_chunks_store = lightrag_instance.text_chunks

    _original_get_by_id = full_docs.get_by_id

    async def _lazy_get_by_id(id: str):
        result = await _original_get_by_id(id)
        if result is not None:
            return result

        logger.debug("Reconstructing full_docs for %s from text_chunks", id)
        try:
            return await _reconstruct_from_chunks(
                id, doc_status_store, text_chunks_store
            )
        except Exception:
            logger.exception("Failed to reconstruct full_docs for %s", id)
            return None

    full_docs.get_by_id = _lazy_get_by_id
    logger.debug("Patched full_docs.get_by_id with lazy reconstruction")


def reset_reconstruction_state() -> None:
    """Clear reconstruction patch tracking. For test teardown."""
    _reconstruction_patched.clear()
