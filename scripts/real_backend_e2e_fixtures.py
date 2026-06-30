"""Seed and clean disposable Memgraph fixtures for real-backend WebUI e2e."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

from neo4j import GraphDatabase


RETAG_DOC_ID = os.environ.get("REAL_E2E_MUTATION_DOC_ID", "real-e2e-retag-doc")
BULK_DELETE_DOC_ID = os.environ.get(
    "REAL_E2E_BULK_DELETE_DOC_ID", "real-e2e-delete-doc"
)
RETAG_TAG = os.environ.get("REAL_E2E_RETAG_TAG", "real_e2e_tag")
MEMGRAPH_URI = os.environ.get("MEMGRAPH_URI", "bolt://127.0.0.1:7687")
FOLDER_ID = os.environ.get("REAL_BACKEND_FOLDER", "default")


def _driver():
    return GraphDatabase.driver(MEMGRAPH_URI, auth=None)


def seed() -> None:
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    docs = [
        {
            "id": RETAG_DOC_ID,
            "status": "processed",
            "file_path": "/real-e2e/retag-source.txt",
            "content_summary": "Disposable real backend e2e retag source.",
            "content_length": 48,
            "chunks_count": 0,
            "chunks_list": json.dumps([]),
            "metadata": json.dumps({"folder": FOLDER_ID, "tags": [], "e2e": True}),
            "created_at": now,
            "updated_at": now,
            "track_id": "real-e2e-retag-track",
        },
        {
            "id": BULK_DELETE_DOC_ID,
            "status": "processed",
            "file_path": "/real-e2e/delete-source.txt",
            "content_summary": "Disposable real backend e2e delete source.",
            "content_length": 50,
            "chunks_count": 0,
            "chunks_list": json.dumps([]),
            "metadata": json.dumps({"folder": FOLDER_ID, "tags": [], "e2e": True}),
            "created_at": now,
            "updated_at": now,
            "track_id": "real-e2e-delete-track",
        },
    ]
    tag_data = {
        "tag": RETAG_TAG,
        "tier": 3,
        "category": "governance",
        "status": "active",
        "def": "Disposable tag for real backend e2e mutation coverage.",
        "aliases": [],
        "deprecates": [],
        "sources_count": 0,
        "chunks_count": 0,
        "query_freq_30d": 0,
        "related": [],
        "examples": [],
        "created": {"by": "real-e2e", "at": now[:10], "action": "seeded"},
        "last_edit": {"by": "real-e2e", "at": now[:10], "action": "seeded"},
    }

    with _driver() as driver:
        with driver.session() as session:
            cleanup(session=session)
            session.run(
                """
                UNWIND $rows AS row
                MERGE (n:DocStatus_default {id: row.id})
                SET n.status = row.status,
                    n.file_path = row.file_path,
                    n.folder = $folder,
                    n.content_summary = row.content_summary,
                    n.content_length = row.content_length,
                    n.chunks_count = row.chunks_count,
                    n.chunks_list = row.chunks_list,
                    n.metadata = row.metadata,
                    n.created_at = row.created_at,
                    n.updated_at = row.updated_at,
                    n.track_id = row.track_id
                """,
                rows=docs,
                folder=FOLDER_ID,
            ).consume()
            session.run(
                """
                MERGE (folder:Folder_default {id: $folder})
                SET folder.label = $folder
                WITH folder
                MATCH (doc:DocStatus_default)
                WHERE doc.id IN $ids
                MERGE (doc)-[:MEMBER_OF]->(folder)
                """,
                folder=FOLDER_ID,
                ids=[doc["id"] for doc in docs],
            ).consume()
            session.run(
                """
                MERGE (t:WebuiTag_default {id: $tag})
                ON CREATE SET t.__created_at = timestamp()
                SET t.data = $data, t.__updated_at = timestamp()
                """,
                tag=RETAG_TAG,
                data=json.dumps(tag_data, sort_keys=True),
            ).consume()

    print(
        json.dumps(
            {
                "retag_doc": RETAG_DOC_ID,
                "bulk_delete_doc": BULK_DELETE_DOC_ID,
                "tag": RETAG_TAG,
            },
            sort_keys=True,
        )
    )


def cleanup(*, session=None) -> None:
    owns_driver = session is None
    driver = _driver() if owns_driver else None
    active_session = session or driver.session()
    try:
        active_session.run(
            """
            MATCH (n)
            WHERE n.id IN $ids
               OR (n.data IS NOT NULL AND n.data CONTAINS 'real-e2e')
               OR (n.data IS NOT NULL AND n.data CONTAINS 'real_e2e')
            DETACH DELETE n
            """,
            ids=[RETAG_DOC_ID, BULK_DELETE_DOC_ID, RETAG_TAG],
        ).consume()
    finally:
        if owns_driver:
            active_session.close()
            driver.close()


def main() -> int:
    command = sys.argv[1] if len(sys.argv) > 1 else "seed"
    if command == "seed":
        seed()
        return 0
    if command == "cleanup":
        cleanup()
        return 0
    print(f"Unknown command: {command}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
