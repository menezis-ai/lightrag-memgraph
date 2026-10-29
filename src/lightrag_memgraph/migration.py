"""Read-only storage inventory. This command never migrates or adopts data."""

import argparse
import asyncio
import json
from pathlib import Path

from . import _pool
from ._constants import STORAGE_LABEL_PREFIX
from ._identity import REGISTRY_LABEL, StorageIdentityError, _registered_record, _rows


async def inspect_storage_identities():
    """Inventory labels/indexes and provenance without creating schema metadata."""
    async with _pool.get_read_session() as session:
        rows = await _rows(
            session, f"MATCH (r:`{REGISTRY_LABEL}`) RETURN properties(r) AS props"
        )
        registered, invalid = [], 0
        for row in rows:
            try:
                registered.append(_registered_record(dict(row["props"])))
            except StorageIdentityError:
                invalid += 1
        counts = await _rows(
            session,
            "MATCH (n) UNWIND labels(n) AS label RETURN label, count(*) AS count ORDER BY label",
        )
        indexes = await _rows(session, "SHOW INDEX INFO")
        vectors = await _rows(session, "SHOW VECTOR INDEX INFO")
    owned = {record["id"] for record in registered}
    for row in counts:
        name = row["label"]
        row["classification"] = (
            "registry"
            if name == REGISTRY_LABEL
            else "registered-v2"
            if name in owned
            else "unclaimed-reserved"
            if name.startswith(STORAGE_LABEL_PREFIX)
            else "legacy-or-application"
        )
    return {
        "format": 2,
        "read_only": True,
        "registered": sorted(registered, key=lambda row: row["id"]),
        "invalid_registry_entries": invalid,
        "labels": counts,
        "indexes": indexes,
        "vector_indexes": vectors,
        "migration": "No writes performed. Labels alone do not establish historical tenant ownership; use source manifests and re-ingest into a fresh scope.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, help="Write the JSON inventory locally instead of stdout"
    )
    args = parser.parse_args()

    async def run():
        try:
            return await inspect_storage_identities()
        finally:
            await _pool.close_driver()

    rendered = json.dumps(asyncio.run(run()), indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
