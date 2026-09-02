"""KB portability — the ``twin-kb-bundle`` format and its operations.

Design record: ``docs/adr/010-kb-portability-contract.md``. FastAPI-free by
construction: the CLI (``python -m twindb_lightrag_memgraph.portability``)
runs inside a bank container with only the storage package installed.
Nothing here is imported by the runtime unless an operator invokes it, so the
absence of any ``TWIN_PORTABILITY_*`` variable changes nothing.
"""

from .manifest import (
    EMBEDDING_PROBE_TEXTS,
    FORMAT,
    FORMAT_VERSION,
    Manifest,
    PROBE_TEXT_SET_ID,
)

__all__ = [
    "EMBEDDING_PROBE_TEXTS",
    "FORMAT",
    "FORMAT_VERSION",
    "Manifest",
    "PROBE_TEXT_SET_ID",
]
