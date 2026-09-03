"""Shared contract of the frozen v1 bundle fixture.

``tests/fixtures/bundles/twin-kb-bundle-v1`` is a REAL export: it was produced
by ``scripts/portability_freeze_fixture.py`` against a live Memgraph and then
committed verbatim.  Its whole value is that it does **not** move when the
exporter moves — it is the only artefact in the suite the current code did not
just write, so it is the only one that can prove a bundle released to an
operator still imports.

Both the generator and ``test_golden_bundle.py`` import this module, so the
embedding that sealed the bundle's probe vectors is byte-for-byte the one that
re-probes them at import time.  The function is a pure function of the text —
never of its position in a batch — because the freeze run and the import run do
not batch the probe sentences the same way, and an index-dependent embedding
would make the fixture fail on a cosine blocker for no real reason.
"""

from __future__ import annotations

import hashlib
import struct
import zlib
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any

#: Repository-relative home of every frozen bundle.
FIXTURES_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "bundles"

#: The one frozen v1 bundle.  A v2 format gets its own directory next to it;
#: this one stays for as long as v1 bundles are claimed to be importable.
GOLDEN_V1_DIR = FIXTURES_ROOT / "twin-kb-bundle-v1"

GOLDEN_DIM = 4
GOLDEN_EMBEDDING_MODEL = "twin-golden-fixture"

#: Workspace and folders the fixture was exported from.  The import tests target
#: a *different*, empty workspace — a v1 import refuses a populated target.
GOLDEN_SOURCE_WORKSPACE = "golden_source"
GOLDEN_SOURCE_FOLDERS = ("gf1", "gf2")
GOLDEN_RUNTIME_FOLDER = "gf3"

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

#: The only ``DROP VECTOR INDEX`` failure that means "already gone".  Anything
#: else — a reset connection, an auth error, a permission denial — must reach
#: the caller.  Mirrors ``MemgraphVectorDBStorage.drop()``.
_INDEX_ABSENT_MARKERS = ("does not exist", "doesn't exist")


def golden_vector(text: str) -> list[float]:
    """A deterministic unit-free vector that depends only on *text*."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return [(digest[index] / 255.0) * 2.0 - 1.0 for index in range(GOLDEN_DIM)]


def golden_embedding() -> Any:
    """The ``EmbeddingFunc`` that sealed the fixture and must re-probe it."""
    from lightrag.utils import EmbeddingFunc

    async def embed(texts: list[str]) -> Any:
        import numpy as np

        return np.asarray([golden_vector(text) for text in texts], dtype=np.float32)

    return EmbeddingFunc(embedding_dim=GOLDEN_DIM, max_token_size=8192, func=embed)


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    payload = tag + data
    return (
        struct.pack(">I", len(data))
        + payload
        + struct.pack(">I", zlib.crc32(payload) & 0xFFFFFFFF)
    )


def golden_png(width: int = 8, height: int = 8) -> bytes:
    """A real, decodable, deterministic 8-bit greyscale PNG.

    The fixture's procedure schematic has to be an actual image. Seeding the
    PNG signature followed by arbitrary text produced a file no decoder
    accepts, which reduced the "procedure file plane is covered" claim to
    "opaque bytes are copied" — the export never looks inside, so only a real
    image proves an image survives the round trip.
    """
    if width < 1 or height < 1:
        raise ValueError("PNG dimensions must be positive")
    raw = b"".join(
        b"\x00" + bytes((x * 31 + y * 17) % 256 for x in range(width))
        for y in range(height)
    )
    # bit depth 8, colour type 0 (greyscale), deflate, adaptive filter, no interlace
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    return (
        PNG_SIGNATURE
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", zlib.compress(raw, 9))
        + _png_chunk(b"IEND", b"")
    )


def png_chunks(data: bytes) -> list[tuple[str, int]]:
    """Walk a PNG, verifying every chunk CRC. Raises ``ValueError`` if invalid.

    Deliberately stdlib-only: Pillow lives in the ``vision``/``procedure``
    extras, which the unit-test and integration-test CI jobs do not install, so
    a Pillow-based check would silently not run where it matters. A CRC walk is
    also the stricter test — a lenient decoder can accept a file whose chunk
    checksums no longer match its bytes.
    """
    if not data.startswith(PNG_SIGNATURE):
        raise ValueError("not a PNG: bad signature")
    chunks: list[tuple[str, int]] = []
    offset = len(PNG_SIGNATURE)
    while offset < len(data):
        if offset + 8 > len(data):
            raise ValueError("truncated chunk header")
        (length,) = struct.unpack(">I", data[offset : offset + 4])
        tag = data[offset + 4 : offset + 8]
        body_end = offset + 8 + length
        if body_end + 4 > len(data):
            raise ValueError(f"truncated chunk {tag!r}")
        payload = data[offset + 4 : body_end]
        (expected,) = struct.unpack(">I", data[body_end : body_end + 4])
        if zlib.crc32(payload) & 0xFFFFFFFF != expected:
            raise ValueError(f"chunk {tag!r} fails its CRC")
        chunks.append((tag.decode("ascii", "replace"), length))
        offset = body_end + 4
    tags = [tag for tag, _ in chunks]
    if not tags or tags[0] != "IHDR":
        raise ValueError("first chunk must be IHDR")
    if tags[-1] != "IEND":
        raise ValueError("last chunk must be IEND")
    if "IDAT" not in tags:
        raise ValueError("no IDAT chunk: the PNG carries no image data")
    return chunks


async def drop_vector_store(
    run: Callable[[str], Awaitable[Any]], *, workspace: str, namespace: str
) -> None:
    """Delete one vector store the way ``MemgraphVectorDBStorage.drop()`` does.

    Two guarantees, both load-bearing and both easy to lose by hand:
    ``REMOVE`` the label before ``DETACH DELETE`` (Memgraph 3.10+ otherwise
    keeps stale vector-index references that break the next ingest), and
    swallow only the idempotent "index already gone" failure so a reset
    connection or a permission error cannot be mistaken for a clean slate.
    """
    label = f"Vec_{workspace}_{namespace}"
    await run(f"MATCH (n:`{label}`) REMOVE n:`{label}` WITH n DETACH DELETE n")
    try:
        await run(f"DROP VECTOR INDEX `vec_{workspace}_{namespace}`")
    except Exception as exc:
        message = str(exc).lower()
        if not any(marker in message for marker in _INDEX_ABSENT_MARKERS):
            raise


async def wipe_workspace(
    run: Callable[[str], Awaitable[Any]],
    *,
    workspace: str,
    folders: Sequence[str],
) -> None:
    """Remove every label a golden seed or import can touch. Safe on a clean DB."""
    from twindb_lightrag_memgraph.portability.stores import (
        KV_NAMESPACES,
        KV_NEVER_NAMESPACES,
        VEC_NAMESPACES,
    )

    labels = [
        *(f"KV_{workspace}_{ns}" for ns in (*KV_NAMESPACES, *KV_NEVER_NAMESPACES)),
        f"DocStatus_{workspace}",
        f"Folder_{workspace}",
        f"GraphOverride_{workspace}",
        f"GraphRelOverride_{workspace}",
        f"WebuiSettings_{workspace}",
        f"TwinSourceLink_{workspace}",
        f"WebuiApiKey_{workspace}",
        workspace,
        # Folder-scoped labels are shared across workspaces: a leftover would
        # make the next run see a non-empty target and block.
        *(f"WebuiTag_{folder}" for folder in folders),
        *(f"WebuiTagCategory_{folder}" for folder in folders),
        *(f"WebuiActivity_{folder}" for folder in folders),
        *(f"WebuiNotification_{folder}" for folder in folders),
    ]
    for label in labels:
        await run(f"MATCH (n:`{label}`) DETACH DELETE n")
    for namespace in VEC_NAMESPACES:
        await drop_vector_store(run, workspace=workspace, namespace=namespace)
