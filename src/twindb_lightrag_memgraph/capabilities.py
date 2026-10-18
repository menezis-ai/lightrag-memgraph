"""Read-only feature discovery, scoped to the connected Memgraph database.

Procedure signatures, rather than version numbers alone, determine which
optional search APIs are usable. No server settings or modules are changed.
"""

import re
from dataclasses import dataclass

from . import _pool


@dataclass(frozen=True)
class MemgraphCapabilities:
    version: str
    procedures: tuple[tuple[str, str], ...]

    @property
    def version_tuple(self) -> tuple[int, int, int]:
        match = re.search(r"(\d+)\.(\d+)\.(\d+)", self.version)
        return tuple(map(int, match.groups())) if match else (0, 0, 0)

    def has_procedure(self, name: str) -> bool:
        return any(proc == name for proc, _ in self.procedures)

    def accepts_config(self, name: str) -> bool:
        return any(
            proc == name
            and re.search(r"\bconfig\b[^,)]*::\s*MAP", signature, re.IGNORECASE)
            for proc, signature in self.procedures
        )

    def returns_score(self, name: str) -> bool:
        return any(
            proc == name and re.search(r"\bscore\s*::", signature)
            for proc, signature in self.procedures
        )

    @property
    def vector_search(self) -> bool:
        return self.has_procedure("vector_search.search")

    @property
    def vector_quantization(self) -> bool:
        return self.version_tuple >= (3, 4, 0)

    @property
    def text_search(self) -> bool:
        # Earlier versions require an experimental startup flag. Never add it:
        # that same flag prevents Memgraph >= 3.6 from starting.
        return self.version_tuple >= (3, 6, 0) and self.has_procedure(
            "text_search.search_all"
        )


async def inspect_memgraph(session=None) -> MemgraphCapabilities:
    """Inspect the current server; call again after an upgrade/module reload.

    Auth, connectivity and procedure-inspection errors deliberately propagate.
    A deployment must not silently lose features because discovery failed.
    """
    if session is None:
        async with _pool.get_session() as active_session:
            return await inspect_memgraph(active_session)

    result = await session.run("SHOW VERSION")
    record = await result.single()
    await result.consume()
    version = str(record[0]) if record else "unknown"

    result = await session.run(
        "CALL mg.procedures() YIELD name, signature RETURN name, signature"
    )
    procedures = tuple(
        [(record["name"], record["signature"]) async for record in result]
    )
    await result.consume()
    return MemgraphCapabilities(version=version, procedures=procedures)
