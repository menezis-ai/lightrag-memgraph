"""
twin_rag_intelligence/thesaurus/loader.py
==========================================
Thesaurus loader with LRU cache.
"""

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger("twin_rag_intelligence.thesaurus")

_THESAURUS_PATH = Path(__file__).parent / "it_ops_thesaurus.json"


class ThesaurusLoader:
    """Loads and caches the IT/Ops thesaurus from JSON."""

    def load(self) -> dict[str, Any]:
        return _load_thesaurus()


@lru_cache(maxsize=1)
def _load_thesaurus() -> dict[str, Any]:
    """Load the IT/Ops thesaurus from the JSON file (cached)."""
    if not _THESAURUS_PATH.exists():
        logger.warning("Thesaurus not found: %s", _THESAURUS_PATH)
        return {"version": "0.0", "glossaire": []}

    with open(_THESAURUS_PATH, encoding="utf-8") as f:
        data = json.load(f)
        logger.info(
            "IT/Ops thesaurus loaded: v%s, %d terms",
            data.get("version"),
            len(data.get("glossaire", [])),
        )
        return data
