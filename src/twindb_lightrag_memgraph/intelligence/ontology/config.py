"""
intelligence/ontology/config.py
================================
Loads ontology configuration from ontology.json.
Returns None if file absent = feature disabled.
No JSON file = zero behavior change.

Uses stdlib json -- no extra dependency needed.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger("twin_rag_intelligence.ontology.config")


@dataclass
class WorkspaceOntologyConfig:
    """Per-workspace ontology configuration."""

    mode: str  # "dedicated" | "emergence" | "deep_extraction"
    subject: str = ""
    context: str = ""
    dsep_operators: list[str] = field(default_factory=list)
    dsep_operators_global: list[str] = field(default_factory=list)
    dsep_operators_local: list[str] = field(default_factory=list)


@dataclass
class OntologyConfig:
    """Top-level ontology configuration."""

    enabled: bool = False
    confidence_threshold: float = 0.7
    require_review: bool = True
    dsep_enabled: bool = True
    dual_pass: bool = False
    global_max_tokens: int = 20000
    workspaces: dict[str, WorkspaceOntologyConfig] = field(default_factory=dict)


_VALID_MODES = ("dedicated", "emergence", "deep_extraction")


def load_ontology_config(path: Optional[Path] = None) -> Optional[OntologyConfig]:
    """Load ontology configuration from JSON.

    Args:
        path: Path to ontology.json. Defaults to project root.

    Returns:
        OntologyConfig if file exists and is valid, None otherwise.
    """
    if path is None:
        path = Path("ontology.json")

    if not path.exists():
        logger.debug("No ontology.json found at %s, feature disabled", path)
        return None

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid ontology.json: {exc}") from exc

    if not isinstance(raw, dict):
        raise ValueError("ontology.json must be a JSON object")

    workspaces = {}
    for ws_name, ws_data in raw.get("workspaces", {}).items():
        if not isinstance(ws_data, dict):
            continue
        mode = ws_data.get("mode", "emergence")
        if mode not in _VALID_MODES:
            raise ValueError(
                f"Invalid mode '{mode}' for workspace '{ws_name}'. "
                f"Must be: {', '.join(_VALID_MODES)}"
            )
        workspaces[ws_name] = WorkspaceOntologyConfig(
            mode=mode,
            subject=ws_data.get("subject", ""),
            context=ws_data.get("context", ""),
            dsep_operators=ws_data.get("dsep_operators", []),
            dsep_operators_global=ws_data.get("dsep_operators_global", []),
            dsep_operators_local=ws_data.get("dsep_operators_local", []),
        )

    return OntologyConfig(
        enabled=raw.get("enabled", False),
        confidence_threshold=raw.get("confidence_threshold", 0.7),
        require_review=raw.get("require_review", True),
        dsep_enabled=raw.get("dsep_enabled", True),
        dual_pass=raw.get("dual_pass", False),
        global_max_tokens=raw.get("global_max_tokens", 20000),
        workspaces=workspaces,
    )
