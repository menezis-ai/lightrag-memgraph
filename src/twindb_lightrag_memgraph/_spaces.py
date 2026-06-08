"""Twin folder catalog parsing.

This module is deliberately free of FastAPI/server imports: runtime WebUI
config generation uses it even when only ``replace_ui=True`` is enabled.

``space`` remains as a legacy wire name for existing deployments. Product and
new API surfaces use ``folder``.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass

from ._constants import validate_identifier

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TwinSpace:
    id: str
    label: str
    kind: str = "custom"
    description: str = ""
    sources: int = 0

    def as_runtime_config(self) -> dict[str, object]:
        return {
            "id": self.id,
            "label": self.label,
            "kind": self.kind,
            "description": self.description,
            "sources": self.sources,
        }

    def as_workspace_compat(self, *, current: bool) -> dict[str, object]:
        return {
            "id": self.id,
            "kb": self.label,
            "visibility": "private" if self.kind == "sandbox" else "internal",
            "sources": self.sources,
            "role": "admin / steward",
            "current": current,
        }


@dataclass(frozen=True)
class TwinSpaceCatalog:
    default_space_id: str
    max_spaces: int
    spaces: tuple[TwinSpace, ...]
    explicit: bool

    @property
    def ids(self) -> frozenset[str]:
        return frozenset(space.id for space in self.spaces)


def _parse_max_spaces() -> int:
    raw = os.environ.get("TWIN_MAX_FOLDERS") or os.environ.get(
        "TWIN_MAX_SPACES",
        "5",
    )
    try:
        value = int(raw)
    except ValueError:
        logger.exception("Invalid TWIN_MAX_FOLDERS/TWIN_MAX_SPACES; falling back to 5")
        value = 5
    return max(1, min(5, value))


def _parse_default_space() -> str:
    raw = (
        os.environ.get("TWIN_DEFAULT_FOLDER")
        or os.environ.get("TWIN_DEFAULT_SPACE")
        or os.environ.get("WORKSPACE")
        or "default"
    ).strip()
    try:
        return validate_identifier(raw, "folder")
    except ValueError:
        logger.exception("Invalid default Twin folder; falling back to 'default'")
        return "default"


def load_space_catalog() -> TwinSpaceCatalog:
    """Load the configured Twin folders from env vars.

    Preferred env vars are ``TWIN_DEFAULT_FOLDER``, ``TWIN_FOLDERS_JSON`` and
    ``TWIN_MAX_FOLDERS``. Legacy ``*_SPACE*`` names are still accepted.
    """
    default_space = _parse_default_space()
    max_spaces = _parse_max_spaces()
    spaces_raw = os.environ.get("TWIN_FOLDERS_JSON") or os.environ.get(
        "TWIN_SPACES_JSON"
    )
    explicit = bool(spaces_raw)
    spaces: list[TwinSpace] = []

    if spaces_raw:
        try:
            parsed = json.loads(spaces_raw)
            if not isinstance(parsed, list):
                raise ValueError("TWIN_FOLDERS_JSON must be a JSON array")
            for item in parsed[:max_spaces]:
                if not isinstance(item, dict):
                    continue
                sid_raw = str(item.get("id") or "").strip()
                if not sid_raw:
                    continue
                try:
                    sid = validate_identifier(sid_raw, "folder")
                except ValueError:
                    logger.exception("Skipping invalid Twin folder id")
                    continue
                spaces.append(
                    TwinSpace(
                        id=sid,
                        label=str(item.get("label") or sid),
                        kind=str(item.get("kind") or "custom"),
                        description=str(item.get("description") or ""),
                        sources=int(item.get("sources") or 0),
                    )
                )
        except Exception:
            logger.exception(
                "Invalid TWIN_FOLDERS_JSON/TWIN_SPACES_JSON; falling back to default folder"
            )
            spaces = []

    if not spaces:
        spaces = [
            TwinSpace(
                id=default_space,
                label=(
                    os.environ.get("TWIN_DEFAULT_FOLDER_LABEL")
                    or os.environ.get("TWIN_DEFAULT_SPACE_LABEL")
                    or "Default folder"
                ),
                kind="primary",
                description="SRE-provisioned default folder for this KB.",
                sources=0,
            )
        ]

    if default_space not in {space.id for space in spaces}:
        default_space = spaces[0].id

    return TwinSpaceCatalog(
        default_space_id=default_space,
        max_spaces=max_spaces,
        spaces=tuple(spaces),
        explicit=explicit,
    )


def build_runtime_space_config() -> dict[str, object]:
    catalog = load_space_catalog()
    folders = [space.as_runtime_config() for space in catalog.spaces]
    return {
        "defaultFolderId": catalog.default_space_id,
        "folders": folders,
        "maxFolders": catalog.max_spaces,
        "defaultSpaceId": catalog.default_space_id,
        "spaces": folders,
        "maxSpaces": catalog.max_spaces,
    }


TwinFolder = TwinSpace
TwinFolderCatalog = TwinSpaceCatalog
load_folder_catalog = load_space_catalog
build_runtime_folder_config = build_runtime_space_config
