"""Twin folder catalog parsing.

This module is deliberately free of FastAPI/server imports: runtime WebUI
config generation uses it even when only ``replace_ui=True`` is enabled.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass

from ._constants import (
    TWIN_DEFAULT_FOLDER_ENV,
    WORKSPACE_ENV,
    validate_identifier,
)

logger = logging.getLogger(__name__)

# Hard ceiling on configurable Twin folders; also the fallback when
# TWIN_MAX_FOLDERS is unset or invalid.
MAX_FOLDERS_CEILING = 5


@dataclass(frozen=True)
class TwinFolder:
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

    def as_api(self, *, current: bool) -> dict[str, object]:
        return {
            "id": self.id,
            "kb": self.label,
            "visibility": "private" if self.kind == "sandbox" else "internal",
            "sources": self.sources,
            "role": "admin / steward",
            "current": current,
        }


@dataclass(frozen=True)
class TwinFolderCatalog:
    default_folder_id: str
    max_folders: int
    folders: tuple[TwinFolder, ...]
    explicit: bool

    @property
    def ids(self) -> frozenset[str]:
        return frozenset(folder.id for folder in self.folders)


def _parse_max_folders() -> int:
    raw = os.environ.get("TWIN_MAX_FOLDERS", str(MAX_FOLDERS_CEILING))
    try:
        value = int(raw)
    except ValueError:
        logger.exception(
            "Invalid TWIN_MAX_FOLDERS; falling back to %d", MAX_FOLDERS_CEILING
        )
        value = MAX_FOLDERS_CEILING
    return max(1, min(MAX_FOLDERS_CEILING, value))


def _parse_default_folder() -> str:
    raw = (
        os.environ.get(TWIN_DEFAULT_FOLDER_ENV)
        or os.environ.get(WORKSPACE_ENV)
        or "default"
    ).strip()
    try:
        return validate_identifier(raw, "folder")
    except ValueError:
        logger.exception("Invalid default Twin folder; falling back to 'default'")
        return "default"


def load_folder_catalog() -> TwinFolderCatalog:
    """Load the configured Twin folders from env vars.

    Supported env vars are ``TWIN_DEFAULT_FOLDER``, ``TWIN_FOLDERS_JSON`` and
    ``TWIN_MAX_FOLDERS``.
    """
    default_folder = _parse_default_folder()
    max_folders = _parse_max_folders()
    folders_raw = os.environ.get("TWIN_FOLDERS_JSON")
    explicit = bool(folders_raw)
    folders: list[TwinFolder] = []

    if folders_raw:
        try:
            parsed = json.loads(folders_raw)
            if not isinstance(parsed, list):
                raise ValueError("TWIN_FOLDERS_JSON must be a JSON array")
            for item in parsed[:max_folders]:
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
                folders.append(
                    TwinFolder(
                        id=sid,
                        label=str(item.get("label") or sid),
                        kind=str(item.get("kind") or "custom"),
                        description=str(item.get("description") or ""),
                        sources=int(item.get("sources") or 0),
                    )
                )
        except Exception:
            logger.exception("Invalid TWIN_FOLDERS_JSON; falling back to default folder")
            folders = []

    if not folders:
        folders = [
            TwinFolder(
                id=default_folder,
                label=(
                    os.environ.get("TWIN_DEFAULT_FOLDER_LABEL")
                    or "Default folder"
                ),
                kind="primary",
                description="SRE-provisioned default folder for this KB.",
                sources=0,
            )
        ]

    if default_folder not in {folder.id for folder in folders}:
        default_folder = folders[0].id

    return TwinFolderCatalog(
        default_folder_id=default_folder,
        max_folders=max_folders,
        folders=tuple(folders),
        explicit=explicit,
    )


def build_runtime_folder_config() -> dict[str, object]:
    catalog = load_folder_catalog()
    folders = [folder.as_runtime_config() for folder in catalog.folders]
    return {
        "defaultFolderId": catalog.default_folder_id,
        "folders": folders,
        "maxFolders": catalog.max_folders,
    }
