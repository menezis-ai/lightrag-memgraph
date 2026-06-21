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
    DEFAULT_TWIN_MAX_FOLDERS,
    TWIN_DEFAULT_FOLDER_ENV,
    TWIN_DEFAULT_FOLDER_LABEL_ENV,
    TWIN_MAX_FOLDERS_ENV,
    WORKSPACE_ENV,
    validate_identifier,
)

logger = logging.getLogger(__name__)


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
    raw = os.environ.get(TWIN_MAX_FOLDERS_ENV, str(DEFAULT_TWIN_MAX_FOLDERS))
    try:
        value = int(raw)
    except ValueError:
        logger.exception(
            "Invalid %s; falling back to %s",
            TWIN_MAX_FOLDERS_ENV,
            DEFAULT_TWIN_MAX_FOLDERS,
        )
        value = DEFAULT_TWIN_MAX_FOLDERS
    return max(1, min(DEFAULT_TWIN_MAX_FOLDERS, value))


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


def _folder_from_item(item: dict[str, object]) -> TwinFolder | None:
    sid_raw = str(item.get("id") or "").strip()
    if not sid_raw:
        return None
    try:
        sid = validate_identifier(sid_raw, "folder")
    except ValueError:
        logger.exception("Skipping invalid Twin folder id")
        return None
    return TwinFolder(
        id=sid,
        label=str(item.get("label") or sid),
        kind=str(item.get("kind") or "custom"),
        description=str(item.get("description") or ""),
        sources=int(item.get("sources") or 0),
    )


def _parse_configured_folders(raw: str, max_folders: int) -> list[TwinFolder]:
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, list):
            raise ValueError("TWIN_FOLDERS_JSON must be a JSON array")
    except Exception:
        logger.exception("Invalid TWIN_FOLDERS_JSON; falling back to default folder")
        return []

    folders: list[TwinFolder] = []
    try:
        for item in parsed[:max_folders]:
            if not isinstance(item, dict):
                continue
            folder = _folder_from_item(item)
            if folder is not None:
                folders.append(folder)
    except Exception:
        logger.exception("Invalid TWIN_FOLDERS_JSON; falling back to default folder")
        return []
    return folders


def _default_folder_entry(default_folder: str) -> TwinFolder:
    return TwinFolder(
        id=default_folder,
        label=(
            os.environ.get(TWIN_DEFAULT_FOLDER_LABEL_ENV)
            or "Default folder"
        ),
        kind="primary",
        description="SRE-provisioned default folder for this KB.",
        sources=0,
    )


def load_folder_catalog() -> TwinFolderCatalog:
    """Load the configured Twin folders from env vars.

    Supported env vars are ``TWIN_DEFAULT_FOLDER``, ``TWIN_FOLDERS_JSON`` and
    ``TWIN_MAX_FOLDERS``.
    """
    default_folder = _parse_default_folder()
    max_folders = _parse_max_folders()
    folders_raw = os.environ.get("TWIN_FOLDERS_JSON")
    explicit = bool(folders_raw)
    folders = (
        _parse_configured_folders(folders_raw, max_folders)
        if folders_raw
        else []
    )

    if not folders:
        folders = [_default_folder_entry(default_folder)]

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
