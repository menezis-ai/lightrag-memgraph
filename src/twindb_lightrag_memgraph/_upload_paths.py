"""Safe relative-path identities shared by storage and server surfaces.

This module belongs to the public storage slice because ``docstatus_impl``
persists the path metadata. Server code imports the compatibility shim at
``server.upload_paths``; keeping the implementation here prevents a public
slice export from acquiring a private ``server`` dependency.
"""

from __future__ import annotations

import base64
import binascii
import unicodedata
from pathlib import PurePosixPath

MAX_FOLDER_DEPTH = 20
# A nested path is base64url-expanded by roughly 4/3 and stored as one
# filesystem basename. 140 UTF-8 bytes leaves room for ``twinrel_``, the
# original suffix, and filesystem variance under the common 255-byte limit.
MAX_RELATIVE_PATH_BYTES = 140
RELATIVE_PATH_HEADER = "X-Twin-Relative-Path"
_STORAGE_PREFIX = "twinrel_"


def normalize_relative_upload_path(value: str) -> str:
    """Return one NFC, POSIX relative path or reject ambiguous input."""
    raw = unicodedata.normalize("NFC", str(value or "").strip())
    if not raw or raw.startswith("/") or "\\" in raw:
        raise ValueError("relative upload path must be a non-empty POSIX path")
    if any(ord(char) < 32 or char == "\x7f" for char in raw):
        raise ValueError("relative upload path contains control characters")
    parts = raw.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("relative upload path contains an empty or traversal segment")
    if len(parts) > MAX_FOLDER_DEPTH:
        raise ValueError(f"relative upload path exceeds {MAX_FOLDER_DEPTH} levels")
    if len(raw.encode("utf-8")) > MAX_RELATIVE_PATH_BYTES:
        raise ValueError(
            f"relative upload path exceeds {MAX_RELATIVE_PATH_BYTES} UTF-8 bytes"
        )
    return "/".join(parts)


def canonical_upload_file_name(relative_path: str) -> str:
    """Losslessly encode a nested path as a safe LightRAG basename identity."""
    normalized = normalize_relative_upload_path(relative_path)
    if "/" not in normalized:
        return normalized
    token = base64.urlsafe_b64encode(normalized.encode("utf-8")).decode().rstrip("=")
    suffix = PurePosixPath(normalized).suffix
    return f"{_STORAGE_PREFIX}{token}{suffix}"


def relative_path_from_storage_name(file_name: str) -> str | None:
    """Decode a folder-upload storage key; ordinary upload names return None."""
    if not file_name.startswith(_STORAGE_PREFIX):
        return None
    payload = file_name[len(_STORAGE_PREFIX) :]
    suffix = PurePosixPath(payload).suffix
    token = payload[: -len(suffix)] if suffix else payload
    try:
        padded = token + "=" * (-len(token) % 4)
        decoded = base64.urlsafe_b64decode(padded).decode("utf-8")
        normalized = normalize_relative_upload_path(decoded)
    except (binascii.Error, ValueError, UnicodeError):
        return None
    return normalized if canonical_upload_file_name(normalized) == file_name else None


def display_upload_file_path(file_path: str) -> str:
    """Return the original relative path for encoded folder-upload names."""
    return relative_path_from_storage_name(file_path) or file_path


__all__ = [
    "MAX_FOLDER_DEPTH",
    "MAX_RELATIVE_PATH_BYTES",
    "RELATIVE_PATH_HEADER",
    "canonical_upload_file_name",
    "display_upload_file_path",
    "normalize_relative_upload_path",
    "relative_path_from_storage_name",
]
