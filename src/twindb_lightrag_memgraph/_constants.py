"""
Centralized constants and helpers shared across all Memgraph storage backends.

Single source of truth for default values, environment variable keys,
and workspace resolution logic.

## Naming: ``workspace`` here ≠ Twin folder

The token ``workspace`` in this module refers strictly to the LightRAG-core
notion: the backtick-safe identifier that becomes the Memgraph node label
(`KV_{workspace}`, `Vec_{workspace}`, `DocStatus_{workspace}`, ...). That
contract is defined upstream by LightRAG and we don't get to rename it.

The user-facing Twin sub-scope is called **folder** everywhere else in this
codebase. The two concepts overlap today because the deploy maps a single
LightRAG workspace per Twin instance, but the wording in this module
deliberately stays "workspace" so anyone touching the storage backends
recognises it as the LightRAG-aligned label and not the Twin overlay surface.
"""

import os
import re
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Iterator

# Environment variable keys.
#
# Workspace resolution chain (`resolve_workspace()`):
#   1. ``MEMGRAPH_WORKSPACE`` — historical alias kept for back-compat
#      with deploys that set it explicitly alongside ``WORKSPACE``.
#   2. ``WORKSPACE`` — the canonical LightRAG-core variable. Setting
#      this single value is now enough for both LightRAG core *and*
#      our Memgraph storage backends.
#   3. ``TWIN_DEFAULT_FOLDER`` — Twin overlay's source of truth; honoured
#      as a fallback so a "folder-only" deploy boots without setting a
#      legacy variable.
#   4. ``DEFAULT_WORKSPACE`` ("base") — the LightRAG-internal default.
#
# Aligning on the chain lets new deploys ship a single ``WORKSPACE``
# or ``TWIN_DEFAULT_FOLDER`` without the old "set both" footgun
# of earlier deployment templates.
MEMGRAPH_WORKSPACE_ENV = "MEMGRAPH_WORKSPACE"
WORKSPACE_ENV = "WORKSPACE"
TWIN_DEFAULT_FOLDER_ENV = "TWIN_DEFAULT_FOLDER"
TWIN_DEFAULT_FOLDER_LABEL_ENV = "TWIN_DEFAULT_FOLDER_LABEL"
TWIN_MAX_FOLDERS_ENV = "TWIN_MAX_FOLDERS"

# Default values
DEFAULT_WORKSPACE = "base"
DEFAULT_MEMGRAPH_URI = "bolt://localhost:7687"
CONNECTION_POOL_SIZE = 50
VECTOR_INDEX_CAPACITY = 100_000
DEFAULT_PAGE_SIZE = 50
DEFAULT_TWIN_MAX_FOLDERS = 5

# Write throttle — limits concurrent write operations (upsert, delete, drop)
# to avoid saturating the Bolt pool during bulk uploads.
MEMGRAPH_WRITE_CONCURRENCY_ENV = "MEMGRAPH_WRITE_CONCURRENCY"
DEFAULT_WRITE_CONCURRENCY = 8
MEMGRAPH_WRITE_SLOT_ACQUIRE_TIMEOUT_ENV = "MEMGRAPH_WRITE_SLOT_ACQUIRE_TIMEOUT"
DEFAULT_WRITE_SLOT_ACQUIRE_TIMEOUT = 5.0

# Pool Bolt tuning — configurable pool size and connection acquire timeout.
MEMGRAPH_POOL_SIZE_ENV = "MEMGRAPH_POOL_SIZE"
MEMGRAPH_CONNECTION_ACQUIRE_TIMEOUT_ENV = "MEMGRAPH_CONNECTION_ACQUIRE_TIMEOUT"
DEFAULT_CONNECTION_ACQUIRE_TIMEOUT = 5.0  # seconds — fail fast, don't hang
MEMGRAPH_OPERATION_TIMEOUT_ENV = "MEMGRAPH_OPERATION_TIMEOUT"
DEFAULT_OPERATION_TIMEOUT = 60.0  # seconds — bound an acquired Bolt session
MEMGRAPH_IDLE_DISCONNECT_SECONDS_ENV = "MEMGRAPH_IDLE_DISCONNECT_SECONDS"
DEFAULT_IDLE_DISCONNECT_SECONDS = 3600.0

# Read pool — dedicated connection pool for read operations.
MEMGRAPH_READ_POOL_SIZE_ENV = "MEMGRAPH_READ_POOL_SIZE"
DEFAULT_READ_POOL_SIZE = 20

# LLM extraction-cache hygiene — when a document lands in FAILED status,
# purge the entity-extraction LLM cache rows tied to its chunks so a
# re-ingestion re-calls the LLM instead of replaying the cached (possibly
# truncated / imparsable) responses. Default ON; set to "0"/"false" to keep
# LightRAG-native behavior (cache rows survive the failure).
# Audit 2026-07-02 addendum, finding B.
TWIN_PURGE_LLM_CACHE_ON_FAILED_ENV = "TWIN_PURGE_LLM_CACHE_ON_FAILED"

# MAGE capability tier — controls detection of optional MAGE query modules
# (community_detection/Louvain, katz_centrality, …) that back additive
# graph-algorithm curation features. "auto" (default) probes the connected
# instance via CALL mg.procedures(); "off" forces the floor tier (base
# memgraph image, LLM-only curation); "on" trusts the operator that MAGE is
# present and skips the probe. The floor tier is ALWAYS fully functional —
# MAGE never gates the storage backends (which are MAGE-free). See
# _capabilities.py.
TWIN_MAGE_ENV = "TWIN_MAGE"

# MarkItDown pre-conversion tier (MARKITDOWN-INGESTION-PLAN.md, PR 1).
# "auto" (default) enables conversion iff the optional markitdown dependency
# ([convert] extra) is importable; "on" forces it (warns and degrades to the
# native path if the import fails); "off" disables it entirely — the native
# LightRAG extraction path is then byte-identical to an unpatched install.
TWIN_CONVERT_ENV = "TWIN_CONVERT"
TWIN_CONVERT_FORMATS_ENV = "TWIN_CONVERT_FORMATS"
TWIN_CONVERT_MAX_BYTES_ENV = "TWIN_CONVERT_MAX_BYTES"
TWIN_CONVERT_TIMEOUT_ENV = "TWIN_CONVERT_TIMEOUT"

# Vision image-ingestion tier (MARKITDOWN-INGESTION-PLAN.md, PR 2).
# Knowledge-Bot pattern: RapidOCR pre-filter -> vision LLM (OpenAI-compatible,
# JSON {image_classification, content}) -> drop noise classes -> markdown.
TWIN_VISION_ENV = "TWIN_VISION"
TWIN_VISION_BASE_URL_ENV = "TWIN_VISION_BASE_URL"
TWIN_VISION_API_KEY_ENV = "TWIN_VISION_API_KEY"
TWIN_VISION_MODEL_ENV = "TWIN_VISION_MODEL"
TWIN_VISION_FORMATS_ENV = "TWIN_VISION_FORMATS"
TWIN_VISION_MAX_BYTES_ENV = "TWIN_VISION_MAX_BYTES"
TWIN_VISION_TIMEOUT_ENV = "TWIN_VISION_TIMEOUT"
TWIN_VISION_MIN_OCR_CHARS_ENV = "TWIN_VISION_MIN_OCR_CHARS"
TWIN_VISION_DROP_CLASSES_ENV = "TWIN_VISION_DROP_CLASSES"

_FALSE_FLAG_VALUES = frozenset({"0", "false", "no", "off"})


def purge_llm_cache_on_failed_enabled() -> bool:
    """Feature flag (default ON) for the FAILED-doc LLM-cache purge."""
    raw = os.environ.get(TWIN_PURGE_LLM_CACHE_ON_FAILED_ENV, "1")
    return raw.strip().lower() not in _FALSE_FLAG_VALUES


_SAFE_IDENTIFIER_RE = re.compile(r"^\w+$", re.ASCII)
_active_storage_folder: ContextVar[str | None] = ContextVar(
    "twin_active_storage_folder",
    default=None,
)
_active_duplicate_share_folder: ContextVar[str | None] = ContextVar(
    "twin_active_duplicate_share_folder",
    default=None,
)
_active_operator_classification: ContextVar[str | None] = ContextVar(
    "twin_active_operator_classification",
    default=None,
)


@dataclass(frozen=True)
class RetrievalFilters:
    """Storage-layer retrieval filters bound for the current query context.

    These are the ``tag_filter`` / ``doc_filter`` / ``min_score`` knobs the
    WebUI sends. They used to be attached to ``QueryParam`` and read by nothing
    in the retrieval path (the LLM was grounded on the *unfiltered* context and
    only the Sources panel was trimmed afterwards — a "faux grounding" lie).
    They are now enforced in :meth:`MemgraphVectorDBStorage._build_search_cypher`
    via :func:`storage_filter_context`, alongside folder membership scoping.

    Sets are normalised by the route boundary:

    - ``doc_*`` hold document ids (case-preserving).
    - ``tag_*`` hold **lower-cased** tag ids (the ``TAGGED_WITH`` graph ids are
      compared case-insensitively, matching the legacy post-filter).
    - ``min_score`` is an explicit cosine-similarity floor. Unfiltered /
      folder-only retrieval keeps the backend's configured
      ``cosine_better_than_threshold``. Doc/tag filtered retrieval treats the
      filter as the candidate corpus and applies only an explicit
      ``min_score`` floor, so the default floor cannot hide an otherwise
      matching tagged document.

    ``all`` vs ``any`` semantics are pinned in
    ``tests/test_retrieval_filters_scoping.py`` — notably ``doc_all`` is strict
    (a chunk has a single ``full_doc_id``, so ``doc_all`` with ≥2 docs is empty),
    not the union-as-``any`` the old post-filter conflated.
    """

    doc_all: frozenset[str] = field(default_factory=frozenset)
    doc_any: frozenset[str] = field(default_factory=frozenset)
    tag_all: frozenset[str] = field(default_factory=frozenset)
    tag_any: frozenset[str] = field(default_factory=frozenset)
    min_score: float = 0.0

    @property
    def has_doc(self) -> bool:
        return bool(self.doc_all or self.doc_any)

    @property
    def has_tag(self) -> bool:
        return bool(self.tag_all or self.tag_any)

    @property
    def is_empty(self) -> bool:
        return not self.has_doc and not self.has_tag and self.min_score <= 0.0


_active_retrieval_filters: ContextVar[RetrievalFilters | None] = ContextVar(
    "twin_active_retrieval_filters",
    default=None,
)


def validate_identifier(value: str, name: str = "identifier") -> str:
    """Validate that a Cypher identifier contains only safe characters.

    Prevents Cypher injection via label names, database names, and
    relationship types that cannot use ``$param`` parameterization.

    Raises:
        ValueError: If *value* contains characters outside ``[a-zA-Z0-9_]``.
    """
    if not value or not _SAFE_IDENTIFIER_RE.match(value):
        raise ValueError(
            f"Invalid {name}: must be non-empty and contain only "
            f"alphanumeric characters or underscores, got {value!r}"
        )
    return value


def resolve_workspace() -> str:
    """Resolve the active workspace (Memgraph label) from the
    environment.

    Falls through the alias chain in order, returning the first
    non-empty value:

      1. ``MEMGRAPH_WORKSPACE``
      2. ``WORKSPACE`` (LightRAG-core canonical)
      3. ``TWIN_DEFAULT_FOLDER`` (Twin overlay)
      4. :data:`DEFAULT_WORKSPACE` (``"base"``)

    Raises:
        ValueError: If the resolved workspace contains unsafe
        characters.
    """
    for env_key in (
        MEMGRAPH_WORKSPACE_ENV,
        WORKSPACE_ENV,
        TWIN_DEFAULT_FOLDER_ENV,
    ):
        candidate = os.environ.get(env_key, "").strip()
        if candidate:
            return validate_identifier(candidate, "workspace")
    return validate_identifier(DEFAULT_WORKSPACE, "workspace")


def default_twin_folder() -> str:
    """Return the fallback Twin folder id used when no request scope exists."""
    candidate = (
        os.environ.get(TWIN_DEFAULT_FOLDER_ENV)
        or os.environ.get(WORKSPACE_ENV)
        or "default"
    ).strip()
    try:
        return validate_identifier(candidate, "folder")
    except ValueError:
        return "default"


def get_active_storage_folder() -> str | None:
    """Folder captured for storage writes in the current async context."""
    return _active_storage_folder.get()


@contextmanager
def storage_folder_context(folder: str | None) -> Iterator[None]:
    """Temporarily bind a Twin folder for low-level storage writes.

    This lives in the storage constants module, not in ``server.folder``, so
    Memgraph storage backends can read it without importing FastAPI/server
    code. Callers must pass an already validated folder id.
    """
    token = _active_storage_folder.set(
        validate_identifier(folder, "folder") if folder else None
    )
    try:
        yield
    finally:
        _active_storage_folder.reset(token)


def get_active_retrieval_filters() -> "RetrievalFilters | None":
    """Retrieval filters captured for read scoping in the current context."""
    return _active_retrieval_filters.get()


@contextmanager
def storage_filter_context(
    filters: "RetrievalFilters | None",
) -> Iterator[None]:
    """Temporarily bind storage-layer retrieval filters.

    Bound by the query routes around ``aquery_llm`` / ``aquery_data`` so every
    vector retrieval LightRAG issues is constrained to the requested docs/tags
    and ``min_score`` *before* the prompt is built — not trimmed afterwards.

    Lives in the storage constants module (FastAPI-free) so the Memgraph vector
    backend can read it without importing server code, mirroring
    :func:`storage_folder_context`. A ``None`` or empty filter set is a no-op so
    the legacy / native LightRAG path is byte-for-byte unchanged.
    """
    token = _active_retrieval_filters.set(
        filters if (filters is not None and not filters.is_empty) else None
    )
    try:
        yield
    finally:
        _active_retrieval_filters.reset(token)


def get_active_duplicate_share_folder() -> str | None:
    """Folder allowed to turn duplicate-file lookups into memberships.

    This is deliberately separate from :func:`get_active_storage_folder`.
    Query/retrieval code also binds a storage folder for read scoping and must
    never mutate memberships just because a low-level getter is called.
    """
    return _active_duplicate_share_folder.get()


@contextmanager
def duplicate_share_folder_context(folder: str | None) -> Iterator[None]:
    """Temporarily allow duplicate upload compatibility sharing.

    Bound only by ingestion routes. It lets LightRAG's synchronous
    duplicate checks become "existing doc joins this folder" without making
    ordinary folder-scoped reads write to the graph.
    """
    token = _active_duplicate_share_folder.set(
        validate_identifier(folder, "folder") if folder else None
    )
    try:
        yield
    finally:
        _active_duplicate_share_folder.reset(token)


def get_active_operator_classification() -> str | None:
    """Operator-selected MIP class captured for the current ingestion context.

    Set from the ``X-Twin-Classification`` upload header. ``None`` means the
    operator made no explicit choice ("no MIP"), so auto-detection alone
    decides. The combination policy (embedded label is a floor — operator can
    raise, never downgrade) lives in
    :func:`classification.apply_operator_classification`.
    """
    return _active_operator_classification.get()


@contextmanager
def operator_classification_context(class_id: str | None) -> Iterator[None]:
    """Temporarily bind an operator-selected MIP classification for ingestion.

    Lives in the storage constants module (not ``server``) so the
    classification hook can read it without importing FastAPI. The raw value is
    validated for safe characters here; ladder membership and the floor policy
    are resolved later in ``classification.apply_operator_classification``.
    An unsafe/garbage value is dropped (treated as "no operator choice").
    """
    cleaned: str | None = None
    if class_id:
        try:
            cleaned = validate_identifier(class_id, "classification")
        except ValueError:
            cleaned = None
    token = _active_operator_classification.set(cleaned)
    try:
        yield
    finally:
        _active_operator_classification.reset(token)
