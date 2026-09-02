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
# Vector index capacity (Memgraph ``CREATE VECTOR INDEX ... "capacity"``).
# Fixed at index creation time: changing the env var does not resize an
# existing index (drop + recreate, see README). ``VECTOR_INDEX_CAPACITY`` is
# the historical default kept for import compatibility; the runtime value is
# ``resolve_vector_index_capacity()``.
TWIN_VECTOR_INDEX_CAPACITY_ENV = "TWIN_VECTOR_INDEX_CAPACITY"
DEFAULT_VECTOR_INDEX_CAPACITY = 100_000
VECTOR_INDEX_CAPACITY = DEFAULT_VECTOR_INDEX_CAPACITY
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
# Max age of a pooled Bolt connection before the driver recycles it without
# any I/O. Deployments whose network path kills idle connections earlier than
# the default (e.g. Docker Swarm overlay/IPVS ~900s) can lower this below the
# kill window so recycling happens by age check instead of by a logged
# reset-by-peer on next use. Purely a log-noise knob: the liveness probe
# already keeps defunct sockets from reaching callers.
MEMGRAPH_MAX_CONNECTION_LIFETIME_ENV = "MEMGRAPH_MAX_CONNECTION_LIFETIME"
DEFAULT_MAX_CONNECTION_LIFETIME = 1800.0  # seconds

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

# MarkItDown pre-conversion tier (docs/adr/005-markitdown-ingestion-supply-chain.md).
# "auto" (default) enables conversion iff the optional markitdown dependency
# ([convert] extra) is importable; "on" forces it (warns and degrades to the
# native path if the import fails); "off" disables it entirely — the native
# LightRAG extraction path is then byte-identical to an unpatched install.
TWIN_CONVERT_ENV = "TWIN_CONVERT"
TWIN_CONVERT_FORMATS_ENV = "TWIN_CONVERT_FORMATS"
# Kill-switch for the 1.5.x preconverted-markdown parse seam (B1);
# "off" forces the raw enqueue path even when the capability exists.
TWIN_PRECONVERTED_PARSE_ENV = "TWIN_PRECONVERTED_PARSE"
TWIN_CONVERT_MAX_BYTES_ENV = "TWIN_CONVERT_MAX_BYTES"
TWIN_CONVERT_TIMEOUT_ENV = "TWIN_CONVERT_TIMEOUT"

# Vision image-ingestion tier (docs/adr/005-markitdown-ingestion-supply-chain.md).
# Knowledge-Bot pattern: RapidOCR pre-filter -> vision LLM (OpenAI-compatible,
# JSON {image_classification, content}) -> drop noise classes -> markdown.
TWIN_VISION_ENV = "TWIN_VISION"
TWIN_VISION_BASE_URL_ENV = "TWIN_VISION_BASE_URL"
TWIN_VISION_API_KEY_ENV = "TWIN_VISION_API_KEY"
TWIN_VISION_MODEL_ENV = "TWIN_VISION_MODEL"
TWIN_VISION_FORMATS_ENV = "TWIN_VISION_FORMATS"
TWIN_VISION_MAX_BYTES_ENV = "TWIN_VISION_MAX_BYTES"
TWIN_VISION_TIMEOUT_ENV = "TWIN_VISION_TIMEOUT"
TWIN_VISION_EXTRA_BODY_ENV = "TWIN_VISION_EXTRA_BODY"
TWIN_VISION_MIN_OCR_CHARS_ENV = "TWIN_VISION_MIN_OCR_CHARS"
TWIN_VISION_DROP_CLASSES_ENV = "TWIN_VISION_DROP_CLASSES"

# Generic visual enrichment for standard PDFs.  The procedure profile keeps
# first refusal on a PDF; this tier only sees documents that were not parked
# by that profile.  It uses the shared vision endpoint but deliberately does
# not inherit the standalone-image OCR rejection threshold: low-text diagrams
# are valid PDF knowledge.
TWIN_PDF_VISION_ENV = "TWIN_PDF_VISION"
TWIN_PDF_VISION_MAX_BYTES_ENV = "TWIN_PDF_VISION_MAX_BYTES"
TWIN_PDF_VISION_MAX_PAGES_ENV = "TWIN_PDF_VISION_MAX_PAGES"
TWIN_PDF_VISION_MAX_VISUALS_ENV = "TWIN_PDF_VISION_MAX_VISUALS"
TWIN_PDF_VISION_MAX_RENDERS_ENV = "TWIN_PDF_VISION_MAX_RENDERS"
TWIN_PDF_VISION_RENDER_SCALE_ENV = "TWIN_PDF_VISION_RENDER_SCALE"
TWIN_PDF_VISION_TIMEOUT_ENV = "TWIN_PDF_VISION_TIMEOUT"
TWIN_PDF_VISION_CONCURRENCY_ENV = "TWIN_PDF_VISION_CONCURRENCY"

# Procedure-PDF ingestion profile (docs/adr/007-procedure-pdf-profile.md).
# BNP "IT Group" level-2 procedures get a dedicated path: deterministic
# template detection (or X-Twin-Doc-Type forcing), per-schematic dual vision
# pass, and a human-approval bundle parked BEFORE enqueue.
TWIN_PROCEDURE_ENV = "TWIN_PROCEDURE"
TWIN_PROCEDURE_STORE_FILE_ENV = "TWIN_PROCEDURE_STORE_FILE"
TWIN_PROCEDURE_RENDER_SCALE_ENV = "TWIN_PROCEDURE_RENDER_SCALE"
TWIN_PROCEDURE_MAX_SCHEMATICS_ENV = "TWIN_PROCEDURE_MAX_SCHEMATICS"
TWIN_PROCEDURE_MAX_TOKENS_ENV = "TWIN_PROCEDURE_MAX_TOKENS"
TWIN_PROCEDURE_MAX_BYTES_ENV = "TWIN_PROCEDURE_MAX_BYTES"

_FALSE_FLAG_VALUES = frozenset({"0", "false", "no", "off"})


def purge_llm_cache_on_failed_enabled() -> bool:
    """Feature flag (default ON) for the FAILED-doc LLM-cache purge."""
    raw = os.environ.get(TWIN_PURGE_LLM_CACHE_ON_FAILED_ENV, "1")
    return raw.strip().lower() not in _FALSE_FLAG_VALUES


# Matched with re.fullmatch, never `^...$`: in Python's re, `$` also matches
# just BEFORE a trailing newline, so the previous `^\w+$` accepted exactly one
# hostile shape -- a single trailing newline, "workspace\n" -- and returned it
# intact. Scope, measured rather than assumed: NOT statement injection, because
# nothing may follow that newline ("ws\nMATCH (n) DETACH DELETE n" and
# "workspace\n\n" were both already rejected). What it did allow is a second,
# visually identical namespace, since `Vec_workspace` and `Vec_workspace\n` are
# distinct labels. No live caller was proven to reach it -- the env paths
# .strip() first, and the installed LightRAG strips entity_type before it gets
# here (operate.py:533) -- so this is defence in depth on the last gate before
# f-string interpolation into Cypher, not a patched exploit.
_SAFE_IDENTIFIER_RE = re.compile(r"\w+", re.ASCII)
_active_storage_folder: ContextVar[str | None] = ContextVar(
    "twin_active_storage_folder",
    default=None,
)
_active_duplicate_share_folder: ContextVar[str | None] = ContextVar(
    "twin_active_duplicate_share_folder",
    default=None,
)
_confirmed_content_doc_ids: ContextVar[frozenset[str]] = ContextVar(
    "twin_confirmed_content_doc_ids",
    default=frozenset(),
)
_active_operator_classification: ContextVar[str | None] = ContextVar(
    "twin_active_operator_classification",
    default=None,
)
_active_doc_type: ContextVar[str | None] = ContextVar(
    "twin_active_doc_type",
    default=None,
)
_active_upload_actor: ContextVar[str | None] = ContextVar(
    "twin_active_upload_actor",
    default=None,
)
_active_upload_relative_path: ContextVar[str | None] = ContextVar(
    "twin_active_upload_relative_path",
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

    ``tag_groups`` is the OR-of-groups form of the tag filter: each group is a
    ``(required, optional)`` pair carrying the flat semantics above, and a
    document matches when *at least one* group matches. The route boundary
    populates either ``tag_all``/``tag_any`` (flat form) or ``tag_groups``,
    never both.
    """

    doc_all: frozenset[str] = field(default_factory=frozenset)
    doc_any: frozenset[str] = field(default_factory=frozenset)
    tag_all: frozenset[str] = field(default_factory=frozenset)
    tag_any: frozenset[str] = field(default_factory=frozenset)
    tag_groups: tuple[tuple[frozenset[str], frozenset[str]], ...] = ()
    min_score: float = 0.0

    @property
    def has_doc(self) -> bool:
        return bool(self.doc_all or self.doc_any)

    @property
    def has_tag(self) -> bool:
        return bool(self.tag_all or self.tag_any or self.tag_groups)

    @property
    def is_empty(self) -> bool:
        return not self.has_doc and not self.has_tag and self.min_score <= 0.0


_active_retrieval_filters: ContextVar[RetrievalFilters | None] = ContextVar(
    "twin_active_retrieval_filters",
    default=None,
)
_active_chunk_retrieval_scores: ContextVar[dict[str, float] | None] = ContextVar(
    "twin_active_chunk_retrieval_scores",
    default=None,
)


def validate_identifier(value: str, name: str = "identifier") -> str:
    """Validate that a Cypher identifier contains only safe characters.

    Prevents Cypher injection via label names, database names, and
    relationship types that cannot use ``$param`` parameterization.

    Raises:
        ValueError: If *value* contains characters outside ``[a-zA-Z0-9_]``.
    """
    if not value or not _SAFE_IDENTIFIER_RE.fullmatch(value):
        raise ValueError(
            f"Invalid {name}: must be non-empty and contain only "
            f"alphanumeric characters or underscores, got {value!r}"
        )
    return value


# ---------------------------------------------------------------------------
# KB portability (docs/adr/010-kb-portability-contract.md). None is required; the
# CLI is the only consumer in PR-P1. Malformed numeric values fail at boot
# (register() calls validate_portability_env()) — same posture as the vector
# capacity above.
# ---------------------------------------------------------------------------
TWIN_PORTABILITY_DIR_ENV = "TWIN_PORTABILITY_DIR"
TWIN_PORTABILITY_MAX_BYTES_ENV = "TWIN_PORTABILITY_MAX_BYTES"
TWIN_PORTABILITY_BATCH_SIZE_ENV = "TWIN_PORTABILITY_BATCH_SIZE"
TWIN_PORTABILITY_INCLUDE_ACTIVITY_ENV = "TWIN_PORTABILITY_INCLUDE_ACTIVITY"
TWIN_PORTABILITY_INCLUDE_PROCEDURES_ENV = "TWIN_PORTABILITY_INCLUDE_PROCEDURES"
TWIN_PORTABILITY_ALLOW_UNVERIFIED_ENV = "TWIN_PORTABILITY_ALLOW_UNVERIFIED"
DEFAULT_PORTABILITY_MAX_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB, decompressed
DEFAULT_PORTABILITY_BATCH_SIZE = 1000
_TRUE_FLAG_VALUES = frozenset({"1", "true", "yes", "on"})


def _positive_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if value < 1:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}")
    return value


def resolve_portability_max_bytes() -> int:
    """``TWIN_PORTABILITY_MAX_BYTES`` — decompressed bundle / upload ceiling."""
    return _positive_int_env(
        TWIN_PORTABILITY_MAX_BYTES_ENV, DEFAULT_PORTABILITY_MAX_BYTES
    )


def resolve_portability_batch_size() -> int:
    """``TWIN_PORTABILITY_BATCH_SIZE`` — keyset page size of export/import."""
    return _positive_int_env(
        TWIN_PORTABILITY_BATCH_SIZE_ENV, DEFAULT_PORTABILITY_BATCH_SIZE
    )


def portability_flag_enabled(name: str) -> bool:
    """A default-OFF portability flag (Q6: activity/procedures opt-in)."""
    return os.environ.get(name, "").strip().lower() in _TRUE_FLAG_VALUES


def resolve_portability_dir(working_dir: str | None = None) -> str:
    """``TWIN_PORTABILITY_DIR`` — bundles, uploads, jobs (``<WORKING_DIR>/portability``)."""
    raw = os.environ.get(TWIN_PORTABILITY_DIR_ENV, "").strip()
    if raw:
        return raw
    base = working_dir or os.environ.get("WORKING_DIR", "").strip() or os.getcwd()
    return os.path.join(base, "portability")


def validate_portability_env() -> None:
    """Fail at boot on a malformed numeric portability knob (never defaulted)."""
    resolve_portability_max_bytes()
    resolve_portability_batch_size()


def resolve_vector_index_capacity() -> int:
    """Resolve the vector-index capacity from ``TWIN_VECTOR_INDEX_CAPACITY``.

    Unset or blank → :data:`DEFAULT_VECTOR_INDEX_CAPACITY` (100 000). Any
    other value must be a positive integer — a malformed value is a
    configuration error, raised (not defaulted) so it fails at boot rather
    than silently creating an index of the wrong size.

    Raises:
        ValueError: If the variable is set to something other than a
        positive integer.
    """
    raw = os.environ.get(TWIN_VECTOR_INDEX_CAPACITY_ENV, "").strip()
    if not raw:
        return DEFAULT_VECTOR_INDEX_CAPACITY
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if value < 1:
        raise ValueError(
            f"{TWIN_VECTOR_INDEX_CAPACITY_ENV} must be a positive integer, "
            f"got {raw!r}"
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
    # Strip each alias BEFORE testing it, exactly like resolve_workspace. The
    # previous form ran the `or` chain on the RAW values and stripped only the
    # winner, so a whitespace-only TWIN_DEFAULT_FOLDER was truthy, captured the
    # chain, stripped to empty, failed validation and returned "default" --
    # silently ignoring a perfectly valid WORKSPACE. This value feeds folder
    # membership reads and writes (docstatus_impl), so routing to the wrong
    # folder is not a cosmetic difference.
    #
    # A blank alias falls through to the next one; an alias that is present but
    # UNSAFE still lands on "default" rather than promoting the next alias --
    # a hostile value must not silently hand control to a different source.
    for env_key in (TWIN_DEFAULT_FOLDER_ENV, WORKSPACE_ENV):
        candidate = os.environ.get(env_key, "").strip()
        if not candidate:
            continue
        try:
            return validate_identifier(candidate, "folder")
        except ValueError:
            return "default"
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


def get_active_chunk_retrieval_scores() -> dict[str, float] | None:
    """Measured chunk similarities captured by the current grounding call."""
    return _active_chunk_retrieval_scores.get()


@contextmanager
def retrieval_score_context() -> Iterator[dict[str, float]]:
    """Capture measured chunk similarities in one request-local mapping.

    The mapping is deliberately mutable so concurrent async retrieval tasks
    spawned by one LightRAG call contribute to the same trace. Nested query
    scopes reuse the outer trace; unrelated requests receive distinct mappings
    through :class:`ContextVar`.
    """
    active_scores = _active_chunk_retrieval_scores.get()
    if active_scores is not None:
        yield active_scores
        return

    scores: dict[str, float] = {}
    token = _active_chunk_retrieval_scores.set(scores)
    try:
        yield scores
    finally:
        _active_chunk_retrieval_scores.reset(token)


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


def get_confirmed_content_doc_ids() -> frozenset[str]:
    """Content-derived doc ids confirmed by the active ingestion call.

    LightRAG 1.4.9.11 filters already-known content ids without invoking a
    duplicate getter or persisting a duplicate record. The enqueue seam binds
    the ids computed from the actual normalized input so ``filter_keys`` can
    distinguish that legacy proof of equality from arbitrary ids.
    """
    return _confirmed_content_doc_ids.get()


@contextmanager
def confirmed_content_doc_ids_context(doc_ids: set[str] | frozenset[str]):
    """Bind content-equality evidence for the duration of one enqueue call."""
    token = _confirmed_content_doc_ids.set(frozenset(doc_ids))
    try:
        yield
    finally:
        _confirmed_content_doc_ids.reset(token)


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


def get_active_doc_type() -> str | None:
    """Operator-selected document profile for the current ingestion context.

    Set from the ``X-Twin-Doc-Type`` upload header: ``"procedure"`` forces
    the procedure profile, ``"standard"`` bypasses it, ``None`` (no header)
    lets template auto-detection decide (``_procedure.detect_procedure``).
    """
    return _active_doc_type.get()


@contextmanager
def upload_actor_context(actor: str | None) -> Iterator[None]:
    """Temporarily bind the request-resolved actor for ingestion audit.

    Audit 2026-08-06, R-03a: the authoritative ``source-uploaded`` activity
    event is emitted server-side by the enqueue pipeline, so the actor must
    travel from the HTTP middleware (which can resolve credentials) to the
    patched enqueue method (which cannot). Lives here, not in ``server``,
    for the same reason as :func:`storage_folder_context`. ``None`` means
    "no request context" — the event then records ``actor="unknown"``
    rather than trusting anything client-declared.
    """
    token = _active_upload_actor.set(actor)
    try:
        yield
    finally:
        _active_upload_actor.reset(token)


def get_active_upload_actor() -> str | None:
    """Request-resolved actor for the current ingestion context, if any."""
    return _active_upload_actor.get()


@contextmanager
def upload_relative_path_context(relative_path: str | None) -> Iterator[None]:
    """Bind a server-validated browser-folder path during ingestion."""
    token = _active_upload_relative_path.set(relative_path)
    try:
        yield
    finally:
        _active_upload_relative_path.reset(token)


def get_active_upload_relative_path() -> str | None:
    return _active_upload_relative_path.get()


@contextmanager
def doc_type_context(doc_type: str | None) -> Iterator[None]:
    """Temporarily bind the operator-selected document profile.

    Lives here (not ``server``) so the registry enqueue seam can read it
    without importing FastAPI. Values outside {procedure, standard} are
    dropped (treated as "no operator choice"); the middleware already 400s
    them at the route boundary.
    """
    cleaned = doc_type.strip().lower() if doc_type else None
    if cleaned not in {"procedure", "standard"}:
        cleaned = None
    token = _active_doc_type.set(cleaned)
    try:
        yield
    finally:
        _active_doc_type.reset(token)
