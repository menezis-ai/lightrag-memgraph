"""Pydantic v2 models for the WebUI phase-1 API surface.

These are the wire-format shapes that the ``lightrag_webui_twin/`` frontend
expects. They mirror the TypeScript interfaces in
``lightrag_webui_twin/src/types/`` and the seed data in ``webui_seed.py`` —
backend phase 1 contract = these models.

Why a separate models module instead of reusing ``LightRAG``'s shapes:
- The WebUI is the operator console, not the LLM API surface. Its endpoints
  return rich human-oriented payloads (tag tier, palier, RBAC, audit feed)
  that have no equivalent in LightRAG core.
- Keeping the WebUI shapes Pydantic-typed (rather than ``dict[str, Any]``)
  lets FastAPI generate accurate OpenAPI docs and the WebUI dev experience
  stays self-documenting via curl.
"""

from __future__ import annotations

from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


class _Base(BaseModel):
    """Tolerant base — accepts extra keys, exports aliases by camelCase."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)


# ---------------------------------------------------------------------------
# Generic list envelope (used by paginated listings)
# ---------------------------------------------------------------------------


class ListEnvelope(_Base, Generic[T]):
    items: list[T]
    total: int


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------


class Document(_Base):
    id: str
    type: Literal["file", "confluence", "sharepoint", "url"]
    source: str
    """File name, URL, or path identifier of the source."""
    summary: str
    """Short human-readable summary of the content."""
    tags: list[str]
    status: Literal["pending", "processing", "completed", "failed"]
    chunks: int
    """Number of chunks the document was split into."""
    updated: str
    """Human-readable relative timestamp (e.g. "2h ago")."""
    visibility: Literal["private", "internal", "public"]
    workspace: str
    """Workspace id this document belongs to."""


# ---------------------------------------------------------------------------
# Topbar: workspaces + notifications
# ---------------------------------------------------------------------------


class Workspace(_Base):
    id: str
    kb: str
    visibility: Literal["private", "internal", "public"]
    sources: int
    role: str
    current: bool


class Notification(_Base):
    id: str
    kind: str
    title: str
    tagname: str | None = None
    suffix: str | None = None
    sub: str | None = None
    rel: str
    read: bool


# ---------------------------------------------------------------------------
# Thesaurus (autocomplete + tag-add)
# ---------------------------------------------------------------------------


class ThesaurusEntry(_Base):
    tag: str
    category: str
    def_: str = Field(alias="def")
    """Short tag definition surfaced in autocomplete tooltips."""


# ---------------------------------------------------------------------------
# Tags governance
# ---------------------------------------------------------------------------


class TagAudit(_Base):
    by: str
    at: str
    action: str | None = None


class TagRelated(_Base):
    tag: str
    strength: float


class TagEntry(_Base):
    tag: str
    tier: int | str
    """1, 2, 3 or "requested"."""
    category: str
    status: Literal[
        "active", "pending-promotion", "pending-review", "deprecated", "rejected"
    ]
    def_: str = Field(alias="def")
    aliases: list[str] = Field(default_factory=list)
    deprecates: list[str] = Field(default_factory=list)
    sources_count: int = 0
    chunks_count: int = 0
    query_freq_30d: int = 0
    created: TagAudit
    last_edit: TagAudit
    related: list[TagRelated] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)
    # Only present for tier="requested"
    requested_by: str | None = None
    requested_at: str | None = None
    justification: str | None = None


class TagCategory(_Base):
    id: str
    label: str
    color: str


# ---------------------------------------------------------------------------
# Activity audit feed
# ---------------------------------------------------------------------------


class ActivityActor(_Base):
    user: str
    role: str


class ActivityTarget(_Base):
    type: str
    label: str
    id: str | None = None


class ActivityEvent(_Base):
    id: str
    ts: str
    rel: str
    day: str
    kind: Literal[
        "retrieval",
        "tag-mutation",
        "doc-retagged",
        "doc-approved",
        "doc-rejected",
        "doc-deleted",
        "source-uploaded",
        "source-ready",
        "source-failed",
        "pipeline-warning",
        "graph-entity-edited",
        "graph-relation-edited",
        "auth",
        "settings",
    ]
    sev: Literal["info", "warning", "error", "critical"]
    actor: ActivityActor
    target: ActivityTarget
    summary: str
    meta: dict[str, Any] = Field(default_factory=dict)


class ActivityEnvelope(_Base):
    items: list[ActivityEvent]
    total: int
    nowMs: int  # noqa: N815 — wire contract for the WebUI fixture-pinned now


# ---------------------------------------------------------------------------
# OpenAPI (curated for the WebUI surface, not FastAPI's own openapi.json)
# ---------------------------------------------------------------------------


class OpenApiEndpoint(_Base):
    m: Literal["GET", "POST", "PUT", "PATCH", "DELETE"]
    p: str
    s: str


class OpenApiGroup(_Base):
    id: str
    name: str
    desc: str
    endpoints: list[OpenApiEndpoint]


class OpenApiEnvelope(_Base):
    groups: list[OpenApiGroup]
    version: str


# ---------------------------------------------------------------------------
# Knowledge graph teaser
# ---------------------------------------------------------------------------


class GraphEntity(_Base):
    id: str
    name: str
    type: Literal["PRODUCT", "TECHNOLOGY", "CONCEPT", "ORG", "PERSON", "LOCATION"]
    x: float
    y: float
    mentions: int
    sources: int
    summary: str


class GraphRelation(_Base):
    id: str
    source: str
    target: str
    label: str
    strength: float


class GraphEntityPatch(_Base):
    """Partial update payload for a graph entity. Every field optional;
    only the keys present in the request body are applied."""

    name: str | None = None
    type: Literal[
        "PRODUCT", "TECHNOLOGY", "CONCEPT", "ORG", "PERSON", "LOCATION"
    ] | None = None
    summary: str | None = None
    tags: list[str] | None = None
    properties: dict[str, str] | None = None


class GraphRelationPatch(_Base):
    """Partial update payload for a graph relation."""

    label: str | None = None
    strength: float | None = None
    properties: dict[str, str] | None = None


class GraphEntityCreate(_Base):
    """Create payload for a manual graph entity addition.

    ``name`` is also used as the LightRAG ``entity_id`` (the PK). A
    409 is returned if a node with this id already exists in the
    workspace — manual creation deliberately doesn't silently overwrite
    an LLM-extracted entity.
    """

    name: str
    type: Literal[
        "PRODUCT", "TECHNOLOGY", "CONCEPT", "ORG", "PERSON", "LOCATION"
    ]
    summary: str | None = None
    tags: list[str] | None = None
    properties: dict[str, str] | None = None


class GraphRelationCreate(_Base):
    """Create payload for a manual graph relation addition.

    ``source`` and ``target`` are WebUI ids (the ``kg_`` prefixed form
    returned by `/graph/entities`). Both endpoints must already exist
    in Memgraph — 422 otherwise.
    """

    source: str
    target: str
    label: str
    strength: float | None = None
    properties: dict[str, str] | None = None


# ---------------------------------------------------------------------------
# Simple ack envelopes for mutations
# ---------------------------------------------------------------------------


class AckResponse(_Base):
    ok: bool = True


# ---------------------------------------------------------------------------
# Tag mutation request bodies (S4c slice 2)
# ---------------------------------------------------------------------------


class TagRequestBody(_Base):
    """POST /tags — propose a new tag for palier-3 review."""

    tag: str
    def_: str = Field(alias="def")
    category: str
    aliases: list[str] = Field(default_factory=list)
    justification: str | None = None
    actor: str | None = None
    """Optional explicit actor for the audit event. Otherwise 'system'."""


class TagEditBody(_Base):
    """PATCH /tags/{name} — edit a tag in place (palier-3 only)."""

    def_: str | None = Field(default=None, alias="def")
    category: str | None = None
    aliases: list[str] | None = None
    deprecates: list[str] | None = None
    actor: str | None = None


class TagApproveBody(_Base):
    """POST /tags/{name}/approve."""

    actor: str | None = None


class TagRejectBody(_Base):
    """POST /tags/{name}/reject."""

    reason: str
    actor: str | None = None


class TagDeprecateBody(_Base):
    """POST /tags/{name}/deprecate."""

    reason: str | None = None
    actor: str | None = None


class TagSynonymsBody(_Base):
    """POST /tags/{name}/synonyms — replace alias list."""

    aliases: list[str]
    actor: str | None = None


class TagDeleteBody(_Base):
    """DELETE /tags/{name} body — migration strategy."""

    strategy: Literal["migrate", "untag"] = "untag"
    to: str | None = None
    actor: str | None = None
