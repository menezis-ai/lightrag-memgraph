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

from pydantic import BaseModel, ConfigDict, Field, field_validator

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
    # Legacy WebUI seed shape.
    id: str | None = Field(default=None, description="Document id.")
    type: Literal["file", "confluence", "sharepoint", "url"] = Field(
        description="Where the document came from."
    )
    source: str | None = Field(
        default=None,
        description="File name, URL, or path identifier of the source.",
    )
    summary: str | None = Field(
        default=None, description="Short human-readable summary of the content."
    )
    tags: list[str] = Field(description="Tags attached to the document.")
    status: str = Field(
        description=(
            "Ingestion status (e.g. `processed`, `pending`, `processing`, " "`failed`)."
        )
    )
    chunks: int | None = Field(
        default=None, description="Number of chunks the document was split into."
    )
    updated: str | None = Field(
        default=None,
        description='Human-readable relative timestamp (e.g. "2h ago").',
    )
    visibility: Literal["private", "internal", "public"] = Field(
        description="Visibility level of the document."
    )
    folder: str = Field(description="Folder id this document belongs to.")
    # LightRAG-native shape consumed by the React port.
    doc_id: str | None = None
    track_id: str | None = Field(
        default=None, description="Ingestion tracking id from the upload."
    )
    file_path: str | None = None
    content_summary: str | None = None
    content_length: int | None = None
    chunks_count: int | None = None
    created_at: str | None = None
    updated_at: str | None = None
    error_msg: str | None = Field(
        default=None,
        description="Failure or rejection reason when `status` is `failed`.",
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Free-form metadata: review state, sensitivity classification, ..."
        ),
    )


# ---------------------------------------------------------------------------
# Topbar: folders + notifications
# ---------------------------------------------------------------------------


class Folder(_Base):
    id: str = Field(
        description="Folder id — the value to send in the `X-Twin-Folder` header.",
        examples=["general"],
    )
    kb: str = Field(description="Display label of the folder.")
    visibility: Literal["private", "internal", "public"] = Field(
        description="Visibility level of the folder."
    )
    sources: int = Field(description="Number of documents currently in the folder.")
    role: str = Field(description="The caller's role on this folder.")
    current: bool = Field(
        description="Whether this is the folder the request resolved to."
    )


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
    long_description: str = ""
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
    proposal_kind: Literal["edit"] | None = None
    target_tag: str | None = None
    proposed_fields: list[str] = Field(default_factory=list)


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
        "doc-folder-added",
        "doc-folder-removed",
        "classification-rejected",
        "source-uploaded",
        "source-ready",
        "source-failed",
        "pipeline-warning",
        "graph-entity-edited",
        "graph-relation-edited",
        "auth",
        "settings",
        "api-key-created",
        "api-key-revoked",
        "vision-settings-updated",
        "procedure-parked",
        "procedure-failed",
        "procedure-approved",
        "procedure-rejected",
        "procedure-retried",
        "procedure-rerouted",
        "procedure-store-recovered",
        "linked-source-declared",
        "linked-source-updated",
        "linked-source-disabled",
        "kb-exported",
        "kb-imported",
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
    source_docs: list[str] = Field(default_factory=list)
    summary: str
    tags: list[str] = Field(default_factory=list)
    properties: dict[str, str] = Field(default_factory=dict)


class GraphRelation(_Base):
    id: str
    source: str
    target: str
    label: str
    strength: float
    properties: dict[str, str] = Field(default_factory=dict)


class GraphEntityPatch(_Base):
    """Partial update payload for a graph entity. Every field optional;
    only the keys present in the request body are applied."""

    name: str | None = None
    type: (
        Literal["PRODUCT", "TECHNOLOGY", "CONCEPT", "ORG", "PERSON", "LOCATION"] | None
    ) = None
    summary: str | None = None
    tags: list[str] | None = None
    properties: dict[str, str] | None = None


class GraphRelationPatch(_Base):
    """Partial update payload for a graph relation."""

    label: str | None = None
    strength: float | None = None
    properties: dict[str, str] | None = None


class GraphEntityCreate(_Base):
    """Request body of ``POST /graph/entities``."""

    # ``name`` doubles as the entity's primary key: a 409 is returned when a
    # node with this id already exists in the workspace — manual creation
    # deliberately doesn't silently overwrite an LLM-extracted entity. An
    # empty/whitespace name is rejected at 422 by the validator below,
    # before the handler runs (TR-KG-01).

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
        json_schema_extra={
            "examples": [
                {
                    "name": "Change Advisory Board",
                    "type": "ORG",
                    "summary": "Committee approving production changes.",
                    "tags": ["governance"],
                }
            ]
        },
    )

    name: str = Field(
        ...,
        max_length=255,
        description="Entity name — must be unique in the knowledge graph.",
    )
    type: Literal["PRODUCT", "TECHNOLOGY", "CONCEPT", "ORG", "PERSON", "LOCATION"] = (
        Field(description="Entity type.")
    )
    summary: str | None = Field(
        default=None, description="Short description of the entity."
    )
    tags: list[str] | None = Field(
        default=None, description="Active catalog tags to attach."
    )
    properties: dict[str, str] | None = Field(
        default=None, description="Free-form key/value properties."
    )

    @field_validator("name", mode="before")
    @classmethod
    def _strip_and_require_name(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("name must be a string")
        stripped = value.strip()
        if not stripped:
            raise ValueError("name must not be empty or whitespace")
        return stripped


class GraphRelationCreate(_Base):
    """Request body of ``POST /graph/relations``."""

    source: str = Field(
        description="Id of the source entity (from `GET /graph/entities`)."
    )
    target: str = Field(
        description="Id of the target entity (from `GET /graph/entities`)."
    )
    label: str = Field(description="Relation label.", examples=["approves"])
    strength: float | None = Field(default=None, description="Relation strength (0-1).")
    properties: dict[str, str] | None = Field(
        default=None, description="Free-form key/value properties."
    )


class FolderCreate(_Base):
    """Request body of ``POST /folders``."""

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
        json_schema_extra={
            "examples": [
                {
                    "id": "hr_policies",
                    "label": "HR Policies",
                    "kind": "custom",
                    "description": "Human-resources policy documents.",
                }
            ]
        },
    )

    id: str = Field(
        description=(
            "Folder id. Letters, digits and underscores only — it becomes "
            "part of the storage namespace."
        ),
        examples=["hr_policies"],
    )
    label: str = Field(description="Human-facing name shown in the folder picker.")
    kind: str = Field(default="custom", description="Folder kind label.")
    description: str = Field(default="", description="Optional description.")


class FolderPatch(_Base):
    """Request body of ``PATCH /folders/{folder_id}``. Only provided
    fields change."""

    label: str | None = Field(default=None, description="New display label.")
    kind: str | None = Field(default=None, description="New folder kind.")
    description: str | None = Field(default=None, description="New description.")


# ---------------------------------------------------------------------------
# KB portability admin jobs (docs/adr/010-kb-portability-contract.md)
# ---------------------------------------------------------------------------


class PortabilityExportCreate(_Base):
    """Start a workspace-wide canonical KB export."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    workspace: str | None = Field(
        default=None,
        description=(
            "Runtime workspace assertion. Omit normally; a value different "
            "from the server WORKSPACE is refused."
        ),
        examples=["base"],
    )
    include_activity: bool = Field(
        default=False,
        description="Include the optional folder-scoped Activity ledger.",
    )
    include_procedures: bool = Field(
        default=False,
        description="Include optional procedure bundles and schematic files.",
    )
    force: bool = Field(
        default=False,
        description=(
            "Allow export while the ingestion pipeline is busy. The resulting "
            "bundle is explicitly marked unverified."
        ),
    )


class PortabilityApproval(_Base):
    """Bind an explicit admin approval to the displayed dry-run report."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    report_hash: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
        description="Exact SHA-256 report hash returned by the dry-run job.",
    )


class PortabilityJobResponse(_Base):
    """Public, path-free view of a persisted portability job."""

    id: str = Field(description="Opaque portability job identifier.")
    kind: Literal["export", "import"] = Field(description="Job operation kind.")
    workspace: str = Field(description="Target/source Memgraph workspace.")
    status: Literal[
        "queued",
        "uploading",
        "running",
        "dry-running",
        "awaiting-approval",
        "approved",
        "applying",
        "applied",
        "validating",
        "completed",
        "failed",
        "cancelled",
        "validated",
        "validation-failed",
    ] = Field(description="Current persisted state-machine status.")
    created_at: str = Field(description="UTC creation timestamp.")
    updated_at: str = Field(description="UTC last-transition timestamp.")
    actor: str = Field(description="Authenticated administrator identity.")
    approved_report_hash: str | None = Field(
        default=None, description="Dry-run report hash explicitly approved by an admin."
    )
    approved_by: str | None = Field(
        default=None, description="Administrator who approved the dry-run."
    )
    applied_by: str | None = Field(
        default=None, description="Administrator who started the apply transition."
    )
    validated_by: str | None = Field(
        default=None, description="Administrator who started validation."
    )
    cancelled_by: str | None = Field(
        default=None, description="Administrator who cancelled the job."
    )
    options: dict[str, Any] = Field(
        default_factory=dict,
        description="Non-secret options approved for the operation.",
    )
    result: dict[str, Any] | None = Field(
        default=None, description="Export/apply receipt when available."
    )
    report: dict[str, Any] | None = Field(
        default=None, description="Sealed import dry-run report when available."
    )
    validation: dict[str, Any] | None = Field(
        default=None, description="Post-import validation report when available."
    )
    error: str | None = Field(
        default=None, description="Operator-safe failure reason, if the job failed."
    )
    download_available: bool = Field(
        default=False,
        description="Whether GET with download=true can return the export archive.",
    )


# ---------------------------------------------------------------------------
# Simple ack envelopes for mutations
# ---------------------------------------------------------------------------


class AckResponse(_Base):
    ok: bool = True


# ---------------------------------------------------------------------------
# Tag mutation request bodies (S4c slice 2)
# ---------------------------------------------------------------------------


_ACTOR_FIELD = Field(
    default=None,
    description=(
        "Accepted for backward compatibility and ignored: the audit trail "
        "always records the authenticated identity, never this value."
    ),
)


def _reject_blank_definition(value: str | None) -> str | None:
    """QA TAG-V8-001: the required ``def`` invariant enforced at creation must
    hold on every write path — an edit could previously blank the definition
    of a validated tag. ``None`` still means "no change" on the edit bodies;
    an explicit empty/whitespace-only string is rejected (422).
    """
    if value is not None and not value.strip():
        raise ValueError("Tag definition cannot be empty or whitespace-only.")
    return value


class TagRequestBody(_Base):
    """Request body of ``POST /tags``."""

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
        json_schema_extra={
            "examples": [
                {
                    "tag": "data-retention",
                    "def": "Rules governing how long records are kept.",
                    "category": "compliance",
                    "aliases": ["retention-policy"],
                    "justification": "Recurring theme across audit documents.",
                }
            ]
        },
    )

    tag: str = Field(description="Tag name (the catalog key).")
    def_: str = Field(alias="def", description="Short definition of the tag.")
    long_description: str | None = Field(
        default=None, description="Longer explanation shown in the tag detail."
    )
    category: str = Field(
        description="Category id from `GET /tags/categories`.",
        examples=["compliance"],
    )
    aliases: list[str] = Field(
        default_factory=list, description="Synonyms that resolve to this tag."
    )
    justification: str | None = Field(
        default=None, description="Why this tag should exist (shown to reviewers)."
    )
    actor: str | None = _ACTOR_FIELD

    @field_validator("def_")
    @classmethod
    def _def_not_blank(cls, value: str | None) -> str | None:
        return _reject_blank_definition(value)


class TagEditBody(_Base):
    """Request body of ``PATCH /tags/{name}``. Only provided fields change."""

    tag: str | None = Field(default=None, description="New name for the tag (rename).")
    def_: str | None = Field(
        default=None, alias="def", description="New short definition."
    )
    long_description: str | None = None
    category: str | None = Field(default=None, description="New category id.")
    aliases: list[str] | None = Field(
        default=None, description="Replacement synonym list."
    )
    deprecates: list[str] | None = Field(
        default=None, description="Tags this one supersedes."
    )
    actor: str | None = _ACTOR_FIELD

    @field_validator("def_")
    @classmethod
    def _def_not_blank(cls, value: str | None) -> str | None:
        return _reject_blank_definition(value)


class TagSuggestEditBody(_Base):
    """Request body of ``POST /tags/{name}/suggest-edit``. Only provided
    fields become part of the proposal."""

    def_: str | None = Field(
        default=None, alias="def", description="Proposed short definition."
    )
    long_description: str | None = Field(
        default=None, description="Proposed longer explanation."
    )
    category: str | None = Field(default=None, description="Proposed category id.")
    aliases: list[str] | None = Field(
        default=None, description="Proposed synonym list."
    )
    justification: str | None = Field(
        default=None, description="Why the change is needed (shown to reviewers)."
    )
    actor: str | None = _ACTOR_FIELD

    @field_validator("def_")
    @classmethod
    def _def_not_blank(cls, value: str | None) -> str | None:
        return _reject_blank_definition(value)


class TagApproveBody(_Base):
    """Request body of ``POST /tags/{name}/approve``."""

    actor: str | None = _ACTOR_FIELD


class TagRejectBody(_Base):
    """Request body of ``POST /tags/{name}/reject``."""

    reason: str = Field(
        description="Why the request or proposal is rejected.",
        examples=["Overlaps with the existing 'records-management' tag."],
    )
    actor: str | None = _ACTOR_FIELD


class TagDeprecateBody(_Base):
    """Request body of ``POST /tags/{name}/deprecate``."""

    reason: str | None = Field(
        default=None, description="Why the tag is being retired."
    )
    actor: str | None = _ACTOR_FIELD


class TagReactivateBody(_Base):
    """Request body of ``POST /tags/{name}/reactivate``."""

    actor: str | None = _ACTOR_FIELD


class TagSynonymsBody(_Base):
    """Request body of ``POST /tags/{name}/synonyms``."""

    aliases: list[str] = Field(
        description="The full replacement synonym list.",
        examples=[["retention-policy", "records-retention"]],
    )
    actor: str | None = _ACTOR_FIELD


class TagDeleteBody(_Base):
    """Request body of ``DELETE /tags/{name}``."""

    strategy: Literal["migrate", "untag"] = Field(
        default="untag",
        description=(
            "`untag` removes the tag from its documents; `migrate` re-links "
            "them to the tag named in `to`."
        ),
    )
    to: str | None = Field(
        default=None,
        description="Migration target tag (required when `strategy` is `migrate`).",
    )
    actor: str | None = _ACTOR_FIELD
