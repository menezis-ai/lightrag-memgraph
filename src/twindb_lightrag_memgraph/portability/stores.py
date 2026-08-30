"""Store registry and the ``PortableStore`` contract — KB-PORTABILITY-PLAN §3.5/§3.6.

Every Memgraph label prefix the runtime writes is declared here with its
*scoping* (``workspace`` / ``folder`` / ``global``) and its *portability*
class (``always`` / ``optional`` / ``never``). The registry is the
single answer to "what is in a bundle": ``exportable_stores()`` never yields
a ``never`` store, and the parity test (``tests/test_portability/
test_store_registry.py``) fails as soon as the code grows a label prefix that
is not declared here — an undeclared store is a silent leak waiting to happen.

Each store declares a :class:`StoreSchema`: the record key, the **allow-list**
of *raw* node/edge properties that may travel (the record a store emits may
reshape them — ``data`` becomes a parsed ``value``, endpoints become
``doc_id``/``folder_id`` — but a property name outside the list never
does), and the transient properties dropped on export. :func:`project_record` is the choke point — a node property outside
both sets aborts the export with the property name (:class:`SchemaViolation`),
never an omission. That is the secret detector the plan asks for: not a regex
over values, a closed schema over names.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from .._constants import validate_identifier

Plane = Literal["memgraph", "overlay"]
Scoping = Literal["workspace", "folder", "global"]
Portability = Literal["always", "optional", "never"]

# Field names that can never be part of an allow-list, whatever the store.
# Matched per ``_``-separated segment (``tokens`` — a chunk's token count — is
# not ``token``) plus the two-word forms. ``hash`` is deliberately absent:
# ``content_hash`` (DocStatus) is a public dedup key; the API-key ``hash`` is
# protected by its store being ``never``.
_FORBIDDEN_FIELD_SEGMENTS = frozenset(
    {"password", "passwd", "secret", "token", "credential", "credentials", "bearer"}
)
_FORBIDDEN_FIELD_SUBSTRINGS = ("api_key", "apikey", "private_key")


def _is_forbidden_field(name: str) -> bool:
    lowered = name.lower()
    if any(sub in lowered for sub in _FORBIDDEN_FIELD_SUBSTRINGS):
        return True
    return any(
        seg in _FORBIDDEN_FIELD_SEGMENTS for seg in lowered.strip("_").split("_")
    )


class PortabilityError(RuntimeError):
    """Base class of every portability failure."""


class SchemaViolation(PortabilityError):
    """A stored property is outside the store's allow-list — export refused."""


@dataclass(frozen=True)
class StoreSchema:
    key: tuple[str, ...]
    fields: frozenset[str]
    transient: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        for name in self.fields | self.transient:
            if _is_forbidden_field(name):
                raise ValueError(
                    f"field {name!r} can never be part of a portable schema"
                )
        missing = set(self.key) - self.fields
        if missing:
            raise ValueError(f"schema key {sorted(missing)} must be listed in fields")
        if self.fields & self.transient:
            raise ValueError("a field cannot be both portable and transient")


@dataclass(frozen=True)
class Scope:
    """What an export/import covers: one workspace, its folders, a folder map."""

    workspace: str
    folder_ids: tuple[str, ...] = ()
    folder_map: dict[str, str] = field(default_factory=dict)
    batch_size: int = 1000
    bundle_id: str | None = None
    embedding_dim: int | None = None

    def __post_init__(self) -> None:
        validate_identifier(self.workspace, "workspace")
        for fid in self.folder_ids:
            validate_identifier(fid, "folder")
        for src, dst in self.folder_map.items():
            validate_identifier(src, "folder")
            validate_identifier(dst, "folder")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.bundle_id is not None and not self.bundle_id.strip():
            raise ValueError("bundle_id must be non-empty when provided")
        if self.embedding_dim is not None and self.embedding_dim < 1:
            raise ValueError("embedding_dim must be positive when provided")

    def mapped_folder(self, folder_id: str) -> str:
        return self.folder_map.get(folder_id, folder_id)


@dataclass(frozen=True)
class StoreSpec:
    name: str
    plane: Plane
    scoping: Scoping
    portability: Portability
    schema: StoreSchema
    label_prefixes: tuple[str, ...]
    file: str | None = None  # bundle path; None for never-exported stores
    namespace: str | None = None  # KV/Vec namespace

    def __post_init__(self) -> None:
        if self.portability == "never" and self.file is not None:
            raise ValueError(f"{self.name}: a never store has no bundle file")
        if self.portability != "never" and self.file is None:
            raise ValueError(f"{self.name}: an exportable store needs a bundle file")

    def label(self, scope_id: str) -> str:
        """The Memgraph label for a workspace or folder id (validated)."""
        validate_identifier(scope_id, "identifier")
        prefix = self.label_prefixes[0]
        if self.namespace is not None:
            return f"{prefix}_{scope_id}_{self.namespace}"
        return f"{prefix}_{scope_id}"


class PortableStore(Protocol):
    """§3.6 — implemented per store in ``stores_memgraph`` / ``stores_overlay``."""

    spec: StoreSpec

    def export_records(self, scope: Scope) -> AsyncIterator[dict[str, Any]]: ...
    async def import_records(
        self, records: AsyncIterator[dict[str, Any]], scope: Scope
    ) -> int: ...
    async def fingerprint(self, scope: Scope) -> dict[str, Any]: ...
    async def count(self, scope: Scope) -> int: ...


def project_record(spec: StoreSpec, props: dict[str, Any]) -> dict[str, Any]:
    """Keep the allow-listed properties, drop the transient ones, refuse the rest."""
    unknown = sorted(
        k
        for k in props
        if k not in spec.schema.fields and k not in spec.schema.transient
    )
    if unknown:
        raise SchemaViolation(
            f"{spec.name}: property {unknown} is not in the portable schema"
        )
    return {k: v for k, v in props.items() if k in spec.schema.fields}


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------

_KV_SCHEMA = StoreSchema(
    key=("id",),
    fields=frozenset({"id", "data", "__created_at", "__updated_at"}),
)
# Vector nodes are a pass-through of the LightRAG payload (vector_impl
# ``_build_entry``): the allow-list is the union of what LightRAG 1.5.6 writes
# per namespace plus the properties Twin adds. Anything new upstream trips
# SchemaViolation on the next export — extend deliberately, after reading the
# upstream diff.
_VEC_COMMON = {
    "id",
    "embedding",
    "content",
    "file_path",
    "source_id",
    "created_at",
    "__created_at",
    "timestamp",
}
_VEC_SCHEMAS = {
    "chunks": StoreSchema(
        key=("id",),
        fields=frozenset(
            _VEC_COMMON
            | {"full_doc_id", "chunk_order_index", "tokens", "llm_cache_list"}
        ),
    ),
    "entities": StoreSchema(
        key=("id",),
        fields=frozenset(
            _VEC_COMMON | {"entity_name", "entity_type", "description", "chunk_ids"}
        ),
    ),
    "relationships": StoreSchema(
        key=("id",),
        fields=frozenset(
            _VEC_COMMON
            | {"src_id", "tgt_id", "keywords", "description", "weight", "chunk_ids"}
        ),
    ),
}
_DOCSTATUS_SCHEMA = StoreSchema(
    key=("id",),
    fields=frozenset(
        {
            "id",
            "status",
            "file_path",
            "content_hash",
            "content_summary",
            "content_length",
            "chunks_count",
            "chunks_list",
            "metadata",
            "track_id",
            "error_msg",
            "created_at",
            "updated_at",
            "folder",
            "multimodal_processed",
        }
    ),
    transient=frozenset({"__membership_epoch", "__delete_claim"}),
)
_FOLDER_SCHEMA = StoreSchema(key=("id",), fields=frozenset({"id"}))
_MEMBER_OF_SCHEMA = StoreSchema(
    key=("doc_id", "folder_id"), fields=frozenset({"doc_id", "folder_id", "updated_at"})
)  # edge props allow-list = fields minus the endpoint keys
_TAGGED_WITH_SCHEMA = StoreSchema(
    key=("doc_id", "folder_id", "tag_id"),
    fields=frozenset({"doc_id", "folder_id", "tag_id", "at", "actor", "migrated_from"}),
)
_GRAPH_NODE_SCHEMA = StoreSchema(
    key=("entity_id",),
    fields=frozenset(
        {
            "entity_id",
            "entity_type",
            "description",
            "source_id",
            "file_path",
            "created_at",
            "timestamp",
            "truncate",
            "display_name",
            "twin_tags_json",
            "twin_props_json",
        }
    ),
    transient=frozenset({"__twin_create_marker"}),
)
_GRAPH_EDGE_SCHEMA = StoreSchema(
    key=("src", "tgt"),
    fields=frozenset(
        {
            "src",
            "tgt",
            "weight",
            "description",
            "keywords",
            "source_id",
            "file_path",
            "created_at",
            "timestamp",
            "truncate",
            "twin_props_json",
            "twin_folder_json",
            "twin_relation_id",
        }
    ),
)
_GRAPH_MEMBER_OF_SCHEMA = StoreSchema(
    key=("entity_id", "folder_id"), fields=frozenset({"entity_id", "folder_id"})
)
_GRAPH_OVERRIDE_SCHEMA = StoreSchema(
    key=("kind", "entity_id", "src", "tgt", "folder"),
    fields=frozenset(
        {
            "kind",
            "entity_id",
            "src",
            "tgt",
            "folder",
            "deleted",
            "description",
            "entity_type",
            "display_name",
            "twin_tags_json",
            "twin_props_json",
            "keywords",
            "weight",
        }
    ),
)
# Overlay stores keep their whole record in one ``data`` JSON string and stamp
# server-side ``__created_at``/``__updated_at`` (Memgraph ``timestamp()``) that
# no public write API can restore — they are transient here, so the same
# state exported from two instances hashes identically.
_OVERLAY_DATA_TRANSIENT = frozenset({"__created_at", "__updated_at"})
_TAG_SCHEMA = StoreSchema(
    key=("folder_id", "id"),
    fields=frozenset({"folder_id", "id", "data"}),
    transient=_OVERLAY_DATA_TRANSIENT | {"__bulk_retag_lock"},
)
_TAG_CATEGORY_SCHEMA = StoreSchema(
    key=("folder_id", "id"),
    fields=frozenset({"folder_id", "id", "data"}),
    transient=_OVERLAY_DATA_TRANSIENT,
)
_ACTIVITY_SCHEMA = StoreSchema(
    key=("folder_id", "id"),
    fields=frozenset({"folder_id", "id", "data", "origin"}),
    # every scalar is re-derived from ``data`` by MemgraphActivityStore.append()
    transient=_OVERLAY_DATA_TRANSIENT
    | {
        "kind",
        "sev",
        "actor_user",
        "target_id",
        "target_label",
        "meta_doc_id",
        "meta_doc_ids",
        "summary",
        "ts_ms",
        "__scalars_version",
    },
)
_SETTINGS_SCHEMA = StoreSchema(key=("id",), fields=frozenset({"id", "data"}))
_SOURCE_LINK_SCHEMA = StoreSchema(
    key=("doc_id", "id"),
    fields=frozenset(
        {
            "id",
            "doc_id",
            "url",
            "label",
            "created_by",
            "created_at",
            "updated_by",
            "updated_at",
            "version",
            "deleted",
            "deleted_by",
            "deleted_at",
        }
    ),
)
_RUNTIME_FOLDER_SCHEMA = StoreSchema(
    key=("id",),
    fields=frozenset({"id", "label", "kind", "description", "sources"}),
)
_PROCEDURE_SCHEMA = StoreSchema(
    key=("id",),
    fields=frozenset(
        {
            "id",
            "file_name",
            "original_path",
            "track_id",
            "state",
            "reason",
            "source",
            "folder",
            "content_hash",
            "full_text",
            "schematics",
            "schematics_total",
            "classification",
            "operator_classification",
            "created_at",
            "updated_at",
            "duplicate_requests",
        }
    ),
)
_NEVER_SCHEMA = StoreSchema(key=("id",), fields=frozenset({"id"}))

# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

KV_NAMESPACES: tuple[str, ...] = (
    "full_docs",
    "text_chunks",
    "full_entities",
    "full_relations",
    "entity_chunks",
    "relation_chunks",
)
KV_NEVER_NAMESPACES: tuple[str, ...] = ("llm_response_cache",)  # Q1
VEC_NAMESPACES: tuple[str, ...] = ("chunks", "entities", "relationships")


def _kv(ns: str, portability: Portability) -> StoreSpec:
    return StoreSpec(
        name=f"kv.{ns}",
        plane="memgraph",
        scoping="workspace",
        portability=portability,
        schema=_KV_SCHEMA if portability != "never" else _NEVER_SCHEMA,
        label_prefixes=("KV",),
        file=f"memgraph/kv.{ns}.jsonl" if portability != "never" else None,
        namespace=ns,
    )


def _vec(ns: str) -> StoreSpec:
    return StoreSpec(
        name=f"vec.{ns}",
        plane="memgraph",
        scoping="workspace",
        portability="always",
        schema=_VEC_SCHEMAS[ns],
        label_prefixes=("Vec",),
        file=f"memgraph/vec.{ns}.jsonl",
        namespace=ns,
    )


STORES: tuple[StoreSpec, ...] = (
    *(_kv(ns, "always") for ns in KV_NAMESPACES),
    *(_kv(ns, "never") for ns in KV_NEVER_NAMESPACES),
    *(_vec(ns) for ns in VEC_NAMESPACES),
    StoreSpec(
        "docstatus",
        "memgraph",
        "workspace",
        "always",
        _DOCSTATUS_SCHEMA,
        ("DocStatus",),
        "memgraph/docstatus.jsonl",
    ),
    StoreSpec(
        "folders",
        "memgraph",
        "workspace",
        "always",
        _FOLDER_SCHEMA,
        ("Folder",),
        "memgraph/folders.jsonl",
    ),
    StoreSpec(
        "member_of",
        "memgraph",
        "workspace",
        "always",
        _MEMBER_OF_SCHEMA,
        ("DocStatus", "Folder"),
        "memgraph/member_of.jsonl",
    ),
    StoreSpec(
        "tagged_with",
        "memgraph",
        "folder",
        "always",
        _TAGGED_WITH_SCHEMA,
        ("DocStatus", "WebuiTag"),
        "memgraph/tagged_with.jsonl",
    ),
    StoreSpec(
        "graph.nodes",
        "memgraph",
        "workspace",
        "always",
        _GRAPH_NODE_SCHEMA,
        (),
        "memgraph/graph.nodes.jsonl",
    ),
    StoreSpec(
        "graph.edges",
        "memgraph",
        "workspace",
        "always",
        _GRAPH_EDGE_SCHEMA,
        (),
        "memgraph/graph.edges.jsonl",
    ),
    StoreSpec(
        "graph.member_of",
        "memgraph",
        "workspace",
        "always",
        _GRAPH_MEMBER_OF_SCHEMA,
        ("Folder",),
        "memgraph/graph.member_of.jsonl",
    ),
    StoreSpec(
        "graph.overrides",
        "memgraph",
        "workspace",
        "always",
        _GRAPH_OVERRIDE_SCHEMA,
        ("GraphOverride", "GraphRelOverride"),
        "memgraph/graph.overrides.jsonl",
    ),
    StoreSpec(
        "runtime_folders",
        "overlay",
        "global",
        "always",
        _RUNTIME_FOLDER_SCHEMA,
        (),
        "overlay/folders.jsonl",
    ),
    StoreSpec(
        "tags",
        "overlay",
        "folder",
        "always",
        _TAG_SCHEMA,
        ("WebuiTag",),
        "overlay/tags.jsonl",
    ),
    StoreSpec(
        "tag_categories",
        "overlay",
        "folder",
        "always",
        _TAG_CATEGORY_SCHEMA,
        ("WebuiTagCategory",),
        "overlay/tag_categories.jsonl",
    ),
    StoreSpec(
        "settings",
        "overlay",
        "workspace",
        "always",
        _SETTINGS_SCHEMA,
        ("WebuiSettings",),
        "overlay/settings.jsonl",
    ),
    StoreSpec(
        "source_links",
        "overlay",
        "workspace",
        "always",
        _SOURCE_LINK_SCHEMA,
        ("TwinSourceLink",),
        "overlay/source_links.jsonl",
    ),
    StoreSpec(
        "activity",
        "overlay",
        "folder",
        "optional",
        _ACTIVITY_SCHEMA,
        ("WebuiActivity",),
        "overlay/activity.jsonl",
    ),
    StoreSpec(
        "procedures",
        "overlay",
        "global",
        "optional",
        _PROCEDURE_SCHEMA,
        (),
        "overlay/procedures.jsonl",
    ),
    StoreSpec(
        "api_keys", "overlay", "workspace", "never", _NEVER_SCHEMA, ("WebuiApiKey",)
    ),
    StoreSpec(
        "notifications",
        "overlay",
        "folder",
        "never",
        _NEVER_SCHEMA,
        ("WebuiNotification",),
    ),
)

_BY_NAME = {spec.name: spec for spec in STORES}
if len(_BY_NAME) != len(STORES):
    raise RuntimeError("duplicate store names in the portability registry")
_FILES = [spec.file for spec in STORES if spec.file]
if len(set(_FILES)) != len(_FILES):
    raise RuntimeError("duplicate bundle files in the portability registry")


def store_by_name(name: str) -> StoreSpec:
    return _BY_NAME[name]


def store_by_file(path: str) -> StoreSpec | None:
    for spec in STORES:
        if spec.file == path:
            return spec
    return None


def exportable_stores(
    *, include_activity: bool = False, include_procedures: bool = False
) -> tuple[StoreSpec, ...]:
    """Stores that travel: every ``always`` plus the enabled ``optional`` ones."""
    enabled = {"activity": include_activity, "procedures": include_procedures}
    return tuple(
        spec
        for spec in STORES
        if spec.portability == "always"
        or (spec.portability == "optional" and enabled.get(spec.name, False))
    )


def declared_label_prefixes() -> frozenset[str]:
    return frozenset(prefix for spec in STORES for prefix in spec.label_prefixes)


def never_label_prefixes() -> frozenset[str]:
    """Prefixes whose nodes must never appear in a bundle (test §9.4)."""
    return frozenset(
        prefix
        for spec in STORES
        if spec.portability == "never"
        for prefix in spec.label_prefixes
        if prefix not in {"KV"}  # KV is shared with always namespaces
    )


def folder_scoped_labels(spec: StoreSpec, scope: Scope) -> list[tuple[str, str]]:
    """``(folder_id, label)`` for a folder-scoped store, one per folder in scope."""
    if spec.scoping != "folder":
        raise ValueError(f"{spec.name} is not folder-scoped")
    return [(fid, spec.label(fid)) for fid in scope.folder_ids]


def portable_store(
    spec: StoreSpec,
    *,
    bundle_writer: Any | None = None,
    bundle_root: Any | None = None,
) -> PortableStore:
    """Instantiate the implementation for one registry entry.

    Imports stay local so merely importing the format/manifest layer never
    pulls server extras.  ``never`` stores intentionally have no implementation
    reachable through this factory: construction is another hard boundary in
    addition to :func:`exportable_stores`.
    """
    if spec.portability == "never":
        raise PortabilityError(f"{spec.name} is never portable")

    from .stores_graph import (
        GraphEdgeStore,
        GraphMemberOfStore,
        GraphNodeStore,
        GraphOverrideStore,
    )
    from .stores_memgraph import (
        DocStatusStore,
        FolderStore,
        KvStore,
        MemberOfStore,
        TaggedWithStore,
        VecStore,
    )
    from .stores_overlay import (
        ActivityStore,
        ProcedureStore,
        RuntimeFolderStore,
        SettingsStore,
        SourceLinkStore,
        TagCategoryStore,
        TagStore,
    )

    if spec.name.startswith("kv."):
        return KvStore(spec.namespace or spec.name.removeprefix("kv."))
    if spec.name.startswith("vec."):
        return VecStore(spec.namespace or spec.name.removeprefix("vec."))
    implementations: dict[str, Any] = {
        "docstatus": DocStatusStore,
        "folders": FolderStore,
        "member_of": MemberOfStore,
        "tagged_with": TaggedWithStore,
        "graph.nodes": GraphNodeStore,
        "graph.edges": GraphEdgeStore,
        "graph.member_of": GraphMemberOfStore,
        "graph.overrides": GraphOverrideStore,
        "runtime_folders": RuntimeFolderStore,
        "tags": TagStore,
        "tag_categories": TagCategoryStore,
        "settings": SettingsStore,
        "source_links": SourceLinkStore,
        "activity": ActivityStore,
    }
    if spec.name == "procedures":
        return ProcedureStore(bundle_writer=bundle_writer, bundle_root=bundle_root)
    try:
        return implementations[spec.name]()
    except KeyError as exc:  # registry growth must add a deliberate implementation
        raise PortabilityError(f"no portable implementation for {spec.name}") from exc
