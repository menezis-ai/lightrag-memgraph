"""``manifest.json`` of a ``twin-kb-bundle`` — ADR 010, decisions 1 and 2.

Design record: ``docs/adr/010-kb-portability-contract.md``.

Stdlib dataclasses only (no pydantic, no FastAPI: the CLI must run inside a
bank container with nothing but the storage package installed). Every field
the contract names is a field here; :func:`Manifest.from_json` is the reader
and it is strict — an unknown key anywhere, a path outside ``memgraph/``,
``overlay/``, ``files/``, a ``source`` key outside the whitelist, a probe with
other than three vectors: all refused, because a bundle that was accepted once
freezes what future readers must understand.

``state_hash`` identifies the *state*: ``sha256(JCS({path: sha256}))``
over the ``memgraph/`` and ``overlay/`` files, so it ignores ``bundle_id``,
``created_at``, ``created_by`` and ``consistency``. ``manifest_hash`` covers
the manifest itself (integrity) and is the only field excluded from its own
input. Reference copies: ``docs/templates/kb-bundle.manifest.schema.json``.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from .canonical import jcs_dumps, jcs_loads, jcs_sha256

FORMAT = "twin-kb-bundle"
FORMAT_VERSION = "1.0"
TOOL_NAME = "twindb_lightrag_memgraph.portability"

# Q2 (2026-08-25): three fixed probe sentences, FR + EN, one phrased as a
# question and one as a passage, so instruction-prefixed embedding models
# (query/passage) are exercised on both sides. Changing a sentence changes
# PROBE_TEXT_SET_ID — a bundle carries the id, never the texts.
PROBE_TEXT_SET_ID = "twin-kb-probe-v1"
EMBEDDING_PROBE_TEXTS: tuple[str, str, str] = (
    "Quelle est la procédure à suivre lorsqu'un incident de production affecte "
    "une application critique pendant la fenêtre de maintenance du week-end ?",
    "The knowledge base groups operational documents by folder; each document "
    "keeps its source identity, its classification label and the chunks that "
    "retrieval grounds an answer on.",
    "Passage : le schéma d'architecture technique décrit les flux entre les "
    "composants, leurs dépendances et les points de reprise après incident.",
)

STATE_HASH_PLANES = ("memgraph/", "overlay/")
_PATH_RE = re.compile(r"^(memgraph|overlay|files)/[a-z0-9_.\-/]+$")
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_KEYS = frozenset(
    {"workspace", "memgraph", "lightrag_version", "package_version"}
)
_SOURCE_MEMGRAPH_KEYS = frozenset({"database", "version", "mage"})
_MAGE_VALUES = frozenset({"unknown", "present", "absent"})
_CONSISTENCY_STATUS = frozenset({"verified", "unverified"})
_SCOPE_KINDS = frozenset({"workspace"})
_PLANES = frozenset({"memgraph", "overlay", "files"})
CLASSIFICATION_LADDER: tuple[str, ...] = ("C0", "C1", "C2", "C3", "C4")


class ManifestError(ValueError):
    """The manifest is not a valid ``twin-kb-bundle`` 1.x manifest."""


@dataclass(frozen=True)
class FileEntry:
    path: str
    store: str
    records: int
    sha256: str
    bytes: int


@dataclass(frozen=True)
class Probe:
    text_set_id: str
    vectors: list[list[float]]


@dataclass(frozen=True)
class Embedding:
    model: str
    dim: int
    metric: str
    probe: Probe


@dataclass(frozen=True)
class Scope:
    kind: str = "workspace"
    include_activity: bool = False
    include_procedures: bool = False


@dataclass(frozen=True)
class Classification:
    max_detected: str | None
    ladder: list[str] = field(default_factory=lambda: list(CLASSIFICATION_LADDER))
    unknown_present: bool = False


@dataclass(frozen=True)
class Consistency:
    status: str
    pipeline_idle: bool
    fingerprints_before: dict[str, Any] = field(default_factory=dict)
    fingerprints_after: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Manifest:
    bundle_id: str
    created_at: str
    created_by: dict[str, str]
    source: dict[str, Any]
    embedding: Embedding
    scope: Scope
    classification: Classification
    consistency: Consistency
    folders: list[dict[str, Any]]
    files: list[FileEntry]
    counts: dict[str, int]
    state_hash: str
    manifest_hash: str = ""
    format: str = FORMAT
    format_version: str = FORMAT_VERSION

    # -- serialisation -------------------------------------------------
    def to_dict(self, *, with_hash: bool = True) -> dict[str, Any]:
        data = asdict(self)
        if not with_hash:
            data.pop("manifest_hash")
        return data

    def compute_manifest_hash(self) -> str:
        return jcs_sha256(self.to_dict(with_hash=False))

    def sealed(self) -> Manifest:
        """Return a copy whose ``manifest_hash`` matches its content."""
        from dataclasses import replace

        return replace(self, manifest_hash=self.compute_manifest_hash())

    def to_json(self) -> str:
        """JCS text of the sealed manifest (always ends with a newline)."""
        sealed = self.sealed()
        return jcs_dumps(sealed.to_dict()) + "\n"

    @classmethod
    def from_json(cls, text: str) -> Manifest:
        try:
            data = jcs_loads(text)
        except ValueError as exc:
            raise ManifestError(f"manifest is not valid JSON ({exc})") from exc
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Any) -> Manifest:
        if not isinstance(data, dict):
            raise ManifestError("manifest must be a JSON object")
        expected = {
            "format",
            "format_version",
            "bundle_id",
            "created_at",
            "created_by",
            "source",
            "embedding",
            "scope",
            "classification",
            "consistency",
            "folders",
            "files",
            "counts",
            "state_hash",
            "manifest_hash",
        }
        _exact_keys(data, expected, "manifest")
        if data["format"] != FORMAT:
            raise ManifestError(f"format must be {FORMAT!r}, got {data['format']!r}")
        version = _str(data, "format_version", "manifest")
        major = version.split(".", 1)[0]
        if major != FORMAT_VERSION.split(".", 1)[0] or not re.fullmatch(
            r"\d+\.\d+", version
        ):
            raise ManifestError(f"unsupported format_version {version!r}")
        manifest = cls(
            format=FORMAT,
            format_version=version,
            bundle_id=_str(data, "bundle_id", "manifest"),
            created_at=_str(data, "created_at", "manifest"),
            created_by=_created_by(data["created_by"]),
            source=_source(data["source"]),
            embedding=_embedding(data["embedding"]),
            scope=_scope(data["scope"]),
            classification=_classification(data["classification"]),
            consistency=_consistency(data["consistency"]),
            folders=_folders(data["folders"]),
            files=_files(data["files"]),
            counts=_counts(data["counts"]),
            state_hash=_hex(data, "state_hash", "manifest"),
            manifest_hash=_hex(data, "manifest_hash", "manifest"),
        )
        if manifest.compute_manifest_hash() != manifest.manifest_hash:
            raise ManifestError("manifest_hash does not match the manifest content")
        _validate_store_roster(manifest.files, manifest.scope)
        if state_hash_of_entries(manifest.files) != manifest.state_hash:
            raise ManifestError("state_hash does not match files[]")
        return manifest


# ---------------------------------------------------------------- hashes


def state_hash_of(hashes: dict[str, str]) -> str:
    """§3.4 — ``sha256(JCS({path: sha256}))`` over the data-plane files only."""
    kept = {
        path: digest
        for path, digest in hashes.items()
        if path.startswith(STATE_HASH_PLANES)
    }
    return jcs_sha256(dict(sorted(kept.items())))


def state_hash_of_entries(files: list[FileEntry]) -> str:
    return state_hash_of({f.path: f.sha256 for f in files})


# ------------------------------------------------------------ validators


def _exact_keys(data: Any, expected: set[str], where: str) -> None:
    if not isinstance(data, dict):
        raise ManifestError(f"{where} must be an object")
    keys = set(data)
    if keys != expected:
        missing = sorted(expected - keys)
        extra = sorted(keys - expected)
        raise ManifestError(
            f"{where}: keys must be exactly {sorted(expected)} (missing {missing}, extra {extra})"
        )


def _str(
    data: dict[str, Any], key: str, where: str, *, allow_empty: bool = False
) -> str:
    value = data.get(key)
    if not isinstance(value, str) or (not allow_empty and not value):
        raise ManifestError(f"{where}.{key} must be a non-empty string")
    return value


def _hex(data: dict[str, Any], key: str, where: str) -> str:
    value = _str(data, key, where)
    if not _HEX64_RE.match(value):
        raise ManifestError(f"{where}.{key} must be a lower-case sha256 hex digest")
    return value


def _int(value: Any, where: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ManifestError(f"{where} must be an integer >= {minimum}")
    return value


def _bool(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        raise ManifestError(f"{where} must be a boolean")
    return value


def _created_by(data: Any) -> dict[str, str]:
    _exact_keys(data, {"tool", "version", "actor"}, "created_by")
    return {
        k: _str(data, k, "created_by", allow_empty=(k == "actor"))
        for k in ("tool", "version", "actor")
    }


def _source(data: Any) -> dict[str, Any]:
    _exact_keys(data, set(_SOURCE_KEYS), "source")
    memgraph = data["memgraph"]
    _exact_keys(memgraph, set(_SOURCE_MEMGRAPH_KEYS), "source.memgraph")
    mage = _str(memgraph, "mage", "source.memgraph")
    if mage not in _MAGE_VALUES:
        raise ManifestError(
            f"source.memgraph.mage must be one of {sorted(_MAGE_VALUES)}"
        )
    return {
        "workspace": _str(data, "workspace", "source"),
        "memgraph": {
            "database": _str(memgraph, "database", "source.memgraph"),
            "version": _str(memgraph, "version", "source.memgraph", allow_empty=True),
            "mage": mage,
        },
        "lightrag_version": _str(data, "lightrag_version", "source", allow_empty=True),
        "package_version": _str(data, "package_version", "source"),
    }


def _vector(value: Any, dim: int, where: str) -> list[float]:
    if not isinstance(value, list) or len(value) != dim:
        raise ManifestError(f"{where} must be a list of {dim} floats")
    out: list[float] = []
    for i, x in enumerate(value):
        if (
            isinstance(x, bool)
            or not isinstance(x, int | float)
            or not math.isfinite(x)
        ):
            raise ManifestError(f"{where}[{i}] must be a finite number")
        out.append(float(x))
    return out


def _embedding(data: Any) -> Embedding:
    _exact_keys(data, {"model", "dim", "metric", "probe"}, "embedding")
    dim = _int(data["dim"], "embedding.dim", minimum=1)
    probe = data["probe"]
    _exact_keys(probe, {"text_set_id", "vectors"}, "embedding.probe")
    vectors = probe["vectors"]
    if not isinstance(vectors, list) or len(vectors) != len(EMBEDDING_PROBE_TEXTS):
        raise ManifestError(
            f"embedding.probe.vectors must hold exactly {len(EMBEDDING_PROBE_TEXTS)} vectors"
        )
    return Embedding(
        model=_str(data, "model", "embedding", allow_empty=True),
        dim=dim,
        metric=_str(data, "metric", "embedding"),
        probe=Probe(
            text_set_id=_str(probe, "text_set_id", "embedding.probe"),
            vectors=[
                _vector(v, dim, f"embedding.probe.vectors[{i}]")
                for i, v in enumerate(vectors)
            ],
        ),
    )


def _scope(data: Any) -> Scope:
    _exact_keys(data, {"kind", "include_activity", "include_procedures"}, "scope")
    kind = _str(data, "kind", "scope")
    if kind not in _SCOPE_KINDS:
        raise ManifestError(f"scope.kind must be one of {sorted(_SCOPE_KINDS)}")
    return Scope(
        kind=kind,
        include_activity=_bool(data["include_activity"], "scope.include_activity"),
        include_procedures=_bool(
            data["include_procedures"], "scope.include_procedures"
        ),
    )


def _classification(data: Any) -> Classification:
    _exact_keys(data, {"max_detected", "ladder", "unknown_present"}, "classification")
    ladder = data["ladder"]
    if (
        not isinstance(ladder, list)
        or not ladder
        or any(not isinstance(x, str) or not x for x in ladder)
        or len(set(ladder)) != len(ladder)
    ):
        raise ManifestError(
            "classification.ladder must be a non-empty list of distinct strings"
        )
    max_detected = data["max_detected"]
    if max_detected is not None and max_detected not in ladder:
        raise ManifestError(
            "classification.max_detected must be null or a ladder value"
        )
    return Classification(
        max_detected=max_detected,
        ladder=list(ladder),
        unknown_present=_bool(
            data["unknown_present"], "classification.unknown_present"
        ),
    )


def _consistency(data: Any) -> Consistency:
    _exact_keys(
        data,
        {"status", "pipeline_idle", "fingerprints_before", "fingerprints_after"},
        "consistency",
    )
    status = _str(data, "status", "consistency")
    if status not in _CONSISTENCY_STATUS:
        raise ManifestError(
            f"consistency.status must be one of {sorted(_CONSISTENCY_STATUS)}"
        )
    for key in ("fingerprints_before", "fingerprints_after"):
        if not isinstance(data[key], dict):
            raise ManifestError(f"consistency.{key} must be an object")
    return Consistency(
        status=status,
        pipeline_idle=_bool(data["pipeline_idle"], "consistency.pipeline_idle"),
        fingerprints_before=dict(data["fingerprints_before"]),
        fingerprints_after=dict(data["fingerprints_after"]),
    )


def _folders(data: Any) -> list[dict[str, Any]]:
    if not isinstance(data, list):
        raise ManifestError("folders must be a list")
    out = []
    seen: set[str] = set()
    for i, item in enumerate(data):
        _exact_keys(item, {"id", "label", "kind"}, f"folders[{i}]")
        fid = _str(item, "id", f"folders[{i}]")
        if fid in seen:
            raise ManifestError(f"folders[{i}]: duplicate folder id {fid!r}")
        seen.add(fid)
        out.append(
            {
                "id": fid,
                "label": _str(item, "label", f"folders[{i}]", allow_empty=True),
                "kind": _str(item, "kind", f"folders[{i}]", allow_empty=True),
            }
        )
    return out


def validate_bundle_path(path: str) -> str:
    """A bundle member path: plane prefix, safe charset, no ``..`` segment."""
    if not isinstance(path, str) or not _PATH_RE.match(path):
        raise ManifestError(f"invalid bundle path {path!r}")
    segments = path.split("/")
    if any(seg in ("", ".", "..") for seg in segments):
        raise ManifestError(f"invalid bundle path {path!r}")
    return path


def _files(data: Any) -> list[FileEntry]:
    if not isinstance(data, list):
        raise ManifestError("files must be a list")
    out: list[FileEntry] = []
    seen: set[str] = set()
    for i, item in enumerate(data):
        where = f"files[{i}]"
        _exact_keys(item, {"path", "store", "records", "sha256", "bytes"}, where)
        path = validate_bundle_path(item["path"])
        if path in seen:
            raise ManifestError(f"{where}: duplicate path {path!r}")
        seen.add(path)
        out.append(
            FileEntry(
                path=path,
                store=_str(item, "store", where),
                records=_int(item["records"], f"{where}.records"),
                sha256=_hex(item, "sha256", where),
                bytes=_int(item["bytes"], f"{where}.bytes"),
            )
        )
    return out


_PROCEDURE_FILE_RE = re.compile(
    r"^files/procedures/[a-z0-9][a-z0-9_.-]*/[1-9][0-9]*\.png$"
)


def _validate_store_roster(files: list[FileEntry], scope: Scope) -> None:
    """Require the closed canonical store roster for the declared scope.

    Empty stores still have an empty JSONL member.  Therefore absence can
    never mean "empty": it is a truncated/non-canonical bundle and must fail
    before an importer can mistake it for a partial workspace.
    """
    from .stores import exportable_stores

    expected = {
        spec.file: spec.name
        for spec in exportable_stores(
            include_activity=scope.include_activity,
            include_procedures=scope.include_procedures,
        )
        if spec.file is not None
    }
    canonical = {
        entry.path: entry
        for entry in files
        if entry.path.startswith(("memgraph/", "overlay/"))
    }
    missing = sorted(set(expected) - set(canonical))
    unexpected = sorted(set(canonical) - set(expected))
    if missing:
        raise ManifestError(f"files[]: missing canonical store member(s): {missing}")
    if unexpected:
        raise ManifestError(
            f"files[]: unexpected canonical store member(s): {unexpected}"
        )
    for path, store in expected.items():
        declared = canonical[path].store
        if declared != store:
            raise ManifestError(
                f"files[]: canonical path {path!r} must declare store {store!r}, "
                f"got {declared!r}"
            )

    procedure_files = [entry for entry in files if entry.path.startswith("files/")]
    if procedure_files and not scope.include_procedures:
        raise ManifestError(
            "files[]: procedure members require scope.include_procedures=true"
        )
    for entry in procedure_files:
        if (
            entry.store != "procedures"
            or entry.records != 0
            or not _PROCEDURE_FILE_RE.fullmatch(entry.path)
        ):
            raise ManifestError(
                f"files[]: invalid procedure file member {entry.path!r}"
            )


def _counts(data: Any) -> dict[str, int]:
    if not isinstance(data, dict):
        raise ManifestError("counts must be an object")
    return {str(k): _int(v, f"counts.{k}") for k, v in data.items()}
