"""Target facts and compatibility verdicts for KB import (plan T2.1).

The source manifest is descriptive; the target is probed at dry-run time.
Model names are deliberately not trusted as an embedding contract: the three
versioned probe vectors are recomputed with the target's real LightRAG binding
and all three cosine similarities must reach ``0.999``.  Facts contain no
credentials and are stable enough to participate in the dry-run report hash.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import asdict, dataclass
from typing import Any

from .._constants import (
    TWIN_PORTABILITY_ALLOW_UNVERIFIED_ENV,
    portability_flag_enabled,
    resolve_vector_index_capacity,
    validate_identifier,
)
from .._folders import load_folder_catalog
from ..server.folder_store import list_runtime_folders
from ._io import read_rows, read_scalar
from .exporter import _embedding_probe, _memgraph_source, _pipeline_is_idle
from .manifest import Manifest, PROBE_TEXT_SET_ID
from .stores import STORES, Scope, portable_store

MIN_PROBE_COSINE = 0.999
MIN_MEMGRAPH_VERSION = (3, 12, 0)


@dataclass(frozen=True)
class TargetFacts:
    workspace: str
    database: str
    memgraph_version: str
    lightrag_version: str
    embedding_model: str
    embedding_dim: int
    embedding_probe: list[list[float]]
    vector_capacity: int
    vector_indexes: list[dict[str, Any]]
    classification_ceiling: str
    allow_unverified: bool
    pipeline_idle: bool
    store_counts: dict[str, int]
    env_folders: list[dict[str, str]]
    runtime_folders: list[dict[str, Any]]
    max_folders: int

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        # Raw endpoint floats are diagnostic, not an approval fact. Their
        # semantic result lives in compat.embedding (probe id, dim, verdict).
        result.pop("embedding_probe")
        return result


def _index_label(value: Any) -> str:
    label = str(value or "").strip()
    if label.startswith(":"):
        label = label[1:]
    if len(label) >= 2 and label[0] == label[-1] == "`":
        label = label[1:-1]
    return label


def _index_property(value: Any) -> str:
    prop = str(value or "").strip()
    if len(prop) >= 2 and prop[0] == prop[-1] == "`":
        prop = prop[1:-1]
    return prop


def _index_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def check_vector_index_contract(
    row: dict[str, Any] | None,
    *,
    workspace: str,
    namespace: str,
    embedding_dim: int,
    minimum_capacity: int,
    expected_size: int,
) -> dict[str, Any]:
    """Check every invariant of the index DDL emitted by ``vector_impl``."""
    expected = {
        "label": f"Vec_{workspace}_{namespace}",
        "property": "embedding",
        "metric": "cos",
        "dimension": embedding_dim,
        "minimum_capacity": minimum_capacity,
        "size": expected_size,
    }
    actual = {
        "label": _index_label(row.get("label")) if row else "",
        "property": _index_property(row.get("property")) if row else "",
        "metric": str(row.get("metric") or "") if row else "",
        "dimension": _index_int(row.get("dimension")) if row else None,
        "capacity": _index_int(row.get("capacity")) if row else None,
        "size": _index_int(row.get("size")) if row else None,
    }
    problems: list[str] = []
    for field in ("label", "property", "metric", "dimension"):
        if actual[field] != expected[field]:
            problems.append(field)
    if actual["capacity"] is None or actual["capacity"] < minimum_capacity:
        problems.append("capacity")
    if actual["size"] != expected_size:
        problems.append("size")
    return {
        "ok": row is not None and not problems,
        "expected": expected,
        "actual": actual,
        "problems": problems,
    }


def _version_tuple(raw: str) -> tuple[int, ...]:
    match = re.search(r"(\d+)(?:\.(\d+))?(?:\.(\d+))?", raw or "")
    if match is None:
        return ()
    return tuple(int(value or 0) for value in match.groups())


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return -1.0
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return -1.0
    return numerator / (left_norm * right_norm)


def _index_facts(rows: list[dict[str, Any]], workspace: str) -> list[dict[str, Any]]:
    prefix = f"vec_{workspace}_"
    fields = (
        "index_name",
        "label",
        "property",
        "dimension",
        "capacity",
        "metric",
        "size",
    )
    result = []
    for row in rows:
        name = str(row.get("index_name") or "")
        if not name.startswith(prefix):
            continue
        result.append({key: row.get(key) for key in fields if key in row})
    return sorted(result, key=lambda item: str(item.get("index_name") or ""))


async def _never_store_counts(scope: Scope) -> dict[str, int]:
    counts: dict[str, int] = {}
    for spec in STORES:
        if spec.portability != "never":
            continue
        if spec.name == "notifications":
            total = 0
            for folder in scope.folder_ids:
                total += int(
                    await read_scalar(
                        f"MATCH (n:`{spec.label(folder)}`) RETURN count(n)"
                    )
                    or 0
                )
            counts[spec.name] = total
        else:
            counts[spec.name] = int(
                await read_scalar(
                    f"MATCH (n:`{spec.label(scope.workspace)}`) RETURN count(n)"
                )
                or 0
            )
    return counts


async def collect_target_facts(
    manifest: Manifest,
    *,
    workspace: str,
    candidate_folder_ids: tuple[str, ...] = (),
    embedding_func: Any | None = None,
    allow_unverified: bool | None = None,
) -> TargetFacts:
    """Probe the target without writing anything."""
    workspace = validate_identifier(workspace, "workspace")
    env_catalog = load_folder_catalog()
    runtime = sorted(list_runtime_folders(), key=lambda folder: folder.id)
    folder_ids = tuple(
        dict.fromkeys(
            [
                *(folder.id for folder in env_catalog.folders),
                *(folder.id for folder in runtime),
                *candidate_folder_ids,
            ]
        )
    )
    scope = Scope(workspace=workspace, folder_ids=folder_ids)

    counts: dict[str, int] = {}
    for spec in STORES:
        if spec.portability == "never":
            continue
        counts[spec.name] = await portable_store(spec).count(scope)
    counts.update(await _never_store_counts(scope))
    # GraphNodeStore intentionally counts only nodes with an ``entity_id``.
    # Target emptiness is stricter: a legacy/corrupt workspace-labelled node
    # must block the import rather than be overwritten or hidden.
    counts["__workspace_nodes__"] = int(
        await read_scalar(f"MATCH (n:`{workspace}`) RETURN count(n)") or 0
    )

    source = await _memgraph_source()
    target_embedding = await _embedding_probe(embedding_func, None)
    try:
        index_rows = await read_rows("SHOW VECTOR INDEX INFO")
    except Exception as exc:
        # Index introspection is a compatibility fact, not an optional About
        # card: hiding its failure could approve an index with the wrong dim.
        raise RuntimeError(f"cannot inspect target vector indexes: {exc}") from exc

    return TargetFacts(
        workspace=workspace,
        database=source["database"],
        memgraph_version=source["version"],
        lightrag_version=source["lightrag_version"],
        embedding_model=target_embedding.model,
        embedding_dim=target_embedding.dim,
        embedding_probe=target_embedding.probe.vectors,
        vector_capacity=resolve_vector_index_capacity(),
        vector_indexes=_index_facts(index_rows, workspace),
        classification_ceiling=os.environ.get(
            "TWIN_MIP_MAX_CLASSIFICATION", "C2"
        ).strip()
        or "C2",
        allow_unverified=(
            portability_flag_enabled(TWIN_PORTABILITY_ALLOW_UNVERIFIED_ENV)
            if allow_unverified is None
            else bool(allow_unverified)
        ),
        pipeline_idle=await _pipeline_is_idle(workspace),
        store_counts=dict(sorted(counts.items())),
        env_folders=[
            {"id": folder.id, "label": folder.label, "kind": folder.kind}
            for folder in env_catalog.folders
        ],
        runtime_folders=[folder.as_runtime_config() for folder in runtime],
        max_folders=env_catalog.max_folders,
    )


def _verdict(
    dimension: str,
    *,
    ok: bool,
    source: Any,
    target: Any,
    reason: str,
) -> dict[str, Any]:
    return {
        "dimension": dimension,
        "ok": ok,
        "source": source,
        "target": target,
        "reason": reason,
    }


def _classification_ok(manifest: Manifest, ceiling: str) -> bool:
    maximum = manifest.classification.max_detected
    if maximum is None:
        return True
    ladder = manifest.classification.ladder
    if maximum not in ladder or ceiling not in ladder:
        return False
    return ladder.index(maximum) <= ladder.index(ceiling)


def check(manifest: Manifest, facts: TargetFacts) -> list[dict[str, Any]]:
    """Return one deterministic verdict for every compatibility dimension."""
    probe_id_ok = manifest.embedding.probe.text_set_id == PROBE_TEXT_SET_ID
    probe_dim_ok = manifest.embedding.dim == facts.embedding_dim
    similarities = (
        [
            _cosine(source, target)
            for source, target in zip(
                manifest.embedding.probe.vectors,
                facts.embedding_probe,
                strict=True,
            )
        ]
        if probe_id_ok
        and probe_dim_ok
        and len(manifest.embedding.probe.vectors) == len(facts.embedding_probe)
        else []
    )
    minimum = min(similarities, default=-1.0)
    embedding_ok = probe_id_ok and probe_dim_ok and minimum >= MIN_PROBE_COSINE

    source_lr = _version_tuple(manifest.source["lightrag_version"])
    target_lr = _version_tuple(facts.lightrag_version)
    lightrag_ok = (
        len(source_lr) >= 2 and len(target_lr) >= 2 and source_lr[:2] == target_lr[:2]
    )
    memgraph_ok = _version_tuple(facts.memgraph_version) >= MIN_MEMGRAPH_VERSION
    consistency_ok = manifest.consistency.status == "verified" or facts.allow_unverified
    classification_ok = (
        not manifest.classification.unknown_present
        and _classification_ok(manifest, facts.classification_ceiling)
    )

    return [
        _verdict(
            "format",
            ok=manifest.format == "twin-kb-bundle"
            and manifest.format_version.split(".", 1)[0] == "1",
            source=f"{manifest.format}/{manifest.format_version}",
            target="twin-kb-bundle/1.x",
            reason="canonical bundle format v1",
        ),
        _verdict(
            "embedding",
            ok=embedding_ok,
            source={
                "model": manifest.embedding.model,
                "dim": manifest.embedding.dim,
                "text_set_id": manifest.embedding.probe.text_set_id,
            },
            target={
                "model": facts.embedding_model,
                "dim": facts.embedding_dim,
                "min_cosine": round(minimum, 12),
            },
            reason=(f"all three probe cosines must be >= {MIN_PROBE_COSINE}"),
        ),
        _verdict(
            "lightrag_version",
            ok=lightrag_ok,
            source=manifest.source["lightrag_version"],
            target=facts.lightrag_version,
            reason="source and target must use the same LightRAG minor",
        ),
        _verdict(
            "memgraph_version",
            ok=memgraph_ok,
            source=manifest.source["memgraph"]["version"],
            target=facts.memgraph_version,
            reason="target Memgraph must be >= 3.12.0",
        ),
        _verdict(
            "consistency",
            ok=consistency_ok,
            source=manifest.consistency.status,
            target={"allow_unverified": facts.allow_unverified},
            reason="unverified bundles are refused unless explicitly allowed",
        ),
        _verdict(
            "classification",
            ok=classification_ok,
            source={
                "max_detected": manifest.classification.max_detected,
                "unknown_present": manifest.classification.unknown_present,
            },
            target={"ceiling": facts.classification_ceiling},
            reason="the whole bundle must fit below the target ceiling",
        ),
    ]


__all__ = [
    "MIN_PROBE_COSINE",
    "TargetFacts",
    "check",
    "check_vector_index_contract",
    "collect_target_facts",
]
