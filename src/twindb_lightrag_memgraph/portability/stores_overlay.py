"""Operational/overlay portable stores (KB-PORTABILITY-PLAN T1.3).

Folder-scoped labels are enumerated from :class:`~.stores.Scope`, never from a
global label scan: two folders may legitimately contain the same tag id and
must remain two independent catalogues.  JSON blobs are decoded at the bundle
boundary and written back through the owning store APIs.  The two stores whose
normal create API cannot preserve identity (settings and procedure bundles)
use narrow, keyed restore seams instead of a generic property merge.
"""

from __future__ import annotations

import base64
import copy
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from .._constants import validate_identifier
from ._io import batched, keyset, read_rows, read_scalar, write
from .manifest import validate_bundle_path
from .stores import PortabilityError, Scope, StoreSpec, project_record, store_by_name


def _json_value(raw: Any, where: str) -> Any:
    if not isinstance(raw, str):
        raise PortabilityError(f"{where}: stored data is not a JSON string")
    try:
        return json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise PortabilityError(f"{where}: stored JSON is not readable ({exc})") from exc


def _encoded(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


# ----------------------------------------------------------- tags/categories


class TagStore:
    spec: StoreSpec = store_by_name("tags")

    async def export_records(self, scope: Scope) -> AsyncIterator[dict[str, Any]]:
        for folder_id in scope.folder_ids:
            label = self.spec.label(folder_id)

            async def fetch(
                after: str | None, limit: int, *, label: str = label
            ) -> list[dict[str, Any]]:
                return await read_rows(
                    f"MATCH (n:`{label}`) WHERE $after IS NULL OR n.id > $after "
                    f"RETURN properties(n) AS p ORDER BY n.id LIMIT {int(limit)}",
                    after=after,
                )

            async for row in keyset(fetch, lambda r: r["p"]["id"], scope.batch_size):
                props = project_record(self.spec, {**row["p"], "folder_id": folder_id})
                yield {
                    "folder_id": folder_id,
                    "id": props["id"],
                    "value": _json_value(
                        props["data"], f"tags[{folder_id},{props['id']}]"
                    ),
                }

    async def import_records(
        self, records: AsyncIterator[dict[str, Any]], scope: Scope
    ) -> int:
        from ..server.webui_tagstore import MemgraphTagStore

        stores: dict[str, Any] = {}
        total = 0
        async for chunk in batched(records, scope.batch_size):
            by_folder: dict[str, list[dict[str, Any]]] = {}
            for record in chunk:
                folder = validate_identifier(
                    scope.mapped_folder(record["folder_id"]), "folder"
                )
                value = copy.deepcopy(record["value"])
                if not isinstance(value, dict) or str(value.get("tag") or "") != str(
                    record["id"]
                ):
                    raise PortabilityError(
                        f"tags[{record['folder_id']},{record['id']}]: value.tag must equal id"
                    )
                by_folder.setdefault(folder, []).append(value)
            for folder, values in by_folder.items():
                store = stores.get(folder)
                if store is None:
                    store = stores[folder] = MemgraphTagStore(workspace=folder)
                    await store.initialize()
                for value in values:
                    await store.upsert_tag(value)
            total += len(chunk)
        return total

    async def count(self, scope: Scope) -> int:
        total = 0
        for folder in scope.folder_ids:
            total += int(
                await read_scalar(
                    f"MATCH (n:`{self.spec.label(folder)}`) RETURN count(n)"
                )
                or 0
            )
        return total

    async def fingerprint(self, scope: Scope) -> dict[str, Any]:
        count = 0
        updated: list[int] = []
        for folder in scope.folder_ids:
            rows = await read_rows(
                f"MATCH (n:`{self.spec.label(folder)}`) "
                "RETURN count(n) AS c, max(n.__updated_at) AS m"
            )
            if rows:
                count += int(rows[0]["c"])
                if rows[0]["m"] is not None:
                    updated.append(int(rows[0]["m"]))
        return {"count": count, "max_updated": max(updated, default=None)}


class TagCategoryStore:
    spec: StoreSpec = store_by_name("tag_categories")

    async def export_records(self, scope: Scope) -> AsyncIterator[dict[str, Any]]:
        for folder_id in scope.folder_ids:
            label = self.spec.label(folder_id)

            async def fetch(
                after: str | None, limit: int, *, label: str = label
            ) -> list[dict[str, Any]]:
                return await read_rows(
                    f"MATCH (n:`{label}`) WHERE $after IS NULL OR n.id > $after "
                    f"RETURN properties(n) AS p ORDER BY n.id LIMIT {int(limit)}",
                    after=after,
                )

            async for row in keyset(fetch, lambda r: r["p"]["id"], scope.batch_size):
                props = project_record(self.spec, {**row["p"], "folder_id": folder_id})
                yield {
                    "folder_id": folder_id,
                    "id": props["id"],
                    "value": _json_value(
                        props["data"],
                        f"tag_categories[{folder_id},{props['id']}]",
                    ),
                }

    async def import_records(
        self, records: AsyncIterator[dict[str, Any]], scope: Scope
    ) -> int:
        from ..server.webui_tagstore import MemgraphTagStore

        stores: dict[str, Any] = {}
        total = 0
        async for chunk in batched(records, scope.batch_size):
            by_folder: dict[str, list[dict[str, Any]]] = {}
            for record in chunk:
                folder = validate_identifier(
                    scope.mapped_folder(record["folder_id"]), "folder"
                )
                value = copy.deepcopy(record["value"])
                if not isinstance(value, dict) or str(value.get("id") or "") != str(
                    record["id"]
                ):
                    raise PortabilityError(
                        "tag_categories"
                        f"[{record['folder_id']},{record['id']}]: value.id must equal id"
                    )
                by_folder.setdefault(folder, []).append(value)
            for folder, values in by_folder.items():
                store = stores.get(folder)
                if store is None:
                    store = stores[folder] = MemgraphTagStore(workspace=folder)
                    await store.initialize()
                # This is the store's own canonical serializer/write path.  A
                # replace API cannot be used per streaming batch because it
                # would delete categories imported by the preceding batch.
                await store._write_many(store._cat_label, "id", values)
            total += len(chunk)
        return total

    async def count(self, scope: Scope) -> int:
        total = 0
        for folder in scope.folder_ids:
            total += int(
                await read_scalar(
                    f"MATCH (n:`{self.spec.label(folder)}`) RETURN count(n)"
                )
                or 0
            )
        return total

    async def fingerprint(self, scope: Scope) -> dict[str, Any]:
        count = 0
        updated: list[int] = []
        for folder in scope.folder_ids:
            rows = await read_rows(
                f"MATCH (n:`{self.spec.label(folder)}`) "
                "RETURN count(n) AS c, max(n.__updated_at) AS m"
            )
            if rows:
                count += int(rows[0]["c"])
                if rows[0]["m"] is not None:
                    updated.append(int(rows[0]["m"]))
        return {"count": count, "max_updated": max(updated, default=None)}


# ---------------------------------------------------------------- activity


class ActivityStore:
    spec: StoreSpec = store_by_name("activity")

    async def export_records(self, scope: Scope) -> AsyncIterator[dict[str, Any]]:
        for folder_id in scope.folder_ids:
            label = self.spec.label(folder_id)

            async def fetch(
                after: str | None, limit: int, *, label: str = label
            ) -> list[dict[str, Any]]:
                return await read_rows(
                    f"MATCH (n:`{label}`) WHERE $after IS NULL OR n.id > $after "
                    f"RETURN properties(n) AS p ORDER BY n.id LIMIT {int(limit)}",
                    after=after,
                )

            async for row in keyset(fetch, lambda r: r["p"]["id"], scope.batch_size):
                props = project_record(
                    self.spec,
                    {
                        **row["p"],
                        "folder_id": folder_id,
                        "origin": {},
                    },
                )
                value = _json_value(
                    props["data"], f"activity[{folder_id},{props['id']}]"
                )
                if not isinstance(value, dict):
                    raise PortabilityError(
                        f"activity[{folder_id},{props['id']}]: value must be an object"
                    )
                value = copy.deepcopy(value)
                origin = value.pop("origin", None)
                origin_workspace = (
                    str(origin.get("workspace") or scope.workspace)
                    if isinstance(origin, dict)
                    else scope.workspace
                )
                yield {
                    "folder_id": folder_id,
                    "id": props["id"],
                    "value": value,
                    # The current transport bundle_id lives in manifest.json.
                    # Keeping it out of semantic JSONL makes state_hash stable
                    # across consecutive exports of unchanged activity.
                    "origin": {"workspace": origin_workspace},
                }

    async def import_records(
        self, records: AsyncIterator[dict[str, Any]], scope: Scope
    ) -> int:
        from ..server.webui_activitystore import MemgraphActivityStore

        if not scope.bundle_id:
            raise PortabilityError("activity import requires scope.bundle_id")
        stores: dict[str, Any] = {}
        total = 0
        async for record in records:
            folder = validate_identifier(
                scope.mapped_folder(record["folder_id"]), "folder"
            )
            event = copy.deepcopy(record["value"])
            if not isinstance(event, dict) or str(event.get("id") or "") != str(
                record["id"]
            ):
                raise PortabilityError(
                    f"activity[{record['folder_id']},{record['id']}]: value.id must equal id"
                )
            origin = record.get("origin")
            if not isinstance(origin, dict) or set(origin) != {"workspace"}:
                raise PortabilityError("activity origin must contain workspace")
            origin_workspace = str(origin.get("workspace") or "")
            if not origin_workspace:
                raise PortabilityError("activity origin.workspace must be non-empty")
            event["origin"] = {
                "bundle_id": scope.bundle_id,
                "workspace": origin_workspace,
            }
            store = stores.get(folder)
            if store is None:
                store = stores[folder] = MemgraphActivityStore(workspace=folder)
                await store.initialize()
            await store.append(event)
            total += 1
        return total

    async def count(self, scope: Scope) -> int:
        total = 0
        for folder in scope.folder_ids:
            total += int(
                await read_scalar(
                    f"MATCH (n:`{self.spec.label(folder)}`) RETURN count(n)"
                )
                or 0
            )
        return total

    async def fingerprint(self, scope: Scope) -> dict[str, Any]:
        count = 0
        updated: list[int] = []
        for folder in scope.folder_ids:
            rows = await read_rows(
                f"MATCH (n:`{self.spec.label(folder)}`) "
                "RETURN count(n) AS c, max(n.__updated_at) AS m"
            )
            if rows:
                count += int(rows[0]["c"])
                if rows[0]["m"] is not None:
                    updated.append(int(rows[0]["m"]))
        return {"count": count, "max_updated": max(updated, default=None)}


# ---------------------------------------------------------------- settings


_SETTING_FIELDS = frozenset(
    {
        "min_ocr_chars",
        "drop_classes",
        "procedure_enabled",
        "updated_at",
        "updated_by",
    }
)


def _validate_settings_value(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PortabilityError("settings.value must be an object")
    unknown = sorted(set(value) - _SETTING_FIELDS)
    if unknown:
        raise PortabilityError(
            f"settings.value contains non-portable field(s): {unknown}"
        )
    return value


class SettingsStore:
    spec: StoreSpec = store_by_name("settings")

    async def export_records(self, scope: Scope) -> AsyncIterator[dict[str, Any]]:
        label = self.spec.label(scope.workspace)
        rows = await read_rows(
            f"MATCH (n:`{label}`) RETURN properties(n) AS p ORDER BY n.id"
        )
        for row in rows:
            props = project_record(self.spec, row["p"])
            yield {
                "id": props["id"],
                "value": _validate_settings_value(
                    _json_value(props["data"], f"settings[{props['id']}]")
                ),
            }

    async def import_records(
        self, records: AsyncIterator[dict[str, Any]], scope: Scope
    ) -> int:
        from ..server.vision_settings_store import SETTINGS_ID, initialize

        await initialize(scope.workspace)
        label = self.spec.label(scope.workspace)
        total = 0
        async for chunk in batched(records, scope.batch_size):
            rows = []
            for record in chunk:
                if record["id"] != SETTINGS_ID:
                    raise PortabilityError(
                        f"settings: unsupported record id {record['id']!r}"
                    )
                rows.append(
                    {
                        "id": SETTINGS_ID,
                        "data": _encoded(_validate_settings_value(record["value"])),
                    }
                )
            await write(
                f"portability.settings[{label}]",
                f"UNWIND $rows AS r MERGE (n:`{label}` {{id: r.id}}) SET n.data = r.data",
                rows=rows,
            )
            total += len(rows)
        return total

    async def count(self, scope: Scope) -> int:
        return int(
            await read_scalar(
                f"MATCH (n:`{self.spec.label(scope.workspace)}`) RETURN count(n)"
            )
            or 0
        )

    async def fingerprint(self, scope: Scope) -> dict[str, Any]:
        records = [record async for record in self.export_records(scope)]
        updated = [
            int(record["value"]["updated_at"])
            for record in records
            if record["value"].get("updated_at") is not None
        ]
        return {"count": len(records), "max_updated": max(updated, default=None)}


# ------------------------------------------------------------ source links


class SourceLinkStore:
    spec: StoreSpec = store_by_name("source_links")

    async def export_records(self, scope: Scope) -> AsyncIterator[dict[str, Any]]:
        from ..server.source_links_store import _public_source_link

        label = self.spec.label(scope.workspace)

        async def fetch(
            after: tuple[str, str] | None, limit: int
        ) -> list[dict[str, Any]]:
            a_doc, a_id = after if after else (None, None)
            return await read_rows(
                f"MATCH (n:`{label}`) WITH properties(n) AS p "
                "WHERE $a_doc IS NULL OR p.doc_id > $a_doc "
                "OR (p.doc_id = $a_doc AND p.id > $a_id) "
                f"RETURN p ORDER BY p.doc_id, p.id LIMIT {int(limit)}",
                a_doc=a_doc,
                a_id=a_id,
            )

        async for row in keyset(
            fetch, lambda r: (r["p"]["doc_id"], r["p"]["id"]), scope.batch_size
        ):
            yield _public_source_link(project_record(self.spec, row["p"]))

    async def import_records(
        self, records: AsyncIterator[dict[str, Any]], scope: Scope
    ) -> int:
        from ..server.source_links_store import MemgraphSourceLinkStore

        store = MemgraphSourceLinkStore(scope.workspace)
        await store.initialize()
        total = 0
        async for record in records:
            portable = project_record(self.spec, dict(record))
            current = await store._current(portable["doc_id"], portable["id"])
            if current is None:
                await store.create(portable)
            elif current != portable:
                raise PortabilityError(
                    "source_links"
                    f"[{portable['doc_id']},{portable['id']}]: target record differs"
                )
            total += 1
        return total

    async def count(self, scope: Scope) -> int:
        return int(
            await read_scalar(
                f"MATCH (n:`{self.spec.label(scope.workspace)}`) RETURN count(n)"
            )
            or 0
        )

    async def fingerprint(self, scope: Scope) -> dict[str, Any]:
        label = self.spec.label(scope.workspace)
        rows = await read_rows(
            f"MATCH (n:`{label}`) RETURN count(n) AS c, max(n.updated_at) AS m"
        )
        return {
            "count": int(rows[0]["c"]) if rows else 0,
            "max_updated": rows[0]["m"] if rows else None,
        }


# ----------------------------------------------------------- runtime folders


class RuntimeFolderStore:
    spec: StoreSpec = store_by_name("runtime_folders")

    async def export_records(self, scope: Scope) -> AsyncIterator[dict[str, Any]]:
        from ..server.folder_store import list_runtime_folders

        del scope
        for folder in sorted(list_runtime_folders(), key=lambda item: item.id):
            yield project_record(self.spec, folder.as_runtime_config())

    async def import_records(
        self, records: AsyncIterator[dict[str, Any]], scope: Scope
    ) -> int:
        from ..server.folder_store import add_runtime_folder, get_runtime_folder

        total = 0
        async for record in records:
            mapped = {
                **project_record(self.spec, dict(record)),
                "id": validate_identifier(scope.mapped_folder(record["id"]), "folder"),
            }
            existing = get_runtime_folder(mapped["id"])
            if existing is None:
                add_runtime_folder(
                    folder_id=mapped["id"],
                    label=mapped["label"],
                    kind=mapped.get("kind") or "custom",
                    description=mapped.get("description") or "",
                    sources=int(mapped.get("sources") or 0),
                )
            elif existing.as_runtime_config() != mapped:
                raise PortabilityError(
                    f"runtime_folders[{mapped['id']}]: target record differs"
                )
            total += 1
        return total

    async def count(self, scope: Scope) -> int:
        from ..server.folder_store import list_runtime_folders

        del scope
        return len(list_runtime_folders())

    async def fingerprint(self, scope: Scope) -> dict[str, Any]:
        return {"count": await self.count(scope)}


# --------------------------------------------------------------- procedures


class ProcedureStore:
    spec: StoreSpec = store_by_name("procedures")

    def __init__(
        self,
        *,
        bundle_writer: Any | None = None,
        bundle_root: Path | None = None,
    ) -> None:
        self.bundle_writer = bundle_writer
        self.bundle_root = Path(bundle_root) if bundle_root is not None else None

    async def export_records(self, scope: Scope) -> AsyncIterator[dict[str, Any]]:
        from .. import _procedure_store

        del scope
        for stored in sorted(_procedure_store.list_bundles(), key=lambda r: r["id"]):
            record = project_record(self.spec, copy.deepcopy(stored))
            schematics = record.get("schematics") or []
            if not isinstance(schematics, list):
                raise PortabilityError(
                    f"procedures[{record['id']}].schematics must be a list"
                )
            for index, schematic in enumerate(schematics, start=1):
                if not isinstance(schematic, dict):
                    raise PortabilityError(
                        f"procedures[{record['id']}].schematics[{index}] must be an object"
                    )
                encoded = schematic.pop("png_base64", None)
                if not encoded:
                    continue
                if self.bundle_writer is None:
                    raise PortabilityError(
                        "procedure export with schematics requires a BundleWriter"
                    )
                try:
                    png = base64.b64decode(encoded, validate=True)
                except (TypeError, ValueError) as exc:
                    raise PortabilityError(
                        f"procedures[{record['id']}].schematics[{index}] has invalid PNG base64"
                    ) from exc
                path = f"files/procedures/{record['id']}/{index}.png"
                self.bundle_writer.add_file(path, png, store=self.spec.name)
                schematic["file"] = path
            yield record

    def _inflate_files(self, record: dict[str, Any]) -> dict[str, Any]:
        restored = copy.deepcopy(record)
        for index, schematic in enumerate(restored.get("schematics") or [], start=1):
            path = schematic.pop("file", None)
            if path is None:
                continue
            validate_bundle_path(path)
            expected_prefix = f"files/procedures/{restored['id']}/"
            if not path.startswith(expected_prefix) or self.bundle_root is None:
                raise PortabilityError(
                    f"procedures[{restored['id']}].schematics[{index}]: invalid file reference"
                )
            target = self.bundle_root / path
            try:
                data = target.read_bytes()
            except OSError as exc:
                raise PortabilityError(
                    f"procedures[{restored['id']}].schematics[{index}]: cannot read {path}"
                ) from exc
            schematic["png_base64"] = base64.b64encode(data).decode("ascii")
        return restored

    async def import_records(
        self, records: AsyncIterator[dict[str, Any]], scope: Scope
    ) -> int:
        from .. import _procedure_store

        total = 0
        async for record in records:
            portable = project_record(self.spec, dict(record))
            folder = portable.get("folder")
            if folder:
                portable["folder"] = validate_identifier(
                    scope.mapped_folder(str(folder)), "folder"
                )
            requests = portable.get("duplicate_requests")
            if isinstance(requests, list):
                for request in requests:
                    if isinstance(request, dict) and request.get("folder"):
                        request["folder"] = validate_identifier(
                            scope.mapped_folder(str(request["folder"])), "folder"
                        )
            try:
                _procedure_store.restore_bundle(self._inflate_files(portable))
            except (ValueError, OSError) as exc:
                raise PortabilityError(
                    f"procedures[{portable.get('id')}]: {exc}"
                ) from exc
            total += 1
        return total

    async def count(self, scope: Scope) -> int:
        from .. import _procedure_store

        del scope
        return len(_procedure_store.list_bundles())

    async def fingerprint(self, scope: Scope) -> dict[str, Any]:
        from .. import _procedure_store

        del scope
        bundles = _procedure_store.list_bundles()
        return {
            "count": len(bundles),
            "max_updated": max(
                (str(bundle.get("updated_at") or "") for bundle in bundles),
                default=None,
            ),
        }
