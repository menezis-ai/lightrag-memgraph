"""Tests for the WebUI tag-mutation endpoints (S4c slice 2).

Each mutation must:
- update the tag store (verified via subsequent GET /tags/{name} look-up
  through GET /tags),
- emit an activity event (verified via GET /activity),
- push a notification (verified via GET /notifications).
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import json

import pytest
from httpx import ASGITransport, AsyncClient

from twindb_lightrag_memgraph import _twindb_state
from twindb_lightrag_memgraph.server import webui_router
from twindb_lightrag_memgraph.server.app import create_app
from twindb_lightrag_memgraph.server.settings import LightRAGServerSettings


def _make_settings() -> LightRAGServerSettings:
    return LightRAGServerSettings(
        working_dir="/tmp/lightrag_webui_mutation_test",
        workspace="cib",
        enable_langsmith_tracing=False,
        api_key="test-infra-root",
        jwt_secret=None,
        enable_webui_routes=True,
    )


@pytest.fixture(autouse=True)
def _reset_store():
    webui_router.reset_store()
    yield
    webui_router.reset_store()


@pytest.fixture()
async def client():
    app = create_app(_make_settings())
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
        headers={"Authorization": "Bearer test-infra-root"},
    ) as c:
        yield c


async def _get_tag(client: AsyncClient, name: str) -> dict | None:
    r = await client.get("/tags")
    for t in r.json():
        if t["tag"] == name:
            return t
    return None


async def _get_activity(client: AsyncClient) -> list[dict]:
    r = await client.get("/activity")
    return r.json()["items"]


async def _get_notifications(client: AsyncClient) -> list[dict]:
    r = await client.get("/notifications")
    return r.json()


class _FakeResult:
    def __init__(self, rows: list[dict]):
        self._rows = iter(rows)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._rows)
        except StopIteration as exc:
            raise StopAsyncIteration from exc

    async def consume(self) -> None:
        return None


class _FakeReadSession:
    def __init__(self):
        self.calls = 0

    async def run(self, _query: str, **_params):
        self.calls += 1
        if self.calls == 1:
            return _FakeResult(
                [{"id": "doc-hyphen", "file_path": "hyphen-source.pdf"}],
            )
        return _FakeResult(
            [{"docId": "doc-hyphen", "tags": ["rmf-validated"]}],
        )


class _NoMemberReadSession:
    async def run(self, _query: str, **_params):
        return _FakeResult([])


class _FakeTx:
    def __init__(self, *, tag_status: str = "active") -> None:
        self.tag_status = tag_status
        self.queries: list[str] = []
        self.committed = False
        self.rolled_back = False

    async def run(self, _query: str, **_params):
        self.queries.append(_query)
        if "RETURN t.id AS id, t.data AS data" in _query:
            return _FakeResult(
                [
                    {
                        "id": "rmf-validated",
                        "data": json.dumps(
                            {
                                "tag": "rmf-validated",
                                "status": self.tag_status,
                            }
                        ),
                    }
                ]
            )
        return _FakeResult([])

    async def commit(self) -> None:
        self.committed = True

    async def rollback(self) -> None:
        self.rolled_back = True


class _FakeWriteSession:
    def __init__(self, tx: _FakeTx | None = None) -> None:
        self.tx = tx or _FakeTx()

    async def begin_transaction(self):
        return self.tx


class _FakeDocStatus:
    def __init__(self):
        self.docs = {
            "doc-reject": {
                "id": "doc-reject",
                "file_path": "/kb/reject-me.pdf",
                "metadata": {},
            }
        }
        self.upserts: list[dict] = []

    async def get_by_id(self, doc_id: str):
        return self.docs.get(doc_id)

    async def upsert(self, docs: dict[str, dict]) -> None:
        self.upserts.append(docs)
        self.docs.update(docs)


class _FakeRag:
    def __init__(self):
        self.doc_status = _FakeDocStatus()


# ---------------------------------------------------------------------------
# POST /documents/_bulk-retag
# ---------------------------------------------------------------------------


class TestBulkRetag:
    async def test_accepts_hyphenated_tag_ids(self, monkeypatch, client):
        from twindb_lightrag_memgraph import _pool

        read_session = _FakeReadSession()
        tx = _FakeTx()

        @asynccontextmanager
        async def fake_read_session():
            yield read_session

        @asynccontextmanager
        async def fake_write_session():
            yield _FakeWriteSession(tx)

        @asynccontextmanager
        async def fake_write_slot():
            yield None

        monkeypatch.setattr(_pool, "get_read_session", fake_read_session)
        monkeypatch.setattr(_pool, "get_session", fake_write_session)
        monkeypatch.setattr(_pool, "acquire_write_slot", fake_write_slot)

        r = await client.post(
            "/documents/_bulk-retag",
            json={
                "targets": ["doc-hyphen"],
                "adds": ["rmf-validated"],
                "removes": [],
                "actor": "claire.benoit",
            },
        )

        assert r.status_code == 200
        assert r.json() == {"updated": 1, "failed": []}
        assert tx.committed is True
        assert tx.rolled_back is False
        assert all("MERGE (t:" not in query for query in tx.queries)

    @pytest.mark.parametrize(
        "status",
        ["pending-review", "pending-promotion", "deprecated", "rejected", None],
    )
    async def test_rejects_every_non_active_catalog_status(
        self, status, monkeypatch, client
    ):
        from twindb_lightrag_memgraph import _pool

        await webui_router.get_store().tags.upsert_tag(
            {"tag": "not-active", "status": status}
        )

        @asynccontextmanager
        async def fail_if_session_is_opened():
            raise AssertionError("non-active retag must fail before graph access")
            yield  # pragma: no cover

        monkeypatch.setattr(_pool, "get_read_session", fail_if_session_is_opened)

        r = await client.post(
            "/documents/_bulk-retag",
            json={
                "targets": ["doc-hyphen"],
                "adds": ["not-active"],
                "removes": [],
            },
        )

        assert r.status_code == 422
        assert r.json()["detail"]["unapproved_tags"] == ["not-active"]

    async def test_rejects_unapproved_tag_without_creating_it(
        self, monkeypatch, client
    ):
        from twindb_lightrag_memgraph import _pool

        async def fail_if_session_is_opened():
            raise AssertionError("unapproved retag must fail before graph access")

        monkeypatch.setattr(_pool, "get_read_session", fail_if_session_is_opened)

        r = await client.post(
            "/documents/_bulk-retag",
            json={
                "targets": ["doc-hyphen"],
                "adds": ["attacker-created-tag"],
                "removes": [],
            },
        )

        assert r.status_code == 422
        assert r.json()["detail"] == {
            "message": "Only active, approved tags may be attached",
            "unapproved_tags": ["attacker-created-tag"],
        }

    async def test_revalidates_status_inside_write_transaction(
        self, monkeypatch, client
    ):
        """A tag deprecated after the list pre-check must not be attached."""
        from twindb_lightrag_memgraph import _pool

        read_session = _FakeReadSession()
        tx = _FakeTx(tag_status="deprecated")

        @asynccontextmanager
        async def fake_read_session():
            yield read_session

        @asynccontextmanager
        async def fake_write_session():
            yield _FakeWriteSession(tx)

        @asynccontextmanager
        async def fake_write_slot():
            yield None

        monkeypatch.setattr(_pool, "get_read_session", fake_read_session)
        monkeypatch.setattr(_pool, "get_session", fake_write_session)
        monkeypatch.setattr(_pool, "acquire_write_slot", fake_write_slot)

        r = await client.post(
            "/documents/_bulk-retag",
            json={
                "targets": ["doc-hyphen"],
                "adds": ["rmf-validated"],
                "removes": [],
            },
        )

        assert r.status_code == 422
        assert r.json()["detail"]["unapproved_tags"] == ["rmf-validated"]
        assert tx.committed is False
        assert tx.rolled_back is True
        assert not any("MERGE (d)-[r:TAGGED_WITH]" in query for query in tx.queries)

    async def test_remove_only_allows_cleanup_of_non_active_legacy_edge(
        self, monkeypatch, client
    ):
        """Cleanup must not require a legacy tag to remain catalog-active."""
        from twindb_lightrag_memgraph import _pool

        read_session = _FakeReadSession()
        tx = _FakeTx(tag_status="deprecated")

        @asynccontextmanager
        async def fake_read_session():
            yield read_session

        @asynccontextmanager
        async def fake_write_session():
            yield _FakeWriteSession(tx)

        @asynccontextmanager
        async def fake_write_slot():
            yield None

        monkeypatch.setattr(_pool, "get_read_session", fake_read_session)
        monkeypatch.setattr(_pool, "get_session", fake_write_session)
        monkeypatch.setattr(_pool, "acquire_write_slot", fake_write_slot)

        r = await client.post(
            "/documents/_bulk-retag",
            json={
                "targets": ["doc-hyphen"],
                "adds": [],
                "removes": ["retired-legacy-tag"],
            },
        )

        assert r.status_code == 200
        assert tx.committed is True
        assert any("DELETE r" in query for query in tx.queries)
        assert not any("__bulk_retag_lock" in query for query in tx.queries)

    async def test_reports_existing_cross_folder_doc_as_failed(
        self, monkeypatch, client
    ):
        from twindb_lightrag_memgraph import _pool

        read_session = _NoMemberReadSession()

        @asynccontextmanager
        async def fake_read_session():
            yield read_session

        monkeypatch.setattr(_pool, "get_read_session", fake_read_session)

        r = await client.post(
            "/documents/_bulk-retag",
            json={
                "targets": ["doc-in-other-folder"],
                "adds": ["rmf-validated"],
                "removes": [],
                "actor": "claire.benoit",
            },
        )

        assert r.status_code == 200
        assert r.json() == {"updated": 0, "failed": ["doc-in-other-folder"]}


# ---------------------------------------------------------------------------
# POST /documents/{doc_id}/reject
# ---------------------------------------------------------------------------


class TestRejectDocument:
    async def test_emits_warning_event_with_doc_id_and_reason(self, client):
        _twindb_state["rag"] = _FakeRag()
        try:
            r = await client.post(
                "/documents/doc-reject/reject",
                json={"reason": "classification too high", "actor": "auditor"},
            )

            assert r.status_code == 200
            body = r.json()
            assert body["review"]["state"] == "rejected"
            assert body["review"]["justification"] == "classification too high"

            events = await _get_activity(client)
            event = events[0]
            assert event["kind"] == "doc-rejected"
            assert event["sev"] == "warning"
            assert event["target"]["id"] == "doc-reject"
            assert event["meta"]["doc_id"] == "doc-reject"
            assert event["meta"]["reason"] == "classification too high"
            assert "classification too high" in event["summary"]
        finally:
            _twindb_state.pop("rag", None)


# ---------------------------------------------------------------------------
# POST /tags — request new tag
# ---------------------------------------------------------------------------


class TestRequestTag:
    async def test_creates_pending_tag(self, client):
        r = await client.post(
            "/tags",
            json={
                "tag": "newtag",
                "def": "A brand new tag",
                "category": "infra",
                "justification": "Needed for upcoming sprint",
                "actor": "marc.berthier",
            },
        )
        assert r.status_code == 201
        body = r.json()
        assert body["tag"] == "newtag"
        assert body["tier"] == "requested"
        assert body["status"] == "pending-review"
        # GET round-trip
        tag = await _get_tag(client, "newtag")
        assert tag is not None
        assert tag["tier"] == "requested"

    async def test_duplicate_returns_409(self, client):
        r = await client.post(
            "/tags",
            json={"tag": "rman", "def": "dup", "category": "oracle"},
        )
        assert r.status_code == 409

    async def test_emits_activity_event_without_trusting_client_actor(self, client):
        await client.post(
            "/tags",
            json={
                "tag": "newtag",
                "def": "test",
                "category": "infra",
                "actor": "marc.berthier",
            },
        )
        events = await _get_activity(client)
        assert events
        assert events[0]["kind"] == "tag-mutation"
        assert "newtag" in events[0]["summary"]
        assert events[0]["actor"]["user"] == "api_key"

    async def test_pushes_notification(self, client):
        await client.post(
            "/tags",
            json={"tag": "newtag", "def": "test", "category": "infra"},
        )
        notifs = await _get_notifications(client)
        assert notifs
        assert notifs[0]["tagname"] == "newtag"
        assert notifs[0]["suffix"] == "requested"
        assert notifs[0]["read"] is False

    async def test_tag_mutation_is_isolated_by_folder(self, monkeypatch, client):
        monkeypatch.setenv("TWIN_DEFAULT_FOLDER", "default")
        monkeypatch.setenv(
            "TWIN_FOLDERS_JSON",
            json.dumps(
                [
                    {"id": "default", "label": "Default folder", "kind": "primary"},
                    {"id": "sandbox", "label": "Sandbox", "kind": "sandbox"},
                ]
            ),
        )
        webui_router.reset_store()

        r = await client.post(
            "/tags",
            headers={"X-Twin-Folder": "default"},
            json={
                "tag": "folderonly",
                "def": "Only in the default folder",
                "category": "infra",
            },
        )
        assert r.status_code == 201

        default_tags = (
            await client.get(
                "/tags",
                headers={"X-Twin-Folder": "default"},
            )
        ).json()
        sandbox_tags = (
            await client.get(
                "/tags",
                headers={"X-Twin-Folder": "sandbox"},
            )
        ).json()
        assert any(tag["tag"] == "folderonly" for tag in default_tags)
        assert all(tag["tag"] != "folderonly" for tag in sandbox_tags)


# ---------------------------------------------------------------------------
# POST /tags/{name}/approve
# ---------------------------------------------------------------------------


class TestApproveTag:
    async def test_flips_tier_and_status(self, client):
        # argocd is seeded as a requested tag
        r = await client.post(
            "/tags/argocd/approve",
            json={"actor": "claire.benoit"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["tier"] == 3
        assert body["status"] == "active"
        # Requested-only fields are dropped
        assert "requested_by" not in body or body.get("requested_by") is None
        assert "justification" not in body or body.get("justification") is None

    async def test_404_when_unknown(self, client):
        r = await client.post(
            "/tags/zzz-no-tag/approve", json={"actor": "claire.benoit"}
        )
        assert r.status_code == 404

    async def test_emits_event_and_notification(self, client):
        await client.post("/tags/argocd/approve", json={"actor": "claire.benoit"})
        events = await _get_activity(client)
        notifs = await _get_notifications(client)
        assert events[0]["summary"].startswith("Tag argocd approved")
        assert notifs[0]["suffix"] == "approved"


# ---------------------------------------------------------------------------
# POST /tags/{name}/reject
# ---------------------------------------------------------------------------


class TestRejectTag:
    async def test_purges_rejected_tag_and_allows_recreate(self, client):
        r = await client.post(
            "/tags/argocd/reject",
            json={"reason": "duplicate of k8s", "actor": "claire.benoit"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "rejected"
        assert body.get("reject_reason") == "duplicate of k8s"
        assert await _get_tag(client, "argocd") is None

        recreated = await client.post(
            "/tags",
            json={
                "tag": "argocd",
                "def": "GitOps controller",
                "category": "infra",
                "actor": "claire.benoit",
            },
        )
        assert recreated.status_code == 201
        assert recreated.json()["status"] == "pending-review"

    async def test_emits_warning_event(self, client):
        await client.post(
            "/tags/argocd/reject",
            json={"reason": "scope creep", "actor": "claire.benoit"},
        )
        events = await _get_activity(client)
        assert events[0]["sev"] == "warning"
        assert events[0]["target"]["id"] == "argocd"
        assert events[0]["meta"]["reason"] == "scope creep"
        assert "scope creep" in events[0]["summary"]

    async def test_missing_reason_is_422(self, client):
        r = await client.post("/tags/argocd/reject", json={"actor": "claire.benoit"})
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# PATCH /tags/{name}
# ---------------------------------------------------------------------------


class TestEditTag:
    async def test_updates_def_and_aliases(self, client):
        r = await client.patch(
            "/tags/rman",
            json={
                "def": "Updated definition",
                "aliases": ["recovery-manager", "rmgr"],
                "actor": "claire.benoit",
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["def"] == "Updated definition"
        assert body["aliases"] == ["recovery-manager", "rmgr"]

    async def test_renames_tag_and_persists_long_description(self, client):
        r = await client.patch(
            "/tags/rman",
            json={
                "tag": "rman-v2",
                "long_description": "Long governance note for RMAN.",
                "actor": "claire.benoit",
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["tag"] == "rman-v2"
        assert body["long_description"] == "Long governance note for RMAN."

        assert await _get_tag(client, "rman") is None
        persisted = await _get_tag(client, "rman-v2")
        assert persisted is not None
        assert persisted["long_description"] == "Long governance note for RMAN."

        docs = (await client.get("/documents", params={"tag": "rman-v2"})).json()
        assert docs["total"] > 0
        stale_docs = (await client.get("/documents", params={"tag": "rman"})).json()
        assert stale_docs["total"] == 0

    async def test_no_op_is_still_successful(self, client):
        r = await client.patch("/tags/rman", json={"actor": "claire.benoit"})
        assert r.status_code == 200

    async def test_rename_conflict_returns_409(self, client):
        r = await client.patch(
            "/tags/rman",
            json={"tag": "oracle", "actor": "claire.benoit"},
        )
        assert r.status_code == 409


# ---------------------------------------------------------------------------
# POST /tags/{name}/suggest-edit
# ---------------------------------------------------------------------------


class TestSuggestTagEdit:
    async def test_creates_pending_edit_proposal(self, client):
        r = await client.post(
            "/tags/rman/suggest-edit",
            json={
                "def": "Updated RMAN definition",
                "aliases": ["rmgr", "recovery-manager"],
                "justification": "clarify recovery manager wording",
                "actor": "alberto",
            },
        )

        assert r.status_code == 201
        body = r.json()
        assert body["tier"] == "requested"
        assert body["status"] == "pending-review"
        assert body["proposal_kind"] == "edit"
        assert body["target_tag"] == "rman"
        assert body["def"] == "Updated RMAN definition"
        assert body["aliases"] == ["rmgr", "recovery-manager"]
        assert set(body["proposed_fields"]) == {"def", "aliases"}

        persisted = await _get_tag(client, body["tag"])
        assert persisted is not None
        events = await _get_activity(client)
        assert events[0]["summary"] == "Tag rman edit suggested for palier-3 review"
        notifications = await _get_notifications(client)
        assert notifications[0]["title"] == "Tag"

    async def test_approve_applies_proposal_to_target_and_removes_proposal(
        self, client
    ):
        proposal = (
            await client.post(
                "/tags/rman/suggest-edit",
                json={
                    "def": "Updated RMAN definition",
                    "long_description": "Longer approved note",
                    "actor": "alberto",
                },
            )
        ).json()

        r = await client.post(
            f"/tags/{proposal['tag']}/approve",
            json={"actor": "claire.benoit"},
        )

        assert r.status_code == 200
        body = r.json()
        assert body["tag"] == "rman"
        assert body["def"] == "Updated RMAN definition"
        assert body["long_description"] == "Longer approved note"
        assert body["last_edit"]["action"] == "edit-approved"
        assert await _get_tag(client, proposal["tag"]) is None
        assert (await _get_tag(client, "rman"))["def"] == "Updated RMAN definition"

    async def test_reject_marks_edit_proposal_rejected(self, client):
        proposal = (
            await client.post(
                "/tags/rman/suggest-edit",
                json={"def": "Rejected wording", "actor": "alberto"},
            )
        ).json()

        r = await client.post(
            f"/tags/{proposal['tag']}/reject",
            json={"reason": "too broad", "actor": "claire.benoit"},
        )

        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "rejected"
        assert body["proposal_kind"] == "edit"
        assert body["target_tag"] == "rman"
        assert body["reject_reason"] == "too broad"

    async def test_unknown_target_returns_404(self, client):
        r = await client.post(
            "/tags/does-not-exist/suggest-edit",
            json={"def": "new", "actor": "alberto"},
        )
        assert r.status_code == 404

    async def test_no_changed_field_returns_400(self, client):
        current = await _get_tag(client, "rman")
        r = await client.post(
            "/tags/rman/suggest-edit",
            json={"def": current["def"], "actor": "alberto"},
        )
        assert r.status_code == 400


# ---------------------------------------------------------------------------
# POST /tags/{name}/deprecate
# ---------------------------------------------------------------------------


class TestDeprecateTag:
    async def test_sets_status_deprecated(self, client):
        r = await client.post(
            "/tags/rman/deprecate",
            json={"reason": "superseded by rman-v2"},
        )
        body = r.json()
        assert body["status"] == "deprecated"
        assert body.get("deprecate_reason") == "superseded by rman-v2"

    async def test_emits_warning_event(self, client):
        await client.post("/tags/rman/deprecate", json={})
        events = await _get_activity(client)
        assert events[0]["sev"] == "warning"


# ---------------------------------------------------------------------------
# POST /tags/{name}/reactivate
# ---------------------------------------------------------------------------


class TestReactivateTag:
    async def test_sets_status_active_and_clears_reason(self, client):
        await client.post(
            "/tags/rman/deprecate",
            json={"reason": "superseded by rman-v2"},
        )
        r = await client.post("/tags/rman/reactivate", json={"actor": "claire"})
        body = r.json()
        assert body["status"] == "active"
        assert "deprecate_reason" not in body
        assert body["last_edit"]["action"] == "reactivated"

    async def test_emits_info_event(self, client):
        await client.post("/tags/rman/deprecate", json={})
        await client.post("/tags/rman/reactivate", json={})
        events = await _get_activity(client)
        assert events[0]["sev"] == "info"
        assert events[0]["summary"] == "Tag rman reactivated"


# ---------------------------------------------------------------------------
# POST /tags/{name}/synonyms
# ---------------------------------------------------------------------------


class TestSynonyms:
    async def test_replaces_alias_list(self, client):
        r = await client.post(
            "/tags/rman/synonyms",
            json={"aliases": ["recovery-mgr", "rmgr"]},
        )
        body = r.json()
        assert body["aliases"] == ["recovery-mgr", "rmgr"]


# ---------------------------------------------------------------------------
# DELETE /tags/{name}
# ---------------------------------------------------------------------------


class TestDeleteTag:
    async def test_migrate_requires_to(self, client):
        r = await client.request(
            "DELETE",
            "/tags/rman",
            json={"strategy": "migrate"},
        )
        assert r.status_code == 422

    async def test_migrate_with_to_succeeds(self, client):
        r = await client.request(
            "DELETE",
            "/tags/rman",
            json={"strategy": "migrate", "to": "oracle"},
        )
        assert r.status_code == 200
        # GET confirms it's gone
        tag = await _get_tag(client, "rman")
        assert tag is None
        rman_docs = await client.get("/documents", params={"tag": "rman"})
        assert rman_docs.json()["items"] == []
        oracle_docs = await client.get("/documents", params={"tag": "oracle"})
        assert oracle_docs.json()["total"] >= 1
        assert all("rman" not in doc["tags"] for doc in oracle_docs.json()["items"])
        events = await _get_activity(client)
        assert "migrated to oracle" in events[0]["summary"]
        assert "docs migrated to oracle" in events[0]["summary"]
        assert events[0]["target"]["id"] == "rman"
        assert events[0]["meta"]["operation"] == "delete-tag"
        assert events[0]["meta"]["strategy"] == "migrate"
        assert events[0]["meta"]["to"] == "oracle"
        assert "docs migrated to oracle" in events[0]["meta"]["result"]
        assert events[0]["meta"]["affected_docs"] >= 1

    async def test_untag_strategy_default(self, client):
        r = await client.request("DELETE", "/tags/vault")
        assert r.status_code == 200
        vault_docs = await client.get("/documents", params={"tag": "vault"})
        assert vault_docs.json()["items"] == []
        events = await _get_activity(client)
        assert "deleted (docs untagged)" in events[0]["summary"]
        assert events[0]["target"]["id"] == "vault"
        assert events[0]["meta"]["strategy"] == "untag"
        assert "docs untagged" in events[0]["meta"]["result"]

    async def test_migrate_requires_existing_target(self, client):
        r = await client.request(
            "DELETE",
            "/tags/rman",
            json={"strategy": "migrate", "to": "zzz-no-tag"},
        )
        assert r.status_code == 404

    async def test_unknown_returns_404(self, client):
        r = await client.request("DELETE", "/tags/zzz-no-tag")
        assert r.status_code == 404
