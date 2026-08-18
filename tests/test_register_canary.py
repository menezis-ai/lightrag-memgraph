"""Tests for ``patches/canary.py`` — register()'s upstream-symbol canaries.

Audit 2026-07-02 COMPAT-3/COMPAT-4: ``register()`` used to read upstream
LightRAG symbols bare, so any upstream rename crashed the boot
(``AttributeError``) or silently decoupled a patch. The canary classifies each
patched symbol:

* REQUIRED (the 3 ``lightrag.kg`` registry dicts) → ``RuntimeError`` with an
  actionable message naming the symbol and the installed lightrag version;
* DEGRADABLE (buffered-merge, ``_insert_done`` hook, document-routes capture,
  ``create_app`` overlay) → loud warning, patch skipped, boot continues;
* DRIFT (the two ``operate`` private copies) → warning-only body-hash check.

Compat doctrine (docs/test-doctrine-lightrag-compat.md): when every symbol is
present, the canary must be behavior-neutral — the patches apply exactly as
before and no canary warning is emitted.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import logging
import sys
import types
from pathlib import Path

import pytest

import twindb_lightrag_memgraph
from twindb_lightrag_memgraph.patches import canary, registry

_TWIN_OVERLAY_ENV = ("TWIN_REPLACE_UI", "TWIN_MOUNT_SERVER", "TWIN_SHIM_NATIVE_ROUTES")


def _reset_registration():
    twindb_lightrag_memgraph._registered = False
    registry._registered = False


def _canary_messages(caplog, *, include_drift: bool = True) -> list[str]:
    msgs = [
        record.getMessage()
        for record in caplog.records
        if "twindb canary:" in record.getMessage()
    ]
    if not include_drift:
        msgs = [m for m in msgs if "known-good" not in m]
    return msgs


def _healthy_fake_kg() -> types.ModuleType:
    mod = types.ModuleType("lightrag.kg")
    mod.STORAGE_IMPLEMENTATIONS = {
        key: {"implementations": []}
        for key in ("KV_STORAGE", "VECTOR_STORAGE", "DOC_STATUS_STORAGE")
    }
    mod.STORAGE_ENV_REQUIREMENTS = {}
    mod.STORAGES = {}
    return mod


def _install_fake_kg(monkeypatch, fake: types.ModuleType) -> None:
    """Make ``import lightrag.kg`` resolve to ``fake`` (attr + sys.modules)."""
    import lightrag

    monkeypatch.setitem(sys.modules, "lightrag.kg", fake)
    monkeypatch.setattr(lightrag, "kg", fake, raising=False)


# ---------------------------------------------------------------------------
# REQUIRED class — the 3 lightrag.kg registry dicts
# ---------------------------------------------------------------------------


class TestRequiredStorageRegistries:
    @pytest.mark.parametrize(
        "missing", ["STORAGE_IMPLEMENTATIONS", "STORAGE_ENV_REQUIREMENTS", "STORAGES"]
    )
    def test_missing_dict_raises_actionable_runtime_error(self, missing):
        fake = _healthy_fake_kg()
        delattr(fake, missing)

        with pytest.raises(RuntimeError) as exc:
            canary.assert_storage_registries(fake)

        message = str(exc.value)
        assert missing in message
        assert canary.installed_lightrag_version() in message
        assert "register() cannot" in message

    def test_malformed_storage_implementations_shape_raises(self):
        fake = _healthy_fake_kg()
        fake.STORAGE_IMPLEMENTATIONS["VECTOR_STORAGE"] = ["not-the-dict-shape"]

        with pytest.raises(RuntimeError, match="VECTOR_STORAGE"):
            canary.assert_storage_registries(fake)

    def test_healthy_registries_pass_silently(self, caplog):
        with caplog.at_level(logging.WARNING, logger="twindb_lightrag_memgraph"):
            canary.assert_storage_registries(_healthy_fake_kg())
        assert _canary_messages(caplog) == []

    def test_register_raises_actionably_when_registry_dict_missing(self, monkeypatch):
        """Full register() path: a missing REQUIRED dict fails loud, not with
        the bare AttributeError this used to produce."""
        for var in _TWIN_OVERLAY_ENV:
            monkeypatch.delenv(var, raising=False)
        fake = _healthy_fake_kg()
        delattr(fake, "STORAGES")
        _install_fake_kg(monkeypatch, fake)
        _reset_registration()

        with pytest.raises(RuntimeError) as exc:
            twindb_lightrag_memgraph.register()

        assert "STORAGES" in str(exc.value)
        assert registry._registered is False


# ---------------------------------------------------------------------------
# DEGRADABLE class — warn + skip the individual patch, boot continues
# ---------------------------------------------------------------------------


class TestDegradableSymbols:
    def test_missing_merge_symbol_warns_and_skips(self, monkeypatch, caplog):
        import lightrag.lightrag as lr_mod
        import lightrag.operate as operate

        lr_binding_before = getattr(lr_mod, "merge_nodes_and_edges", None)
        monkeypatch.delattr(operate, "merge_nodes_and_edges")

        with caplog.at_level(logging.WARNING, logger="twindb_lightrag_memgraph"):
            registry._patch_merge_write_path()

        assert not hasattr(operate, "merge_nodes_and_edges")  # patch skipped
        assert getattr(lr_mod, "merge_nodes_and_edges", None) is lr_binding_before
        msgs = _canary_messages(caplog)
        assert any(
            "merge_nodes_and_edges" in m and "skipping" in m.lower() for m in msgs
        )

    def test_missing_insert_done_warns_and_skips(self, monkeypatch, caplog):
        from lightrag.lightrag import LightRAG

        monkeypatch.delattr(LightRAG, "_insert_done")

        with caplog.at_level(logging.WARNING, logger="twindb_lightrag_memgraph"):
            registry._patch_insert_done()

        assert not hasattr(LightRAG, "_insert_done")  # patch skipped
        msgs = _canary_messages(caplog)
        assert any("_insert_done" in m and "skipping" in m.lower() for m in msgs)

    def test_insert_done_arity_break_warns_and_skips(self, monkeypatch, caplog):
        """The wrapper hardcodes ``(self, pipeline_status, pipeline_status_lock)``;
        a signature that can no longer bind it must skip (warn) instead of
        installing a patch that would crash every ingestion."""
        from lightrag.lightrag import LightRAG

        async def _new_shape(self, *, mandatory_new_kwarg):  # pragma: no cover
            raise AssertionError("never called")

        monkeypatch.setattr(LightRAG, "_insert_done", _new_shape)

        with caplog.at_level(logging.WARNING, logger="twindb_lightrag_memgraph"):
            registry._patch_insert_done()

        assert LightRAG._insert_done is _new_shape  # patch skipped
        msgs = _canary_messages(caplog)
        assert any("_insert_done" in m and "call shape" in m for m in msgs)

    def test_missing_create_document_routes_warns_and_skips(self, monkeypatch, caplog):
        saved_argv = sys.argv
        sys.argv = ["lightrag"]
        try:
            import lightrag.api.routers.document_routes as dr
        finally:
            sys.argv = saved_argv

        monkeypatch.setattr(dr, "_twindb_capture_rag_patched", False, raising=False)
        monkeypatch.delattr(dr, "create_document_routes")

        with caplog.at_level(logging.WARNING, logger="twindb_lightrag_memgraph"):
            registry._patch_capture_rag()

        assert getattr(dr, "_twindb_capture_rag_patched", False) is False
        msgs = _canary_messages(caplog)
        assert any(
            "create_document_routes" in m and "skipping" in m.lower() for m in msgs
        )

    def test_missing_create_app_warns_and_skips(self, monkeypatch, caplog):
        saved_argv = sys.argv
        sys.argv = ["lightrag"]
        try:
            import lightrag.api.lightrag_server as srv
        finally:
            sys.argv = saved_argv

        monkeypatch.setattr(srv, "_twindb_create_app_patched", False, raising=False)
        monkeypatch.delattr(srv, "create_app")

        with caplog.at_level(logging.WARNING, logger="twindb_lightrag_memgraph"):
            registry._patch_lightrag_server_create_app()

        assert getattr(srv, "_twindb_create_app_patched", False) is False
        msgs = _canary_messages(caplog)
        assert any("create_app" in m and "skipping" in m.lower() for m in msgs)

    def test_register_completes_degraded_when_degradable_symbols_missing(
        self, monkeypatch, caplog
    ):
        """register() must finish (storage backends registered) even when
        every DEGRADABLE symbol is gone — warn-and-degrade, never crash."""
        import lightrag.kg as kg_registry
        import lightrag.operate as operate
        from lightrag.lightrag import LightRAG

        for var in _TWIN_OVERLAY_ENV:
            monkeypatch.delenv(var, raising=False)
        monkeypatch.delattr(operate, "merge_nodes_and_edges")
        monkeypatch.delattr(LightRAG, "_insert_done")
        _reset_registration()

        with caplog.at_level(logging.WARNING, logger="twindb_lightrag_memgraph"):
            twindb_lightrag_memgraph.register()

        assert registry._registered is True
        assert (
            kg_registry.STORAGES["MemgraphKVStorage"]
            == "twindb_lightrag_memgraph.kv_impl"
        )
        msgs = _canary_messages(caplog)
        assert any("merge_nodes_and_edges" in m for m in msgs)
        assert any("_insert_done" in m for m in msgs)


# ---------------------------------------------------------------------------
# Behavior-neutrality — everything present ⇒ no canary warning, patches apply
# ---------------------------------------------------------------------------


class TestCanaryNeutrality:
    def test_register_all_present_emits_no_symbol_canary_warning(
        self, monkeypatch, caplog
    ):
        """Compat doctrine: with a healthy installed lightrag, register() must
        behave exactly as before the canary — patches applied, zero symbol
        warnings. (Drift warnings are excluded: on CI-matrix versions without
        a recorded hash — 1.4.11/1.4.12 — the drift canary legitimately
        fires; that is its job, not a neutrality break.)"""
        import lightrag.operate as operate
        from lightrag.kg.memgraph_impl import MemgraphStorage
        from lightrag.lightrag import LightRAG

        for var in _TWIN_OVERLAY_ENV:
            monkeypatch.delenv(var, raising=False)
        _reset_registration()

        with caplog.at_level(logging.WARNING, logger="twindb_lightrag_memgraph"):
            twindb_lightrag_memgraph.register()

        assert _canary_messages(caplog, include_drift=False) == []
        # The DEGRADABLE patches actually applied (not skipped):
        assert (
            operate.merge_nodes_and_edges.__name__ == "buffered_merge_nodes_and_edges"
        )
        assert LightRAG._insert_done.__name__ == "hooked_insert_done"
        assert operate._get_node_data is registry._fused_get_node_data
        assert (
            operate._find_most_related_edges_from_entities is registry._fused_find_edges
        )
        assert getattr(
            MemgraphStorage.__init__, "_twindb_explicit_workspace_patch", False
        )

    def test_register_stays_idempotent_with_canary(self, monkeypatch):
        import lightrag.kg as kg_registry

        for var in _TWIN_OVERLAY_ENV:
            monkeypatch.delenv(var, raising=False)
        _reset_registration()
        twindb_lightrag_memgraph.register()
        twindb_lightrag_memgraph.register()

        count = kg_registry.STORAGE_IMPLEMENTATIONS["KV_STORAGE"][
            "implementations"
        ].count("MemgraphKVStorage")
        assert count == 1


# ---------------------------------------------------------------------------
# DRIFT class — body-hash canary for the two operate private copies
# ---------------------------------------------------------------------------


def _fake_operate(fn, name: str = "_get_node_data") -> types.ModuleType:
    mod = types.ModuleType("fake_operate")
    setattr(mod, name, fn)
    return mod


class TestPrivateCopyDriftCanary:
    def test_known_hash_is_silent(self, monkeypatch, caplog):
        def sample_upstream_body():
            return "known"

        digest = canary.normalized_source_hash(sample_upstream_body)
        monkeypatch.setitem(
            canary.KNOWN_PRIVATE_COPY_SOURCE_HASHES,
            "_get_node_data",
            {digest: "test baseline"},
        )

        with caplog.at_level(logging.WARNING, logger="twindb_lightrag_memgraph"):
            canary.warn_on_private_copy_drift(
                _fake_operate(sample_upstream_body), "_get_node_data"
            )

        assert _canary_messages(caplog) == []

    def test_unknown_hash_warns_drift(self, monkeypatch, caplog):
        def sample_altered_body():
            return "altered"

        monkeypatch.setitem(
            canary.KNOWN_PRIVATE_COPY_SOURCE_HASHES,
            "_get_node_data",
            {"0" * 64: "some other body"},
        )

        with caplog.at_level(logging.WARNING, logger="twindb_lightrag_memgraph"):
            canary.warn_on_private_copy_drift(
                _fake_operate(sample_altered_body), "_get_node_data"
            )

        msgs = _canary_messages(caplog)
        assert len(msgs) == 1
        assert "_get_node_data" in msgs[0]
        assert "may have drifted" in msgs[0]
        assert canary.installed_lightrag_version() in msgs[0]

    def test_absent_symbol_warns(self, caplog):
        empty = types.ModuleType("fake_operate")

        with caplog.at_level(logging.WARNING, logger="twindb_lightrag_memgraph"):
            canary.warn_on_private_copy_drift(empty, "_get_node_data")

        msgs = _canary_messages(caplog)
        assert len(msgs) == 1
        assert "absent" in msgs[0]

    def test_own_replacement_is_silent(self, monkeypatch, caplog):
        """Idempotent re-register(): once operate carries our fused copy,
        re-hashing it must not fire a spurious drift warning."""

        def sample():
            return "ours"

        sample.__module__ = "twindb_lightrag_memgraph.patches.registry"
        monkeypatch.setitem(
            canary.KNOWN_PRIVATE_COPY_SOURCE_HASHES, "_get_node_data", {}
        )

        with caplog.at_level(logging.WARNING, logger="twindb_lightrag_memgraph"):
            canary.warn_on_private_copy_drift(_fake_operate(sample), "_get_node_data")

        assert _canary_messages(caplog) == []

    def test_wired_into_patch_operate_hot_paths(self, monkeypatch):
        """Wiring proof: _patch_operate_hot_paths runs the drift canary on
        both private-copy targets before overwriting them."""
        seen: list[str] = []
        monkeypatch.setattr(
            canary,
            "warn_on_private_copy_drift",
            lambda owner, attr: seen.append(attr),
        )

        registry._patch_operate_hot_paths()

        assert set(seen) == {
            "_get_node_data",
            "_find_most_related_edges_from_entities",
        }

    def test_final_answer_cache_is_consulted_after_retrieval(self):
        """Supported LightRAG pins rebuild grounding before a query-cache hit.

        This ordering guarantees that ``references`` and Twin's chunk-query
        trace describe the same request even when LightRAG reuses the final
        answer.  Keep this source-order canary in the compatibility matrix:
        moving ``handle_cache`` ahead of retrieval would make an empty trace
        ambiguous instead of proving graph-only provenance.
        """
        import lightrag.operate as operate

        kg_source = inspect.getsource(operate.kg_query)
        naive_source = inspect.getsource(operate.naive_query)

        assert kg_source.index("_build_query_context(") < kg_source.index(
            "cached_result = await handle_cache("
        )
        assert naive_source.index("_get_vector_context(") < naive_source.index(
            "cached_result = await handle_cache("
        )

    def test_recorded_hashes_match_installed_lightrag(self):
        """Independent recomputation (ast over the installed operate.py file,
        immune to runtime monkeypatching) must match the recorded baselines.
        Only meaningful on the versions we recorded — 1.4.9.11 (historical
        BNP pin, computed from the exact PyPI wheel) and 1.5.4–1.5.6 (the
        supported line; identical bodies, recomputation confirmed on 1.5.6)."""
        version = canary.installed_lightrag_version()
        if version not in ("1.4.9.11", "1.5.4", "1.5.5", "1.5.6"):
            pytest.skip(
                f"no recorded hash baseline for lightrag-hku {version} "
                "(only 1.4.9.11 and 1.5.4–1.5.6 were computed; the drift "
                "warning is the intended signal on other versions)"
            )
        import lightrag.operate as operate

        source = Path(operate.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        found: dict[str, str] = {}
        for node in ast.walk(tree):
            if (
                isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef))
                and node.name in canary.KNOWN_PRIVATE_COPY_SOURCE_HASHES
            ):
                segment = ast.get_source_segment(source, node)
                found[node.name] = hashlib.sha256(
                    "".join(segment.split()).encode("utf-8")
                ).hexdigest()

        assert set(found) == set(canary.KNOWN_PRIVATE_COPY_SOURCE_HASHES)
        for name, digest in found.items():
            assert digest in canary.KNOWN_PRIVATE_COPY_SOURCE_HASHES[name], (
                f"installed lightrag-hku {version} body of operate.{name} "
                f"hashes to {digest}, not recorded in "
                "canary.KNOWN_PRIVATE_COPY_SOURCE_HASHES"
            )


# ---------------------------------------------------------------------------
# Memgraph constructor — reviewed ABI/body + minimal delegating wrapper
# ---------------------------------------------------------------------------


class _ReviewedMemgraphStorage:
    def __init__(
        self, namespace, global_config, embedding_func, workspace=None
    ):  # pragma: no cover - signature fixture only
        self.workspace = workspace


class TestMemgraphInitCanary:
    @pytest.mark.parametrize(
        ("version", "digest"),
        [
            (
                "1.4.9.11",
                "feb3429c45ef0360e25900926b4a132abc6260f7eac6cfb5a5a43c9f398e622d",
            ),
            (
                "1.5.6",
                "a0c43427a1013f0d24f4e4ce1ad41558a5c13af7a50929faab419c32fb79b47b",
            ),
        ],
    )
    def test_recorded_signature_and_body_are_silent(
        self, version, digest, monkeypatch, caplog
    ):
        monkeypatch.setattr(canary, "installed_lightrag_version", lambda: version)
        monkeypatch.setattr(canary, "normalized_source_hash", lambda _fn: digest)

        with caplog.at_level(logging.WARNING, logger="twindb_lightrag_memgraph"):
            reviewed = canary.reviewed_memgraph_init(_ReviewedMemgraphStorage)

        assert reviewed is _ReviewedMemgraphStorage.__init__
        assert str(inspect.signature(reviewed)) in canary.KNOWN_MEMGRAPH_INIT_SIGNATURES
        assert _canary_messages(caplog) == []

    def test_reviewed_fingerprint_provenance_covers_recorded_versions(self):
        provenance = " ".join(canary.KNOWN_MEMGRAPH_INIT_SOURCE_HASHES.values())
        assert "1.4.9.11" in provenance
        assert "1.4.11" in provenance
        assert "1.4.12" in provenance
        assert "1.5.6" in provenance
        assert (
            "feb3429c45ef0360e25900926b4a132abc6260f7eac6cfb5a5a43c9f398e622d"
            in canary.KNOWN_MEMGRAPH_INIT_SOURCE_HASHES
        )
        assert (
            "a0c43427a1013f0d24f4e4ce1ad41558a5c13af7a50929faab419c32fb79b47b"
            in canary.KNOWN_MEMGRAPH_INIT_SOURCE_HASHES
        )

    def test_recorded_fingerprint_matches_installed_supported_wheel(self):
        version = canary.installed_lightrag_version()
        recorded_versions = {
            "1.4.9.11",
            "1.4.11",
            "1.4.12",
            "1.5.3",
            "1.5.4",
            "1.5.5",
            "1.5.6",
        }
        if version not in recorded_versions:
            pytest.skip(f"no Memgraph constructor baseline recorded for {version}")

        import lightrag.kg.memgraph_impl as memgraph_impl

        source = Path(memgraph_impl.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        storage_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "MemgraphStorage"
        )
        init_node = next(
            node
            for node in storage_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "__init__"
        )
        init_source = ast.get_source_segment(source, init_node)
        digest = hashlib.sha256(
            "".join(init_source.split()).encode("utf-8")
        ).hexdigest()

        assert digest in canary.KNOWN_MEMGRAPH_INIT_SOURCE_HASHES
        assert (
            str(inspect.signature(memgraph_impl.MemgraphStorage.__init__))
            in canary.KNOWN_MEMGRAPH_INIT_SIGNATURES
        )

    def test_reviewed_signature_with_unknown_body_warns_and_skips(
        self, monkeypatch, caplog
    ):
        monkeypatch.setattr(canary, "normalized_source_hash", lambda _fn: "f" * 64)

        with caplog.at_level(logging.WARNING, logger="twindb_lightrag_memgraph"):
            reviewed = canary.reviewed_memgraph_init(_ReviewedMemgraphStorage)

        assert reviewed is None
        msgs = _canary_messages(caplog)
        assert len(msgs) == 1
        assert "no reviewed constructor body" in msgs[0]
        assert "skipping" in msgs[0]

    def test_unreviewed_signature_warns_and_skips(self, caplog):
        class ChangedMemgraphStorage:
            def __init__(self, namespace, *, new_required):  # pragma: no cover
                self.namespace = namespace

        with caplog.at_level(logging.WARNING, logger="twindb_lightrag_memgraph"):
            reviewed = canary.reviewed_memgraph_init(ChangedMemgraphStorage)

        assert reviewed is None
        msgs = _canary_messages(caplog)
        assert len(msgs) == 1
        assert "unreviewed signature" in msgs[0]
        assert "skipping" in msgs[0]

    def test_unavailable_source_warns_and_skips(self, monkeypatch, caplog):
        monkeypatch.setattr(canary, "normalized_source_hash", lambda _fn: None)

        with caplog.at_level(logging.WARNING, logger="twindb_lightrag_memgraph"):
            reviewed = canary.reviewed_memgraph_init(_ReviewedMemgraphStorage)

        assert reviewed is None
        msgs = _canary_messages(caplog)
        assert len(msgs) == 1
        assert "source" in msgs[0]
        assert "unavailable" in msgs[0]
        assert "skipping" in msgs[0]

    def test_registry_does_not_replace_init_when_canary_skips(self, monkeypatch):
        from lightrag.kg.memgraph_impl import MemgraphStorage

        def unreviewed_init(self, *, changed):  # pragma: no cover
            self.changed = changed

        monkeypatch.setattr(MemgraphStorage, "__init__", unreviewed_init)
        monkeypatch.setattr(canary, "reviewed_memgraph_init", lambda _owner: None)
        monkeypatch.setattr(registry, "_patch_operate_hot_paths", lambda: None)

        registry._patch_builtin_memgraph_storage()

        assert MemgraphStorage.__init__ is unreviewed_init
