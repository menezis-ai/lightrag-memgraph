"""Boot-time canaries for the upstream LightRAG symbols ``register()`` patches.

``patches/registry.py`` monkey-patches a dozen upstream LightRAG symbols. An
upstream rename historically produced one of two failure modes (audit
2026-07-02, COMPAT-3/COMPAT-4):

* **crash-at-boot** — a bare attribute read (``operate.merge_nodes_and_edges``,
  ``LightRAG._insert_done``, ``create_document_routes``, ``create_app``) raises
  ``AttributeError`` inside ``register()`` and the whole instance serves
  nothing;
* **silent decoupling** — a ``setattr``-style patch keeps "succeeding" while
  upstream stopped calling the patched name (the SKEW-1 buffered-merge case),
  or the private copies of two ``operate`` internals drift from the upstream
  bodies they replicate (silent retrieval-quality regression).

This module converts both into explicit, classified behavior:

* **REQUIRED** symbols (the 3 ``lightrag.kg`` registry dicts): without them
  ``register()`` cannot register the storage backends at all. A missing one
  raises ``RuntimeError`` with an actionable message naming the symbol and the
  installed lightrag version. This does NOT make any today-working boot fail:
  the same absence crashed with a bare ``AttributeError``/``KeyError`` before.
* **DEGRADABLE** symbols (buffered-merge, ``_insert_done`` hook, document-routes
  capture, ``create_app`` overlay, unreviewed Memgraph constructor): a loud
  ``logger.warning`` naming the symbol and version, then the individual patch
  is SKIPPED and boot continues.
* **DRIFT** (private copies): a sha256 over the whitespace-normalized source of
  the upstream function is compared against recorded known-good hashes; an
  unknown hash logs a warning ("private copy may have drifted") and never
  fails.

Hard behavioral contract (auth-posture doctrine, 2026-06-10 BNP crash-loop):
when every symbol exists with the expected shape, the canary is a pure
read-only no-op — zero warnings, zero behavior change. It never turns a
today-working boot into a refusal.
"""

from __future__ import annotations

import hashlib
import inspect
import logging

logger = logging.getLogger("twindb_lightrag_memgraph")

_CANARY_PREFIX = "twindb canary:"

#: The lightrag.kg registry dicts register() mutates. Without any one of them
#: the storage backends cannot be registered — REQUIRED class.
REQUIRED_KG_REGISTRY_DICTS = (
    "STORAGE_IMPLEMENTATIONS",
    "STORAGE_ENV_REQUIREMENTS",
    "STORAGES",
)

#: STORAGE_IMPLEMENTATIONS keys register() indexes with a hardcoded shape
#: (``[key]["implementations"]`` list). Part of the REQUIRED contract.
STORAGE_IMPLEMENTATION_KEYS = (
    "KV_STORAGE",
    "VECTOR_STORAGE",
    "DOC_STATUS_STORAGE",
)

#: Known-good sha256 hashes of the whitespace-normalized source of the two
#: ``lightrag.operate`` internals that ``patches/registry.py`` carries private
#: copies of (``_fused_get_node_data`` / ``_fused_find_edges``). Keyed
#: ``function name -> {hash: provenance}``. An installed lightrag whose body
#: hashes to none of these triggers a drift warning (never a failure).
#:
#: Computed with :func:`normalized_source_hash` over:
#: * the exact PyPI wheel ``lightrag-hku==1.4.9.11`` (historical BNP pin);
#: * the locally installed ``lightrag-hku==1.5.4``, whose bodies are
#:   byte-identical on ``1.5.5`` and ``1.5.6`` — the current single supported
#:   pin (recomputed and confirmed against the exact 1.5.6 PyPI wheel).
#:
#: NB: ``_get_node_data`` differs between 1.4.9.11 and 1.5.4–1.5.6 (1.5.x
#: added the ``query_embedding`` passthrough parameter — the private copy
#: implements the 1.5.x body and is arg-compatible with both).
#: ``_find_most_related_edges_from_entities`` is byte-identical across all
#: recorded versions.
KNOWN_PRIVATE_COPY_SOURCE_HASHES: dict[str, dict[str, str]] = {
    "_get_node_data": {
        "ec4e925117576ef600aa0dc8ae6d474785425778471879aa9ee47a82970ebf9a": (
            "lightrag-hku 1.4.9.11 (PyPI wheel)"
        ),
        "063fe2a6dba05bc2472c3b0b132237e47d94ea4bc3c91dd24cc5fd1d80ab0a11": (
            "lightrag-hku 1.5.4 == 1.5.5 == 1.5.6 (identical body)"
        ),
    },
    "_find_most_related_edges_from_entities": {
        "de7f1680d7b215accaed9adaaeb6a95a3f60e5b070480f6af6d43cb05176d842": (
            "lightrag-hku 1.4.9.11 (PyPI wheel) == 1.5.4 == 1.5.5 == 1.5.6 "
            "(identical body)"
        ),
    },
}

#: Reviewed constructor contracts for ``MemgraphStorage.__init__``.  Unlike
#: the operate hot-path patches above, the Twin patch wraps and delegates to
#: this constructor instead of carrying a private copy.  We still record both
#: its call shape and body hashes because a future constructor may derive more
#: state from the environment-selected workspace before the wrapper can apply
#: the explicit per-instance value.
KNOWN_MEMGRAPH_INIT_SIGNATURES: dict[str, str] = {
    "(self, namespace, global_config, embedding_func, workspace=None)": (
        "lightrag-hku 1.4.9.11 / 1.4.11 / 1.4.12 / 1.5.3 / 1.5.4 / 1.5.5 / 1.5.6"
    ),
}

KNOWN_MEMGRAPH_INIT_SOURCE_HASHES: dict[str, str] = {
    "feb3429c45ef0360e25900926b4a132abc6260f7eac6cfb5a5a43c9f398e622d": (
        "lightrag-hku 1.4.9.11 / 1.4.11 / 1.4.12 (exact cached PyPI wheel bodies)"
    ),
    "a0c43427a1013f0d24f4e4ce1ad41558a5c13af7a50929faab419c32fb79b47b": (
        "lightrag-hku 1.5.3 / 1.5.4 / 1.5.5 / 1.5.6 "
        "(same body plus upstream validate_workspace call)"
    ),
}


def installed_lightrag_version() -> str:
    """Best-effort installed lightrag version for canary messages.

    ``importlib.metadata`` first (unaffected by ``_patch_version_string``),
    then ``lightrag.__version__`` with our ``+memgraph-*`` marker stripped.
    """
    try:
        from importlib.metadata import version

        return version("lightrag-hku")
    except Exception:  # pragma: no cover - metadata lookup is environment-bound
        try:
            import lightrag

            raw = str(getattr(lightrag, "__version__", "unknown"))
            return raw.split("+memgraph-")[0]
        except Exception:
            return "unknown"


def _owner_name(owner: object) -> str:
    return getattr(owner, "__name__", None) or repr(owner)


def assert_storage_registries(kg_module: object) -> None:
    """REQUIRED-class canary for the 3 ``lightrag.kg`` registry dicts.

    Raises ``RuntimeError`` with an actionable message when a dict is missing
    or ``STORAGE_IMPLEMENTATIONS`` lost the hardcoded shape register() indexes.
    Read-only — when everything is present this is a no-op.
    """
    version = installed_lightrag_version()
    missing = [
        name
        for name in REQUIRED_KG_REGISTRY_DICTS
        if not isinstance(getattr(kg_module, name, None), dict)
    ]
    if missing:
        raise RuntimeError(
            f"{_CANARY_PREFIX} lightrag.kg is missing the required storage "
            f"registry dict(s) {missing} in the installed lightrag-hku "
            f"{version}. register() cannot plug the Memgraph KV/Vector/"
            "DocStatus backends without them — the runtime would boot on "
            "LightRAG's default storages and silently write elsewhere. "
            "Pin the supported lightrag-hku 1.5.6, whose registry layout "
            "is known-good, or port "
            "twindb-lightrag-memgraph to the new registry layout."
        )

    impls = kg_module.STORAGE_IMPLEMENTATIONS
    for key in STORAGE_IMPLEMENTATION_KEYS:
        entry = impls.get(key)
        if not isinstance(entry, dict) or not isinstance(
            entry.get("implementations"), list
        ):
            raise RuntimeError(
                f"{_CANARY_PREFIX} lightrag.kg.STORAGE_IMPLEMENTATIONS[{key!r}] "
                "no longer has the {'implementations': [...]} shape in the "
                f"installed lightrag-hku {version}. register() cannot declare "
                "the Memgraph backends as valid implementations. Pin the "
                "supported known-good lightrag-hku 1.5.6 or port "
                "twindb-lightrag-memgraph to the new registry layout."
            )


def degradable_symbol(
    owner: object,
    attr: str,
    *,
    patch_name: str,
    call_args: tuple | None = None,
):
    """DEGRADABLE-class canary: return ``owner.attr`` or warn-and-``None``.

    ``call_args`` optionally probes the hardcoded call shape the patch relies
    on (e.g. the 2-arg ``_insert_done`` forward). A symbol whose signature can
    no longer bind those args is treated as absent (warn + skip) — applying
    the patch anyway would crash at call time, not at boot.

    Signature-introspection failures are treated as compatible: the canary
    must never degrade a boot that works today.
    """
    sym = getattr(owner, attr, None)
    if sym is None:
        logger.warning(
            "%s upstream symbol %s.%s is absent from the installed "
            "lightrag-hku %s — skipping the %s patch. The runtime keeps "
            "booting DEGRADED (native LightRAG behavior, without this Twin "
            "extension).",
            _CANARY_PREFIX,
            _owner_name(owner),
            attr,
            installed_lightrag_version(),
            patch_name,
        )
        return None
    if call_args is not None and not _accepts_call(sym, call_args):
        logger.warning(
            "%s upstream symbol %s.%s exists but no longer accepts the "
            "%d-argument call shape hardcoded by the %s patch "
            "(installed lightrag-hku %s) — skipping the patch. The runtime "
            "keeps booting DEGRADED (native LightRAG behavior, without this "
            "Twin extension).",
            _CANARY_PREFIX,
            _owner_name(owner),
            attr,
            len(call_args),
            patch_name,
            installed_lightrag_version(),
        )
        return None
    return sym


def _accepts_call(sym: object, call_args: tuple) -> bool:
    try:
        sig = inspect.signature(sym)
    except (TypeError, ValueError):
        return True  # introspection failed — assume compatible (status quo)
    try:
        sig.bind(*call_args)
    except TypeError:
        return False
    return True


def normalized_source_hash(fn: object) -> str | None:
    """sha256 over the whitespace-normalized source of ``fn`` (or ``None``)."""
    try:
        source = inspect.getsource(fn)
    except (OSError, TypeError):
        return None
    normalized = "".join(source.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def reviewed_memgraph_init(owner: object):
    """Return a compatible ``MemgraphStorage.__init__`` or warn-and-skip.

    The explicit-workspace patch calls the upstream constructor with the
    reviewed five-argument ABI, then corrects only ``self.workspace``.  An ABI
    change is therefore DEGRADABLE: leave the upstream constructor untouched
    rather than install a wrapper that could crash every graph-storage
    construction.  The multi-workspace intelligence engine separately checks
    that the wrapper is present and fails closed before constructing LightRAG.

    A source-body change, or unavailable source, is also DEGRADABLE: the wrapper
    is skipped because a new constructor could derive other workspace-dependent
    state before the wrapper corrects ``self.workspace``.  ``register()`` still
    boots on native behavior, while the multi-workspace engine checks the patch
    marker and fails closed.
    """
    init = getattr(owner, "__init__", None)
    if init is None:
        logger.warning(
            "%s upstream symbol %s.__init__ is absent from the installed "
            "lightrag-hku %s — skipping the explicit Memgraph workspace "
            "patch. Multi-workspace callers must fail closed.",
            _CANARY_PREFIX,
            _owner_name(owner),
            installed_lightrag_version(),
        )
        return None
    if getattr(init, "_twindb_explicit_workspace_patch", False):
        return init

    try:
        signature_text = str(inspect.signature(init))
    except (TypeError, ValueError):
        signature_text = None
    if (
        signature_text is not None
        and signature_text not in KNOWN_MEMGRAPH_INIT_SIGNATURES
    ):
        logger.warning(
            "%s upstream symbol %s.__init__ has unreviewed signature %s in "
            "the installed lightrag-hku %s — skipping the explicit Memgraph "
            "workspace patch. Expected one of %s; multi-workspace callers "
            "must fail closed.",
            _CANARY_PREFIX,
            _owner_name(owner),
            signature_text,
            installed_lightrag_version(),
            sorted(KNOWN_MEMGRAPH_INIT_SIGNATURES),
        )
        return None

    digest = normalized_source_hash(init)
    if digest is None:
        logger.warning(
            "%s source of %s.__init__ is unavailable in the installed "
            "lightrag-hku %s — skipping the explicit Memgraph workspace "
            "patch because its constructor body cannot be verified. "
            "Multi-workspace callers must fail closed.",
            _CANARY_PREFIX,
            _owner_name(owner),
            installed_lightrag_version(),
        )
        return None
    if digest not in KNOWN_MEMGRAPH_INIT_SOURCE_HASHES:
        logger.warning(
            "%s %s.__init__ in the installed lightrag-hku %s hashes to %s, "
            "which matches no reviewed constructor body — skipping the "
            "explicit Memgraph workspace patch. Review whether upstream "
            "derives other state from the environment-selected workspace, "
            "then record the new hash. Multi-workspace callers must fail "
            "closed.",
            _CANARY_PREFIX,
            _owner_name(owner),
            installed_lightrag_version(),
            digest,
        )
        return None
    return init


def warn_on_private_copy_drift(owner: object, attr: str) -> None:
    """COMPAT-3 canary for the two ``operate`` private copies. Warning-only.

    Called by ``_patch_operate_hot_paths`` *before* the fused replacements are
    installed. Skips silently when the attribute is already one of our own
    replacements (idempotent re-``register()`` in tests) and when the source
    is unavailable (frozen/zipped install).
    """
    fn = getattr(owner, attr, None)
    if fn is None:
        logger.warning(
            "%s lightrag.operate.%s is absent from the installed lightrag-hku "
            "%s — the fused private copy in patches/registry.py is installed "
            "anyway but upstream may no longer call it (silent decoupling; "
            "cf. audit 2026-07-02 COMPAT-3/SKEW-1).",
            _CANARY_PREFIX,
            attr,
            installed_lightrag_version(),
        )
        return
    module = getattr(fn, "__module__", "") or ""
    if module.startswith("twindb_lightrag_memgraph"):
        return  # already our replacement — first-pass verification stands
    digest = normalized_source_hash(fn)
    if digest is None:
        logger.debug(
            "%s source of lightrag.operate.%s unavailable — drift check skipped",
            _CANARY_PREFIX,
            attr,
        )
        return
    if digest not in KNOWN_PRIVATE_COPY_SOURCE_HASHES.get(attr, {}):
        logger.warning(
            "%s lightrag.operate.%s in the installed lightrag-hku %s hashes "
            "to %s, which matches no recorded known-good body — the private "
            "copy in patches/registry.py may have drifted from upstream. "
            "Diff the upstream function against the _fused_* replacement, "
            "then record the new hash in patches/canary.py.",
            _CANARY_PREFIX,
            attr,
            installed_lightrag_version(),
            digest,
        )
