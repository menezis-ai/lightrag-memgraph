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
  capture, ``create_app`` overlay): a loud ``logger.warning`` naming the symbol
  and version, then the individual patch is SKIPPED and boot continues.
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
#: * the exact PyPI wheel ``lightrag-hku==1.4.9.11`` (BNP prod pin);
#: * the locally installed ``lightrag-hku==1.5.4``.
#:
#: 1.4.11 / 1.4.12 (CI matrix) are deliberately absent — not computable from
#: this workstation. The CI matrix surfaces them through the drift warning;
#: once reviewed, record their hashes here.
#:
#: NB: ``_get_node_data`` differs between 1.4.9.11 and 1.5.4 (1.5.4 added the
#: ``query_embedding`` passthrough parameter — the private copy already
#: implements the 1.5.4 body and is arg-compatible with both).
#: ``_find_most_related_edges_from_entities`` is byte-identical across the two.
KNOWN_PRIVATE_COPY_SOURCE_HASHES: dict[str, dict[str, str]] = {
    "_get_node_data": {
        "ec4e925117576ef600aa0dc8ae6d474785425778471879aa9ee47a82970ebf9a": (
            "lightrag-hku 1.4.9.11 (PyPI wheel)"
        ),
        "063fe2a6dba05bc2472c3b0b132237e47d94ea4bc3c91dd24cc5fd1d80ab0a11": (
            "lightrag-hku 1.5.4"
        ),
    },
    "_find_most_related_edges_from_entities": {
        "de7f1680d7b215accaed9adaaeb6a95a3f60e5b070480f6af6d43cb05176d842": (
            "lightrag-hku 1.4.9.11 (PyPI wheel) == 1.5.4 (identical body)"
        ),
    },
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
            "Pin a lightrag-hku version that still exposes these dicts "
            "(1.4.9.11 / 1.4.11 / 1.4.12 are known-good) or port "
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
                "the Memgraph backends as valid implementations. Pin a "
                "known-good lightrag-hku (1.4.9.11 / 1.4.11 / 1.4.12) or port "
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
