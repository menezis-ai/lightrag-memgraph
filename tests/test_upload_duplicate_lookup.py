"""LightRAG-compat test for the upload duplicate-lookup cache.

``_patch_upload_duplicate_lookup`` replaces LightRAG's
``find_existing_file_by_file_path`` (an O(n) per-upload ``iterdir()`` scan) with
an mtime-keyed canonical-name index. Per ``docs/test-doctrine-lightrag-compat.md``
a patched LightRAG internal must behave IDENTICALLY to the native one — this
test pins that equivalence (a benchmark proves speed, not correctness).

Architecture note: the patch is applied at server-boot (from the create_app
wrapper), never at ``register()``-time — importing ``lightrag.api.*`` runs its
argv-based config init, which aborts when ``register()`` is imported under
pytest's argv. This test mirrors that by doing all lightrag imports inside the
fixture under a neutralized argv, so the module stays import-safe at collection.

Known limitation (documented, not a correctness bug for the BNP Linux target):
the cache is keyed on the input-dir ``st_mtime_ns``. On a coarse-mtime
filesystem a file added within the same mtime tick as the last index build can
be missed until the next dir change. The upload flow writes the file (bumping
dir mtime) before the dedupe lookup, so a single upload is always fresh; only
concurrent uploads on a coarse-mtime FS share the window.
"""

from __future__ import annotations

import sys

import pytest

# Set by the fixture, under an argv guard, before any test body runs.
dr = None


@pytest.fixture
def patched():
    """Import lightrag + apply the cache patch fresh (new index), then restore."""
    global dr
    saved_argv = sys.argv
    sys.argv = ["lightrag"]
    try:
        import lightrag.api.routers.document_routes as _dr

        from twindb_lightrag_memgraph.patches.registry import (
            _patch_upload_duplicate_lookup,
        )

        dr = _dr
        original = dr.find_existing_file_by_file_path
        had_flag = getattr(dr, "_twindb_upload_lookup_cached", False)
        dr._twindb_upload_lookup_cached = False  # force (re)patch
        _patch_upload_duplicate_lookup()
        patched_fn = dr.find_existing_file_by_file_path
    finally:
        sys.argv = saved_argv
    try:
        yield patched_fn
    finally:
        dr.find_existing_file_by_file_path = original
        dr._twindb_upload_lookup_cached = had_flag


def _native(input_dir, file_path):
    """Faithful copy of upstream find_existing_file_by_file_path."""
    if not file_path or file_path == dr.UNKNOWN_FILE_SOURCE:
        return None
    try:
        for candidate in input_dir.iterdir():
            if not candidate.is_file():
                continue
            if dr.normalize_file_path(candidate.name) == file_path:
                return candidate
    except FileNotFoundError:
        return None
    return None


def _same(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return str(a) == str(b)


def _assert_parity(patched_fn, input_dir, file_path):
    p = patched_fn(input_dir, file_path)
    n = _native(input_dir, file_path)
    assert _same(p, n), f"divergence for {file_path!r}: patched={p} native={n}"
    return p


def test_exact_match_parity(patched, tmp_path):
    (tmp_path / "report.pdf").write_text("x")
    result = _assert_parity(patched, tmp_path, "report.pdf")
    assert result is not None and result.name == "report.pdf"


def test_miss_parity(patched, tmp_path):
    (tmp_path / "report.pdf").write_text("x")
    assert _assert_parity(patched, tmp_path, "absent.pdf") is None


def test_unknown_and_empty_parity(patched, tmp_path):
    (tmp_path / "report.pdf").write_text("x")
    assert _assert_parity(patched, tmp_path, dr.UNKNOWN_FILE_SOURCE) is None
    assert _assert_parity(patched, tmp_path, "") is None


def test_non_file_entries_ignored_parity(patched, tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "report.pdf").write_text("x")
    # A directory named like the lookup must not match in either impl.
    (tmp_path / "folder.pdf").mkdir()
    assert _assert_parity(patched, tmp_path, "folder.pdf") is None
    assert _assert_parity(patched, tmp_path, "report.pdf") is not None


def test_noncanonical_file_path_parity(patched, tmp_path):
    """The fixed divergence: a whitespace/non-canonical file_path must NOT match
    (upstream compares against the raw file_path, so neither does)."""
    (tmp_path / "report.pdf").write_text("x")
    assert _assert_parity(patched, tmp_path, "  report.pdf  ") is None
    assert _assert_parity(patched, tmp_path, "report.pdf") is not None


def test_missing_input_dir_parity(patched, tmp_path):
    missing = tmp_path / "does-not-exist"
    assert _assert_parity(patched, missing, "report.pdf") is None


def test_cache_refreshes_on_dir_change(patched, tmp_path):
    # First lookup misses; after a new file lands (dir mtime bumps) the cache
    # rebuilds and the patched impl sees it — parity with native throughout.
    assert _assert_parity(patched, tmp_path, "late.pdf") is None
    (tmp_path / "late.pdf").write_text("x")
    found = _assert_parity(patched, tmp_path, "late.pdf")
    assert found is not None and found.name == "late.pdf"


def test_stale_positive_revalidated(patched, tmp_path):
    # File present then removed → both impls return None (the patched impl's
    # .is_file() revalidation must not hand back a path to a vanished file).
    target = tmp_path / "gone.pdf"
    target.write_text("x")
    assert _assert_parity(patched, tmp_path, "gone.pdf") is not None
    target.unlink()
    assert _assert_parity(patched, tmp_path, "gone.pdf") is None


def test_idempotent_patch(patched):
    # Re-applying must not double-wrap (flag guards it).
    from twindb_lightrag_memgraph.patches.registry import (
        _patch_upload_duplicate_lookup,
    )

    first = dr.find_existing_file_by_file_path
    _patch_upload_duplicate_lookup()
    assert dr.find_existing_file_by_file_path is first


def test_patch_wired_at_server_boot_not_register():
    """The optimization must be applied at server boot (inside the create_app
    wrapper), NOT at register-time — importing lightrag.api eagerly there breaks
    register() for callers under a non-server argv. Pin that structurally: the
    patch is called inside the create_app wrapper, after the native create_app
    runs, and is NOT called from register() itself.

    (The behavioural equivalent — driving the real wrapped create_app — is
    flaky under full-suite collection because prior tests desync the global
    lightrag server module identity; this structural check is deterministic.)
    """
    from pathlib import Path

    from twindb_lightrag_memgraph.patches import registry

    # Read the source FILE directly — runtime attributes on the module get
    # monkeypatched by other tests (the register() re-export sync propagates
    # those into this module), so inspect.getsource(<attr>) is unreliable here.
    src = Path(registry.__file__).read_text(encoding="utf-8")

    # The patch is applied inside the create_app wrapper, right after the native
    # create_app builds the app (so document_routes is imported under the
    # server's argv) — not at register-time.
    boot_idx = src.index("app = orig_create_app(args)")
    call_idx = src.index("_patch_upload_duplicate_lookup()", boot_idx)
    assert 0 < call_idx - boot_idx < 500, (
        "upload-lookup patch must be applied inside the create_app wrapper, "
        "just after the native create_app"
    )

    # And register() itself must NOT call it (eager lightrag.api import there
    # breaks register() for callers under a non-server argv, e.g. pytest).
    register_start = src.index("\ndef register(")
    register_end = src.index("\ndef ", register_start + 1)
    assert "_patch_upload_duplicate_lookup" not in src[register_start:register_end]
