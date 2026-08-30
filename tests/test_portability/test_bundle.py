"""T0.3 — bundle container: dir <-> tar.gz round-trip, hostile archives refused."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest

from twindb_lightrag_memgraph._constants import (
    DEFAULT_PORTABILITY_BATCH_SIZE,
    DEFAULT_PORTABILITY_MAX_BYTES,
    resolve_portability_batch_size,
    resolve_portability_dir,
    resolve_portability_max_bytes,
    validate_portability_env,
)
from twindb_lightrag_memgraph.portability.bundle import (
    MANIFEST_NAME,
    BundleError,
    BundleReader,
    BundleWriter,
    inspect_bundle,
)
from twindb_lightrag_memgraph.portability.manifest import (
    Manifest,
    state_hash_of_entries,
)
from twindb_lightrag_memgraph.portability.stores import exportable_stores

from ._fixtures import manifest_dict


def _build(root: Path, *, with_png: bool = True) -> tuple[Manifest, BundleWriter]:
    writer = BundleWriter(root)
    for spec in exportable_stores(include_procedures=with_png):
        assert spec.file is not None
        with writer.open_jsonl(spec.file, store=spec.name) as jsonl:
            if spec.name == "docstatus":
                jsonl.write({"id": "doc-1", "status": "processed"})
                jsonl.write({"id": "doc-2", "status": "processed"})
            elif spec.name == "tags":
                jsonl.write({"folder_id": "f1", "id": "t1", "value": {"tag": "t1"}})
    if with_png:
        writer.add_file(
            "files/procedures/x/1.png",
            b"\x89PNG\r\n\x1a\n" + b"0" * 32,
            store="procedures",
        )
    entries = writer.entries
    data = manifest_dict(scope={"include_procedures": with_png})
    data["files"] = [e.__dict__ for e in entries]
    data["state_hash"] = state_hash_of_entries(entries)
    from twindb_lightrag_memgraph.portability.canonical import jcs_sha256

    data["manifest_hash"] = jcs_sha256(
        {k: v for k, v in data.items() if k != "manifest_hash"}
    )
    return writer.finalize(Manifest.from_dict(data)), writer


def _no_residue(parent: Path) -> None:
    assert not [p for p in parent.iterdir() if p.name.startswith(".kb-bundle-")], list(
        parent.iterdir()
    )


def test_dir_and_archive_round_trip(tmp_path):
    manifest, writer = _build(tmp_path / "bundle")
    assert (tmp_path / "bundle" / MANIFEST_NAME).is_file()
    assert oct((tmp_path / "bundle").stat().st_mode & 0o777) == "0o700"
    assert oct((tmp_path / "bundle" / MANIFEST_NAME).stat().st_mode & 0o777) == "0o600"
    insp = inspect_bundle(tmp_path / "bundle")
    assert insp.ok, insp.problems
    archive = writer.archive(tmp_path / "out" / "kb.tar.gz")
    with tarfile.open(archive) as tar:
        names = tar.getnames()
    assert names[0] == MANIFEST_NAME and names[1:] == sorted(names[1:])
    with BundleReader(archive) as reader:
        insp2 = reader.inspect()
        assert insp2.ok, insp2.problems
        assert insp2.manifest == manifest
        assert (
            reader.path_of("memgraph/docstatus.jsonl").read_bytes()
            == (tmp_path / "bundle/memgraph/docstatus.jsonl").read_bytes()
        )
        assert insp2.as_dict()["workspace"] == "base"
    _no_residue(tmp_path / "out")


def test_writer_refuses_non_empty_dir_duplicate_path_and_unlisted_member(tmp_path):
    (tmp_path / "used").mkdir()
    (tmp_path / "used" / "x").write_text("x")
    with pytest.raises(BundleError, match="not empty"):
        BundleWriter(tmp_path / "used")
    writer = BundleWriter(tmp_path / "b")
    writer.open_jsonl("memgraph/a.jsonl", store="kv").close()
    with pytest.raises(BundleError, match="duplicate"):
        writer.open_jsonl("memgraph/a.jsonl", store="kv")
    with pytest.raises(BundleError, match="files/ plane"):
        writer.add_file("memgraph/b.bin", b"x", store="kv")
    data = manifest_dict()  # lists members the writer never wrote
    from twindb_lightrag_memgraph.portability.canonical import jcs_sha256

    data["manifest_hash"] = jcs_sha256(
        {k: v for k, v in data.items() if k != "manifest_hash"}
    )
    with pytest.raises(BundleError, match="exactly the written members"):
        writer.finalize(Manifest.from_dict(data))
    with pytest.raises(BundleError, match="finalize"):
        writer.archive(tmp_path / "x.tar.gz")


def test_directory_bundle_reports_stray_and_tampered_files(tmp_path):
    _build(tmp_path / "bundle")
    (tmp_path / "bundle" / "overlay" / "stray.jsonl").write_text("{}\n")
    insp = inspect_bundle(tmp_path / "bundle")
    assert not insp.ok and any("not listed" in p for p in insp.problems)
    (tmp_path / "bundle" / "overlay" / "stray.jsonl").unlink()
    target = tmp_path / "bundle" / "memgraph" / "docstatus.jsonl"
    target.write_bytes(target.read_bytes().replace(b"doc-2", b"doc-3"))
    insp = inspect_bundle(tmp_path / "bundle")
    assert not insp.ok and any("hash/size mismatch" in p for p in insp.problems)
    target.unlink()
    insp = inspect_bundle(tmp_path / "bundle")
    assert not insp.ok and any("missing file" in p for p in insp.problems)


def test_directory_bundle_rejects_symlinked_parent_component(tmp_path):
    _build(tmp_path / "bundle")
    root = tmp_path / "bundle"
    external = tmp_path / "external-memgraph"
    (root / "memgraph").rename(external)
    (root / "memgraph").symlink_to(external, target_is_directory=True)

    inspection = inspect_bundle(root)
    assert not inspection.ok
    assert any("symlink path component: memgraph" in p for p in inspection.problems)
    with BundleReader(root) as reader:
        with pytest.raises(BundleError, match="symlink path component"):
            reader.path_of("memgraph/docstatus.jsonl")


def _tar_from(root: Path, out: Path, *, mutate=None) -> Path:
    """Repack a bundle dir, letting *mutate(tar)* inject hostile members."""
    with tarfile.open(out, "w:gz") as tar:
        tar.add(root / MANIFEST_NAME, arcname=MANIFEST_NAME, recursive=False)
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.name != MANIFEST_NAME:
                tar.add(p, arcname=p.relative_to(root).as_posix(), recursive=False)
        if mutate:
            mutate(tar)
    return out


def _add_bytes(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def test_hostile_archives_are_refused_without_residue(tmp_path):
    _build(tmp_path / "bundle")
    out = tmp_path / "out"
    out.mkdir()
    root = tmp_path / "bundle"

    # 1. a member outside the manifest (also covers ../ escapes: such a path
    #    can never be listed, so it is refused before the data filter runs)
    a1 = _tar_from(
        root,
        out / "unlisted.tar.gz",
        mutate=lambda t: _add_bytes(t, "overlay/evil.jsonl", b"{}\n"),
    )
    with pytest.raises(BundleError, match="not listed"):
        BundleReader(a1).__enter__()
    a1b = _tar_from(
        root,
        out / "escape.tar.gz",
        mutate=lambda t: _add_bytes(t, "../escape.txt", b"x"),
    )
    with pytest.raises(BundleError):
        BundleReader(a1b).__enter__()

    # 2. a symlink in place of a listed member
    def _symlink(t: tarfile.TarFile) -> None:
        info = tarfile.TarInfo("overlay/tags.jsonl")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        t.addfile(info)

    a2 = _tar_from(root, out / "symlink.tar.gz", mutate=_symlink)
    with pytest.raises(BundleError, match="not a regular file"):
        BundleReader(a2).__enter__()

    # 3. manifest not first
    a3 = out / "manifest-last.tar.gz"
    with tarfile.open(a3, "w:gz") as tar:
        tar.add(root / "memgraph/docstatus.jsonl", arcname="memgraph/docstatus.jsonl")
        tar.add(root / MANIFEST_NAME, arcname=MANIFEST_NAME)
    with pytest.raises(BundleError, match="first archive member"):
        BundleReader(a3).__enter__()

    # 4. decompressed size above the ceiling (member by member)
    a4 = _tar_from(root, out / "big.tar.gz")
    with pytest.raises(BundleError, match="TWIN_PORTABILITY_MAX_BYTES"):
        BundleReader(a4, max_bytes=200).__enter__()

    # 5. a tampered member: extraction succeeds, inspect() reports the mismatch
    tampered = tmp_path / "tampered"
    import shutil

    shutil.copytree(root, tampered)
    f = tampered / "memgraph/docstatus.jsonl"
    f.write_bytes(f.read_bytes().replace(b"doc-2", b"doc-9"))
    a5 = _tar_from(tampered, out / "tampered.tar.gz")
    with pytest.raises(BundleError, match="size differs"):
        # size is checked from the tar header before extraction …
        BundleReader(
            _tar_from(
                tampered,
                out / "tampered2.tar.gz",
                mutate=lambda t: _add_bytes(t, "overlay/tags.jsonl", b"{}\n"),
            )
        ).__enter__()
    with BundleReader(a5) as reader:
        insp = reader.inspect()
    assert not insp.ok and any("hash/size mismatch" in p for p in insp.problems)

    # 6. not an archive at all
    (out / "junk.tar.gz").write_bytes(b"not a tarball")
    with pytest.raises(BundleError, match="tar.gz"):
        BundleReader(out / "junk.tar.gz").__enter__()
    with pytest.raises(BundleError, match="no such bundle"):
        BundleReader(out / "absent.tar.gz").__enter__()

    _no_residue(out)


def test_portability_env_knobs(monkeypatch):
    for name in (
        "TWIN_PORTABILITY_MAX_BYTES",
        "TWIN_PORTABILITY_BATCH_SIZE",
        "TWIN_PORTABILITY_DIR",
    ):
        monkeypatch.delenv(name, raising=False)
    assert resolve_portability_max_bytes() == DEFAULT_PORTABILITY_MAX_BYTES
    assert resolve_portability_batch_size() == DEFAULT_PORTABILITY_BATCH_SIZE
    monkeypatch.setenv("WORKING_DIR", "/srv/kb")
    assert resolve_portability_dir() == "/srv/kb/portability"
    monkeypatch.setenv("TWIN_PORTABILITY_DIR", "/mnt/bundles")
    assert resolve_portability_dir() == "/mnt/bundles"
    monkeypatch.setenv("TWIN_PORTABILITY_MAX_BYTES", "1024")
    monkeypatch.setenv("TWIN_PORTABILITY_BATCH_SIZE", "50")
    validate_portability_env()
    assert (resolve_portability_max_bytes(), resolve_portability_batch_size()) == (
        1024,
        50,
    )
    for bad in ("0", "-1", "abc", "1.5"):
        monkeypatch.setenv("TWIN_PORTABILITY_BATCH_SIZE", bad)
        with pytest.raises(ValueError, match="TWIN_PORTABILITY_BATCH_SIZE"):
            validate_portability_env()
