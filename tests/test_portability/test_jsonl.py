"""T0.2 — canonical JSONL: byte determinism, integrity, streaming memory."""

from __future__ import annotations

import tracemalloc

import pytest

from twindb_lightrag_memgraph.portability.jsonl import (
    IntegrityError,
    JsonlWriter,
    iter_jsonl,
    sha256_of_file,
)


def test_same_records_different_key_order_are_byte_identical(tmp_path):
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    with JsonlWriter(a, store="kv", bundle_path="memgraph/a.jsonl") as w:
        w.write({"id": "1", "value": {"z": 1, "a": [1.5, "é"]}})
    with JsonlWriter(b, store="kv", bundle_path="memgraph/b.jsonl") as w:
        w.write({"value": {"a": [1.5, "é"], "z": 1}, "id": "1"})
    assert a.read_bytes() == b.read_bytes()
    entry = JsonlWriter(a, store="kv", bundle_path="memgraph/a.jsonl").close()
    assert (
        entry.records == 0
    )  # a fresh writer truncates: close() reports its own stream


def test_writer_entry_matches_file_and_reader_verifies_digest(tmp_path):
    path = tmp_path / "x.jsonl"
    with JsonlWriter(path, store="tags", bundle_path="overlay/tags.jsonl") as w:
        for i in range(5):
            w.write({"id": str(i), "n": i})
        entry = w.close()
    digest, size = sha256_of_file(path)
    assert (entry.sha256, entry.bytes, entry.records, entry.store) == (
        digest,
        size,
        5,
        "tags",
    )
    assert [r["n"] for r in iter_jsonl(path, entry.sha256)] == [0, 1, 2, 3, 4]
    with pytest.raises(IntegrityError, match="sha256"):
        list(iter_jsonl(path, "0" * 64))


def test_reader_refuses_non_canonical_and_non_object_lines(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_bytes(b'{"b":1,"a":2}\n')
    with pytest.raises(IntegrityError, match="canonical"):
        list(iter_jsonl(path))
    path.write_bytes(b"[1]\n")
    with pytest.raises(IntegrityError, match="not an object"):
        list(iter_jsonl(path))
    path.write_bytes(b'{"a":1}')
    with pytest.raises(IntegrityError, match="newline"):
        list(iter_jsonl(path))
    path.write_bytes(b'{"a":NaN}\n')
    with pytest.raises(IntegrityError, match="invalid JSON"):
        list(iter_jsonl(path))


def test_writer_rejects_non_dict_and_use_after_close(tmp_path):
    w = JsonlWriter(tmp_path / "w.jsonl", store="kv", bundle_path="memgraph/w.jsonl")
    with pytest.raises(TypeError):
        w.write(["not", "a", "dict"])  # type: ignore[arg-type]
    w.close()
    with pytest.raises(ValueError):
        w.write({"id": "x"})


def test_streaming_100k_lines_stays_under_20_mib(tmp_path):
    path = tmp_path / "big.jsonl"
    payload = {"id": "", "value": {"content": "x" * 200, "meta": [1, 2, 3]}}
    tracemalloc.start()
    with JsonlWriter(path, store="kv", bundle_path="memgraph/big.jsonl") as w:
        for i in range(100_000):
            payload["id"] = f"{i:08d}"
            w.write(payload)
        entry = w.close()
    n = sum(1 for _ in iter_jsonl(path, entry.sha256))
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert n == 100_000
    assert peak < 20 * 1024 * 1024, peak
