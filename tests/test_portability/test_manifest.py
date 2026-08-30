"""T0.1 — manifest contract: round-trip, hash invariants, strict reader, schema."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from twindb_lightrag_memgraph.portability.canonical import (
    CanonicalisationError,
    es6_number,
    jcs_dumps,
    jcs_loads,
)
from twindb_lightrag_memgraph.portability.manifest import (
    EMBEDDING_PROBE_TEXTS,
    FileEntry,
    Manifest,
    ManifestError,
    state_hash_of,
    validate_bundle_path,
)

from ._fixtures import SHA_A, manifest_dict, sealed_manifest_dict

SCHEMA = json.loads(
    (
        Path(__file__).resolve().parents[2]
        / "docs/templates/kb-bundle.manifest.schema.json"
    ).read_text()
)
_VALIDATOR = jsonschema.Draft202012Validator(SCHEMA)


def _schema_errors(data: dict) -> list[str]:
    return [e.message for e in _VALIDATOR.iter_errors(data)]


def _reseal(data: dict) -> dict:
    from twindb_lightrag_memgraph.portability.canonical import jcs_sha256

    data["state_hash"] = state_hash_of(
        {entry["path"]: entry["sha256"] for entry in data["files"]}
    )
    data["manifest_hash"] = jcs_sha256(
        {key: value for key, value in data.items() if key != "manifest_hash"}
    )
    return data


# ------------------------------------------------------------------ canonical


def test_jcs_sorts_keys_by_utf16_and_keeps_unicode():
    assert (
        jcs_dumps({"é": 1, "z": 2, "a": [True, None, 1.5]})
        == '{"a":[true,null,1.5],"z":2,"é":1}'
    )


def test_jcs_refuses_nan_and_unknown_types():
    with pytest.raises(CanonicalisationError):
        jcs_dumps({"x": float("nan")})
    with pytest.raises(CanonicalisationError):
        jcs_dumps({"x": object()})
    with pytest.raises(ValueError):
        jcs_loads('{"x": NaN}')


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.0, "0"),
        (-0.0, "0"),
        (1.0, "1"),
        (1e21, "1e+21"),
        (1e16, "10000000000000000"),
        (1e-7, "1e-7"),
        (0.000001, "0.000001"),
        (5e-324, "5e-324"),
        (-2.5e-10, "-2.5e-10"),
        (0.30000000000000004, "0.30000000000000004"),
    ],
)
def test_es6_number_layout(value, expected):
    assert es6_number(value) == expected


def test_probe_texts_are_three_fixed_sentences():
    assert len(EMBEDDING_PROBE_TEXTS) == 3
    assert any(t.rstrip().endswith("?") for t in EMBEDDING_PROBE_TEXTS)
    assert any(t.startswith("Passage") for t in EMBEDDING_PROBE_TEXTS)


# ------------------------------------------------------------------- manifest


def test_round_trip_and_hashes():
    m = Manifest.from_dict(sealed_manifest_dict())
    text = m.to_json()
    again = Manifest.from_json(text)
    assert again == m
    assert text.endswith("\n") and again.to_json() == text
    assert m.manifest_hash == m.compute_manifest_hash()


def test_state_hash_ignores_volatile_fields_and_files_plane():
    base = Manifest.from_dict(sealed_manifest_dict())
    other = Manifest.from_dict(
        sealed_manifest_dict(
            bundle_id="ffffffff-0000-4000-8000-000000000000",
            created_at="2030-01-01T00:00:00Z",
            created_by={"actor": "someone-else"},
            consistency={"status": "unverified", "pipeline_idle": False},
        )
    )
    assert other.state_hash == base.state_hash
    assert other.manifest_hash != base.manifest_hash
    # the files/ plane never enters the state hash
    assert state_hash_of(
        {"memgraph/a.jsonl": SHA_A, "files/x.png": "0" * 64}
    ) == state_hash_of({"memgraph/a.jsonl": SHA_A})


def test_manifest_hash_changes_when_a_file_digest_changes():
    data = sealed_manifest_dict()
    data["files"][0]["sha256"] = "d" * 64
    with pytest.raises(ManifestError, match="manifest_hash"):
        Manifest.from_dict(data)
    resealed = sealed_manifest_dict()
    resealed["files"][0]["sha256"] = "d" * 64
    # re-seal both hashes: state hash must follow the file digest too
    body = {
        k: v for k, v in resealed.items() if k not in ("manifest_hash", "state_hash")
    }
    body["state_hash"] = state_hash_of(
        {f["path"]: f["sha256"] for f in resealed["files"]}
    )
    from twindb_lightrag_memgraph.portability.canonical import jcs_sha256

    body["manifest_hash"] = jcs_sha256(body)
    assert (
        Manifest.from_dict(body).manifest_hash
        != Manifest.from_dict(sealed_manifest_dict()).manifest_hash
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"classification": {"max_detected": None, "unknown_present": True}},
        {
            "scope": {"include_activity": True, "include_procedures": True},
            "source": {"memgraph": {"mage": "present", "version": ""}},
        },
    ],
)
def test_valid_manifests_pass_reader_and_schema(overrides):
    data = sealed_manifest_dict(**overrides)
    Manifest.from_dict(data)
    assert _schema_errors(data) == []


@pytest.mark.parametrize(
    ("overrides", "reason", "schema_catches"),
    [
        ({"format": "twin-catalog-bundle"}, "format", True),
        ({"format_version": "2.0"}, "format_version", True),
        (
            {
                "files": [
                    {
                        "path": "memgraph/../x.jsonl",
                        "store": "kv",
                        "records": 0,
                        "sha256": SHA_A,
                        "bytes": 0,
                    }
                ]
            },
            "path",
            True,
        ),
        (
            {
                "files": [
                    {
                        "path": "memgraph/kv.jsonl",
                        "store": "kv",
                        "records": 0,
                        "bytes": 0,
                    }
                ]
            },
            "sha256",
            True,
        ),
        ({"source": {"instance": "about-payload"}}, "source", True),
        # vector length == embedding.dim is a cross-field rule JSON Schema cannot
        # express: only the Python reader refuses it (the schema is a mirror, the
        # reader is the authority).
        (
            {"embedding": {"probe": {"vectors": [[0.1, 0.2], [0.1, 0.2], [0.1, 0.2]]}}},
            "probe-dim",
            False,
        ),
        (
            {"embedding": {"probe": {"vectors": [[0.1] * 4, [0.1] * 4]}}},
            "probe-cardinality",
            True,
        ),
    ],
)
def test_invalid_manifests_are_refused_by_reader_and_schema(
    overrides, reason, schema_catches
):
    data = sealed_manifest_dict(**overrides)
    with pytest.raises(ManifestError):
        Manifest.from_dict(data)
    assert bool(_schema_errors(data)) is schema_catches, reason


def test_reader_refuses_unknown_top_level_key_and_non_object():
    data = sealed_manifest_dict()
    data["extra"] = 1
    with pytest.raises(ManifestError, match="extra"):
        Manifest.from_dict(data)
    with pytest.raises(ManifestError):
        Manifest.from_dict([])
    with pytest.raises(ManifestError):
        Manifest.from_json("{not json")


@pytest.mark.parametrize(
    "path", ["memgraph/kv.full_docs.jsonl", "files/procedures/abc/1.png"]
)
def test_bundle_path_accepts_planes(path):
    assert validate_bundle_path(path) == path


@pytest.mark.parametrize(
    "path",
    [
        "../x",
        "memgraph//x",
        "memgraph/./x",
        "etc/passwd",
        "memgraph/X.jsonl",
        "memgraph/",
    ],
)
def test_bundle_path_refuses_escapes(path):
    with pytest.raises(ManifestError):
        validate_bundle_path(path)


def test_file_entry_is_frozen_and_reader_rejects_duplicate_paths():
    entry = FileEntry(
        path="memgraph/a.jsonl", store="kv", records=1, sha256=SHA_A, bytes=1
    )
    with pytest.raises(AttributeError):
        entry.records = 2  # type: ignore[misc]
    data = manifest_dict()
    data["files"].append(dict(data["files"][0]))
    with pytest.raises(ManifestError, match="duplicate"):
        Manifest.from_dict(data)


def test_manifest_refuses_missing_and_empty_store_rosters():
    missing = sealed_manifest_dict()
    missing["files"].pop()
    with pytest.raises(ManifestError, match="missing canonical store"):
        Manifest.from_dict(_reseal(missing))

    empty = sealed_manifest_dict(files=[])
    with pytest.raises(ManifestError, match="missing canonical store"):
        Manifest.from_dict(_reseal(empty))


def test_manifest_refuses_unexpected_store_and_path_store_mismatch():
    unexpected = sealed_manifest_dict()
    unexpected["files"].append(
        {
            "path": "overlay/unexpected.jsonl",
            "store": "unexpected",
            "records": 0,
            "sha256": SHA_A,
            "bytes": 0,
        }
    )
    with pytest.raises(ManifestError, match="unexpected canonical store"):
        Manifest.from_dict(_reseal(unexpected))

    mismatch = sealed_manifest_dict()
    mismatch["files"][0]["store"] = "tags"
    with pytest.raises(ManifestError, match="must declare store"):
        Manifest.from_dict(_reseal(mismatch))


def test_manifest_optional_flags_and_procedure_file_contract():
    activity_missing = sealed_manifest_dict(
        scope={"include_activity": True, "include_procedures": False}
    )
    activity_missing["files"] = [
        entry for entry in activity_missing["files"] if entry["store"] != "activity"
    ]
    with pytest.raises(ManifestError, match="missing canonical store"):
        Manifest.from_dict(_reseal(activity_missing))

    procedure = sealed_manifest_dict(
        scope={"include_activity": False, "include_procedures": True}
    )
    procedure["files"].append(
        {
            "path": "files/procedures/id/not-a-number.png",
            "store": "procedures",
            "records": 0,
            "sha256": SHA_A,
            "bytes": 1,
        }
    )
    with pytest.raises(ManifestError, match="invalid procedure file"):
        Manifest.from_dict(_reseal(procedure))
