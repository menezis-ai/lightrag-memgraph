"""Contract tests for the private release SBOM evidence tooling."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "release_sbom.py"
GENERATOR_PATH = REPO_ROOT / "scripts" / "generate_release_sbom.sh"
DOCKERFILE_PATH = REPO_ROOT / "Dockerfile"
RUNBOOK_PATH = REPO_ROOT / "docs" / "operations" / "install-runbook.md"
IMAGE_REF = "registry.example/twin-kms@sha256:" + "a" * 64


def _load_release_sbom_module():
    spec = importlib.util.spec_from_file_location("release_sbom", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


release_sbom = _load_release_sbom_module()


def _image_document(*, serial: str, timestamp: str) -> dict:
    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.6",
        "serialNumber": serial,
        "metadata": {
            "timestamp": timestamp,
            "component": {
                "type": "container",
                "name": "twin-kms-runtime",
                "version": "sha256:abc",
                "bom-ref": "runtime-image",
            },
            "tools": {
                "components": [
                    {
                        "type": "application",
                        "name": "syft",
                        "version": "1.31.0",
                        "bom-ref": serial,
                    }
                ]
            },
        },
        "components": [
            {
                "type": "library",
                "name": "requests",
                "version": "2.32.5",
                "purl": "pkg:pypi/requests@2.32.5",
                "bom-ref": "image-requests",
                "hashes": [{"alg": "SHA-256", "content": "b" * 64}],
                "evidence": {"occurrences": [{"location": "/usr/lib/python/requests"}]},
            }
        ],
        "dependencies": [{"ref": "runtime-image", "dependsOn": ["image-requests"]}],
    }


def _webui_document() -> dict:
    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.6",
        "metadata": {"tools": [{"name": "syft", "version": "1.31.0"}]},
        "components": [
            {
                "type": "library",
                "name": "requests",
                "version": "2.32.5",
                "purl": "pkg:pypi/requests@2.32.5",
                "bom-ref": "webui-duplicate",
                "licenses": [{"license": {"id": "Apache-2.0"}}],
            },
            {
                "type": "library",
                "name": "react",
                "version": "19.1.1",
                "purl": "pkg:npm/react@19.1.1",
                "bom-ref": "webui-react",
                "licenses": [{"license": {"id": "MIT"}}],
            },
        ],
        "dependencies": [{"ref": "webui-react", "dependsOn": ["webui-duplicate"]}],
    }


def test_merge_is_byte_reproducible_and_preserves_package_evidence():
    first = release_sbom.merge_documents(
        [
            _image_document(serial="urn:uuid:first", timestamp="2026-08-31T08:00:00Z"),
            _webui_document(),
        ],
        image_ref=IMAGE_REF,
        version="1.2.0",
    )
    second = release_sbom.merge_documents(
        [
            _image_document(serial="urn:uuid:second", timestamp="2030-01-01T00:00:00Z"),
            _webui_document(),
        ],
        image_ref=IMAGE_REF,
        version="1.2.0",
    )

    assert release_sbom._canonical_json(first) == release_sbom._canonical_json(second)
    release_sbom.verify_document(first)
    components = {
        component["purl"]: component
        for component in first["components"]
        if "purl" in component
    }
    requests = components["pkg:pypi/requests@2.32.5"]
    assert requests["hashes"] == [{"alg": "SHA-256", "content": "b" * 64}]
    assert requests["licenses"] == [{"license": {"id": "Apache-2.0"}}]
    assert "evidence" not in requests
    assert "pkg:npm/react@19.1.1" in components
    assert first["serialNumber"].startswith("urn:uuid:")
    assert first["metadata"]["component"]["properties"][0]["value"] == IMAGE_REF


@pytest.mark.parametrize(
    "unsafe_value",
    [
        "/Users/alice/project/package.json",
        "/root/private-build/package.json",
        "API_KEY=top-secret",
        "-----BEGIN PRIVATE KEY-----",
    ],
)
def test_verify_rejects_local_paths_and_secret_like_material(unsafe_value):
    document = release_sbom.merge_documents(
        [_image_document(serial="urn:uuid:first", timestamp="ignored")],
        image_ref=IMAGE_REF,
        version="1.2.0",
    )
    document["components"][0]["description"] = unsafe_value

    with pytest.raises(release_sbom.SbomError):
        release_sbom.verify_document(document)


@pytest.mark.parametrize(
    "property_name",
    [
        "OPENAI_API_KEY",
        "TOKEN_SECRET",
        "client_secret",
        "access_token",
    ],
)
def test_verify_rejects_sensitive_cyclonedx_name_value_properties(property_name):
    document = release_sbom.merge_documents(
        [_image_document(serial="urn:uuid:first", timestamp="ignored")],
        image_ref=IMAGE_REF,
        version="1.2.0",
    )
    document["components"][0]["properties"] = [
        {"name": property_name, "value": "live-secret-value"}
    ]

    with pytest.raises(release_sbom.SbomError, match="sensitive property name"):
        release_sbom.verify_document(document)


@pytest.mark.parametrize("field_name", ["secret", "TOKEN_SECRET", "auth_token"])
def test_verify_rejects_direct_sensitive_dictionary_keys(field_name):
    document = release_sbom.merge_documents(
        [_image_document(serial="urn:uuid:first", timestamp="ignored")],
        image_ref=IMAGE_REF,
        version="1.2.0",
    )
    document["components"][0][field_name] = "live-secret-value"

    with pytest.raises(release_sbom.SbomError, match="sensitive field name"):
        release_sbom.verify_document(document)


def test_evidence_archive_is_deterministic(tmp_path):
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (evidence / "manifest.txt").write_text("result=pass\n", encoding="utf-8")
    (evidence / "sbom.json").write_text('{"bomFormat":"CycloneDX"}\n')
    (evidence / "stale-previous-run.txt").write_text("must not ship\n")
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"
    members = ["manifest.txt", "sbom.json"]

    release_sbom.archive_evidence(evidence, first, members)
    release_sbom.archive_evidence(evidence, second, members)

    assert first.read_bytes() == second.read_bytes()
    with zipfile.ZipFile(first) as archive:
        assert archive.namelist() == ["manifest.txt", "sbom.json"]
        assert {entry.date_time for entry in archive.infolist()} == {
            (1980, 1, 1, 0, 0, 0)
        }


def test_evidence_archive_rejects_missing_or_escaping_allowlist_member(tmp_path):
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (evidence / "manifest.txt").write_text("result=pass\n", encoding="utf-8")

    with pytest.raises(release_sbom.SbomError, match="not a file"):
        release_sbom.archive_evidence(
            evidence, tmp_path / "missing.zip", ["missing.txt"]
        )
    with pytest.raises(release_sbom.SbomError, match="Invalid"):
        release_sbom.archive_evidence(
            evidence, tmp_path / "escape.zip", ["../manifest.txt"]
        )


def test_generator_contract_covers_both_surfaces_and_fresh_image_smoke():
    script = GENERATOR_PATH.read_text(encoding="utf-8")
    dockerfile = DOCKERFILE_PATH.read_text(encoding="utf-8")

    assert script.count('"${SYFT_BIN}" scan') == 2
    assert 'scan "${IMAGE_REF}"' in script
    assert 'scan "dir:${WEBUI_SOURCE_DIR}"' in script
    assert 'git archive "${RELEASE_COMMIT}"' in script
    assert "org.opencontainers.image.revision" in script
    assert 'IMAGE_REVISION}" != "${RELEASE_COMMIT}' in script
    assert "cmp " in script
    assert "docker run --rm" in script
    assert "fresh_image_smoke=pass" in script
    assert '"${NORMALIZER}" archive' in script
    assert script.count("--file ") == 3
    assert 'mv "${STAGING_DIR}" "${EVIDENCE_DIR}"' in script
    assert "ARG TWIN_RELEASE_COMMIT=unversioned" in dockerfile
    assert 'org.opencontainers.image.revision="${TWIN_RELEASE_COMMIT}"' in dockerfile


def test_install_runbook_targets_one_runtime_and_only_real_log_variables():
    runbook = RUNBOOK_PATH.read_text(encoding="utf-8")

    assert runbook.startswith(
        "# Runbook d'installation — `twindb-lightrag-memgraph` v1.2.0"
    )
    assert "| **Version cible du package** | `1.2.0` |" in runbook
    assert "| **Version cible de LightRAG** | `1.5.6` |" in runbook
    assert "TWIN_LOG_FORMAT=json" in runbook
    assert "TWIN_LOG_LEVEL" not in runbook


def _prepare_release_script_fixture(tmp_path: Path) -> tuple[Path, Path, str]:
    release_repo = tmp_path / "release-repo"
    scripts = release_repo / "scripts"
    webui = release_repo / "lightrag_webui_twin"
    scripts.mkdir(parents=True)
    webui.mkdir()
    shutil.copy2(GENERATOR_PATH, scripts / GENERATOR_PATH.name)
    shutil.copy2(SCRIPT_PATH, scripts / SCRIPT_PATH.name)
    shutil.copy2(REPO_ROOT / "pyproject.toml", release_repo / "pyproject.toml")
    (release_repo / "Dockerfile").write_text("FROM scratch\n", encoding="utf-8")
    (webui / "package.json").write_text(
        '{"name":"release-fixture","version":"1.0.0"}\n', encoding="utf-8"
    )
    (webui / "bun.lock").write_text("# fixture\n", encoding="utf-8")

    subprocess.run(["git", "init", "-q"], cwd=release_repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "release-test@example.invalid"],
        cwd=release_repo,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Release Test"],
        cwd=release_repo,
        check=True,
    )
    subprocess.run(
        ["git", "config", "commit.gpgsign", "false"],
        cwd=release_repo,
        check=True,
    )
    hooks = release_repo / ".test-hooks"
    hooks.mkdir()
    subprocess.run(
        ["git", "config", "core.hooksPath", str(hooks)],
        cwd=release_repo,
        check=True,
    )
    subprocess.run(["git", "add", "."], cwd=release_repo, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "release fixture"],
        cwd=release_repo,
        check=True,
    )
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=release_repo, text=True
    ).strip()

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    syft = fake_bin / "syft"
    syft.write_text(
        """#!/bin/sh
if [ "$1" = "version" ]; then
    echo "syft fixture 1.0"
    exit 0
fi
for argument in "$@"; do
    case "$argument" in
        cyclonedx-json=*) output=${argument#cyclonedx-json=} ;;
    esac
done
printf '%s\n' '{"bomFormat":"CycloneDX","specVersion":"1.6","components":[{"type":"library","name":"fixture","version":"1","bom-ref":"fixture"}]}' > "$output"
""",
        encoding="utf-8",
    )
    docker = fake_bin / "docker"
    docker.write_text(
        """#!/bin/sh
if [ "$1" = "pull" ]; then
    exit 0
fi
if [ "$1" = "image" ] && [ "$2" = "inspect" ]; then
    echo "$FAKE_IMAGE_REVISION"
    exit 0
fi
if [ "$1" = "run" ]; then
    if [ "${FAKE_DOCKER_RUN_FAIL:-0}" = "1" ]; then
        exit 44
    fi
    echo "1.2.0"
    exit 0
fi
exit 45
""",
        encoding="utf-8",
    )
    syft.chmod(0o755)
    docker.chmod(0o755)
    return release_repo, fake_bin, commit


def test_generator_publishes_atomically_and_never_archives_stale_files(tmp_path):
    release_repo, fake_bin, commit = _prepare_release_script_fixture(tmp_path)
    image_ref = "registry.example/twin@sha256:" + "c" * 64
    base_env = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "SYFT_BIN": str(fake_bin / "syft"),
        "TWIN_RELEASE_IMAGE": image_ref,
        "TWIN_RELEASE_COMMIT": commit,
        "FAKE_IMAGE_REVISION": commit,
    }

    failed_destination = tmp_path / "published" / "failed"
    failed = subprocess.run(
        [str(release_repo / "scripts" / "generate_release_sbom.sh")],
        cwd=release_repo,
        env={
            **base_env,
            "TWIN_RELEASE_EVIDENCE_DIR": str(failed_destination),
            "FAKE_DOCKER_RUN_FAIL": "1",
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert failed.returncode == 44
    assert not failed_destination.exists()
    assert not list(
        failed_destination.parent.glob(f".{failed_destination.name}.staging.*")
    )

    destination = tmp_path / "published" / "success"
    completed = subprocess.run(
        [str(release_repo / "scripts" / "generate_release_sbom.sh")],
        cwd=release_repo,
        env={**base_env, "TWIN_RELEASE_EVIDENCE_DIR": str(destination)},
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    archive = destination / "twin-kms-1.2.0-release-evidence.zip"
    assert {path.name for path in destination.iterdir()} == {
        "twin-kms-1.2.0.sbom.cyclonedx.json",
        "fresh-image-smoke.txt",
        "manifest.txt",
        archive.name,
    }
    with zipfile.ZipFile(archive) as evidence:
        assert evidence.namelist() == [
            "fresh-image-smoke.txt",
            "manifest.txt",
            "twin-kms-1.2.0.sbom.cyclonedx.json",
        ]
    manifest = (destination / "manifest.txt").read_text(encoding="utf-8")
    assert f"git_commit={commit}" in manifest
    assert f"image_revision={commit}" in manifest
    assert "webui_tree=" in manifest


def test_cli_merge_and_verify_round_trip(tmp_path):
    source = tmp_path / "source.json"
    output = tmp_path / "release.cyclonedx.json"
    source.write_text(json.dumps(_image_document(serial="raw", timestamp="raw")))

    assert (
        release_sbom.main(
            [
                "merge",
                "--input",
                str(source),
                "--image-ref",
                IMAGE_REF,
                "--version",
                "1.2.0",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert release_sbom.main(["verify", str(output)]) == 0
