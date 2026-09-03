from pathlib import Path

from tests._repo_only import require_repo_path

# Asserts on the private BNP pipeline and Dockerfile; the export ships neither.
require_repo_path(".gitlab-ci.yml")

ROOT = Path(__file__).resolve().parents[1]


def _requirement_names(path: Path) -> set[str]:
    names: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        name = line.split(";", 1)[0].strip()
        for separator in ("==", ">=", "<=", "~=", "!=", ">", "<", "["):
            name = name.split(separator, 1)[0].strip()
        names.add(name.lower().replace("_", "-"))
    return names


def test_bnp_cve_remediation_excludes_lightrag_baseline():
    names = _requirement_names(ROOT / "requirements" / "cve-remediation.txt")

    assert "lightrag-hku" not in names
    assert {
        "pyjwt",
        "aiohttp",
        "cryptography",
        "python-multipart",
        "urllib3",
        "wheel",
    }.issubset(names)


def test_bnp_dockerfile_applies_cve_remediation_overlay():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "apt-get install -y --no-install-recommends --only-upgrade" in dockerfile
    assert "libcap2" in dockerfile
    assert "libssl3t64" in dockerfile
    assert "requirements/cve-remediation.txt" in dockerfile
    assert 'm.version("lightrag-hku") == "1.5.6"' in dockerfile


def _job_block(ci_text: str, job_name: str) -> str:
    start = ci_text.index(f"{job_name}:\n")
    lines = ci_text[start:].splitlines()
    block = [lines[0]]
    for line in lines[1:]:
        if line and not line.startswith((" ", "-")) and line.endswith(":"):
            break
        block.append(line)
    return "\n".join(block)


def test_bnp_gitlab_ci_gates_before_image_push():
    ci_text = (ROOT / ".gitlab-ci.yml").read_text(encoding="utf-8")

    assert ci_text.index("  - test") < ci_text.index("  - docker:build")
    assert ci_text.index("  - docker:security") < ci_text.index("  - docker:push")

    for job_name in ("image:push:prod", "image:push:hprd"):
        block = _job_block(ci_text, job_name)
        assert "tests:cve-remediation" in block
        assert "tests:smoke-runner" in block
        assert "dockerfile:lint" in block
        assert "image:security-check:trivy" in block


def test_bnp_gitlab_ci_trivy_blocks_push():
    ci_text = (ROOT / ".gitlab-ci.yml").read_text(encoding="utf-8")
    trivy = _job_block(ci_text, "image:security-check:trivy")

    assert "allow_failure: false" in trivy
    assert "--exit-code 1" in trivy
