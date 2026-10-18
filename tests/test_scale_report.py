"""Missing, skipped or failed scale measurements must never publish success."""

import json
import xml.etree.ElementTree as ET

import pytest

from scripts.summarize_scale import markdown, summarize


@pytest.mark.parametrize(
    "problem", ["none", "failure", "error", "skipped", "partial", "missing"]
)
def test_scale_summary_fails_closed(tmp_path, problem):
    path = tmp_path / "scale.xml"
    root = ET.Element("testsuites")
    suite = ET.SubElement(root, "testsuite")
    for operation, size in [
        ("paginated", 10_000),
        ("paginated", 100_000),
        ("vector-query", 10_000),
    ]:
        case = ET.SubElement(suite, "testcase", name=f"{operation}-{size}")
        if problem == "partial" and size == 100_000:
            continue
        props = ET.SubElement(case, "properties")
        metrics = {
            "operation": operation,
            "dataset_size": size,
            "samples": 24,
            "concurrency": 8,
            "p50_ms": 10,
            "p95_ms": 20,
            "max_ms": 30,
            "budget_ms": 500,
            "passed": True,
        }
        ET.SubElement(
            props, "property", name="scale_metrics", value=json.dumps(metrics)
        )
        if problem in ("failure", "error", "skipped"):
            ET.SubElement(case, problem, message="qualification did not pass")
    if problem != "missing":
        ET.ElementTree(root).write(path)
    report = summarize(path)
    assert report["passed"] is (problem == "none")
    text = markdown(report)
    assert f"Result: **{'PASS' if problem == 'none' else 'FAIL'}**" in text
    if problem == "none":
        assert len(report["metrics"]) == 3
        assert "| vector-query | 10000 | 8 | 24 |" in text
