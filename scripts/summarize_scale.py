"""Render machine-readable and GitHub summaries from scale-test JUnit metrics."""

import argparse
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET


def summarize(junit: Path) -> dict:
    report = {
        "adapter_ref": os.getenv("SCALE_ADAPTER_REF", "local"),
        "adapter_sha": os.getenv("SCALE_ADAPTER_SHA", "unknown"),
        "harness_sha": os.getenv("GITHUB_SHA", "local"),
        "memgraph": os.getenv("MEMGRAPH_VERSION", "unknown"),
        "metrics": [],
        "errors": [],
    }
    if not junit.is_file():
        report["errors"].append("JUnit report missing: tests did not complete")
    else:
        root = ET.parse(junit).getroot()
        for case in root.iter("testcase"):
            for prop in case.findall("./properties/property"):
                if prop.get("name") == "scale_metrics":
                    report["metrics"].append(json.loads(prop.get("value")))
            for tag in ("failure", "error", "skipped"):
                for error in case.findall(tag):
                    report["errors"].append(
                        f"{case.get('name')}: {tag}: {error.get('message', '')}"
                    )
        for error in root.findall("./error"):
            report["errors"].append(error.get("message", "Collection error"))
        if len(report["metrics"]) != 3:
            report["errors"].append(
                f"Expected three scale measurements, received {len(report['metrics'])}"
            )
    report["passed"] = not report["errors"] and all(
        m["passed"] for m in report["metrics"]
    )
    return report


def markdown(report):
    lines = [
        f"### Scale qualification: {report['adapter_ref']} / Memgraph {report['memgraph']}",
        "",
        f"Adapter commit: `{report['adapter_sha']}`",
        "",
        "| Operation | Entries | Concurrency | Samples | p50 ms | p95 ms | Max ms | Budget ms | Result |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for m in report["metrics"]:
        lines.append(
            f"| {m['operation']} | {m['dataset_size']} | {m['concurrency']} | {m['samples']} | "
            f"{m['p50_ms']:.2f} | {m['p95_ms']:.2f} | {m['max_ms']:.2f} | {m['budget_ms']} | "
            f"{'PASS' if m['passed'] else 'FAIL'} |"
        )
    lines += ["", f"Result: **{'PASS' if report['passed'] else 'FAIL'}**", ""]
    for error in report["errors"]:
        lines.append(f"- {error}")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("junit", type=Path)
    parser.add_argument("--json", type=Path, required=True, dest="json_path")
    parser.add_argument("--markdown", type=Path, required=True, dest="markdown_path")
    args = parser.parse_args()
    report = summarize(args.junit)
    for path in (args.json_path, args.markdown_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.json_path.write_text(json.dumps(report, indent=2) + "\n")
    args.markdown_path.write_text(markdown(report))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
