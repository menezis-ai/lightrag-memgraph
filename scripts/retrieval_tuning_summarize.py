#!/usr/bin/env python3
"""Summarize retrieval_tuning_probe JSONL output as Markdown."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSONL: {exc}") from exc
            if not isinstance(row, dict):
                raise SystemExit(f"{path}:{line_no}: row must be an object")
            rows.append(row)
    if not rows:
        raise SystemExit(f"{path}: no rows found")
    return rows


def _param_label(row: dict[str, Any]) -> str:
    params = row.get("params") if isinstance(row.get("params"), dict) else {}
    return (
        f"{params.get('mode', '?')} "
        f"top={params.get('top_k', '?')} "
        f"chunk={params.get('chunk_top_k', '?')} "
        f"rerank={params.get('enable_rerank', '?')}"
    )


def _question(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("question")
    return value if isinstance(value, dict) else {}


def _metrics(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("metrics")
    return value if isinstance(value, dict) else {}


def _axis(row: dict[str, Any]) -> str:
    return str(_question(row).get("axis") or "?")


def _qid(row: dict[str, Any]) -> str:
    return str(_question(row).get("id") or "?")


def _ok(row: dict[str, Any]) -> bool:
    return bool(row.get("ok"))


def _answer_status(row: dict[str, Any]) -> str:
    return str(row.get("answer_status") or "?")


def _source_count(row: dict[str, Any]) -> int:
    value = _metrics(row).get("source_count", 0)
    return int(value) if isinstance(value, int) else 0


def _unique_source_count(row: dict[str, Any]) -> int:
    value = _metrics(row).get("unique_source_count", 0)
    return int(value) if isinstance(value, int) else 0


def _duplicate_names(row: dict[str, Any]) -> list[str]:
    value = _metrics(row).get("duplicate_source_names")
    return [str(item) for item in value] if isinstance(value, list) else []


def _redundancy_ratio(row: dict[str, Any]) -> float:
    source_count = _source_count(row)
    if source_count <= 0:
        return 0.0
    return _unique_source_count(row) / source_count


def _candidate_score(row: dict[str, Any]) -> tuple[int, int, float, int, int]:
    """Sort key for candidate rows.

    The score is intentionally modest: it prefers successful grounded answers,
    then more unique sources, then less duplicate concentration. Manual review
    of response text and source relevance remains mandatory.
    """
    grounded = 1 if _answer_status(row) == "grounded" else 0
    return (
        1 if _ok(row) else 0,
        grounded,
        _redundancy_ratio(row),
        _unique_source_count(row),
        _source_count(row),
    )


def _flags(row: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    axis = _axis(row)
    status = _answer_status(row)
    if not _ok(row):
        flags.append("HTTP/error")
    if axis == "CONTROL" and status == "grounded":
        flags.append("control grounded")
    if axis != "CONTROL" and status == "insufficient_information":
        flags.append("insufficient")
    if _duplicate_names(row):
        flags.append("duplicate sources")
    if status == "grounded" and _source_count(row) == 0 and axis != "CONTROL":
        flags.append("grounded without sources")
    return flags


def _escape_cell(value: Any) -> str:
    text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def _table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    output = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        output.append("| " + " | ".join(_escape_cell(value) for value in row) + " |")
    return output


def _summarize(path: Path, rows: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append(f"# Retrieval Tuning Summary — {path.name}")
    lines.append("")
    lines.append("## Global")
    lines.append("")
    status_counts = Counter(_answer_status(row) for row in rows)
    axis_counts = Counter(_axis(row) for row in rows)
    error_count = sum(1 for row in rows if not _ok(row))
    lines.extend(
        _table(
            ["metric", "value"],
            [
                ["rows", len(rows)],
                ["errors", error_count],
                ["answer_status", dict(sorted(status_counts.items()))],
                ["axes", dict(sorted(axis_counts.items()))],
            ],
        )
    )
    lines.append("")

    flagged = [row for row in rows if _flags(row)]
    lines.append("## Flags")
    lines.append("")
    if flagged:
        lines.extend(
            _table(
                ["question", "axis", "params", "status", "sources", "flags"],
                [
                    [
                        _qid(row),
                        _axis(row),
                        _param_label(row),
                        _answer_status(row),
                        _source_count(row),
                        ", ".join(_flags(row)),
                    ]
                    for row in flagged[:50]
                ],
            )
        )
        if len(flagged) > 50:
            lines.append("")
            lines.append(f"Truncated: {len(flagged) - 50} additional flagged rows.")
    else:
        lines.append("No automatic flags.")
    lines.append("")

    by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_question[_qid(row)].append(row)

    lines.append("## Best Candidates Per Question")
    lines.append("")
    for qid in sorted(by_question):
        candidates = sorted(
            by_question[qid],
            key=_candidate_score,
            reverse=True,
        )
        top = candidates[:5]
        axis = _axis(top[0]) if top else "?"
        query = _question(top[0]).get("query") if top else ""
        lines.append(f"### {qid} ({axis})")
        lines.append("")
        lines.append(str(query or ""))
        lines.append("")
        lines.extend(
            _table(
                [
                    "params",
                    "ok",
                    "status",
                    "sources",
                    "unique",
                    "redundancy",
                    "duplicates",
                    "duration_ms",
                ],
                [
                    [
                        _param_label(row),
                        _ok(row),
                        _answer_status(row),
                        _source_count(row),
                        _unique_source_count(row),
                        f"{_redundancy_ratio(row):.2f}",
                        ", ".join(_duplicate_names(row)[:3]),
                        row.get("duration_ms"),
                    ]
                    for row in top
                ],
            )
        )
        lines.append("")

    lines.append("## Manual Review Checklist")
    lines.append("")
    lines.append("- For TR-RETQ-01, inspect whether duplicate sources add evidence or repeat the same passage.")
    lines.append("- For TR-RETQ-02, inspect source relevance manually; the current C3 source score is not a semantic judge.")
    lines.append("- For TR-RETQ-03, prefer settings that turn conceptual questions from `insufficient_information` into grounded answers with plausible multi-document coverage.")
    lines.append("- For CONTROL, any grounded answer must be reviewed as a likely false positive unless the corpus really contains the topic.")
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize retrieval_tuning_probe JSONL output."
    )
    parser.add_argument("results", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    rows = _load_rows(args.results)
    markdown = _summarize(args.results, rows)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(markdown + "\n", encoding="utf-8")
        print(str(args.out))
    else:
        print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
