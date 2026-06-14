#!/usr/bin/env python3
"""Run a bounded TwinRAG retrieval-tuning matrix.

This is an operator-side probe for Alberto's TR-RETQ observations. It calls
the Twin overlay ``/twin/api/query`` contract and records raw responses plus
small derived metrics. It intentionally does not send ``tag_filter`` because
LightRAG 1.4.x does not apply it to retrieval and the Twin backend rejects it.
"""

from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_QUESTIONS = Path(
    "docs/audits/retrieval-tuning/questions-tr-retq.jsonl"
)
DEFAULT_MODES = "mix,hybrid,global,local,naive"
DEFAULT_TOP_K = "20,40"
DEFAULT_CHUNK_TOP_K = "10,20"
DEFAULT_RERANK = "true,false"


def _csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _csv_int(value: str) -> list[int]:
    parsed: list[int] = []
    for part in _csv(value):
        try:
            parsed.append(int(part))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"expected comma-separated integers, got {value!r}"
            ) from exc
    return parsed


def _csv_bool(value: str) -> list[bool]:
    parsed: list[bool] = []
    for part in _csv(value):
        lowered = part.lower()
        if lowered in {"1", "true", "yes", "on"}:
            parsed.append(True)
        elif lowered in {"0", "false", "no", "off"}:
            parsed.append(False)
        else:
            raise argparse.ArgumentTypeError(
                f"expected comma-separated booleans, got {value!r}"
            )
    return parsed


def _load_questions(path: Path) -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                item = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSONL: {exc}") from exc
            if not isinstance(item, dict):
                raise SystemExit(f"{path}:{line_no}: question row must be an object")
            if not item.get("id") or not item.get("query"):
                raise SystemExit(
                    f"{path}:{line_no}: question row requires id and query"
                )
            questions.append(item)
    if not questions:
        raise SystemExit(f"{path}: no questions found")
    return questions


def _make_output_path() -> Path:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%SZ")
    return Path(f"docs/audits/retrieval-tuning/results-{stamp}.jsonl")


def _url(base_url: str, endpoint: str) -> str:
    base = base_url.rstrip("/")
    path = "/" + endpoint.lstrip("/")
    return urllib.parse.urljoin(base + "/", path.lstrip("/"))


def _headers(args: argparse.Namespace) -> dict[str, str]:
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    token = args.token
    if token is None and args.token_env:
        token = os.environ.get(args.token_env)
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if args.folder:
        headers["X-Twin-Folder"] = args.folder
    return headers


def _post_json(
    url: str,
    payload: dict[str, Any],
    *,
    headers: dict[str, str],
    timeout: float,
) -> tuple[int, Any]:
    encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=encoded,
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
            return response.status, json.loads(body) if body else None
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            parsed: Any = json.loads(body) if body else None
        except json.JSONDecodeError:
            parsed = body
        return exc.code, parsed


def _source_name(source: dict[str, Any]) -> str:
    value = source.get("name") or source.get("doc_id") or source.get("chunk_id")
    return str(value or "")


def _metrics(sources: list[Any]) -> dict[str, Any]:
    source_dicts = [source for source in sources if isinstance(source, dict)]
    names = [_source_name(source) for source in source_dicts]
    chunk_ids = [
        str(source.get("chunk_id"))
        for source in source_dicts
        if source.get("chunk_id")
    ]
    duplicates = sorted({name for name in names if names.count(name) > 1 and name})
    return {
        "source_count": len(source_dicts),
        "unique_source_count": len({name for name in names if name}),
        "duplicate_source_names": duplicates,
        "chunk_count": len(chunk_ids),
        "unique_chunk_count": len(set(chunk_ids)),
    }


def _iter_matrix(
    questions: list[dict[str, Any]],
    *,
    modes: list[str],
    top_k_values: list[int],
    chunk_top_k_values: list[int],
    rerank_values: list[bool],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    rows: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for question, mode, top_k, chunk_top_k, rerank in itertools.product(
        questions, modes, top_k_values, chunk_top_k_values, rerank_values
    ):
        rows.append(
            (
                question,
                {
                    "mode": mode,
                    "top_k": top_k,
                    "chunk_top_k": chunk_top_k,
                    "enable_rerank": rerank,
                },
            )
        )
    return rows


def _record(
    *,
    run_id: str,
    question: dict[str, Any],
    params: dict[str, Any],
    payload: dict[str, Any],
    started_at: str,
    duration_ms: int | None,
    http_status: int | None,
    response_body: Any,
    error: str | None,
) -> dict[str, Any]:
    sources: list[Any] = []
    answer_status = None
    response_text = None
    if isinstance(response_body, dict):
        maybe_sources = response_body.get("sources")
        if isinstance(maybe_sources, list):
            sources = maybe_sources
        answer_status = response_body.get("answer_status")
        response_value = response_body.get("response")
        if isinstance(response_value, str):
            response_text = response_value
    is_query_response = isinstance(response_body, dict) and (
        "response" in response_body or "sources" in response_body
    )
    return {
        "run_id": run_id,
        "started_at": started_at,
        "question": {
            "id": question["id"],
            "axis": question.get("axis"),
            "query": question["query"],
            "expected_signal": question.get("expected_signal"),
            "notes": question.get("notes"),
        },
        "params": params,
        "request": payload,
        "http_status": http_status,
        "ok": bool(http_status is not None and 200 <= http_status < 300 and not error),
        "duration_ms": duration_ms,
        "answer_status": answer_status,
        "response": response_text,
        "sources": sources,
        "metrics": _metrics(sources),
        "error": error,
        "raw_body": None if is_query_response else response_body,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a TwinRAG retrieval tuning matrix against /twin/api/query."
    )
    parser.add_argument("--base-url", default="http://localhost:9621")
    parser.add_argument("--endpoint", default="/twin/api/query")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--folder", default=None)
    parser.add_argument("--token", default=None)
    parser.add_argument("--token-env", default="TWIN_API_TOKEN")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--modes", type=_csv, default=_csv(DEFAULT_MODES))
    parser.add_argument("--top-k", type=_csv_int, default=_csv_int(DEFAULT_TOP_K))
    parser.add_argument(
        "--chunk-top-k",
        type=_csv_int,
        default=_csv_int(DEFAULT_CHUNK_TOP_K),
    )
    parser.add_argument(
        "--rerank",
        type=_csv_bool,
        default=_csv_bool(DEFAULT_RERANK),
        help="Comma-separated booleans, e.g. true,false.",
    )
    parser.add_argument("--response-type", default=None)
    parser.add_argument("--history-turns", type=int, default=None)
    parser.add_argument(
        "--question-id",
        action="append",
        default=[],
        help="Restrict to one id. Can be passed multiple times.",
    )
    parser.add_argument("--max-calls", type=int, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write planned rows without sending HTTP requests.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    questions = _load_questions(args.questions)
    if args.question_id:
        allowed = set(args.question_id)
        questions = [q for q in questions if q["id"] in allowed]
        missing = allowed - {q["id"] for q in questions}
        if missing:
            raise SystemExit(f"unknown question id(s): {', '.join(sorted(missing))}")

    matrix = _iter_matrix(
        questions,
        modes=args.modes,
        top_k_values=args.top_k,
        chunk_top_k_values=args.chunk_top_k,
        rerank_values=args.rerank,
    )
    if args.max_calls is not None:
        matrix = matrix[: args.max_calls]

    out_path = args.out or _make_output_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    target_url = _url(args.base_url, args.endpoint)
    headers = _headers(args)
    run_id = dt.datetime.now(dt.timezone.utc).strftime("retq-%Y%m%d-%H%M%SZ")

    with out_path.open("w", encoding="utf-8") as handle:
        for index, (question, params) in enumerate(matrix, start=1):
            payload: dict[str, Any] = {
                "query": question["query"],
                **params,
            }
            if args.response_type:
                payload["response_type"] = args.response_type
            if args.history_turns is not None:
                payload["history_turns"] = args.history_turns

            started_at = dt.datetime.now(dt.timezone.utc).isoformat()
            duration_ms: int | None = None
            status: int | None = None
            body: Any = None
            error: str | None = None

            if args.dry_run:
                body = {
                    "response": None,
                    "sources": [],
                    "answer_status": "dry_run",
                }
            else:
                start = time.perf_counter()
                try:
                    status, body = _post_json(
                        target_url,
                        payload,
                        headers=headers,
                        timeout=args.timeout,
                    )
                except Exception as exc:  # noqa: BLE001 - probe must keep matrix running
                    error = f"{type(exc).__name__}: {exc}"
                duration_ms = round((time.perf_counter() - start) * 1000)

            row = _record(
                run_id=run_id,
                question=question,
                params=params,
                payload=payload,
                started_at=started_at,
                duration_ms=duration_ms,
                http_status=status,
                response_body=body,
                error=error,
            )
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            print(
                f"[{index}/{len(matrix)}] {question['id']} "
                f"{params['mode']} top_k={params['top_k']} "
                f"chunk_top_k={params['chunk_top_k']} "
                f"rerank={params['enable_rerank']} "
                f"status={status if status is not None else 'n/a'}",
                file=sys.stderr,
            )

    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
