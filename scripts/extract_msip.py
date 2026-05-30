#!/usr/bin/env python3
"""CLI wrapper around `twindb_lightrag_memgraph.classification`.

Usage
-----
    python scripts/extract_msip.py FILE [FILE ...]
    python scripts/extract_msip.py --label-map labels.json FILE
    python scripts/extract_msip.py --json FILE

The default output is one human-readable block per file. ``--json`` emits
a JSON array of `ClassificationResult.as_dict()` payloads for piping into
other tooling. ``--exit-code-on-above CLASS`` returns non-zero when any
input exceeds the given threshold (useful as a pre-commit / CI gate on a
folder of docs).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from twindb_lightrag_memgraph.classification import (
    detect_classification,
    is_above,
    load_label_map,
)


def _human_block(path: Path, result_dict: dict) -> str:
    lines = [f"=== {path} ==="]
    for k in ("class_id", "class_name", "label_guid", "set_date",
              "method", "source_format", "reason"):
        v = result_dict.get(k)
        if v is not None:
            lines.append(f"  {k:>14} : {v}")
    extra_meta = {
        k: v for k, v in (result_dict.get("meta") or {}).items()
        if k not in {"Name", "SetDate", "Method"}
    }
    if extra_meta:
        lines.append(f"  {'meta':>14} : {extra_meta}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="+", type=Path, help="Files to probe.")
    p.add_argument("--label-map", type=Path,
                   help="JSON file mapping {guid → class_id} or "
                        "{guid → {id, name}}. Defaults to "
                        "$TWIN_MIP_LABEL_MAP or empty.")
    p.add_argument("--json", action="store_true",
                   help="Emit a JSON array instead of human blocks.")
    p.add_argument("--exit-code-on-above", metavar="CLASS",
                   help="Return non-zero if any file's class outranks CLASS "
                        "(e.g. C2). Unknown classes also trigger the gate "
                        "(fail-closed).")
    args = p.parse_args(argv)

    label_map = load_label_map(args.label_map) if args.label_map else load_label_map()
    results = []
    above = False
    for path in args.files:
        if not path.exists():
            print(f"!! {path}: not found", file=sys.stderr)
            results.append({"_path": str(path), "_error": "not-found"})
            continue
        r = detect_classification(path, label_map=label_map)
        d = r.as_dict()
        d["_path"] = str(path)
        results.append(d)
        if args.exit_code_on_above and is_above(r.class_id, args.exit_code_on_above):
            above = True

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        for r in results:
            if r.get("_error"):
                continue
            print(_human_block(Path(r["_path"]), r))
            print()

    if args.exit_code_on_above and above:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
