#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rubric_cot_pipeline.io import read_jsonl


def example_key(row: dict[str, Any]) -> str:
    return str(row.get("example_id") or row.get("user_id") or row.get("id") or row.get("interaction_id") or "")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate sharded CoT candidate-list JSONL files in input order.")
    parser.add_argument("--input", required=True, help="Original examples JSONL for ordering.")
    parser.add_argument("--shards", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-examples", type=int, default=0)
    args = parser.parse_args()

    rows_by_key: dict[str, dict[str, Any]] = {}
    loaded_shards = []
    for raw_path in args.shards:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(path)
        loaded_shards.append(str(path))
        for row in read_jsonl(path):
            key = example_key(row)
            if key and key not in rows_by_key:
                rows_by_key[key] = row

    ordered_rows = list(read_jsonl(args.input, limit=args.max_examples))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    missing = 0
    with output_path.open("w", encoding="utf-8") as f:
        for src in ordered_rows:
            key = example_key(src)
            row = rows_by_key.get(key)
            if row is None:
                missing += 1
                continue
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

    print(
        json.dumps(
            {
                "input": args.input,
                "output": args.output,
                "shards": loaded_shards,
                "input_rows": len(ordered_rows),
                "available_rows": len(rows_by_key),
                "written": written,
                "missing": missing,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if missing:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
