#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


RAW_ASIN_RE = re.compile(r"\bB0[A-Z0-9]{8}\b", re.IGNORECASE)
FORBIDDEN_HISTORY_LABELS = (
    "Rating:",
    "Description:",
    "Details:",
    "Catalog stats:",
    "Amazon ASIN:",
)


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                yield line_number, json.loads(line)


def example_id(row: dict[str, Any]) -> str:
    def text(value: Any, default: str = "") -> str:
        return default if value is None else str(value)

    return ":".join(
        [
            text(row.get("category"), "CDs_and_Vinyl"),
            text(row.get("split")),
            text(row.get("interaction_id")),
            text(row.get("user_id")),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create target-free CoT generation inputs from aligned embedding pairs."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--expected-split", required=True)
    args = parser.parse_args()

    output_rows: list[dict[str, Any]] = []
    failures: dict[str, int] = {}
    seen_ids: set[str] = set()

    def fail(name: str) -> None:
        failures[name] = failures.get(name, 0) + 1

    for _, row in read_jsonl(args.input):
        query = str(row.get("query") or "").strip()
        split = str(row.get("split") or "")
        eid = example_id(row)
        if not query:
            fail("empty_query")
        if split != args.expected_split:
            fail("split_mismatch")
        if eid in seen_ids:
            fail("duplicate_example_id")
        seen_ids.add(eid)
        if "[TRUNCATED]" in query:
            fail("truncated_query")
        if RAW_ASIN_RE.search(query):
            fail("raw_asin_in_query")
        for label in FORBIDDEN_HISTORY_LABELS:
            if label.lower() in query.lower():
                fail(f"forbidden_label_{label.rstrip(':').lower().replace(' ', '_')}")
        if row.get("query_fields") != ["title", "store", "categories"]:
            fail("query_fields_mismatch")

        # Deliberately omit positive, target title/text, ratings, and future-item
        # metadata. The generator receives only user_history plus matching keys.
        output_rows.append(
            {
                "example_id": eid,
                "category": str(row.get("category") or "CDs_and_Vinyl"),
                "split": split,
                "user_id": str(row.get("user_id") or ""),
                "interaction_id": row.get("interaction_id"),
                "target_item_id": row.get("target_item_id"),
                "history_item_ids": row.get("history_item_ids") or [],
                "history_item_count": row.get("history_item_count"),
                "user_history": query,
                "history_metadata_mode": "title_store_categories",
                "history_include_ratings": False,
                "history_include_catalog_stats": False,
                "history_max_item_chars": 0,
            }
        )

    if not output_rows:
        raise ValueError(f"No rows found in {args.input}")
    audit = {
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "expected_split": args.expected_split,
        "rows": len(output_rows),
        "unique_example_ids": len(seen_ids),
        "prompt_content_fields": ["user_history", "category"],
        "target_fields_rendered_in_prompt": [],
        "matching_keys_not_rendered_in_prompt": ["target_item_id"],
        "query_fields": ["title", "store", "categories"],
        "history_include_ratings": False,
        "history_include_catalog_stats": False,
        "history_max_item_chars": 0,
        "failures": failures,
    }
    if failures:
        raise ValueError(json.dumps(audit, ensure_ascii=False, indent=2))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
