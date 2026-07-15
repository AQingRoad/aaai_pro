#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


RAW_ASIN_RE = re.compile(r"\bB0[A-Z0-9]{8}\b", re.IGNORECASE)
FORBIDDEN_BASE_LABELS = (
    "Rating:",
    "Description:",
    "Details:",
    "Catalog stats:",
    "Amazon ASIN:",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    def text(value: Any) -> str:
        return "" if value is None else str(value)

    return (
        text(row.get("split")),
        text(row.get("interaction_id")),
        text(row.get("user_id")),
        text(row.get("target_item_id")),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prove that history-only and history+CoT datasets differ only by appended CoT."
    )
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--cot", type=Path, required=True)
    parser.add_argument("--expected-split", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    history_rows = read_jsonl(args.history)
    cot_rows = read_jsonl(args.cot)
    failures: dict[str, int] = {}

    def fail(name: str) -> None:
        failures[name] = failures.get(name, 0) + 1

    if len(history_rows) != len(cot_rows):
        fail("row_count_mismatch")

    paired_rows = min(len(history_rows), len(cot_rows))
    history_keys: list[tuple[str, str, str, str]] = []
    cot_keys: list[tuple[str, str, str, str]] = []
    for index in range(paired_rows):
        history = history_rows[index]
        cot = cot_rows[index]
        hkey = key(history)
        ckey = key(cot)
        history_keys.append(hkey)
        cot_keys.append(ckey)
        hquery = str(history.get("query") or "")
        base_query = str(cot.get("base_query") or "")
        cot_query = str(cot.get("query") or "")
        cot_text = str(cot.get("cot") or "")
        positive = str(history.get("positive") or "")

        if hkey != ckey:
            fail("ordered_key_mismatch")
        if history.get("split") != args.expected_split or cot.get("split") != args.expected_split:
            fail("split_mismatch")
        if hquery != base_query:
            fail("base_query_byte_mismatch")
        if history.get("positive") != cot.get("positive"):
            fail("positive_byte_mismatch")
        if history.get("history_item_ids") != cot.get("history_item_ids"):
            fail("history_item_ids_mismatch")
        if history.get("target_item_id") != cot.get("target_item_id"):
            fail("target_item_id_mismatch")
        expected_query = f"{hquery.strip()}\n\nRecommendation reasoning:\n{cot_text.strip()}"
        if cot_query != expected_query:
            fail("cot_append_format_mismatch")
        if not all(tag in cot_text for tag in ("<think>", "</think>", "<answer>", "</answer>")):
            fail("missing_tagged_cot")
        if not hquery.strip() or not cot_text.strip() or not positive.strip():
            fail("empty_required_text")
        if "[TRUNCATED]" in hquery or "[TRUNCATED]" in cot_query or "[TRUNCATED]" in positive or "[TRUNCATED]" in cot_text:
            fail("truncated_marker")
        if RAW_ASIN_RE.search(hquery) or RAW_ASIN_RE.search(cot_query):
            fail("raw_asin_in_query")
        for label in FORBIDDEN_BASE_LABELS:
            if label.lower() in hquery.lower():
                fail(f"forbidden_base_label_{label.rstrip(':').lower().replace(' ', '_')}")

    if len(set(history_keys)) != len(history_keys):
        fail("duplicate_history_keys")
    if len(set(cot_keys)) != len(cot_keys):
        fail("duplicate_cot_keys")

    report = {
        "history": str(args.history.resolve()),
        "cot": str(args.cot.resolve()),
        "expected_split": args.expected_split,
        "history_rows": len(history_rows),
        "cot_rows": len(cot_rows),
        "paired_rows": paired_rows,
        "unique_history_keys": len(set(history_keys)),
        "unique_cot_keys": len(set(cot_keys)),
        "base_query_relation": "cot.base_query byte-equals history.query",
        "only_query_delta": "two newlines + Recommendation reasoning label + full tagged CoT",
        "failures": failures,
        "aligned": not failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
