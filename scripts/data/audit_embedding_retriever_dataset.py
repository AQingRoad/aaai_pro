#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from statistics import median


RAW_ASIN_RE = re.compile(r"\bB0[A-Z0-9]{8}\b", re.IGNORECASE)


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield line_number, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc


def percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
    return ordered[max(0, index)]


def length_summary(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"min": 0, "median": 0.0, "p95": 0, "max": 0}
    return {
        "min": min(values),
        "median": median(values),
        "p95": percentile(values, 0.95),
        "max": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit an embedding query/positive JSONL before training.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--expected-split", default="")
    parser.add_argument("--expected-query-fields", default="")
    parser.add_argument("--forbid-cot-tags", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output", default="")
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    path = Path(args.dataset)
    if not path.is_file():
        raise FileNotFoundError(path)

    failures = Counter()
    target_counts: Counter[str] = Counter()
    pair_counts: Counter[tuple[str, str]] = Counter()
    query_lengths: list[int] = []
    positive_lengths: list[int] = []
    rows = 0
    expected_query_fields = [field.strip() for field in args.expected_query_fields.split(",") if field.strip()]

    for _, row in read_jsonl(path):
        rows += 1
        query = str(row.get("query") or "")
        positive = str(row.get("positive") or "")
        target_id = row.get("target_item_id")
        target_key = "" if target_id is None else str(target_id)
        history_ids = {str(item_id) for item_id in (row.get("history_item_ids") or [])}

        query_lengths.append(len(query))
        positive_lengths.append(len(positive))
        if not query.strip():
            failures["empty_query"] += 1
        if not positive.strip():
            failures["empty_positive"] += 1
        if not target_key:
            failures["missing_target_item_id"] += 1
        else:
            target_counts[target_key] += 1
        if args.expected_split and row.get("split") != args.expected_split:
            failures["split_mismatch"] += 1
        if expected_query_fields and row.get("query_fields") != expected_query_fields:
            failures["query_fields_mismatch"] += 1
        if args.forbid_cot_tags and ("<think>" in query.lower() or "<answer>" in query.lower()):
            failures["cot_tags_in_query"] += 1
        if target_key and target_key in history_ids:
            failures["target_in_history_ids"] += 1
        if "[TRUNCATED]" in query:
            failures["query_truncated_marker"] += 1
        if "[TRUNCATED]" in positive:
            failures["positive_truncated_marker"] += 1
        if RAW_ASIN_RE.search(query):
            failures["raw_asin_in_query"] += 1
        pair_counts[(query, positive)] += 1

    if rows == 0:
        raise ValueError(f"No rows found in {path}")

    excess_target_rows = rows - len(target_counts)
    rows_in_repeated_target_groups = sum(count for count in target_counts.values() if count > 1)
    duplicate_pair_rows = sum(count - 1 for count in pair_counts.values() if count > 1)
    same_target_probability = (
        sum(count * (count - 1) for count in target_counts.values()) / (rows * (rows - 1))
        if rows > 1
        else 0.0
    )
    expected_pairs = args.batch_size * (args.batch_size - 1) / 2 * same_target_probability
    report = {
        "dataset": str(path.resolve()),
        "rows": rows,
        "expected_split": args.expected_split or None,
        "expected_query_fields": expected_query_fields or None,
        "forbid_cot_tags": args.forbid_cot_tags,
        "failures": dict(sorted(failures.items())),
        "unique_targets": len(target_counts),
        "excess_target_rows": excess_target_rows,
        "excess_target_row_rate": excess_target_rows / rows,
        "rows_in_repeated_target_groups": rows_in_repeated_target_groups,
        "repeated_target_group_row_rate": rows_in_repeated_target_groups / rows,
        "targets_with_multiple_rows": sum(count > 1 for count in target_counts.values()),
        "max_rows_per_target": max(target_counts.values(), default=0),
        "duplicate_query_positive_rows": duplicate_pair_rows,
        "batch_size": args.batch_size,
        "expected_same_target_pairs_per_batch": expected_pairs,
        "query_chars": length_summary(query_lengths),
        "positive_chars": length_summary(positive_lengths),
    }

    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")

    if args.strict and failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
