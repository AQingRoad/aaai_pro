#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterator

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rubric_cot_pipeline.io import read_jsonl, write_jsonl
from rubric_cot_pipeline.item_metadata import build_item_map, build_item_text


TRUNCATION_MARKER = "[TRUNCATED]"


def target_id(row: dict[str, Any]) -> int | None:
    raw = row.get("target_item_id", row.get("item_id"))
    try:
        return int(raw)
    except Exception:
        return None


def refreshed_rows(
    input_path: Path,
    item_map: dict[int, dict[str, Any]],
    max_target_chars: int,
    stats: dict[str, int],
) -> Iterator[dict[str, Any]]:
    for row in read_jsonl(input_path):
        stats["rows"] += 1
        has_target_text = "target_item_text" in row
        has_positive = "positive" in row
        if not has_target_text and not has_positive:
            yield row
            continue

        old_text = str(row.get("target_item_text") or "")
        old_positive = str(row.get("positive") or "")
        if old_text and TRUNCATION_MARKER in old_text:
            stats["target_truncated_before"] += 1
        if old_positive and TRUNCATION_MARKER in old_positive:
            stats["positive_truncated_before"] += 1

        item_id = target_id(row)
        if item_id is None:
            stats["missing_target_id"] += 1
            yield row
            continue

        item = item_map.get(item_id)
        if item is None:
            stats["missing_item_info"] += 1
            yield row
            continue

        title = str(row.get("target_item_title") or row.get("item_title") or "")
        new_text = build_item_text(item, title, max_target_chars)
        if not new_text:
            stats["empty_rebuilt_text"] += 1
            yield row
            continue

        if has_target_text and new_text != old_text:
            row = dict(row)
            row["target_item_text"] = new_text
            stats["updated"] += 1
        if TRUNCATION_MARKER in new_text:
            stats["target_truncated_after"] += 1
        if has_positive and new_text != old_positive:
            row = dict(row)
            row["positive"] = new_text
            stats["positive_updated"] += 1
        if has_positive and TRUNCATION_MARKER in str(row.get("positive") or ""):
            stats["positive_truncated_after"] += 1
        yield row


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh target item text fields from item_info without changing query/history fields.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="", help="Output JSONL. Defaults to in-place update.")
    parser.add_argument("--item-info", required=True)
    parser.add_argument(
        "--max-target-chars",
        type=int,
        default=0,
        help="Maximum target/positive item text characters; 0 keeps the full item text.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path
    item_map = build_item_map(read_jsonl(args.item_info))
    stats = {
        "rows": 0,
        "updated": 0,
        "positive_updated": 0,
        "target_truncated_before": 0,
        "target_truncated_after": 0,
        "positive_truncated_before": 0,
        "positive_truncated_after": 0,
        "missing_target_id": 0,
        "missing_item_info": 0,
        "empty_rebuilt_text": 0,
    }

    if output_path == input_path:
        tmp_path = input_path.with_suffix(input_path.suffix + ".refresh_target_tmp")
        write_path = tmp_path
    else:
        tmp_path = None
        write_path = output_path

    written = write_jsonl(write_path, refreshed_rows(input_path, item_map, args.max_target_chars, stats))
    if tmp_path is not None:
        os.replace(tmp_path, input_path)

    stats.update(
        {
            "input": str(input_path),
            "output": str(output_path),
            "item_info": args.item_info,
            "max_target_chars": args.max_target_chars,
            "written": written,
        }
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
