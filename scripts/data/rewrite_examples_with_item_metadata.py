#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rubric_cot_pipeline.io import read_jsonl, write_jsonl
from rubric_cot_pipeline.item_metadata import build_item_map, build_item_summary_map, compact, history_text


RATING_RE = re.compile(r"\(([-+]?\d+(?:\.\d+)?)\s+stars?\)", re.IGNORECASE)


def parse_history_ratings(user_history: str) -> list[float]:
    ratings: list[float] = []
    for match in RATING_RE.finditer(user_history or ""):
        try:
            ratings.append(float(match.group(1)))
        except ValueError:
            continue
    return ratings


def item_title(item: dict[str, Any] | None) -> str:
    return compact(item.get("title"), 300) if item else ""


def rewrite_row(
    row: dict[str, Any],
    item_map: dict[int, dict[str, Any]],
    summary_map: dict[int, str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    raw_history_item_ids = row.get("history_item_ids")
    if raw_history_item_ids is None:
        raw_history_item_ids = row.get("history_item_id", [])
    history_item_ids = [int(x) for x in raw_history_item_ids]
    existing_history = str(row.get("user_history", ""))
    expected_history_count = int(row.get("history_item_count") or 0)
    if expected_history_count > 0 and not history_item_ids:
        example_id = row.get("example_id") or row.get("interaction_id") or row.get("user_id") or "<unknown>"
        raise ValueError(
            f"Cannot rebuild non-empty history for example {example_id}: "
            "missing history_item_ids. "
            "Use the original examples.jsonl that still contains history_item_ids."
        )

    titles = [item_title(item_map.get(item_id)) for item_id in history_item_ids]
    if any(not title for title in titles):
        bad_positions = [idx for idx, title in enumerate(titles) if not title]
        bad_item_ids = [history_item_ids[idx] for idx in bad_positions]
        example_id = row.get("example_id") or row.get("interaction_id") or row.get("user_id") or "<unknown>"
        raise ValueError(
            f"Empty item title while rebuilding history for example {example_id}: "
            f"positions={bad_positions} item_ids={bad_item_ids}. "
            "Fix item_info or rebuild source examples before adding metadata."
        )

    ratings = parse_history_ratings(existing_history)
    if len(ratings) != len(titles):
        example_id = row.get("example_id") or row.get("interaction_id") or row.get("user_id") or "<unknown>"
        raise ValueError(
            f"History rating count mismatch for example {example_id}: "
            f"history_item_ids={len(history_item_ids)} ratings_in_user_history={len(ratings)}. "
            "Rebuild examples.jsonl from the source RRec dataset first."
        )

    out = dict(row)
    rewritten_history = history_text(
        row.get("category") or args.category,
        titles,
        ratings,
        args.max_history_items,
        item_ids=history_item_ids,
        item_map=item_map,
        metadata_mode=args.history_metadata_mode,
        max_item_chars=args.history_max_item_chars,
        summary_map=summary_map,
    )
    out["user_history"] = rewritten_history
    out["query"] = rewritten_history
    out["history_metadata_mode"] = args.history_metadata_mode
    out["history_max_item_chars"] = args.history_max_item_chars
    if args.item_summary:
        out["item_summary_source"] = args.item_summary
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Rewrite existing RRec examples with item metadata in user_history.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--item-info", required=True)
    parser.add_argument("--item-summary", default="")
    parser.add_argument("--category", default="CDs_and_Vinyl")
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--max-history-items", type=int, default=20)
    parser.add_argument(
        "--history-metadata-mode",
        choices=["none", "compact", "summary"],
        default="summary",
    )
    parser.add_argument("--history-max-item-chars", type=int, default=420)
    args = parser.parse_args()

    item_map = build_item_map(read_jsonl(args.item_info))
    summary_map = build_item_summary_map(read_jsonl(args.item_summary)) if args.item_summary else {}
    rows = []
    for row in read_jsonl(args.input, limit=args.max_examples):
        rows.append(rewrite_row(row, item_map, summary_map, args))

    count = write_jsonl(args.output, rows)
    stats = {
        "input": args.input,
        "output": args.output,
        "item_info": args.item_info,
        "item_summary": args.item_summary,
        "history_metadata_mode": args.history_metadata_mode,
        "history_max_item_chars": args.history_max_item_chars,
        "written": count,
        "summary_items": len(summary_map),
    }
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
