#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rubric_cot_pipeline.item_metadata import (
    as_text_list,
    build_item_map,
    build_item_summary_map,
    compact,
    format_selected_details,
    history_text,
)
from rubric_cot_pipeline.io import read_jsonl, write_jsonl
from scripts.data.rewrite_examples_with_item_metadata import parse_history_ratings


def item_title(item: Mapping[str, Any] | None, fallback: str = "") -> str:
    if not item:
        return fallback
    return compact(item.get("title") or fallback, 300)


def build_summary_item_text(
    item: Mapping[str, Any] | None,
    title: str,
    item_id: int,
    summary_map: Mapping[int, str],
    max_chars: int,
) -> str:
    if not item:
        return compact(title, max_chars)

    parts: list[str] = []
    title_text = compact(item.get("title") or title, 300)
    if title_text:
        parts.append(title_text)

    main_category = compact(item.get("main_category"), 160)
    if main_category:
        parts.append(f"Main category: {main_category}")

    store = compact(item.get("store"), 300)
    if store:
        parts.append(f"Store/artist/format: {store}")

    categories = " > ".join(as_text_list(item.get("categories"), limit=6))
    if categories:
        parts.append(f"Categories: {categories}")

    summary = compact(summary_map.get(item_id, ""), 0)
    if summary:
        parts.append(f"Summary: {summary}")

    details = format_selected_details(item)
    if details:
        parts.append(f"Details: {details}")

    if not parts:
        parts.append(title)
    return compact(" ".join(parts), max_chars)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build phase0 embedding pairs from prepared examples JSONL.")
    parser.add_argument("--examples", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--item-info", required=True)
    parser.add_argument("--item-summary", default="")
    parser.add_argument("--category", default="CDs_and_Vinyl")
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--max-history-items", type=int, default=20)
    parser.add_argument("--min-history", type=int, default=1)
    parser.add_argument(
        "--history-metadata-mode",
        choices=["none", "compact", "summary"],
        default="summary",
    )
    parser.add_argument("--history-max-item-chars", type=int, default=0)
    parser.add_argument("--max-target-chars", type=int, default=1800)
    parser.add_argument("--default-history-rating", type=float, default=5.0)
    args = parser.parse_args()

    item_map = build_item_map(read_jsonl(args.item_info))
    summary_map = build_item_summary_map(read_jsonl(args.item_summary)) if args.item_summary else {}

    rows = []
    skipped_short_history = 0
    for row in read_jsonl(args.examples, limit=args.max_examples):
        history_item_ids = [int(x) for x in row.get("history_item_ids", [])]
        if len(history_item_ids) < args.min_history:
            skipped_short_history += 1
            continue
        titles = [item_title(item_map.get(item_id), fallback=f"item_{item_id}") for item_id in history_item_ids]
        ratings = parse_history_ratings(str(row.get("user_history", "")))
        if len(ratings) < len(titles):
            ratings = ratings + [args.default_history_rating] * (len(titles) - len(ratings))
        elif len(ratings) > len(titles):
            ratings = ratings[-len(titles) :] if titles else []

        target_item_id = int(row["target_item_id"])
        target_title = str(row.get("target_item_title") or "")
        query = history_text(
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
        positive = build_summary_item_text(
            item_map.get(target_item_id),
            target_title,
            target_item_id,
            summary_map,
            args.max_target_chars,
        )
        rows.append(
            {
                "query": query,
                "positive": positive,
                "category": row.get("category") or args.category,
                "split": row.get("split", "train"),
                "user_id": row.get("user_id", ""),
                "interaction_id": row.get("interaction_id", ""),
                "target_item_id": target_item_id,
                "target_item_title": target_title,
                "target_rating": row.get("target_rating"),
                "history_item_ids": history_item_ids[-args.max_history_items :]
                if args.max_history_items > 0
                else history_item_ids,
                "history_item_count": min(len(history_item_ids), args.max_history_items)
                if args.max_history_items > 0
                else len(history_item_ids),
                "history_metadata_mode": args.history_metadata_mode,
                "history_max_item_chars": args.history_max_item_chars,
                "item_summary_source": args.item_summary,
            }
        )

    count = write_jsonl(args.output, rows)
    stats = {
        "examples": args.examples,
        "output": args.output,
        "item_info": args.item_info,
        "item_summary": args.item_summary,
        "history_metadata_mode": args.history_metadata_mode,
        "history_max_item_chars": args.history_max_item_chars,
        "max_history_items": args.max_history_items,
        "min_history": args.min_history,
        "written": count,
        "skipped_short_history": skipped_short_history,
        "summary_items": len(summary_map),
    }
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
