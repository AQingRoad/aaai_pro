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
HISTORY_ENTRY_RE = re.compile(r"(?:^|[.;]\s*)(?P<title>.+?)\s*\((?P<rating>[-+]?\d+(?:\.\d+)?)\s+stars?\)", re.IGNORECASE)


def parse_history_ratings(user_history: str) -> list[float]:
    ratings: list[float] = []
    for match in RATING_RE.finditer(user_history or ""):
        try:
            ratings.append(float(match.group(1)))
        except ValueError:
            continue
    return ratings


def normalize_title(title: str) -> str:
    return re.sub(r"\s+", " ", str(title or "")).strip().lower()


def parse_history_entries(user_history: str) -> tuple[list[str], list[float]]:
    titles: list[str] = []
    ratings: list[float] = []
    prefix = "This user's Amazon CDs and Vinyl interaction history over time is listed below."
    text = str(user_history or "").replace(prefix, "").strip()
    for match in HISTORY_ENTRY_RE.finditer(text):
        title = re.sub(r"^\d+\.\s*", "", match.group("title")).strip(" ;.\n\t")
        if not title:
            continue
        try:
            rating = float(match.group("rating"))
        except ValueError:
            continue
        titles.append(title)
        ratings.append(rating)
    return titles, ratings


def build_title_to_item_ids(item_map: dict[int, dict[str, Any]]) -> dict[str, list[int]]:
    title_to_ids: dict[str, list[int]] = {}
    for item_id, item in item_map.items():
        title = normalize_title(item.get("title") or "")
        if title:
            title_to_ids.setdefault(title, []).append(item_id)
    return title_to_ids


def item_title(item: dict[str, Any] | None, fallback: str = "") -> str:
    if not item:
        return fallback
    return compact(item.get("title") or fallback, 300)


def rewrite_row(
    row: dict[str, Any],
    item_map: dict[int, dict[str, Any]],
    title_to_item_ids: dict[str, list[int]],
    summary_map: dict[int, str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    history_item_ids = [int(x) for x in row.get("history_item_ids", [])]
    existing_history = str(row.get("user_history", ""))
    if history_item_ids:
        titles = [item_title(item_map.get(item_id), fallback=f"item_{item_id}") for item_id in history_item_ids]
        ratings = parse_history_ratings(existing_history)
    else:
        titles, ratings = parse_history_entries(existing_history)
        history_item_ids = [
            ids[0] if len(ids := title_to_item_ids.get(normalize_title(title), [])) == 1 else -1
            for title in titles
        ]

    expected_history_count = int(row.get("history_item_count") or 0)
    if expected_history_count > 0 and not titles:
        example_id = row.get("example_id") or row.get("interaction_id") or row.get("user_id") or "<unknown>"
        raise ValueError(
            f"Cannot rebuild non-empty history for example {example_id}: "
            "missing history_item_ids and no '(N stars)' entries in user_history. "
            "Use the original examples.jsonl that still contains history_item_ids."
        )

    if len(ratings) < len(titles):
        ratings = ratings + [args.default_history_rating] * (len(titles) - len(ratings))
    elif len(ratings) > len(titles):
        ratings = ratings[-len(titles) :] if titles else []

    metadata_item_ids = [item_id if item_id >= 0 else None for item_id in history_item_ids]

    out = dict(row)
    rewritten_history = history_text(
        row.get("category") or args.category,
        titles,
        ratings,
        args.max_history_items,
        item_ids=[item_id for item_id in metadata_item_ids if item_id is not None]
        if all(item_id is not None for item_id in metadata_item_ids)
        else None,
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
    parser.add_argument("--default-history-rating", type=float, default=5.0)
    args = parser.parse_args()

    item_map = build_item_map(read_jsonl(args.item_info))
    title_to_item_ids = build_title_to_item_ids(item_map)
    summary_map = build_item_summary_map(read_jsonl(args.item_summary)) if args.item_summary else {}
    rows = []
    for row in read_jsonl(args.input, limit=args.max_examples):
        rows.append(rewrite_row(row, item_map, title_to_item_ids, summary_map, args))

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
