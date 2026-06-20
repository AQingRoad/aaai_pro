#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from datasets import Dataset, load_from_disk

from rubric_cot_pipeline.io import write_jsonl


def category_label(category: str) -> str:
    return category.replace("_", " ")


def compact(text: Any, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max_chars - 15].rstrip() + " [TRUNCATED]"


def as_text_list(value: Any, limit: int = 8) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            text = compact(item, 500)
            if text:
                out.append(text)
            if len(out) >= limit:
                break
        return out
    return [compact(value, 500)]


def build_item_text(item: dict[str, Any] | None, title: str, max_chars: int) -> str:
    if not item:
        return compact(title, max_chars)

    parts: list[str] = []
    for key in ("title", "main_category", "store"):
        value = compact(item.get(key), 300)
        if value:
            parts.append(value)
    categories = " > ".join(as_text_list(item.get("categories"), limit=6))
    if categories:
        parts.append(f"Categories: {categories}")
    features = "; ".join(as_text_list(item.get("features"), limit=8))
    if features:
        parts.append(f"Features: {features}")
    description = " ".join(as_text_list(item.get("description"), limit=2))
    if description:
        parts.append(f"Description: {description}")
    if not parts:
        parts.append(title)
    return compact(" ".join(parts), max_chars)


def build_item_map(item_info: Dataset) -> dict[int, dict[str, Any]]:
    item_map: dict[int, dict[str, Any]] = {}
    for item in item_info:
        item_id = item.get("item_id")
        if item_id is not None:
            item_map[int(item_id)] = dict(item)
    return item_map


def history_text(category: str, titles: list[str], ratings: list[float], max_history_items: int) -> str:
    if max_history_items > 0:
        titles = titles[-max_history_items:]
        ratings = ratings[-max_history_items:]

    entries = []
    for title, rating in zip(titles, ratings):
        title = compact(title, 240)
        if title:
            entries.append(f"{title} ({float(rating):g} stars)")

    history = "; ".join(entries)
    return (
        f"This user's Amazon {category_label(category)} interaction history over time is listed below. "
        f"{history}."
    )


def split_names(raw: str) -> list[str]:
    if raw == "all":
        return ["train", "valid", "test"]
    return [raw]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/root/autodl-tmp/rec/RRec_official/data")
    parser.add_argument("--category", required=True)
    parser.add_argument("--dataset-dir", default="")
    parser.add_argument("--split", choices=["train", "valid", "test", "all"], default="train")
    parser.add_argument("--output", default="")
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--max-history-items", type=int, default=20)
    parser.add_argument("--min-history", type=int, default=1)
    parser.add_argument("--min-rating", type=float, default=0.0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=20260614)
    parser.add_argument("--max-target-chars", type=int, default=1400)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else Path(args.data_root) / f"{args.category}_0_2022-10-2023-10"
    if not dataset_dir.exists():
        raise FileNotFoundError(f"RRec dataset directory does not exist: {dataset_dir}")

    output = args.output or f"data/rrec_amazon/{args.category}/examples.jsonl"
    ds = load_from_disk(str(dataset_dir))
    item_map = build_item_map(ds["item_info"])

    rows = []
    skipped = 0
    for split in split_names(args.split):
        split_ds = ds[split]
        if args.shuffle:
            split_ds = split_ds.shuffle(seed=args.seed)
        if args.max_examples > 0:
            split_ds = split_ds.select(range(min(args.max_examples, len(split_ds))))

        for row in split_ds:
            titles = [str(x) for x in row.get("history_item_title", [])]
            ratings = [float(x) for x in row.get("history_rating", [])]
            if len(titles) < args.min_history or float(row.get("rating", 0.0)) < args.min_rating:
                skipped += 1
                continue

            item_id = int(row["item_id"])
            target_title = compact(row.get("item_title", ""), 300)
            item_text = build_item_text(item_map.get(item_id), target_title, args.max_target_chars)
            interaction_id = int(row.get("interaction_id", len(rows)))
            user_id = str(row["user_id"])
            example_id = f"{args.category}:{split}:{interaction_id}:{user_id}"

            rows.append(
                {
                    "example_id": example_id,
                    "dataset": "rrec-amazon-2023",
                    "category": args.category,
                    "split": split,
                    "user_id": user_id,
                    "interaction_id": interaction_id,
                    "target_item_id": item_id,
                    "target_item_asin": row.get("item_asin", ""),
                    "target_item_title": target_title,
                    "target_item_text": item_text,
                    "target_rating": float(row.get("rating", 0.0)),
                    "history_item_ids": [int(x) for x in row.get("history_item_id", [])][-args.max_history_items :],
                    "history_item_asins": list(row.get("item_asins", []))[-args.max_history_items :],
                    "history_item_count": min(len(titles), args.max_history_items) if args.max_history_items > 0 else len(titles),
                    "user_history": history_text(args.category, titles, ratings, args.max_history_items),
                }
            )

    count = write_jsonl(output, rows)
    stats = {
        "category": args.category,
        "split": args.split,
        "dataset_dir": str(dataset_dir),
        "output": output,
        "written": count,
        "skipped": skipped,
    }
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
