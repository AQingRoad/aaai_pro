#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from datasets import load_from_disk

from rubric_cot_pipeline.io import write_jsonl
from scripts.prepare_rrec_amazon_examples import build_item_map, build_item_text, history_text


DEFAULT_CATEGORIES = ["Musical_Instruments", "Video_Games", "CDs_and_Vinyl"]


def iter_rows(args):
    for category in args.categories:
        dataset_dir = Path(args.data_root) / f"{category}_0_2022-10-2023-10"
        if not dataset_dir.exists():
            raise FileNotFoundError(f"RRec dataset directory does not exist: {dataset_dir}")

        ds = load_from_disk(str(dataset_dir))
        item_map = build_item_map(ds["item_info"])
        split_ds = ds[args.split]
        if args.shuffle:
            split_ds = split_ds.shuffle(seed=args.seed)
        if args.max_examples_per_category > 0:
            split_ds = split_ds.select(range(min(args.max_examples_per_category, len(split_ds))))

        written = 0
        skipped = 0
        for row in split_ds:
            titles = [str(x) for x in row.get("history_item_title", [])]
            ratings = [float(x) for x in row.get("history_rating", [])]
            if len(titles) < args.min_history or float(row.get("rating", 0.0)) < args.min_rating:
                skipped += 1
                continue

            item_id = int(row["item_id"])
            target_title = str(row.get("item_title", "") or "")
            positive = build_item_text(item_map.get(item_id), target_title, args.max_target_chars)
            query = history_text(category, titles, ratings, args.max_history_items)
            interaction_id = int(row.get("interaction_id", written))

            yield {
                "query": query,
                "positive": positive,
                "category": category,
                "split": args.split,
                "user_id": str(row["user_id"]),
                "interaction_id": interaction_id,
                "target_item_id": item_id,
                "target_item_title": target_title,
                "target_rating": float(row.get("rating", 0.0)),
                "history_item_count": min(len(titles), args.max_history_items)
                if args.max_history_items > 0
                else len(titles),
            }
            written += 1

        print(json.dumps({"category": category, "written": written, "skipped": skipped}, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/root/autodl-tmp/rec/RRec_official/data")
    parser.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES)
    parser.add_argument("--split", choices=["train", "valid", "test"], default="train")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-examples-per-category", type=int, default=0)
    parser.add_argument("--max-history-items", type=int, default=20)
    parser.add_argument("--min-history", type=int, default=1)
    parser.add_argument("--min-rating", type=float, default=0.0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=20260615)
    parser.add_argument("--max-target-chars", type=int, default=1400)
    args = parser.parse_args()

    count = write_jsonl(args.output, iter_rows(args))
    print(json.dumps({"output": args.output, "written": count}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
