#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from datasets import load_from_disk

from rubric_cot_pipeline.item_metadata import build_item_map, build_item_summary_map, build_item_text, compact, history_text
from rubric_cot_pipeline.io import read_jsonl
from rubric_cot_pipeline.io import write_jsonl


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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-target-chars", type=int, default=1400)
    parser.add_argument(
        "--history-metadata-mode",
        choices=["none", "compact", "summary"],
        default=os.getenv("HISTORY_METADATA_MODE", "none"),
    )
    parser.add_argument("--history-max-item-chars", type=int, default=int(os.getenv("HISTORY_MAX_ITEM_CHARS", "320")))
    parser.add_argument("--item-summary", default=os.getenv("ITEM_METADATA_SUMMARY", ""))
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else Path(args.data_root) / f"{args.category}_0_2022-10-2023-10"
    if not dataset_dir.exists():
        raise FileNotFoundError(f"RRec dataset directory does not exist: {dataset_dir}")

    output = args.output or f"data/rrec_amazon/{args.category}/examples.jsonl"
    ds = load_from_disk(str(dataset_dir))
    item_map = build_item_map(ds["item_info"])
    summary_map = build_item_summary_map(read_jsonl(args.item_summary)) if args.item_summary else {}

    rows = []
    skipped = 0
    for split in split_names(args.split):
        split_ds = ds[split]
        if args.shuffle:
            split_ds = split_ds.shuffle(seed=args.seed)
        if args.max_examples > 0:
            split_ds = split_ds.select(range(min(args.max_examples, len(split_ds))))

        for row in split_ds:
            history_item_ids = [int(x) for x in row.get("history_item_id", [])]
            titles = [compact(x, 300) for x in row.get("history_item_title", [])]
            ratings = [float(x) for x in row.get("history_rating", [])]
            if len(titles) < args.min_history or float(row.get("rating", 0.0)) < args.min_rating:
                skipped += 1
                continue
            if any(not title for title in titles):
                bad_positions = [idx for idx, title in enumerate(titles) if not title]
                bad_item_ids = [history_item_ids[idx] if idx < len(history_item_ids) else None for idx in bad_positions]
                raise ValueError(
                    f"Empty history_item_title in {args.category}:{split}: "
                    f"user_id={row.get('user_id')} interaction_id={row.get('interaction_id')} "
                    f"positions={bad_positions} item_ids={bad_item_ids}. "
                    "Fix the source RRec data or item metadata before building examples."
                )
            if len(ratings) != len(titles):
                raise ValueError(
                    f"history_rating length mismatch in {args.category}:{split}: "
                    f"user_id={row.get('user_id')} interaction_id={row.get('interaction_id')} "
                    f"titles={len(titles)} ratings={len(ratings)}."
                )

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
                    "history_item_ids": history_item_ids[-args.max_history_items :],
                    "history_item_asins": list(row.get("item_asins", []))[-args.max_history_items :],
                    "history_item_count": min(len(titles), args.max_history_items) if args.max_history_items > 0 else len(titles),
                    "user_history": history_text(
                        args.category,
                        titles,
                        ratings,
                        args.max_history_items,
                        item_ids=history_item_ids,
                        item_map=item_map,
                        metadata_mode=args.history_metadata_mode,
                        max_item_chars=args.history_max_item_chars,
                        summary_map=summary_map,
                    ),
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
        "history_metadata_mode": args.history_metadata_mode,
        "history_max_item_chars": args.history_max_item_chars,
        "item_summary": args.item_summary,
    }
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
