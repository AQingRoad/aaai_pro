#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rubric_cot_pipeline.io import load_json, write_jsonl


HISTORY_MARKER = "this user's movie viewing history over time is listed below."


def extract_preamble(prompt: str) -> str:
    if HISTORY_MARKER in prompt:
        return prompt.split(HISTORY_MARKER, 1)[0].strip() + " " + HISTORY_MARKER
    return "This user's movie viewing history over time is listed below."


def build_history_prompt(preamble: str, history_entries: list[str]) -> str:
    history = "; ".join(history_entries)
    return f"{preamble} {history}."


def item_title(datamaps: dict, internal_item_id: int | str) -> str:
    key = str(internal_item_id)
    return datamaps.get("itemid2title", {}).get(key) or f"item-{key}"


def raw_item_id(datamaps: dict, internal_item_id: int | str) -> str:
    key = str(internal_item_id)
    return str(datamaps.get("id2item", {}).get(key, key))


def choose_target_index(
    ratings: list[int],
    min_history: int,
    min_target_rating: int,
    allow_low_target: bool,
) -> int | None:
    positives = [idx for idx, rating in enumerate(ratings) if idx >= min_history and rating >= min_target_rating]
    if positives:
        return positives[-1]
    if allow_low_target and len(ratings) > min_history:
        return len(ratings) - 1
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default="/root/autodl-tmp/rec/Open-World-Knowledge-Augmented-Recommendation")
    parser.add_argument("--dataset", default="ml-1m")
    parser.add_argument("--output", default="data/ml1m_examples.jsonl")
    parser.add_argument("--split", choices=["train", "test", "all"], default="train")
    parser.add_argument("--max-users", type=int, default=1000)
    parser.add_argument("--max-history-items", type=int, default=50)
    parser.add_argument("--min-history", type=int, default=5)
    parser.add_argument("--min-target-rating", type=int, default=4)
    parser.add_argument("--allow-low-target", action="store_true")
    parser.add_argument("--seed", type=int, default=20260614)
    args = parser.parse_args()

    root = Path(args.source_root)
    data_root = root / "data" / args.dataset
    user_klg = load_json(data_root / "knowledge" / "user.klg")
    item_klg = load_json(data_root / "knowledge" / "item.klg")
    datamaps = load_json(data_root / "proc_data" / "datamaps.json")
    sequential = load_json(data_root / "proc_data" / "sequential_data.json")
    split = load_json(data_root / "proc_data" / "train_test_split.json")

    if args.split == "all":
        user_ids = list(user_klg.keys())
    else:
        user_ids = [str(uid) for uid in split[args.split]]
    user_ids = [uid for uid in user_ids if uid in user_klg and uid in datamaps.get("user2id", {})]

    rng = random.Random(args.seed)
    if args.max_users > 0 and args.max_users < len(user_ids):
        user_ids = rng.sample(user_ids, args.max_users)
    user_ids = sorted(user_ids, key=lambda x: int(x))

    rows = []
    skipped = 0
    for user_id in user_ids:
        internal_uid = str(datamaps["user2id"][user_id])
        if internal_uid not in sequential:
            skipped += 1
            continue
        item_ids, ratings = sequential[internal_uid]
        ratings = [int(x) for x in ratings]
        target_idx = choose_target_index(
            ratings,
            min_history=args.min_history,
            min_target_rating=args.min_target_rating,
            allow_low_target=args.allow_low_target,
        )
        if target_idx is None:
            skipped += 1
            continue

        start = max(0, target_idx - args.max_history_items)
        history_entries = [
            f"{item_title(datamaps, iid)}, {rating} stars"
            for iid, rating in zip(item_ids[start:target_idx], ratings[start:target_idx])
        ]
        if len(history_entries) < args.min_history:
            skipped += 1
            continue

        target_internal = item_ids[target_idx]
        target_raw = raw_item_id(datamaps, target_internal)
        target_title = item_title(datamaps, target_internal)
        target_knowledge = item_klg.get(target_raw, {})
        target_text = " ".join(
            part.strip()
            for part in [target_title, target_knowledge.get("ans", "")]
            if part and part.strip()
        )
        source_prompt = user_klg[user_id].get("prompt", "")
        preamble = extract_preamble(source_prompt)
        user_history = build_history_prompt(preamble, history_entries)

        rows.append(
            {
                "dataset": args.dataset,
                "user_id": user_id,
                "internal_user_id": internal_uid,
                "target_item_internal_id": str(target_internal),
                "target_item_raw_id": target_raw,
                "target_item_title": target_title,
                "target_item_text": target_text,
                "target_rating": ratings[target_idx],
                "history_item_count": len(history_entries),
                "user_history": re.sub(r"\s+", " ", user_history).strip(),
                "source_prompt_full": source_prompt,
                "source_answer": user_klg[user_id].get("ans", ""),
            }
        )

    count = write_jsonl(args.output, rows)
    stats = {
        "dataset": args.dataset,
        "split": args.split,
        "selected_users": count,
        "skipped_users": skipped,
        "output": args.output,
        "source_root": str(root),
    }
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
