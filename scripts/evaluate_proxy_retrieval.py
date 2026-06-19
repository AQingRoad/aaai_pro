#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import load_from_disk

from rubric_cot_pipeline.embeddings import append_recommendation_reasoning
from rubric_cot_pipeline.io import read_jsonl, write_jsonl
from rubric_cot_pipeline.rubric import WORD_RE, normalize


def vectorize(text: str) -> Counter[str]:
    return Counter(WORD_RE.findall(normalize(text)))


def cosine(left: Counter[str], right: Counter[str], left_norm: float, right_norm: float) -> float:
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    dot = sum(v * right.get(k, 0) for k, v in left.items())
    return dot / (left_norm * right_norm)


def norm(vec: Counter[str]) -> float:
    return math.sqrt(sum(v * v for v in vec.values()))


def compact(value: Any, limit: int = 1200) -> str:
    if isinstance(value, (list, tuple)):
        text = " ".join(str(x) for x in value[:8])
    else:
        text = str(value or "")
    text = " ".join(text.split())
    return text[:limit]


def item_text(row: dict[str, Any]) -> str:
    parts = [
        compact(row.get("title"), 300),
        compact(row.get("main_category"), 200),
        compact(row.get("categories"), 400),
        compact(row.get("features"), 700),
        compact(row.get("description"), 900),
    ]
    return " ".join(p for p in parts if p)


def load_items(dataset_dir: Path) -> list[dict[str, Any]]:
    ds = load_from_disk(str(dataset_dir))
    items = []
    for row in ds["item_info"]:
        item_id = int(row["item_id"])
        if item_id <= 0:
            continue
        text = item_text(dict(row))
        vec = vectorize(text)
        items.append({"item_id": item_id, "text": text, "vec": vec, "norm": norm(vec)})
    return items


def load_cot_map(path: str) -> dict[str, str]:
    if not path:
        return {}
    cot_map = {}
    for row in read_jsonl(path):
        example_id = row.get("example_id")
        if example_id and row.get("cot"):
            cot_map[str(example_id)] = str(row["cot"])
    return cot_map


def rank_target(query: str, target_item_id: int, items: list[dict[str, Any]]) -> int | None:
    qvec = vectorize(query)
    qnorm = norm(qvec)
    scores = []
    for item in items:
        scores.append((cosine(qvec, item["vec"], qnorm, item["norm"]), item["item_id"]))
    scores.sort(reverse=True)
    for idx, (_, item_id) in enumerate(scores, start=1):
        if item_id == target_item_id:
            return idx
    return None


def metrics_from_ranks(ranks: list[int], ks: list[int]) -> dict[str, float]:
    out: dict[str, float] = {"examples": float(len(ranks))}
    for k in ks:
        hits = [1.0 if r <= k else 0.0 for r in ranks]
        ndcgs = [1.0 / math.log2(r + 1) if r <= k else 0.0 for r in ranks]
        out[f"HR@{k}"] = sum(hits) / len(ranks) if ranks else 0.0
        out[f"NDCG@{k}"] = sum(ndcgs) / len(ranks) if ranks else 0.0
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--data-root", default="/root/autodl-tmp/rec/RRec_official/data")
    parser.add_argument("--dataset-dir", default="")
    parser.add_argument("--cot-file", default="")
    parser.add_argument("--mode", choices=["baseline", "with_cot"], default="baseline")
    parser.add_argument("--require-cot", action="store_true")
    parser.add_argument("--ks", default="5,10,20")
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--output", default="")
    parser.add_argument("--ranks-output", default="")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else Path(args.data_root) / f"{args.category}_0_2022-10-2023-10"
    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    cot_map = load_cot_map(args.cot_file)
    items = load_items(dataset_dir)

    ranks = []
    rank_rows = []
    for row in read_jsonl(args.examples, limit=args.max_examples):
        example_id = str(row.get("example_id", ""))
        cot = cot_map.get(example_id, "")
        if args.require_cot and not cot:
            continue
        query = row["user_history"]
        if args.mode == "with_cot" and cot:
            query = append_recommendation_reasoning(query, cot)
        target_id = int(row["target_item_id"])
        rank = rank_target(query, target_id, items)
        if rank is None:
            continue
        ranks.append(rank)
        rank_rows.append(
            {
                "example_id": example_id,
                "user_id": row.get("user_id"),
                "target_item_id": target_id,
                "rank": rank,
                "mode": args.mode,
            }
        )

    metrics = metrics_from_ranks(ranks, ks)
    metrics.update(
        {
            "category": args.category,
            "mode": args.mode,
            "item_count": len(items),
            "cot_file": args.cot_file,
            "examples_file": args.examples,
        }
    )
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.ranks_output:
        write_jsonl(args.ranks_output, rank_rows)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
