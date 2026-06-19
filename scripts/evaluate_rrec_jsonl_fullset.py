#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from rubric_cot_pipeline.embeddings import DEFAULT_RECOMMENDATION_QUERY_INSTRUCTION, Qwen3TextEmbedder
from rubric_cot_pipeline.io import read_jsonl


WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'-]*")


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


def category_label(category: str) -> str:
    return category.replace("_", " ").replace("And", "and")


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


def counts(text: str) -> Counter[str]:
    return Counter(WORD_RE.findall((text or "").lower()))


def norm(vec: Counter[str]) -> float:
    return math.sqrt(sum(v * v for v in vec.values()))


def cosine(left: Counter[str], left_norm: float, right: Counter[str], right_norm: float) -> float:
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    if len(left) > len(right):
        left, right = right, left
    dot = sum(v * right.get(k, 0) for k, v in left.items())
    return dot / (left_norm * right_norm)


def metrics_at_rank(rank: int, ks: list[int]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in ks:
        hit = 1.0 if rank <= k else 0.0
        ndcg = 1.0 / math.log2(rank + 1) if rank <= k else 0.0
        out[f"HR@{k}"] = hit
        out[f"NDCG@{k}"] = ndcg
    return out


def rank_target_lexical(
    query: str,
    item_ids: list[int],
    item_vecs: list[Counter[str]],
    item_norms: list[float],
    target_id: int,
) -> int:
    query_vec = counts(query)
    query_norm = norm(query_vec)
    scores = [
        (cosine(query_vec, query_norm, item_vec, item_norm), item_id)
        for item_id, item_vec, item_norm in zip(item_ids, item_vecs, item_norms)
    ]
    scores.sort(reverse=True)
    return next((idx for idx, (_, item_id) in enumerate(scores, start=1) if item_id == target_id), len(scores) + 1)


def rank_target_embedding(
    query: str,
    item_ids: list[int],
    item_embs: torch.Tensor,
    target_id: int,
    embedder: Qwen3TextEmbedder,
) -> int:
    query_emb = embedder.encode_queries([query])[0]
    scores = torch.mv(item_embs, query_emb)
    order = torch.argsort(scores, descending=True)
    target_index = item_ids.index(target_id)
    matches = (order == target_index).nonzero(as_tuple=False)
    return int(matches[0].item()) + 1 if len(matches) else len(item_ids) + 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", required=True)
    parser.add_argument("--item-info", required=True)
    parser.add_argument("--category", default="CDs_and_Vinyl")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--max-history-items", type=int, default=20)
    parser.add_argument("--ks", default="5,10,20")
    parser.add_argument("--scorer", choices=["lexical", "qwen3_embedding"], default="qwen3_embedding")
    parser.add_argument("--embedding-model", default="")
    parser.add_argument("--embedding-max-length", type=int, default=2048)
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    parser.add_argument("--embedding-output-dim", type=int, default=0)
    parser.add_argument("--query-instruction", default=DEFAULT_RECOMMENDATION_QUERY_INSTRUCTION)
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--device", default=os.getenv("QWEN3_EMBEDDING_DEVICE", "cuda:0"))
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    ks = [int(x.strip()) for x in args.ks.split(",") if x.strip()]
    examples = list(read_jsonl(args.examples, limit=args.max_examples))
    if not examples:
        raise ValueError(f"No examples loaded from {args.examples}")

    item_rows = list(read_jsonl(args.item_info))
    item_map = {int(row["item_id"]): row for row in item_rows}

    item_ids: list[int] = []
    item_texts: list[str] = []
    item_vecs: list[Counter[str]] = []
    item_norms: list[float] = []
    for item_id, item in sorted(item_map.items()):
        text = build_item_text(item, str(item.get("title", "")), max_chars=1200)
        item_ids.append(item_id)
        item_texts.append(text)
        if args.scorer == "lexical":
            vec = counts(text)
            item_vecs.append(vec)
            item_norms.append(norm(vec))

    embedder = None
    item_embs = None
    if args.scorer == "qwen3_embedding":
        if not args.embedding_model:
            raise ValueError("--embedding-model is required for qwen3_embedding scorer")
        embedder = Qwen3TextEmbedder(
            args.embedding_model,
            max_length=args.embedding_max_length,
            batch_size=args.embedding_batch_size,
            torch_dtype=args.torch_dtype,
            device=args.device,
            query_instruction=args.query_instruction,
            output_dim=args.embedding_output_dim,
        )
        item_embs = embedder.encode_documents(item_texts)

    totals = {f"HR@{k}": 0.0 for k in ks} | {f"NDCG@{k}": 0.0 for k in ks}
    ranks: list[int] = []
    for row in examples:
        prompt = history_text(
            args.category,
            [str(x) for x in row.get("history_item_title", [])],
            [float(x) for x in row.get("history_rating", [])],
            args.max_history_items,
        )
        target_id = int(row["item_id"])
        if args.scorer == "lexical":
            rank = rank_target_lexical(prompt, item_ids, item_vecs, item_norms, target_id)
        else:
            rank = rank_target_embedding(prompt, item_ids, item_embs, target_id, embedder)  # type: ignore[arg-type]
        ranks.append(rank)
        row_metrics = metrics_at_rank(rank, ks)
        for key, value in row_metrics.items():
            totals[key] += value

    n = max(1, len(ranks))
    result = {
        "examples": args.examples,
        "item_info": args.item_info,
        "category": args.category,
        "split": args.split,
        "max_examples": args.max_examples,
        "evaluated": len(ranks),
        "num_items": len(item_ids),
        "mean_rank": sum(ranks) / n if ranks else None,
        "median_rank": sorted(ranks)[len(ranks) // 2] if ranks else None,
        "metrics": {key: value / n for key, value in totals.items()},
        "scorer": args.scorer,
        "embedding_model": args.embedding_model if args.scorer == "qwen3_embedding" else None,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
