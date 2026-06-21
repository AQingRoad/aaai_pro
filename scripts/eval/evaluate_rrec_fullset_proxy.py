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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from datasets import load_from_disk

from rubric_cot_pipeline.embeddings import DEFAULT_RECOMMENDATION_QUERY_INSTRUCTION, Qwen3TextEmbedder
from rubric_cot_pipeline.item_metadata import build_item_map, build_item_text, history_text


WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'-]*")


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


def rank_target_lexical(query: str, item_ids: list[int], item_vecs: list[Counter[str]], item_norms: list[float], target_id: int) -> int:
    query_vec = counts(query)
    query_norm = norm(query_vec)
    scores = [
        (cosine(query_vec, query_norm, item_vec, item_norm), item_id)
        for item_id, item_vec, item_norm in zip(item_ids, item_vecs, item_norms)
    ]
    scores.sort(reverse=True)
    return next((idx for idx, (_, item_id) in enumerate(scores, start=1) if item_id == target_id), len(scores) + 1)


def rank_target_embedding(query: str, item_ids: list[int], item_embs: torch.Tensor, target_id: int, embedder: Qwen3TextEmbedder) -> int:
    query_emb = embedder.encode_queries([query])[0]
    scores = torch.mv(item_embs, query_emb)
    order = torch.argsort(scores, descending=True)
    target_index = item_ids.index(target_id)
    matches = (order == target_index).nonzero(as_tuple=False)
    return int(matches[0].item()) + 1 if len(matches) else len(item_ids) + 1


def metrics_at_rank(rank: int, ks: list[int]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in ks:
        hit = 1.0 if rank <= k else 0.0
        ndcg = 1.0 / math.log2(rank + 1) if rank <= k else 0.0
        out[f"HR@{k}"] = hit
        out[f"NDCG@{k}"] = ndcg
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/root/autodl-tmp/rec/RRec_official/data")
    parser.add_argument("--category", required=True)
    parser.add_argument("--dataset-dir", default="")
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--max-examples", type=int, default=100)
    parser.add_argument("--max-history-items", type=int, default=20)
    parser.add_argument(
        "--history-metadata-mode",
        choices=["none", "compact"],
        default=os.getenv("HISTORY_METADATA_MODE", "none"),
    )
    parser.add_argument("--history-max-item-chars", type=int, default=int(os.getenv("HISTORY_MAX_ITEM_CHARS", "320")))
    parser.add_argument("--ks", default="5,10,20")
    parser.add_argument("--scorer", choices=["lexical", "qwen3_embedding"], default="lexical")
    parser.add_argument("--embedding-model", default="/root/autodl-tmp/modelscope_cache/models/Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--embedding-max-length", type=int, default=8192)
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    parser.add_argument("--embedding-output-dim", type=int, default=0)
    parser.add_argument("--query-instruction", default=DEFAULT_RECOMMENDATION_QUERY_INSTRUCTION)
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--device", default=os.getenv("QWEN3_EMBEDDING_DEVICE", "cuda:0"))
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    ks = [int(x.strip()) for x in args.ks.split(",") if x.strip()]
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else Path(args.data_root) / f"{args.category}_0_2022-10-2023-10"
    ds = load_from_disk(str(dataset_dir))
    item_map = build_item_map(ds["item_info"])

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

    rows = ds[args.split]
    if args.max_examples > 0:
        rows = rows.select(range(min(args.max_examples, len(rows))))

    totals = {f"HR@{k}": 0.0 for k in ks} | {f"NDCG@{k}": 0.0 for k in ks}
    ranks: list[int] = []
    for row in rows:
        prompt = history_text(
            args.category,
            [str(x) for x in row.get("history_item_title", [])],
            [float(x) for x in row.get("history_rating", [])],
            args.max_history_items,
            item_ids=[int(x) for x in (row.get("history_item_id") or row.get("history_item_ids") or [])],
            item_map=item_map,
            metadata_mode=args.history_metadata_mode,
            max_item_chars=args.history_max_item_chars,
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
        "history_metadata_mode": args.history_metadata_mode,
        "history_max_item_chars": args.history_max_item_chars,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
