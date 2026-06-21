#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from rubric_cot_pipeline.embeddings import DEFAULT_RECOMMENDATION_QUERY_INSTRUCTION, Qwen3TextEmbedder
from rubric_cot_pipeline.io import read_jsonl, write_jsonl
from rubric_cot_pipeline.item_metadata import build_item_map, build_item_summary_map
from scripts.data.make_phase0_embedder_dataset_from_examples import build_summary_item_text


def as_int_set(value: Any) -> set[int]:
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        out = set()
        for item in value:
            try:
                out.add(int(item))
            except Exception:
                continue
        return out
    try:
        return {int(value)}
    except Exception:
        return set()


def item_title(item: Mapping[str, Any] | None, fallback: str = "") -> str:
    if not item:
        return fallback
    return str(item.get("title") or fallback)


def build_item_texts(
    item_map: Mapping[int, Mapping[str, Any]],
    summary_map: Mapping[int, str],
    max_item_chars: int,
) -> tuple[list[int], list[str]]:
    item_ids: list[int] = []
    item_texts: list[str] = []
    for item_id, item in sorted(item_map.items()):
        if item_id <= 0:
            continue
        item_ids.append(item_id)
        item_texts.append(
            build_summary_item_text(
                item,
                item_title(item),
                item_id,
                summary_map,
                max_item_chars,
            )
        )
    return item_ids, item_texts


def mine_for_batch(
    queries: list[str],
    rows: list[dict[str, Any]],
    item_ids: list[int],
    item_texts: list[str],
    item_embs: torch.Tensor,
    embedder: Qwen3TextEmbedder,
    num_negatives: int,
    candidate_pool: int,
    only_above_target: bool,
) -> tuple[list[list[int]], list[list[str]], list[int | None]]:
    query_embs = embedder.encode_queries(queries)
    scores = query_embs @ item_embs.T
    k = min(max(candidate_pool, num_negatives + 1), len(item_ids))
    top_indices = torch.topk(scores, k=k, dim=1).indices.cpu().tolist()
    item_index = {item_id: idx for idx, item_id in enumerate(item_ids)}

    batch_negative_ids: list[list[int]] = []
    batch_negative_texts: list[list[str]] = []
    target_ranks: list[int | None] = []
    for row_idx, row in enumerate(rows):
        target_id = int(row.get("target_item_id"))
        excluded = as_int_set(row.get("history_item_ids") or row.get("history_item_id"))
        excluded.add(target_id)
        excluded.add(0)

        target_rank = None
        target_index = item_index.get(target_id)
        if target_index is not None:
            better_count = int((scores[row_idx] > scores[row_idx, target_index]).sum().item())
            target_rank = better_count + 1
        target_ranks.append(target_rank)

        negative_ids: list[int] = []
        negative_texts: list[str] = []
        for index in top_indices[row_idx]:
            item_id = item_ids[index]
            if item_id in excluded:
                continue
            if only_above_target and target_index is not None and scores[row_idx, index] <= scores[row_idx, target_index]:
                continue
            negative_ids.append(item_id)
            negative_texts.append(item_texts[index])
            if len(negative_ids) >= num_negatives:
                break
        batch_negative_ids.append(negative_ids)
        batch_negative_texts.append(negative_texts)
    return batch_negative_ids, batch_negative_texts, target_ranks


def main() -> None:
    parser = argparse.ArgumentParser(description="Add embedding-mined hard negatives to phase0 embedder JSONL rows.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--item-info", required=True)
    parser.add_argument("--item-summary", default="")
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--num-negatives", type=int, default=8)
    parser.add_argument("--candidate-pool", type=int, default=200)
    parser.add_argument("--query-batch-size", type=int, default=32)
    parser.add_argument("--embedding-batch-size", type=int, default=128)
    parser.add_argument("--embedding-max-length", type=int, default=4096)
    parser.add_argument("--max-item-chars", type=int, default=1800)
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--query-instruction", default=DEFAULT_RECOMMENDATION_QUERY_INSTRUCTION)
    parser.add_argument("--only-above-target", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-fewer-negatives", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preview-cases", type=int, default=2)
    args = parser.parse_args()

    rows = list(read_jsonl(args.input, limit=args.max_rows))
    if not rows:
        raise ValueError(f"No rows loaded from {args.input}")
    item_map = build_item_map(read_jsonl(args.item_info))
    summary_map = build_item_summary_map(read_jsonl(args.item_summary)) if args.item_summary else {}
    item_ids, item_texts = build_item_texts(item_map, summary_map, args.max_item_chars)
    if not item_ids:
        raise ValueError(f"No item rows loaded from {args.item_info}")

    embedder = Qwen3TextEmbedder(
        args.embedding_model,
        max_length=args.embedding_max_length,
        batch_size=args.embedding_batch_size,
        torch_dtype=args.torch_dtype,
        device=args.device,
        query_instruction=args.query_instruction,
    )
    item_embs = embedder.encode_documents(item_texts)

    output_rows: list[dict[str, Any]] = []
    skipped_not_enough = 0
    target_ranks_seen: list[int] = []
    for start in range(0, len(rows), args.query_batch_size):
        batch_rows = rows[start : start + args.query_batch_size]
        queries = [str(row.get("query") or "") for row in batch_rows]
        negative_ids_batch, negative_texts_batch, ranks = mine_for_batch(
            queries,
            batch_rows,
            item_ids,
            item_texts,
            item_embs,
            embedder,
            args.num_negatives,
            args.candidate_pool,
            args.only_above_target,
        )
        for row, negative_ids, negative_texts, rank in zip(batch_rows, negative_ids_batch, negative_texts_batch, ranks):
            if rank is not None:
                target_ranks_seen.append(rank)
            if len(negative_texts) < args.num_negatives and not args.allow_fewer_negatives:
                skipped_not_enough += 1
                continue
            out = dict(row)
            out["negatives"] = negative_texts
            out["negative_item_ids"] = negative_ids
            out["hard_negative_mining"] = {
                "embedding_model": args.embedding_model,
                "num_negatives": args.num_negatives,
                "candidate_pool": args.candidate_pool,
                "only_above_target": args.only_above_target,
                "target_rank_before_mining": rank,
            }
            output_rows.append(out)

    count = write_jsonl(args.output, output_rows)
    for index, row in enumerate(output_rows[: args.preview_cases], start=1):
        print(
            json.dumps(
                {
                    "preview_type": "hard_negative_embedder_case",
                    "case_index": index,
                    "target_item_id": row.get("target_item_id"),
                    "target_item_title": row.get("target_item_title"),
                    "query": row.get("query"),
                    "positive": row.get("positive"),
                    "negative_item_ids": row.get("negative_item_ids"),
                    "negatives": row.get("negatives"),
                    "hard_negative_mining": row.get("hard_negative_mining"),
                },
                ensure_ascii=False,
                indent=2,
            )
        )

    rank_stats = {}
    if target_ranks_seen:
        sorted_ranks = sorted(target_ranks_seen)
        rank_stats = {
            "count": len(sorted_ranks),
            "mean": round(sum(sorted_ranks) / len(sorted_ranks), 2),
            "min": sorted_ranks[0],
            "max": sorted_ranks[-1],
            "p50": sorted_ranks[len(sorted_ranks) // 2],
        }
    stats = {
        "input": args.input,
        "output": args.output,
        "item_info": args.item_info,
        "item_summary": args.item_summary,
        "embedding_model": args.embedding_model,
        "source_rows": len(rows),
        "written": count,
        "skipped_not_enough_negatives": skipped_not_enough,
        "num_items": len(item_ids),
        "num_negatives": args.num_negatives,
        "candidate_pool": args.candidate_pool,
        "only_above_target": args.only_above_target,
        "target_rank_before_mining_stats": rank_stats,
    }
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
