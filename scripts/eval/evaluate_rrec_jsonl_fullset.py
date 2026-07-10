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

from rubric_cot_pipeline.embeddings import (
    DEFAULT_RECOMMENDATION_QUERY_INSTRUCTION,
    Qwen3TextEmbedder,
    append_recommendation_reasoning,
)
from rubric_cot_pipeline.io import read_jsonl
from rubric_cot_pipeline.item_metadata import build_item_summary_map
from rubric_cot_pipeline.item_metadata import build_item_text as metadata_build_item_text
from rubric_cot_pipeline.item_metadata import history_text as metadata_history_text


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


def candidate_text(candidate: dict[str, Any], mode: str) -> str:
    think = str(candidate.get("think") or "").strip()
    answer = str(candidate.get("answer") or "").strip()
    cot = str(candidate.get("cot") or "").strip()
    if mode == "answer":
        return answer or cot or think
    if mode == "think":
        return think or cot or answer
    if mode == "tagged":
        if think and answer:
            return f"<think>\n{think}\n</think>\n<answer>\n{answer}\n</answer>"
        return cot or answer or think
    if mode == "full":
        return cot or candidate_text(candidate, "tagged")
    raise ValueError(f"Unsupported cot text mode: {mode}")


def selected_candidate(row: dict[str, Any], candidate_index: int) -> dict[str, Any] | None:
    candidates = row.get("candidates")
    if isinstance(candidates, list):
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            try:
                index = int(candidate.get("candidate_index", -1))
            except (TypeError, ValueError):
                index = -1
            if index == candidate_index:
                return candidate
        return None
    if any(key in row for key in ("think", "answer", "cot")):
        return row
    return None


def row_target_id(row: dict[str, Any]) -> int:
    for key in ("item_id", "target_item_id"):
        value = row.get(key)
        if value is not None:
            return int(value)
    raise KeyError("row must contain item_id or target_item_id")


def row_history_item_ids(row: dict[str, Any]) -> list[int]:
    return [int(x) for x in (row.get("history_item_id") or row.get("history_item_ids") or [])]


def env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def masked_item_indices(
    history_item_ids: list[int],
    target_id: int,
    item_index: dict[int, int],
    mask_history_items: bool,
    mask_pad_item: bool,
    keep_target_unmasked: bool,
) -> set[int]:
    masked_item_ids: set[int] = set()
    if mask_history_items:
        masked_item_ids.update(history_item_ids)
    if mask_pad_item:
        masked_item_ids.add(0)
    if keep_target_unmasked:
        masked_item_ids.discard(target_id)
    return {item_index[item_id] for item_id in masked_item_ids if item_id in item_index}


def build_rebuilt_history_query(row: dict[str, Any], args: argparse.Namespace, item_map: dict[int, dict[str, Any]], summary_map: dict[int, str]) -> str:
    return metadata_history_text(
        args.category,
        [str(x) for x in row.get("history_item_title", [])],
        [float(x) for x in row.get("history_rating", [])],
        args.max_history_items,
        item_ids=row_history_item_ids(row),
        item_map=item_map,
        metadata_mode=args.history_metadata_mode,
        max_item_chars=args.history_max_item_chars,
        summary_map=summary_map,
    )


def build_query(
    row: dict[str, Any],
    args: argparse.Namespace,
    item_map: dict[int, dict[str, Any]],
    summary_map: dict[int, str],
) -> tuple[str, bool]:
    rebuilt = None
    if args.query_mode == "rebuild_history":
        return build_rebuilt_history_query(row, args, item_map, summary_map), False

    history = str(row.get("user_history") or row.get("query") or "").strip()
    if not history:
        rebuilt = build_rebuilt_history_query(row, args, item_map, summary_map)
        history = rebuilt

    if args.query_mode == "user_history":
        return history, False

    candidate = selected_candidate(row, args.candidate_index)
    cot = candidate_text(candidate, args.cot_text_mode) if candidate else ""
    if args.require_cot and not cot:
        raise ValueError(f"Missing CoT candidate for example {row.get('example_id') or row.get('interaction_id')}")
    return append_recommendation_reasoning(history, cot), bool(cot)


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
    item_index: dict[int, int],
    masked_indices: set[int],
) -> int:
    target_index = item_index.get(target_id)
    if target_index is None:
        return len(item_ids) + 1

    query_vec = counts(query)
    query_norm = norm(query_vec)
    scores = [
        cosine(query_vec, query_norm, item_vec, item_norm)
        for item_vec, item_norm in zip(item_vecs, item_norms)
    ]
    target_score = -float("inf") if target_index in masked_indices else scores[target_index]
    return 1 + sum(
        1
        for index, score in enumerate(scores)
        if index not in masked_indices and score > target_score
    )


def rank_target_embedding(
    query: str,
    item_ids: list[int],
    item_embs: torch.Tensor,
    target_id: int,
    embedder: Qwen3TextEmbedder,
    item_index: dict[int, int],
    masked_indices: set[int],
) -> int:
    target_index = item_index.get(target_id)
    if target_index is None:
        return len(item_ids) + 1

    query_emb = embedder.encode_queries([query])[0]
    scores = torch.mv(item_embs, query_emb).clone()
    if masked_indices:
        scores[list(masked_indices)] = -float("inf")
    target_score = scores[target_index]
    return int((scores > target_score).sum().item()) + 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", required=True)
    parser.add_argument("--item-info", required=True)
    parser.add_argument("--category", default="CDs_and_Vinyl")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--max-history-items", type=int, default=20)
    parser.add_argument(
        "--history-metadata-mode",
        choices=["none", "compact", "summary"],
        default=os.getenv("HISTORY_METADATA_MODE", "none"),
    )
    parser.add_argument("--history-max-item-chars", type=int, default=int(os.getenv("HISTORY_MAX_ITEM_CHARS", "320")))
    parser.add_argument("--item-summary", default=os.getenv("ITEM_METADATA_SUMMARY", ""))
    parser.add_argument(
        "--query-mode",
        choices=["rebuild_history", "user_history", "history_plus_cot"],
        default=os.getenv("EVAL_QUERY_MODE", "rebuild_history"),
        help="rebuild_history preserves the original RRec eval path; user_history uses row.user_history; history_plus_cot appends the selected generated CoT.",
    )
    parser.add_argument("--cot-text-mode", choices=["answer", "think", "tagged", "full"], default=os.getenv("EVAL_COT_TEXT_MODE", "tagged"))
    parser.add_argument("--candidate-index", type=int, default=int(os.getenv("EVAL_CANDIDATE_INDEX", "0")))
    parser.add_argument("--require-cot", action=argparse.BooleanOptionalAction, default=os.getenv("EVAL_REQUIRE_COT", "0").lower() in {"1", "true", "yes", "on"})
    parser.add_argument("--ks", default="5,10,20")
    parser.add_argument("--scorer", choices=["lexical", "qwen3_embedding"], default="qwen3_embedding")
    parser.add_argument("--embedding-model", default="")
    parser.add_argument("--embedding-max-length", type=int, default=2048)
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    parser.add_argument("--embedding-output-dim", type=int, default=0)
    parser.add_argument("--max-item-chars", type=int, default=int(os.getenv("MAX_ITEM_CHARS", "0")))
    parser.add_argument("--query-instruction", default=DEFAULT_RECOMMENDATION_QUERY_INSTRUCTION)
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--device", default=os.getenv("QWEN3_EMBEDDING_DEVICE", "cuda:0"))
    parser.add_argument("--mask-history-items", dest="mask_history_items", action="store_true", default=env_flag("EVAL_MASK_HISTORY_ITEMS", True))
    parser.add_argument("--no-mask-history-items", dest="mask_history_items", action="store_false")
    parser.add_argument("--mask-pad-item", dest="mask_pad_item", action="store_true", default=env_flag("EVAL_MASK_PAD_ITEM", True))
    parser.add_argument("--no-mask-pad-item", dest="mask_pad_item", action="store_false")
    parser.add_argument("--keep-target-unmasked", action="store_true", default=env_flag("EVAL_KEEP_TARGET_UNMASKED", False))
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    ks = [int(x.strip()) for x in args.ks.split(",") if x.strip()]
    examples = list(read_jsonl(args.examples, limit=args.max_examples))
    if not examples:
        raise ValueError(f"No examples loaded from {args.examples}")

    item_rows = list(read_jsonl(args.item_info))
    item_map = {int(row["item_id"]): row for row in item_rows}
    summary_map = build_item_summary_map(read_jsonl(args.item_summary)) if args.item_summary else {}

    item_ids: list[int] = []
    item_texts: list[str] = []
    item_vecs: list[Counter[str]] = []
    item_norms: list[float] = []
    for item_id, item in sorted(item_map.items()):
        text = metadata_build_item_text(item, str(item.get("title", "")), max_chars=args.max_item_chars)
        item_ids.append(item_id)
        item_texts.append(text)
        if args.scorer == "lexical":
            vec = counts(text)
            item_vecs.append(vec)
            item_norms.append(norm(vec))
    item_index = {item_id: index for index, item_id in enumerate(item_ids)}

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
    cot_queries = 0
    masked_score_total = 0
    target_in_history_count = 0
    for row in examples:
        prompt, used_cot = build_query(row, args, item_map, summary_map)
        if used_cot:
            cot_queries += 1
        target_id = row_target_id(row)
        history_item_ids = row_history_item_ids(row)
        if target_id in set(history_item_ids):
            target_in_history_count += 1
        masked_indices = masked_item_indices(
            history_item_ids,
            target_id,
            item_index,
            args.mask_history_items,
            args.mask_pad_item,
            args.keep_target_unmasked,
        )
        masked_score_total += len(masked_indices)
        if args.scorer == "lexical":
            rank = rank_target_lexical(prompt, item_ids, item_vecs, item_norms, target_id, item_index, masked_indices)
        else:
            rank = rank_target_embedding(prompt, item_ids, item_embs, target_id, embedder, item_index, masked_indices)  # type: ignore[arg-type]
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
        "max_item_chars": args.max_item_chars,
        "query_mode": args.query_mode,
        "cot_text_mode": args.cot_text_mode if args.query_mode == "history_plus_cot" else None,
        "candidate_index": args.candidate_index if args.query_mode == "history_plus_cot" else None,
        "cot_queries": cot_queries,
        "require_cot": args.require_cot,
        "history_metadata_mode": args.history_metadata_mode,
        "history_max_item_chars": args.history_max_item_chars,
        "item_summary": args.item_summary,
        "mask_history_items": args.mask_history_items,
        "mask_pad_item": args.mask_pad_item,
        "keep_target_unmasked": args.keep_target_unmasked,
        "masked_score_total": masked_score_total,
        "masked_score_mean": masked_score_total / n if ranks else 0.0,
        "target_in_history_count": target_in_history_count,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
