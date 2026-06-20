#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rubric_cot_pipeline.embeddings import append_recommendation_reasoning
from rubric_cot_pipeline.io import read_jsonl, write_jsonl


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


def load_item_map(path: str) -> dict[int, dict[str, Any]]:
    if not path:
        return {}
    item_map: dict[int, dict[str, Any]] = {}
    for row in read_jsonl(path):
        try:
            item_map[int(row["item_id"])] = row
        except Exception:
            continue
    return item_map


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


def row_key(row: dict[str, Any]) -> str:
    return str(row.get("example_id") or row.get("user_id") or row.get("interaction_id") or "")


def target_text(row: dict[str, Any], item_map: dict[int, dict[str, Any]], max_item_chars: int) -> str:
    text = str(row.get("target_item_text") or row.get("positive") or "").strip()
    if text:
        return text
    title = str(row.get("target_item_title") or row.get("item_title") or "").strip()
    target_id = row.get("target_item_id", row.get("item_id"))
    try:
        item = item_map.get(int(target_id))
    except Exception:
        item = None
    return build_item_text(item, title, max_item_chars)


def selected_candidates(row: dict[str, Any], candidate_index: int) -> list[dict[str, Any]]:
    candidates = row.get("candidates")
    if isinstance(candidates, list):
        if candidate_index >= 0:
            return [c for c in candidates if int(c.get("candidate_index", -1)) == candidate_index]
        return [c for c in candidates if isinstance(c, dict)]
    if any(key in row for key in ("think", "answer", "cot")):
        return [row]
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a CoT-aware embedding training dataset from one-CoT candidate lists.")
    parser.add_argument("--candidate-lists", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--item-info", default="")
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--candidate-index", type=int, default=0)
    parser.add_argument("--cot-text-mode", choices=["answer", "think", "tagged", "full"], default="answer")
    parser.add_argument("--max-cot-chars", type=int, default=1200)
    parser.add_argument("--max-item-chars", type=int, default=1400)
    parser.add_argument("--include-history", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-cot", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    item_map = load_item_map(args.item_info)
    rows = []
    stats = {
        "source_rows": 0,
        "history_pairs": 0,
        "cot_pairs": 0,
        "skipped_no_positive": 0,
        "skipped_no_cot": 0,
    }

    for row in read_jsonl(args.candidate_lists, limit=args.max_examples):
        stats["source_rows"] += 1
        history = str(row.get("user_history") or row.get("query") or row.get("source_prompt") or "").strip()
        positive = target_text(row, item_map, args.max_item_chars)
        if not history or not positive:
            stats["skipped_no_positive"] += 1
            continue

        base_meta = {
            "source_example_id": row_key(row),
            "category": row.get("category"),
            "split": row.get("split"),
            "user_id": row.get("user_id"),
            "interaction_id": row.get("interaction_id"),
            "target_item_id": row.get("target_item_id", row.get("item_id")),
            "target_item_title": row.get("target_item_title", row.get("item_title", "")),
            "target_rating": row.get("target_rating", row.get("rating")),
            "history_item_count": row.get("history_item_count"),
        }

        if args.include_history:
            rows.append(
                {
                    **base_meta,
                    "example_id": f"{row_key(row)}:history",
                    "query_type": "history",
                    "query": history,
                    "positive": positive,
                }
            )
            stats["history_pairs"] += 1

        if not args.include_cot:
            continue
        candidates = selected_candidates(row, args.candidate_index)
        if not candidates:
            stats["skipped_no_cot"] += 1
            continue
        for candidate in candidates:
            cot = compact(candidate_text(candidate, args.cot_text_mode), args.max_cot_chars)
            if not cot:
                stats["skipped_no_cot"] += 1
                continue
            query = append_recommendation_reasoning(history, cot)
            cand_idx = candidate.get("candidate_index", args.candidate_index)
            rows.append(
                {
                    **base_meta,
                    "example_id": f"{row_key(row)}:cot:{cand_idx}",
                    "query_type": f"history_plus_cot_{args.cot_text_mode}",
                    "candidate_id": candidate.get("candidate_id"),
                    "candidate_index": cand_idx,
                    "temperature": candidate.get("temperature"),
                    "query": query,
                    "positive": positive,
                }
            )
            stats["cot_pairs"] += 1

    count = write_jsonl(args.output, rows)
    stats["written"] = count
    stats["output"] = args.output
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
