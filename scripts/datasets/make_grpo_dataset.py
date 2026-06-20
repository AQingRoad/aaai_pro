#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rubric_cot_pipeline.embeddings import DEFAULT_RECOMMENDATION_QUERY_INSTRUCTION, Qwen3TextEmbedder
from rubric_cot_pipeline.io import read_jsonl, write_jsonl
from rubric_cot_pipeline.prompts import COT_SYSTEM, build_user_prompt
from rubric_cot_pipeline.rubric import hashed_cosine


def message_user_content(row: dict) -> str:
    for message in row.get("messages", []):
        if message.get("role") == "user":
            return message.get("content", "")
    return ""


def normalize_history(text: str) -> str:
    text = (text or "").strip()
    match = re.search(
        r"<Historical Interactions>\s*(.*?)\s*</Historical Interactions>",
        text,
        flags=re.DOTALL,
    )
    if match:
        text = match.group(1)
    return " ".join(text.split())


def row_history(row: dict) -> str:
    return normalize_history(row.get("user_history") or row.get("source_prompt") or message_user_content(row))


def row_interaction_key(row: dict) -> tuple[str, str] | None:
    user_id = row.get("user_id")
    interaction_id = row.get("interaction_id")
    if user_id is None or interaction_id is None:
        return None
    return str(user_id), str(interaction_id)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/ml1m_examples.jsonl")
    parser.add_argument("--output", default="outputs/grpo_train.jsonl")
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument(
        "--exclude-prompts-from",
        action="append",
        default=[],
        help="JSONL file whose user histories/chat prompts should be excluded from the GRPO dataset.",
    )
    parser.add_argument("--precompute-lexical-baseline", action="store_true")
    parser.add_argument("--baseline-mode", choices=["none", "lexical", "qwen3_embedding"], default="none")
    parser.add_argument("--embedding-model", default="/root/autodl-tmp/modelscope_cache/models/Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--embedding-max-length", type=int, default=8192)
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    parser.add_argument("--embedding-output-dim", type=int, default=0)
    parser.add_argument("--query-instruction", default=DEFAULT_RECOMMENDATION_QUERY_INSTRUCTION)
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--device", default=os.getenv("QWEN3_EMBEDDING_DEVICE", "cuda:0"))
    args = parser.parse_args()

    baseline_mode = "lexical" if args.precompute_lexical_baseline else args.baseline_mode
    embedder = None
    if baseline_mode == "qwen3_embedding":
        embedder = Qwen3TextEmbedder(
            args.embedding_model,
            max_length=args.embedding_max_length,
            batch_size=args.embedding_batch_size,
            torch_dtype=args.torch_dtype,
            device=args.device,
            query_instruction=args.query_instruction,
            output_dim=args.embedding_output_dim,
        )

    excluded_example_ids = set()
    excluded_interactions = set()
    excluded_histories = set()
    for exclude_path in args.exclude_prompts_from:
        for row in read_jsonl(exclude_path):
            example_id = row.get("example_id")
            if example_id:
                excluded_example_ids.add(str(example_id))
            interaction_key = row_interaction_key(row)
            if interaction_key is not None:
                excluded_interactions.add(interaction_key)
            history = row_history(row)
            if history:
                excluded_histories.add(history)

    rows = []
    for row in read_jsonl(args.input):
        history = row_history(row)
        example_id = str(row.get("example_id") or "")
        interaction_key = row_interaction_key(row)
        excluded_by_id = bool(example_id and example_id in excluded_example_ids) or (
            interaction_key is not None and interaction_key in excluded_interactions
        )
        excluded_by_history = (
            not excluded_example_ids
            and not excluded_interactions
            and history in excluded_histories
        )
        if excluded_by_id or excluded_by_history:
            continue
        out = {
            "example_id": row.get("example_id"),
            "user_id": row["user_id"],
            "interaction_id": row.get("interaction_id"),
            "dataset": row.get("dataset"),
            "category": row.get("category"),
            "split": row.get("split"),
            "source_prompt": history,
            "user_history": history,
            "target_item_id": row.get("target_item_id"),
            "target_item_title": row.get("target_item_title", ""),
            "target_item_text": row.get("target_item_text", ""),
            "target_rating": row.get("target_rating"),
            "history_item_ids": row.get("history_item_ids") or row.get("history_item_id") or [],
            "history_item_count": row.get("history_item_count"),
            "messages": [
                {"role": "system", "content": COT_SYSTEM},
                {"role": "user", "content": build_user_prompt(row["user_history"], row.get("category", ""))},
            ],
        }
        target_text = row.get("target_item_text") or row.get("target_item_title", "")
        if baseline_mode == "lexical":
            out["baseline_sim"] = hashed_cosine(history, target_text)
            out["baseline_embedder_mode"] = "lexical"
        rows.append(out)
        if args.max_examples and len(rows) >= args.max_examples:
            break
    if baseline_mode == "qwen3_embedding" and rows:
        histories = [str(row.get("user_history") or "") for row in rows]
        targets = [str(row.get("target_item_text") or row.get("target_item_title") or "") for row in rows]
        sims = embedder.pairwise_cosine(histories, targets)  # type: ignore[union-attr]
        for row, sim in zip(rows, sims):
            row["baseline_sim"] = sim
            row["baseline_embedder_mode"] = "qwen3_embedding"
    count = write_jsonl(args.output, rows)
    print(f"Wrote {count} GRPO rows to {args.output}")


if __name__ == "__main__":
    main()
