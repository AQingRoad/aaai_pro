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
from peft import PeftModel
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from rubric_cot_pipeline.embeddings import (
    DEFAULT_RECOMMENDATION_QUERY_INSTRUCTION,
    Qwen3TextEmbedder,
    append_recommendation_reasoning,
)
from rubric_cot_pipeline.io import read_jsonl
from rubric_cot_pipeline.item_metadata import build_item_summary_map, build_item_text, history_text
from rubric_cot_pipeline.prompts import COT_SYSTEM, build_user_prompt


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


def rank_target(query: str, item_ids: list[int], item_vecs: list[Counter[str]], item_norms: list[float], target_id: int) -> int:
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
    target_index = item_ids.index(target_id)
    return rank_target_embedding_from_emb(query_emb, item_embs, target_index)


def rank_target_embedding_from_emb(query_emb: torch.Tensor, item_embs: torch.Tensor, target_index: int) -> int:
    scores = torch.mv(item_embs, query_emb)
    target_score = scores[target_index]
    return int((scores > target_score).sum().item()) + 1


def update_metrics(totals: dict[str, float], prefix: str, rank: int, ks: list[int]) -> None:
    for k in ks:
        totals[f"{prefix}_HR@{k}"] += 1.0 if rank <= k else 0.0
        totals[f"{prefix}_NDCG@{k}"] += 1.0 / math.log2(rank + 1) if rank <= k else 0.0


def resolve_dtype(name: str):
    name = (name or "bfloat16").lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    if name == "auto":
        return "auto"
    raise ValueError(f"Unsupported dtype: {name}")


def load_reasoner(model_path: str, adapter_path: str, torch_dtype: str, model_device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    for attr in ("base_model_tp_plan", "base_model_pp_plan", "base_model_ep_plan"):
        if hasattr(config, attr):
            setattr(config, attr, None)
    kwargs: dict[str, Any] = {
        "config": config,
        "trust_remote_code": True,
        "torch_dtype": resolve_dtype(torch_dtype),
        "tp_plan": None,
        "tp_size": None,
    }
    if model_device == "auto":
        kwargs["device_map"] = "auto"
    else:
        kwargs["device_map"] = {"": model_device}
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    except TypeError as exc:
        if "unexpected keyword argument" not in str(exc):
            raise
        retry_kwargs = {key: value for key, value in kwargs.items() if key not in {"tp_plan", "tp_size"}}
        model = AutoModelForCausalLM.from_pretrained(model_path, **retry_kwargs)
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def first_device(model) -> torch.device:
    return next(model.parameters()).device


@torch.no_grad()
def generate_cot(model, tokenizer, user_history: str, category: str, args) -> str:
    return generate_cots(model, tokenizer, [user_history], category, args)[0]


@torch.no_grad()
def generate_cots(model, tokenizer, user_histories: list[str], category: str, args) -> list[str]:
    messages = [
        [
            {"role": "system", "content": COT_SYSTEM},
            {"role": "user", "content": build_user_prompt(user_history, category)},
        ]
        for user_history in user_histories
    ]
    prompts = [tokenizer.apply_chat_template(row, tokenize=False, add_generation_prompt=True) for row in messages]
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_prompt_tokens,
    )
    device = first_device(model)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": args.temperature > 0,
    }
    if args.temperature > 0:
        gen_kwargs["temperature"] = args.temperature
        gen_kwargs["top_p"] = args.top_p
    output_ids = model.generate(**inputs, **gen_kwargs)
    generated = output_ids[:, inputs["input_ids"].shape[-1] :]
    return [text.strip() for text in tokenizer.batch_decode(generated, skip_special_tokens=True)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", default="")
    parser.add_argument("--item-info", default="")
    parser.add_argument("--data-root", default="")
    parser.add_argument("--category", required=True)
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--model", default="/root/autodl-tmp/modelscope_cache/models/Qwen/Qwen3-4B")
    parser.add_argument("--adapter", default="")
    parser.add_argument("--adapter-name", default="")
    parser.add_argument("--max-examples", type=int, default=20)
    parser.add_argument("--max-history-items", type=int, default=20)
    parser.add_argument(
        "--history-metadata-mode",
        choices=["none", "compact", "summary"],
        default=os.getenv("HISTORY_METADATA_MODE", "none"),
    )
    parser.add_argument("--history-max-item-chars", type=int, default=int(os.getenv("HISTORY_MAX_ITEM_CHARS", "320")))
    parser.add_argument("--item-summary", default=os.getenv("ITEM_METADATA_SUMMARY", ""))
    parser.add_argument("--max-prompt-tokens", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--generation-batch-size", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--model-device", default=os.getenv("REASONER_MODEL_DEVICE", "cuda:0"))
    parser.add_argument("--ks", default="5,10,20")
    parser.add_argument("--scorer", choices=["lexical", "qwen3_embedding"], default="lexical")
    parser.add_argument("--embedding-model", default="/root/autodl-tmp/modelscope_cache/models/Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--embedding-max-length", type=int, default=8192)
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    parser.add_argument("--embedding-output-dim", type=int, default=0)
    parser.add_argument("--query-instruction", default=DEFAULT_RECOMMENDATION_QUERY_INSTRUCTION)
    parser.add_argument("--embedding-torch-dtype", default="bfloat16")
    parser.add_argument("--embedding-device", default=os.getenv("QWEN3_EMBEDDING_DEVICE", "cuda:0"))
    parser.add_argument("--output", default="")
    parser.add_argument("--predictions-output", default="")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    if args.generation_batch_size < 1:
        raise ValueError("--generation-batch-size must be >= 1")
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")

    ks = [int(x.strip()) for x in args.ks.split(",") if x.strip()]
    examples_path = Path(args.examples) if args.examples else Path("github_artifacts") / args.category / "rrec_eval" / f"{args.split}.jsonl"
    item_info_path = Path(args.item_info) if args.item_info else Path("github_artifacts") / args.category / "rrec_eval" / "item_info.jsonl"
    if not examples_path.exists():
        raise FileNotFoundError(f"Examples JSONL not found: {examples_path}")
    if not item_info_path.exists():
        raise FileNotFoundError(f"Item info JSONL not found: {item_info_path}")
    all_rows = list(read_jsonl(examples_path))
    if not all_rows:
        raise ValueError(f"No examples loaded from {examples_path}")
    item_map = {int(row["item_id"]): row for row in read_jsonl(item_info_path)}
    summary_map = build_item_summary_map(read_jsonl(args.item_summary)) if args.item_summary else {}

    item_ids: list[int] = []
    item_id_to_index: dict[int, int] = {}
    item_texts: list[str] = []
    item_vecs: list[Counter[str]] = []
    item_norms: list[float] = []
    for item_id, item in sorted(item_map.items()):
        item_text = build_item_text(item, str(item.get("title", "")), max_chars=1200)
        item_id_to_index[item_id] = len(item_ids)
        item_ids.append(item_id)
        item_texts.append(item_text)
        if args.scorer == "lexical":
            vec = counts(item_text)
            item_vecs.append(vec)
            item_norms.append(norm(vec))

    embedder = None
    item_embs = None
    if args.scorer == "qwen3_embedding":
        embedder = Qwen3TextEmbedder(
            args.embedding_model,
            max_length=args.embedding_max_length,
            batch_size=args.embedding_batch_size,
            torch_dtype=args.embedding_torch_dtype,
            device=args.embedding_device,
            query_instruction=args.query_instruction,
            output_dim=args.embedding_output_dim,
        )
        item_embs = embedder.encode_documents(item_texts)

    model, tokenizer = load_reasoner(args.model, args.adapter, args.torch_dtype, args.model_device)
    selected_indices = list(range(len(all_rows)))
    if args.max_examples > 0:
        selected_indices = selected_indices[: min(args.max_examples, len(selected_indices))]
    selected_indices = [
        row_index
        for limited_pos, row_index in enumerate(selected_indices)
        if limited_pos % args.num_shards == args.shard_index
    ]
    rows = [all_rows[row_index] for row_index in selected_indices]

    metric_keys = [f"{prefix}_{metric}@{k}" for prefix in ("baseline", "reasoner") for metric in ("HR", "NDCG") for k in ks]
    totals = {key: 0.0 for key in metric_keys}
    pred_rows: list[dict[str, Any]] = []

    for batch_start in range(0, len(rows), args.generation_batch_size):
        batch_end = min(batch_start + args.generation_batch_size, len(rows))
        batch_rows = [rows[i] for i in range(batch_start, batch_end)]
        batch_indices = selected_indices[batch_start:batch_end]
        user_histories = [
            history_text(
                args.category,
                [str(x) for x in row.get("history_item_title", [])],
                [float(x) for x in row.get("history_rating", [])],
                args.max_history_items,
                item_ids=[int(x) for x in (row.get("history_item_id") or row.get("history_item_ids") or [])],
                item_map=item_map,
                metadata_mode=args.history_metadata_mode,
                max_item_chars=args.history_max_item_chars,
                summary_map=summary_map,
            )
            for row in batch_rows
        ]
        target_ids = [int(row["item_id"]) for row in batch_rows]
        cots = generate_cots(model, tokenizer, user_histories, args.category, args)
        reasoner_queries = [
            append_recommendation_reasoning(user_history, cot)
            for user_history, cot in zip(user_histories, cots)
        ]

        if args.scorer == "lexical":
            baseline_ranks = [
                rank_target(user_history, item_ids, item_vecs, item_norms, target_id)
                for user_history, target_id in zip(user_histories, target_ids)
            ]
            reasoner_ranks = [
                rank_target(reasoner_query, item_ids, item_vecs, item_norms, target_id)
                for reasoner_query, target_id in zip(reasoner_queries, target_ids)
            ]
        else:
            baseline_query_embs = embedder.encode_queries(user_histories)  # type: ignore[union-attr]
            reasoner_query_embs = embedder.encode_queries(reasoner_queries)  # type: ignore[union-attr]
            baseline_ranks = []
            reasoner_ranks = []
            for row_pos, target_id in enumerate(target_ids):
                target_index = item_id_to_index.get(target_id)
                if target_index is None:
                    baseline_ranks.append(len(item_ids) + 1)
                    reasoner_ranks.append(len(item_ids) + 1)
                else:
                    baseline_ranks.append(
                        rank_target_embedding_from_emb(baseline_query_embs[row_pos], item_embs, target_index)  # type: ignore[arg-type]
                    )
                    reasoner_ranks.append(
                        rank_target_embedding_from_emb(reasoner_query_embs[row_pos], item_embs, target_index)  # type: ignore[arg-type]
                    )

        for row_pos, row in enumerate(batch_rows):
            local_index = batch_start + row_pos + 1
            global_index = batch_indices[row_pos] + 1
            baseline_rank = baseline_ranks[row_pos]
            reasoner_rank = reasoner_ranks[row_pos]
            update_metrics(totals, "baseline", baseline_rank, ks)
            update_metrics(totals, "reasoner", reasoner_rank, ks)
            pred_rows.append(
                {
                    "category": args.category,
                    "split": args.split,
                    "index": local_index,
                    "global_index": global_index,
                    "shard_index": args.shard_index,
                    "num_shards": args.num_shards,
                    "user_id": row.get("user_id"),
                    "target_item_id": target_ids[row_pos],
                    "target_item_title": row.get("item_title", ""),
                    "baseline_rank": baseline_rank,
                    "reasoner_rank": reasoner_rank,
                    "cot": cots[row_pos],
                }
            )
        print(
            f"evaluated {batch_end}/{len(rows)} "
            f"shard={args.shard_index}/{args.num_shards} "
            f"batch_size={len(batch_rows)}",
            flush=True,
        )

    n = max(1, len(pred_rows))
    result = {
        "category": args.category,
        "split": args.split,
        "examples": str(examples_path),
        "item_info": str(item_info_path),
        "adapter": args.adapter,
        "adapter_name": args.adapter_name or (Path(args.adapter).parent.parent.name if args.adapter else "base"),
        "evaluated": len(pred_rows),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "generation_batch_size": args.generation_batch_size,
        "model_device": args.model_device,
        "num_items": len(item_ids),
        "metrics": {key: value / n for key, value in totals.items()},
        "scorer": args.scorer,
        "embedding_model": args.embedding_model if args.scorer == "qwen3_embedding" else None,
        "history_metadata_mode": args.history_metadata_mode,
        "history_max_item_chars": args.history_max_item_chars,
        "item_summary": args.item_summary,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.predictions_output:
        path = Path(args.predictions_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for row in pred_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
