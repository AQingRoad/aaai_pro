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
from datasets import load_from_disk
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from rubric_cot_pipeline.embeddings import (
    DEFAULT_RECOMMENDATION_QUERY_INSTRUCTION,
    Qwen3TextEmbedder,
    append_recommendation_reasoning,
)
from rubric_cot_pipeline.prompts import COT_SYSTEM, build_user_prompt
from scripts.prepare_rrec_amazon_examples import build_item_map, build_item_text, history_text


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
    scores = torch.mv(item_embs, query_emb)
    order = torch.argsort(scores, descending=True)
    target_index = item_ids.index(target_id)
    matches = (order == target_index).nonzero(as_tuple=False)
    return int(matches[0].item()) + 1 if len(matches) else len(item_ids) + 1


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


def load_reasoner(model_path: str, adapter_path: str, torch_dtype: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=resolve_dtype(torch_dtype),
        device_map="auto",
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def first_device(model) -> torch.device:
    return next(model.parameters()).device


@torch.no_grad()
def generate_cot(model, tokenizer, user_history: str, category: str, args) -> str:
    messages = [
        {"role": "system", "content": COT_SYSTEM},
        {"role": "user", "content": build_user_prompt(user_history, category)},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_prompt_tokens)
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
    generated = output_ids[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/root/autodl-tmp/rec/RRec_official/data")
    parser.add_argument("--category", required=True)
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--model", default="/root/autodl-tmp/modelscope_cache/models/Qwen/Qwen3-4B")
    parser.add_argument("--adapter", default="")
    parser.add_argument("--adapter-name", default="")
    parser.add_argument("--max-examples", type=int, default=20)
    parser.add_argument("--max-history-items", type=int, default=20)
    parser.add_argument("--max-prompt-tokens", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--torch-dtype", default="bfloat16")
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
    args = parser.parse_args()

    ks = [int(x.strip()) for x in args.ks.split(",") if x.strip()]
    dataset_dir = Path(args.data_root) / f"{args.category}_0_2022-10-2023-10"
    ds = load_from_disk(str(dataset_dir))
    item_map = build_item_map(ds["item_info"])

    item_ids: list[int] = []
    item_texts: list[str] = []
    item_vecs: list[Counter[str]] = []
    item_norms: list[float] = []
    for item_id, item in sorted(item_map.items()):
        item_text = build_item_text(item, str(item.get("title", "")), max_chars=1200)
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

    model, tokenizer = load_reasoner(args.model, args.adapter, args.torch_dtype)
    rows = ds[args.split]
    if args.max_examples > 0:
        rows = rows.select(range(min(args.max_examples, len(rows))))

    metric_keys = [f"{prefix}_{metric}@{k}" for prefix in ("baseline", "reasoner") for metric in ("HR", "NDCG") for k in ks]
    totals = {key: 0.0 for key in metric_keys}
    pred_rows: list[dict[str, Any]] = []

    for idx, row in enumerate(rows, start=1):
        user_history = history_text(
            args.category,
            [str(x) for x in row.get("history_item_title", [])],
            [float(x) for x in row.get("history_rating", [])],
            args.max_history_items,
        )
        target_id = int(row["item_id"])
        cot = generate_cot(model, tokenizer, user_history, args.category, args)
        reasoner_query = append_recommendation_reasoning(user_history, cot)
        if args.scorer == "lexical":
            baseline_rank = rank_target(user_history, item_ids, item_vecs, item_norms, target_id)
            reasoner_rank = rank_target(reasoner_query, item_ids, item_vecs, item_norms, target_id)
        else:
            baseline_rank = rank_target_embedding(user_history, item_ids, item_embs, target_id, embedder)  # type: ignore[arg-type]
            reasoner_rank = rank_target_embedding(reasoner_query, item_ids, item_embs, target_id, embedder)  # type: ignore[arg-type]
        update_metrics(totals, "baseline", baseline_rank, ks)
        update_metrics(totals, "reasoner", reasoner_rank, ks)
        pred_rows.append(
            {
                "category": args.category,
                "split": args.split,
                "index": idx,
                "user_id": row.get("user_id"),
                "target_item_id": target_id,
                "target_item_title": row.get("item_title", ""),
                "baseline_rank": baseline_rank,
                "reasoner_rank": reasoner_rank,
                "cot": cot,
            }
        )
        print(f"evaluated {idx}/{len(rows)} baseline_rank={baseline_rank} reasoner_rank={reasoner_rank}", flush=True)

    n = max(1, len(pred_rows))
    result = {
        "category": args.category,
        "split": args.split,
        "adapter": args.adapter,
        "adapter_name": args.adapter_name or (Path(args.adapter).parent.parent.name if args.adapter else "base"),
        "evaluated": len(pred_rows),
        "num_items": len(item_ids),
        "metrics": {key: value / n for key, value in totals.items()},
        "scorer": args.scorer,
        "embedding_model": args.embedding_model if args.scorer == "qwen3_embedding" else None,
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
