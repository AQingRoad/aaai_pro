#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rubric_cot_pipeline.embeddings import (
    DEFAULT_RECOMMENDATION_QUERY_INSTRUCTION,
    Qwen3TextEmbedder,
    append_recommendation_reasoning,
)
from rubric_cot_pipeline.io import append_jsonl, ensure_parent, read_jsonl
from rubric_cot_pipeline.rubric import hashed_cosine


def resolve_dtype(name: str):
    import torch

    name = (name or "auto").lower()
    if name == "auto":
        return "auto"
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def first_device(model):
    return next(model.parameters()).device


class QwenHiddenEmbedder:
    def __init__(self, model_path: str, torch_dtype: str, device: str, max_tokens: int):
        import torch
        import torch.nn.functional as F
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.F = F
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        kwargs = {"trust_remote_code": True, "torch_dtype": resolve_dtype(torch_dtype)}
        if device == "auto":
            kwargs["device_map"] = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        if device != "auto":
            self.model.to(device)
        self.model.eval()
        self.max_tokens = max_tokens

    def encode(self, text: str) -> torch.Tensor:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_tokens,
            padding=False,
        )
        device = first_device(self.model)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with self.torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True, use_cache=False)
        hidden = outputs.hidden_states[-1][0]
        mask = inputs["attention_mask"][0].unsqueeze(-1).to(hidden.dtype)
        emb = (hidden * mask).sum(dim=0) / mask.sum().clamp_min(1.0)
        return self.F.normalize(emb.float(), dim=0).cpu()


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


def load_item_index(path: str, item_max_chars: int) -> tuple[list[int], list[str], dict[int, int]]:
    item_ids: list[int] = []
    item_texts: list[str] = []
    for row in read_jsonl(path):
        item_id = int(row["item_id"])
        item_ids.append(item_id)
        item_texts.append(build_item_text(row, str(row.get("title", "")), item_max_chars))
    order = sorted(range(len(item_ids)), key=lambda idx: item_ids[idx])
    item_ids = [item_ids[idx] for idx in order]
    item_texts = [item_texts[idx] for idx in order]
    return item_ids, item_texts, {item_id: idx for idx, item_id in enumerate(item_ids)}


def as_int_set(value: Any) -> set[int]:
    if value is None:
        return set()
    if isinstance(value, (str, int, float)):
        raw = [value]
    else:
        try:
            raw = list(value)
        except TypeError:
            raw = [value]
    out = set()
    for item in raw:
        try:
            out.add(int(item))
        except (TypeError, ValueError):
            continue
    return out


def ndcg_at_rank(rank: int, k: int) -> float:
    if rank <= 0 or rank > k:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def rank_from_torch_scores(scores, target_index: int, masked_indices: set[int]) -> tuple[int, float]:
    scores = scores.clone()
    if masked_indices:
        scores[list(masked_indices)] = -float("inf")
    target_score = float(scores[target_index].item())
    rank = int((scores > scores[target_index]).sum().item()) + 1
    return rank, target_score


def rank_from_list_scores(scores: list[float], target_index: int, masked_indices: set[int]) -> tuple[int, float]:
    target_score = scores[target_index]
    rank = 1
    for idx, score in enumerate(scores):
        if idx in masked_indices:
            continue
        if score > target_score:
            rank += 1
    return rank, float(target_score)


def row_shard_key(row: dict[str, Any]) -> str:
    return str(row.get("example_id") or row.get("user_id") or row.get("candidate_id") or row.get("interaction_id") or "")


def stable_shard_index(key: str, num_shards: int) -> int:
    digest = hashlib.md5(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % num_shards


def text_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/cot_judged.jsonl")
    parser.add_argument("--output", default="outputs/cot_scored.jsonl")
    parser.add_argument("--embedder-mode", choices=["lexical", "qwen_hidden", "qwen3_embedding"], default="lexical")
    parser.add_argument("--gain-mode", choices=["ndcg", "sim"], default=os.getenv("COT_GAIN_MODE", "ndcg"))
    parser.add_argument("--item-info", default=os.getenv("COT_GAIN_ITEM_INFO", ""))
    parser.add_argument("--ndcg-k", type=int, default=int(os.getenv("COT_GAIN_NDCG_K", "100")))
    parser.add_argument("--item-max-chars", type=int, default=1200)
    parser.add_argument("--mask-history-items", dest="mask_history_items", action="store_true", default=True)
    parser.add_argument("--no-mask-history-items", dest="mask_history_items", action="store_false")
    parser.add_argument("--mask-pad-item", dest="mask_pad_item", action="store_true", default=True)
    parser.add_argument("--no-mask-pad-item", dest="mask_pad_item", action="store_false")
    parser.add_argument(
        "--model",
        default=os.getenv("RUBRIC_COT_MODEL", "/root/autodl-tmp/modelscope_cache/models/Qwen/Qwen3-4B"),
    )
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("QWEN3_EMBEDDING_MODEL", "/root/autodl-tmp/modelscope_cache/models/Qwen/Qwen3-Embedding-0.6B"),
    )
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=int(os.getenv("COT_GAIN_NUM_SHARDS", "1")))
    parser.add_argument("--shard-index", type=int, default=int(os.getenv("COT_GAIN_SHARD_INDEX", "0")))
    parser.add_argument("--row-batch-size", type=int, default=int(os.getenv("COT_GAIN_ROW_BATCH_SIZE", "32")))
    parser.add_argument("--baseline-cache", default=os.getenv("COT_GAIN_BASELINE_CACHE", ""))
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--embedding-max-length", type=int, default=8192)
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    parser.add_argument("--embedding-output-dim", type=int, default=0)
    parser.add_argument(
        "--query-instruction",
        default=os.getenv("QWEN3_EMBEDDING_QUERY_INSTRUCTION", ""),
        help="Override the default Qwen3-Embedding query instruction. Empty uses the recommendation default.",
    )
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--device", default=os.getenv("QWEN3_EMBEDDING_DEVICE", "cuda:0"))
    args = parser.parse_args()

    if args.gain_mode == "ndcg" and not args.item_info:
        raise ValueError("--item-info is required when --gain-mode ndcg")
    if args.ndcg_k <= 0:
        raise ValueError("--ndcg-k must be positive")
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")
    if args.row_batch_size <= 0:
        raise ValueError("--row-batch-size must be positive")

    ensure_parent(args.output).write_text("", encoding="utf-8")
    embedder = None
    if args.embedder_mode == "qwen_hidden":
        embedder = QwenHiddenEmbedder(args.model, args.torch_dtype, args.device, args.max_tokens)
    elif args.embedder_mode == "qwen3_embedding":
        embedder = Qwen3TextEmbedder(
            args.embedding_model,
            max_length=args.embedding_max_length,
            batch_size=args.embedding_batch_size,
            torch_dtype=args.torch_dtype,
            device=args.device,
            query_instruction=args.query_instruction or DEFAULT_RECOMMENDATION_QUERY_INSTRUCTION,
            output_dim=args.embedding_output_dim,
        )

    @lru_cache(maxsize=8192)
    def encode_cached(text: str):
        return embedder.encode(text)  # type: ignore[union-attr]

    item_ids: list[int] = []
    item_texts: list[str] = []
    item_index: dict[int, int] = {}
    item_embs = None
    if args.gain_mode == "ndcg":
        item_ids, item_texts, item_index = load_item_index(args.item_info, args.item_max_chars)
        if args.embedder_mode == "qwen3_embedding":
            item_embs = embedder.encode_documents(item_texts)  # type: ignore[union-attr]
        elif args.embedder_mode == "qwen_hidden":
            import torch

            item_embs = torch.stack([encode_cached(text) for text in item_texts])

    def masked_indices(row: dict[str, Any], target_id: int) -> set[int]:
        masked_item_ids = set()
        if args.mask_history_items:
            masked_item_ids.update(as_int_set(row.get("history_item_ids") or row.get("history_item_id")))
        if args.mask_pad_item:
            masked_item_ids.add(0)
        masked_item_ids.discard(target_id)
        return {item_index[item_id] for item_id in masked_item_ids if item_id in item_index}

    def score_rank_ndcg(query: str, row: dict[str, Any]) -> tuple[float, int, float]:
        target_id = int(row["target_item_id"])
        if target_id not in item_index:
            return 0.0, len(item_ids) + 1, 0.0
        target_index = item_index[target_id]
        mask = masked_indices(row, target_id)

        if args.embedder_mode == "lexical":
            scores = [hashed_cosine(query, item_text) for item_text in item_texts]
            rank, target_score = rank_from_list_scores(scores, target_index, mask)
        elif args.embedder_mode == "qwen_hidden":
            query_emb = encode_cached(query)
            scores = item_embs @ query_emb  # type: ignore[operator]
            rank, target_score = rank_from_torch_scores(scores, target_index, mask)
        else:
            query_emb = embedder.encode_queries([query])[0]  # type: ignore[union-attr]
            scores = item_embs @ query_emb  # type: ignore[operator]
            rank, target_score = rank_from_torch_scores(scores, target_index, mask)
        return target_score, rank, ndcg_at_rank(rank, args.ndcg_k)

    def score_query_embedding_ndcg(query_emb, row: dict[str, Any]) -> tuple[float, int, float]:
        target_id = int(row["target_item_id"])
        if target_id not in item_index:
            return 0.0, len(item_ids) + 1, 0.0
        target_index = item_index[target_id]
        scores = item_embs @ query_emb  # type: ignore[operator]
        rank, target_score = rank_from_torch_scores(scores, target_index, masked_indices(row, target_id))
        return target_score, rank, ndcg_at_rank(rank, args.ndcg_k)

    def row_in_shard(row: dict[str, Any]) -> bool:
        if args.num_shards <= 1:
            return True
        key = row_shard_key(row)
        if not key:
            return False
        return stable_shard_index(key, args.num_shards) == args.shard_index

    def baseline_cache_key(row: dict[str, Any], user_history: str, target_id: int) -> str:
        payload = {
            "example_id": str(row.get("example_id") or ""),
            "target_id": target_id,
            "user_history_hash": text_hash(user_history),
            "history_item_ids": sorted(as_int_set(row.get("history_item_ids") or row.get("history_item_id"))),
            "embedder_mode": args.embedder_mode,
            "embedding_model": str(args.embedding_model) if args.embedder_mode == "qwen3_embedding" else "",
            "gain_mode": args.gain_mode,
            "item_info": str(args.item_info),
            "item_max_chars": args.item_max_chars,
            "ndcg_k": args.ndcg_k,
            "mask_history_items": args.mask_history_items,
            "mask_pad_item": args.mask_pad_item,
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.md5(encoded.encode("utf-8")).hexdigest()

    baseline_cache_path = ensure_parent(args.baseline_cache) if args.baseline_cache else None
    baseline_ndcg_cache: dict[str, tuple[float, int, float]] = {}
    if baseline_cache_path and baseline_cache_path.exists():
        for cache_row in read_jsonl(baseline_cache_path):
            key = str(cache_row.get("cache_key") or "")
            if not key:
                continue
            try:
                baseline_ndcg_cache[key] = (
                    float(cache_row["baseline_sim"]),
                    int(cache_row["baseline_rank"]),
                    float(cache_row["baseline_ndcg"]),
                )
            except (KeyError, TypeError, ValueError):
                continue

    def store_baseline_cache(cache_key: str, row: dict[str, Any], values: tuple[float, int, float]) -> None:
        if not baseline_cache_path:
            return
        baseline_sim, baseline_rank, baseline_ndcg = values
        append_jsonl(
            baseline_cache_path,
            {
                "cache_key": cache_key,
                "example_id": row.get("example_id"),
                "user_id": row.get("user_id"),
                "interaction_id": row.get("interaction_id"),
                "target_item_id": row.get("target_item_id"),
                "baseline_sim": baseline_sim,
                "baseline_rank": baseline_rank,
                "baseline_ndcg": baseline_ndcg,
                "ndcg_k": args.ndcg_k,
                "embedder_mode": args.embedder_mode,
                "embedding_model": args.embedding_model if args.embedder_mode == "qwen3_embedding" else "",
                "gain_mode": args.gain_mode,
            },
        )

    def process_qwen3_ndcg_batch(rows: list[dict[str, Any]]) -> int:
        prepared = []
        missing_base_queries = []
        missing_base_rows = []
        missing_base_keys = []
        cot_queries = []

        for row in rows:
            user_history = row["user_history"]
            cot_text = row.get("cot", "")
            with_cot = append_recommendation_reasoning(user_history, cot_text)
            target_id = int(row["target_item_id"])
            cache_key = baseline_cache_key(row, user_history, target_id)
            prepared.append((row, with_cot, cache_key))
            if cache_key not in baseline_ndcg_cache:
                missing_base_queries.append(user_history)
                missing_base_rows.append(row)
                missing_base_keys.append(cache_key)
            cot_queries.append(with_cot)

        if missing_base_queries:
            base_embs = embedder.encode_queries(missing_base_queries)  # type: ignore[union-attr]
            for cache_key, row, query_emb in zip(missing_base_keys, missing_base_rows, base_embs):
                baseline_values = score_query_embedding_ndcg(query_emb, row)
                baseline_ndcg_cache[cache_key] = baseline_values
                store_baseline_cache(cache_key, row, baseline_values)

        cot_embs = embedder.encode_queries(cot_queries)  # type: ignore[union-attr]
        written = 0
        for (row, _with_cot, cache_key), cot_emb in zip(prepared, cot_embs):
            baseline_sim, baseline_rank, baseline_ndcg = baseline_ndcg_cache[cache_key]
            cot_sim, cot_rank, cot_ndcg = score_query_embedding_ndcg(cot_emb, row)
            out = {
                **row,
                "baseline_sim": baseline_sim,
                "cot_sim": cot_sim,
                "cot_gain": cot_ndcg - baseline_ndcg,
                "gain_mode": args.gain_mode,
                "embedder_mode": args.embedder_mode,
                "baseline_rank": baseline_rank,
                "cot_rank": cot_rank,
                "baseline_ndcg": baseline_ndcg,
                "cot_ndcg": cot_ndcg,
                "ndcg_k": args.ndcg_k,
                "sim_gain": cot_sim - baseline_sim,
                "masked_history_items": args.mask_history_items,
                "masked_pad_item": args.mask_pad_item,
                "num_shards": args.num_shards,
                "shard_index": args.shard_index,
            }
            append_jsonl(args.output, out)
            written += 1
        return written

    if args.gain_mode == "ndcg" and args.embedder_mode == "qwen3_embedding":
        count = 0
        batch: list[dict[str, Any]] = []
        seen = 0
        for row in read_jsonl(args.input):
            if not row_in_shard(row):
                continue
            seen += 1
            if args.max_examples and seen > args.max_examples:
                break
            batch.append(row)
            if len(batch) >= args.row_batch_size:
                count += process_qwen3_ndcg_batch(batch)
                batch = []
                print(f"scored {count} shard={args.shard_index}/{args.num_shards}", flush=True)
        if batch:
            count += process_qwen3_ndcg_batch(batch)
        print(f"Wrote {count} gain-scored rows to {args.output} shard={args.shard_index}/{args.num_shards}")
        return

    count = 0
    for row in read_jsonl(args.input, limit=args.max_examples):
        if not row_in_shard(row):
            continue
        user_history = row["user_history"]
        cot_text = row.get("cot", "")
        target_text = row.get("target_item_text") or row.get("target_item_title", "")
        with_cot = append_recommendation_reasoning(user_history, cot_text)

        if args.gain_mode == "sim" and args.embedder_mode == "lexical":
            baseline_sim = hashed_cosine(user_history, target_text)
            cot_sim = hashed_cosine(with_cot, target_text)
            cot_gain = cot_sim - baseline_sim
            extra = {}
        elif args.gain_mode == "sim" and args.embedder_mode == "qwen_hidden":
            h_base = encode_cached(user_history)
            h_cot = encode_cached(with_cot)
            h_item = encode_cached(target_text)
            baseline_sim = float((h_base * h_item).sum())
            cot_sim = float((h_cot * h_item).sum())
            cot_gain = cot_sim - baseline_sim
            extra = {}
        elif args.gain_mode == "sim":
            baseline_sim, cot_sim = embedder.pairwise_cosine(  # type: ignore[union-attr]
                [user_history, with_cot],
                [target_text, target_text],
            )
            cot_gain = cot_sim - baseline_sim
            extra = {}
        else:
            target_id = int(row["target_item_id"])
            cache_key = baseline_cache_key(row, user_history, target_id)
            if cache_key not in baseline_ndcg_cache:
                baseline_values = score_rank_ndcg(user_history, row)
                baseline_ndcg_cache[cache_key] = baseline_values
                store_baseline_cache(cache_key, row, baseline_values)
            baseline_sim, baseline_rank, baseline_ndcg = baseline_ndcg_cache[cache_key]
            cot_sim, cot_rank, cot_ndcg = score_rank_ndcg(with_cot, row)
            cot_gain = cot_ndcg - baseline_ndcg
            extra = {
                "baseline_rank": baseline_rank,
                "cot_rank": cot_rank,
                "baseline_ndcg": baseline_ndcg,
                "cot_ndcg": cot_ndcg,
                "ndcg_k": args.ndcg_k,
                "sim_gain": cot_sim - baseline_sim,
                "masked_history_items": args.mask_history_items,
                "masked_pad_item": args.mask_pad_item,
            }

        out = {
            **row,
            "baseline_sim": baseline_sim,
            "cot_sim": cot_sim,
            "cot_gain": cot_gain,
            "gain_mode": args.gain_mode,
            "embedder_mode": args.embedder_mode,
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
            **extra,
        }
        append_jsonl(args.output, out)
        count += 1
        if count % 10 == 0:
            print(f"scored {count}", flush=True)
    print(f"Wrote {count} gain-scored rows to {args.output}")


if __name__ == "__main__":
    main()
