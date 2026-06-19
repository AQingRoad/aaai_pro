#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/cot_judged.jsonl")
    parser.add_argument("--output", default="outputs/cot_scored.jsonl")
    parser.add_argument("--embedder-mode", choices=["lexical", "qwen_hidden", "qwen3_embedding"], default="lexical")
    parser.add_argument(
        "--model",
        default=os.getenv("RUBRIC_COT_MODEL", "/root/autodl-tmp/modelscope_cache/models/Qwen/Qwen3-4B"),
    )
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("QWEN3_EMBEDDING_MODEL", "/root/autodl-tmp/modelscope_cache/models/Qwen/Qwen3-Embedding-0.6B"),
    )
    parser.add_argument("--max-examples", type=int, default=0)
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
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

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

    count = 0
    for row in read_jsonl(args.input, limit=args.max_examples):
        user_history = row["user_history"]
        cot_text = row.get("cot", "")
        target_text = row.get("target_item_text") or row.get("target_item_title", "")
        with_cot = append_recommendation_reasoning(user_history, cot_text)

        if args.embedder_mode == "lexical":
            baseline_sim = hashed_cosine(user_history, target_text)
            cot_sim = hashed_cosine(with_cot, target_text)
        elif args.embedder_mode == "qwen_hidden":
            h_base = encode_cached(user_history)
            h_cot = encode_cached(with_cot)
            h_item = encode_cached(target_text)
            baseline_sim = float((h_base * h_item).sum())
            cot_sim = float((h_cot * h_item).sum())
        else:
            baseline_sim, cot_sim = embedder.pairwise_cosine(  # type: ignore[union-attr]
                [user_history, with_cot],
                [target_text, target_text],
            )

        out = {
            **row,
            "baseline_sim": baseline_sim,
            "cot_sim": cot_sim,
            "cot_gain": cot_sim - baseline_sim,
            "embedder_mode": args.embedder_mode,
        }
        append_jsonl(args.output, out)
        count += 1
        if count % 10 == 0:
            print(f"scored {count}", flush=True)
    print(f"Wrote {count} gain-scored rows to {args.output}")


if __name__ == "__main__":
    main()
