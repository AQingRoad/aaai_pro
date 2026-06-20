#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures as futures
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rubric_cot_pipeline.io import append_jsonl, ensure_parent, read_jsonl
from rubric_cot_pipeline.judge_api import build_judge_api_client
from rubric_cot_pipeline.prompts import build_judge_messages
from rubric_cot_pipeline.rubric import normalize_judge_score, parse_judge_json, rule_score


def resolve_dtype(name: str):
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


def load_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs = {"trust_remote_code": True, "torch_dtype": resolve_dtype(args.torch_dtype)}
    if args.device == "auto":
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    if args.device != "auto":
        model.to(args.device)
    model.eval()
    return model, tokenizer


def judge_with_qwen(model, tokenizer, row: dict, args) -> tuple[dict | None, str]:
    target = row.get("target_item_text", "") if args.expose_target_to_judge else ""
    messages = build_judge_messages(row["user_history"], row["cot"], target)
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=args.max_prompt_tokens)
    device = first_device(model)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    raw = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True).strip()
    obj = parse_judge_json(raw)
    return (normalize_judge_score(obj) if obj else None), raw


def score_row(row: dict, args, model=None, tokenizer=None, api_client=None) -> tuple[dict, bool]:
    judge_raw = ""
    fallback_used = False
    if args.judge_mode == "qwen":
        scored, judge_raw = judge_with_qwen(model, tokenizer, row, args)
        if scored is None:
            scored = rule_score(row["user_history"], row["cot"])
            fallback_used = True
            judge_mode_used = "qwen_fallback_rules"
        else:
            judge_mode_used = "qwen"
    elif args.judge_mode == "api":
        target = row.get("target_item_text", "") if args.expose_target_to_judge else ""
        result = api_client.score(row["user_history"], row["cot"], target)
        judge_raw = result.raw
        if result.score is None:
            scored = rule_score(row["user_history"], row["cot"])
            fallback_used = True
            judge_mode_used = f"api_{result.provider}_fallback_rules"
        else:
            scored = result.score
            judge_mode_used = f"api_{result.provider}"
    else:
        scored = rule_score(row["user_history"], row["cot"])
        judge_mode_used = "rules"

    out = {
        **row,
        "rubric": scored,
        "rubric_total": scored["total"],
        "rubric_score_norm": scored["score_norm"],
        "judge_mode": judge_mode_used,
        "judge_api_provider": args.api_provider if args.judge_mode == "api" else "",
        "judge_raw": judge_raw,
    }
    return out, fallback_used


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/cot_candidates.jsonl")
    parser.add_argument("--output", default="outputs/cot_judged.jsonl")
    parser.add_argument("--judge-mode", choices=["rules", "qwen", "api"], default="rules")
    parser.add_argument(
        "--model",
        default=os.getenv("RUBRIC_COT_MODEL", "/root/autodl-tmp/modelscope_cache/models/Qwen/Qwen3-4B"),
    )
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-prompt-tokens", type=int, default=3072)
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--expose-target-to-judge", action="store_true")
    parser.add_argument("--api-provider", default=os.getenv("RUBRIC_JUDGE_API_PROVIDER", "mock"))
    parser.add_argument("--api-base-url", default=os.getenv("RUBRIC_JUDGE_API_BASE_URL", ""))
    parser.add_argument("--api-key", default=os.getenv("RUBRIC_JUDGE_API_KEY", ""))
    parser.add_argument("--api-model", default=os.getenv("RUBRIC_JUDGE_API_MODEL", ""))
    parser.add_argument("--api-timeout", type=float, default=float(os.getenv("RUBRIC_JUDGE_API_TIMEOUT", "60")))
    parser.add_argument("--api-max-retries", type=int, default=int(os.getenv("RUBRIC_JUDGE_API_MAX_RETRIES", "2")))
    parser.add_argument("--api-workers", type=int, default=int(os.getenv("RUBRIC_JUDGE_API_WORKERS", "4")))
    args = parser.parse_args()

    ensure_parent(args.output).write_text("", encoding="utf-8")
    model = tokenizer = None
    if args.judge_mode == "qwen":
        model, tokenizer = load_model(args)
    api_client = None
    if args.judge_mode == "api":
        api_client = build_judge_api_client(
            provider=args.api_provider,
            base_url=args.api_base_url,
            api_key=args.api_key,
            model=args.api_model,
            timeout=args.api_timeout,
            max_retries=args.api_max_retries,
        )

    count = 0
    fallbacks = 0
    rows = list(read_jsonl(args.input, limit=args.max_examples))
    if args.judge_mode == "api" and args.api_workers > 1:
        with futures.ThreadPoolExecutor(max_workers=args.api_workers) as pool:
            futs = [pool.submit(score_row, row, args, None, None, api_client) for row in rows]
            for fut in futures.as_completed(futs):
                out, fallback_used = fut.result()
                append_jsonl(args.output, out)
                count += 1
                fallbacks += int(fallback_used)
                if count % 50 == 0:
                    print(f"judged {count}/{len(rows)}", flush=True)
    else:
        for row in rows:
            out, fallback_used = score_row(row, args, model, tokenizer, api_client)
            append_jsonl(args.output, out)
            count += 1
            fallbacks += int(fallback_used)
            if count % 50 == 0:
                print(f"judged {count}/{len(rows)}", flush=True)
    print(f"Wrote {count} judged rows to {args.output}; fallbacks={fallbacks}")


if __name__ == "__main__":
    main()
