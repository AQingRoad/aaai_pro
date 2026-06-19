#!/usr/bin/env python3
from __future__ import annotations

import argparse
import http.client
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rubric_cot_pipeline.io import append_jsonl, ensure_parent, read_jsonl
from rubric_cot_pipeline.prompts import (
    ANSWER_TAG,
    REASONING_TAG,
    build_generation_messages,
    normalize_cot_tags,
)


def parse_temperatures(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


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


def load_model(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

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


def generate_one(model, tokenizer, messages, args, temperature: float) -> str:
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=args.max_prompt_tokens)
    device = first_device(model)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    do_sample = temperature > 0
    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": do_sample,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = args.top_p
    import torch

    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)
    generated = output_ids[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def _extract_recommendation(content: str) -> str:
    normalized = normalize_cot_tags(content)
    lower = normalized.lower()
    answer_start = lower.rfind(f"<{ANSWER_TAG}>")
    if answer_start >= 0:
        answer = normalized[answer_start + len(ANSWER_TAG) + 2 :]
        answer_end = answer.lower().find(f"</{ANSWER_TAG}>")
        if answer_end >= 0:
            answer = answer[:answer_end]
    else:
        answer = normalized
    answer = re.sub(
        rf"</?(?:{REASONING_TAG}|{ANSWER_TAG}|think|thinking|thoughts|answer)>|</?tool_call>|```[\s\S]*?```",
        "",
        answer,
        flags=re.IGNORECASE,
    ).strip()
    answer = re.sub(r"^(?:recommendation|answer)\s*[:：]\s*", "", answer, flags=re.IGNORECASE).strip()
    return answer


def _build_cot_from_api_message(content: str, reasoning: str) -> str:
    analysis = reasoning.strip()
    recommendation = _extract_recommendation(content)
    if not recommendation:
        raise ValueError("API returned empty content/answer; increase --max-new-tokens or retry")
    return f"<{REASONING_TAG}>\n{analysis}\n</{REASONING_TAG}>\n<{ANSWER_TAG}>\n{recommendation}\n</{ANSWER_TAG}>"


def generate_one_api(messages, args, temperature: float) -> tuple[str, dict]:
    stage_start = time.perf_counter()
    if args.api_provider not in {"openai", "openai_compatible", "chat_completions"}:
        raise ValueError(f"Unsupported generation API provider: {args.api_provider}")
    if not args.api_base_url:
        raise ValueError("--api-base-url is required when --generation-mode api")
    if not args.api_model:
        raise ValueError("--api-model is required when --generation-mode api")

    url = args.api_base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"
    payload = {
        "model": args.api_model,
        "messages": messages,
        "temperature": temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_new_tokens,
    }
    if args.api_thinking:
        payload["thinking"] = {"type": args.api_thinking}
    if args.api_reasoning_effort:
        payload["reasoning_effort"] = args.api_reasoning_effort
    if temperature <= 0:
        payload["temperature"] = 0

    data = json.dumps(payload).encode("utf-8")
    last_error = None
    request_start = time.perf_counter()
    for attempt in range(args.api_max_retries + 1):
        attempt_start = time.perf_counter()
        try:
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=args.api_timeout) as resp:
                raw = resp.read().decode("utf-8")
            response_received = time.perf_counter()
            obj = json.loads(raw)
            parsed_json = time.perf_counter()
            message = obj["choices"][0]["message"]
            content = str(message.get("content") or "").strip()
            reasoning = str(message.get("reasoning_content") or "").strip()
            cot = _build_cot_from_api_message(content, reasoning)
            parsed_cot = time.perf_counter()
            timing = {
                "stage_total_seconds": round(parsed_cot - stage_start, 6),
                "api_request_seconds": round(response_received - attempt_start, 6),
                "api_total_with_retries_seconds": round(response_received - request_start, 6),
                "json_parse_seconds": round(parsed_json - response_received, 6),
                "cot_parse_seconds": round(parsed_cot - parsed_json, 6),
                "attempts": attempt + 1,
            }
            meta = {
                "timing": timing,
                "api_finish_reason": str(obj.get("choices", [{}])[0].get("finish_reason", "")),
                "api_has_reasoning_content": bool(reasoning),
                "api_content_chars": len(content),
                "api_reasoning_chars": len(reasoning),
                "api_usage": obj.get("usage", {}),
            }
            return cot, meta
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, http.client.RemoteDisconnected, KeyError, IndexError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < args.api_max_retries:
                time.sleep(min(2**attempt, 8))
    raise RuntimeError(f"generation API failed after retries: {last_error}")


def mock_cot(row: dict, candidate_idx: int) -> str:
    title = row.get("target_item_title", "the held-out positive item")
    return (
        f"<{REASONING_TAG}>\n"
        "The user's high-rated history should be separated from low-rated counterexamples. "
        "Recurring positive signals indicate stable genre, tone, and era preferences, while "
        "lower ratings help avoid mismatched recommendations. These patterns can generalize "
        "to unseen items with similar narrative and stylistic cues without naming the target.\n"
        f"</{REASONING_TAG}>\n"
        f"<{ANSWER_TAG}>\n"
        f"Candidate {candidate_idx}: recommend items sharing the user's repeated positive cues, "
        f"and avoid weakly supported styles; evaluation target is hidden during generation.\n"
        f"</{ANSWER_TAG}>"
    ).replace(title, "the held-out item")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/ml1m_examples.jsonl")
    parser.add_argument("--output", default="outputs/cot_candidates.jsonl")
    parser.add_argument(
        "--model",
        default=os.getenv("RUBRIC_COT_MODEL", "/root/autodl-tmp/modelscope_cache/models/Qwen/Qwen3-4B"),
    )
    parser.add_argument("--num-candidates", type=int, default=4)
    parser.add_argument("--temperatures", default="0.6,0.8,1.0")
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-prompt-tokens", type=int, default=2048)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generation-mode", choices=["local", "api", "mock"], default=os.getenv("COT_GENERATION_MODE", "local"))
    parser.add_argument("--api-provider", default=os.getenv("COT_GENERATION_API_PROVIDER", "openai_compatible"))
    parser.add_argument("--api-base-url", default=os.getenv("COT_GENERATION_API_BASE_URL", ""))
    parser.add_argument("--api-key", default=os.getenv("COT_GENERATION_API_KEY", ""))
    parser.add_argument("--api-model", default=os.getenv("COT_GENERATION_API_MODEL", ""))
    parser.add_argument("--api-timeout", type=float, default=float(os.getenv("COT_GENERATION_API_TIMEOUT", "120")))
    parser.add_argument("--api-max-retries", type=int, default=int(os.getenv("COT_GENERATION_API_MAX_RETRIES", "2")))
    parser.add_argument("--api-min-interval", type=float, default=float(os.getenv("COT_GENERATION_API_MIN_INTERVAL", "0")))
    parser.add_argument("--api-thinking", default=os.getenv("COT_GENERATION_API_THINKING", "enabled"))
    parser.add_argument("--api-reasoning-effort", default=os.getenv("COT_GENERATION_API_REASONING_EFFORT", ""))
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()
    if args.mock:
        args.generation_mode = "mock"

    random.seed(args.seed)
    rows = list(read_jsonl(args.input, limit=args.max_examples))
    ensure_parent(args.output).write_text("", encoding="utf-8")
    temperatures = parse_temperatures(args.temperatures) or [0.7]

    model = tokenizer = None
    if args.generation_mode == "local":
        model, tokenizer = load_model(args)

    written = 0
    last_api_call = 0.0
    for row in rows:
        row_start = time.perf_counter()
        messages = build_generation_messages(row["user_history"], row.get("category", ""))
        prompt_built = time.perf_counter()
        row_id = row.get("example_id") or row["user_id"]
        for cand_idx in range(args.num_candidates):
            candidate_start = time.perf_counter()
            temperature = temperatures[cand_idx % len(temperatures)]
            generation_meta = {}
            if args.generation_mode == "mock":
                cot = mock_cot(row, cand_idx)
            elif args.generation_mode == "api":
                wait_start = time.perf_counter()
                if args.api_min_interval > 0:
                    elapsed = time.time() - last_api_call
                    if elapsed < args.api_min_interval:
                        time.sleep(args.api_min_interval - elapsed)
                wait_end = time.perf_counter()
                cot, generation_meta = generate_one_api(messages, args, temperature)
                last_api_call = time.time()
                generation_meta.setdefault("timing", {})["rate_limit_wait_seconds"] = round(wait_end - wait_start, 6)
            else:
                cot = generate_one(model, tokenizer, messages, args, temperature)
            candidate_end = time.perf_counter()
            out = {
                **row,
                "candidate_id": f"{row_id}-{cand_idx}",
                "candidate_index": cand_idx,
                "temperature": temperature,
                "cot": cot,
                "generator_model": args.api_model if args.generation_mode == "api" else args.model,
                "generation_mode": args.generation_mode,
                "generation_timing": {
                    "row_prompt_build_seconds": round(prompt_built - row_start, 6),
                    "candidate_total_seconds": round(candidate_end - candidate_start, 6),
                    **generation_meta.get("timing", {}),
                },
                "generation_api_meta": {k: v for k, v in generation_meta.items() if k != "timing"},
            }
            append_jsonl(args.output, out)
            written += 1
            print(f"generated {written}: user={row['user_id']} cand={cand_idx}", flush=True)
    print(f"Wrote {written} candidates to {args.output}")


if __name__ == "__main__":
    main()
