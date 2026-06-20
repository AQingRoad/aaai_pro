#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rubric_cot_pipeline.io import ensure_parent, read_jsonl
from rubric_cot_pipeline.prompts import ANSWER_TAG, REASONING_TAG, build_generation_messages, normalize_cot_tags


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


def example_key(row: dict[str, Any]) -> str:
    return str(row.get("example_id") or row.get("user_id") or row.get("id") or row.get("interaction_id") or "")


def first_device(model):
    return next(model.parameters()).device


def strip_tags(text: str) -> str:
    return re.sub(
        rf"</?(?:{REASONING_TAG}|{ANSWER_TAG}|think|thinking|thoughts|answer|hidden_reasoning|reasoning|analysis)>|</?tool_call>|```[\s\S]*?```",
        "",
        text or "",
        flags=re.IGNORECASE,
    ).strip()


def extract_answer(text: str) -> str:
    normalized = normalize_cot_tags(text)
    lower = normalized.lower()
    answer_start = lower.rfind(f"<{ANSWER_TAG}>")
    if answer_start >= 0:
        answer = normalized[answer_start + len(ANSWER_TAG) + 2 :]
        answer_end = answer.lower().find(f"</{ANSWER_TAG}>")
        if answer_end >= 0:
            answer = answer[:answer_end]
        return strip_tags(answer)

    think_block_re = re.compile(
        r"<\s*(?:hidden_reasoning|reasoning|analysis|think|thinking|thoughts)\s*>[\s\S]*?<\s*/\s*(?:hidden_reasoning|reasoning|analysis|think|thinking|thoughts)\s*>",
        re.IGNORECASE,
    )
    without_think = think_block_re.sub("", normalized)
    answer = strip_tags(without_think)
    answer = re.sub(r"^(?:recommendation|answer|final answer|final)\s*[:：]\s*", "", answer, flags=re.IGNORECASE)
    return answer.strip()


def extract_think(text: str) -> str:
    normalized = normalize_cot_tags(text)
    think_re = re.compile(rf"<\s*{REASONING_TAG}\s*>([\s\S]*?)<\s*/\s*{REASONING_TAG}\s*>", re.IGNORECASE)
    match = think_re.search(normalized)
    return strip_tags(match.group(1)) if match else ""


def normalize_generated_cot(raw: str) -> tuple[str, str, str]:
    think = extract_think(raw)
    answer = extract_answer(raw)
    if think and answer:
        cot = f"<{REASONING_TAG}>\n{think}\n</{REASONING_TAG}>\n<{ANSWER_TAG}>\n{answer}\n</{ANSWER_TAG}>"
    elif answer:
        cot = f"<{ANSWER_TAG}>\n{answer}\n</{ANSWER_TAG}>"
    else:
        answer = strip_tags(raw)
        cot = answer
    return think, answer, cot


def load_existing(path: Path) -> dict[str, dict[str, Any]]:
    existing: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return existing
    for row in read_jsonl(path):
        key = example_key(row)
        candidates = row.get("candidates")
        if key and isinstance(candidates, list) and candidates:
            existing[key] = row
    return existing


def write_output(path: Path, rows: list[dict[str, Any]], generated: dict[str, dict[str, Any]]) -> int:
    ensure_parent(path)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            key = example_key(row)
            out = generated.get(key)
            if out is None:
                continue
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            count += 1
    return count


def load_model(args: argparse.Namespace):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    for attr in ("base_model_tp_plan", "base_model_pp_plan", "base_model_ep_plan"):
        if hasattr(config, attr):
            setattr(config, attr, None)

    kwargs: dict[str, Any] = {
        "config": config,
        "trust_remote_code": True,
        "torch_dtype": resolve_dtype(args.torch_dtype),
        "tp_plan": None,
        "tp_size": None,
    }
    if args.device == "auto":
        kwargs["device_map"] = "auto"
    else:
        kwargs["device_map"] = {"": args.device}
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    except TypeError as exc:
        if "unexpected keyword argument" not in str(exc):
            raise
        retry_kwargs = {key: value for key, value in kwargs.items() if key not in {"tp_plan", "tp_size"}}
        model = AutoModelForCausalLM.from_pretrained(args.model, **retry_kwargs)
    model.eval()
    return model, tokenizer


def build_mock_output(row: dict[str, Any]) -> str:
    return (
        f"<{REASONING_TAG}>\n"
        "The user history contains repeated positive signals. The next item should preserve the strongest supported "
        "style and format cues while avoiding unsupported shifts.\n"
        f"</{REASONING_TAG}>\n"
        f"<{ANSWER_TAG}>\n"
        "A CD or vinyl item matching the user's recurring high-rated style, format, and genre cues.\n"
        f"</{ANSWER_TAG}>"
    )


def generate_batch(model, tokenizer, rows: list[dict[str, Any]], args: argparse.Namespace) -> list[str]:
    if args.mock:
        return [build_mock_output(row) for row in rows]

    import torch

    messages_batch = [build_generation_messages(row["user_history"], row.get("category", "")) for row in rows]
    prompts = [tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) for messages in messages_batch]
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_prompt_tokens,
    )
    device = first_device(model)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    do_sample = args.temperature > 0
    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": do_sample,
    }
    if do_sample:
        gen_kwargs["temperature"] = args.temperature
        gen_kwargs["top_p"] = args.top_p
    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)
    generated = output_ids[:, inputs["input_ids"].shape[-1] :]
    return [text.strip() for text in tokenizer.batch_decode(generated, skip_special_tokens=True)]


def candidate_row(src: dict[str, Any], raw: str, args: argparse.Namespace, elapsed: float) -> dict[str, Any]:
    key = example_key(src)
    think, answer, cot = normalize_generated_cot(raw)
    candidate = {
        "example_id": key,
        "candidate_id": f"{key}-0",
        "candidate_index": 0,
        "temperature": args.temperature,
        "think": think,
        "answer": answer,
        "cot": cot,
        "generator_model": args.model,
        "generation_mode": "local_mock" if args.mock else "local",
        "generation_timing": {
            "candidate_total_seconds": round(elapsed, 6),
        },
        "generation_meta": {
            "raw_output_chars": len(raw or ""),
            "answer_chars": len(answer or ""),
            "think_chars": len(think or ""),
        },
    }
    return {
        **src,
        "example_id": key,
        "candidate_count": 1,
        "candidates": [candidate],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate one local-model CoT candidate per example with sharding and batched inference.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="/mnt/tidal-sh01/usr/xiayu6/xiayu/checkpoint/Qwen3/4B")
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--generation-batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-prompt-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--device", default=os.getenv("COT_MODEL_DEVICE", "cuda:0"))
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--aggregate-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")
    if args.generation_batch_size < 1:
        raise ValueError("--generation-batch-size must be >= 1")

    random.seed(args.seed + args.shard_index)
    all_rows = list(read_jsonl(args.input, limit=args.max_examples))
    selected = [
        row
        for limited_pos, row in enumerate(all_rows)
        if limited_pos % args.num_shards == args.shard_index
    ]
    output_path = Path(args.output)
    generated = load_existing(output_path) if args.resume else {}
    selected_keys = {example_key(row) for row in selected}
    generated = {key: row for key, row in generated.items() if key in selected_keys}
    pending = [row for row in selected if example_key(row) and example_key(row) not in generated]

    print(
        f"loaded={len(all_rows)} selected={len(selected)} existing={len(generated)} pending={len(pending)} "
        f"shard={args.shard_index}/{args.num_shards} output={output_path}",
        flush=True,
    )
    if not pending:
        written = write_output(output_path, selected, generated)
        print(f"no pending rows; wrote={written} output={output_path}", flush=True)
        return

    model = tokenizer = None
    if not args.mock:
        model, tokenizer = load_model(args)

    completed = 0
    for start in range(0, len(pending), args.generation_batch_size):
        batch = pending[start : start + args.generation_batch_size]
        batch_start = time.perf_counter()
        raws = generate_batch(model, tokenizer, batch, args)
        batch_elapsed = time.perf_counter() - batch_start
        per_row_elapsed = batch_elapsed / max(1, len(batch))
        for row, raw in zip(batch, raws):
            key = example_key(row)
            generated[key] = candidate_row(row, raw, args, per_row_elapsed)
            completed += 1
        if completed == len(batch) or completed % args.aggregate_every == 0:
            written = write_output(output_path, selected, generated)
            print(
                f"completed={completed}/{len(pending)} written={written}/{len(selected)} "
                f"shard={args.shard_index}/{args.num_shards}",
                flush=True,
            )

    written = write_output(output_path, selected, generated)
    print(f"done completed={completed} written={written}/{len(selected)} output={output_path}", flush=True)


if __name__ == "__main__":
    main()
