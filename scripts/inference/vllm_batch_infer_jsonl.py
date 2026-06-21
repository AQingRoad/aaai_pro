#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rubric_cot_pipeline.io import read_jsonl
from rubric_cot_pipeline.item_metadata import as_text_list, compact, format_selected_details
from rubric_cot_pipeline.prompts import ANSWER_TAG, COT_SYSTEM, REASONING_TAG, build_generation_messages, normalize_cot_tags


WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'-]*")


def patch_transformers_tokenizer_compat() -> None:
    """Patch older transformers tokenizers for newer vLLM tokenizer caching."""
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    if hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
        return

    @property
    def all_special_tokens_extended(self):  # type: ignore[no-untyped-def]
        token_map = getattr(self, "special_tokens_map_extended", None)
        if token_map is None:
            token_map = getattr(self, "special_tokens_map", {})
        tokens = []
        seen = set()
        for value in token_map.values():
            values = value if isinstance(value, (list, tuple)) else [value]
            for token in values:
                if token is None:
                    continue
                key = str(token)
                if key not in seen:
                    seen.add(key)
                    tokens.append(token)
        return tokens

    PreTrainedTokenizerBase.all_special_tokens_extended = all_special_tokens_extended


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text or ""))


def truncate_words(text: str, max_words: int) -> str:
    if max_words <= 0:
        return compact(text, 0)
    words = WORD_RE.findall(text or "")
    if len(words) <= max_words:
        return compact(text, 0)
    return " ".join(words[:max_words])


def clean_generation(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json|text)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    text = re.sub(r"^(?:summary|description_summary|answer)\s*[:：]\s*", "", text, flags=re.IGNORECASE).strip()
    if len(text) >= 2 and ((text[0] == text[-1] == '"') or (text[0] == text[-1] == "'")):
        text = text[1:-1].strip()
    return re.sub(r"\s+", " ", text).strip()


def example_key(row: dict[str, Any], task: str) -> str:
    if task == "description_summary":
        try:
            item_id = int(row.get("item_id"))
        except (TypeError, ValueError):
            return ""
        return str(item_id) if item_id > 0 else ""
    return str(row.get("example_id") or row.get("user_id") or row.get("id") or row.get("interaction_id") or "")


def load_existing(path: Path, task: str) -> dict[str, dict[str, Any]]:
    existing: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return existing
    for row in read_jsonl(path):
        key = example_key(row, task)
        if not key:
            continue
        if task == "description_summary" and not str(row.get("description_summary") or "").strip():
            continue
        existing[key] = row
    return existing


def write_jsonl_ordered(path: Path, rows: list[dict[str, Any]], generated: dict[str, dict[str, Any]], task: str) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            key = example_key(row, task)
            out = generated.get(key)
            if out is None:
                continue
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            count += 1
    return count


def selected_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.max_examples > 0:
        rows = rows[: args.max_examples]
    return [row for index, row in enumerate(rows) if index % args.num_shards == args.shard_index]


def description_text(item: dict[str, Any]) -> str:
    return " ".join(as_text_list(item.get("description"), limit=8))


def categories_text(item: dict[str, Any]) -> str:
    return " > ".join(as_text_list(item.get("categories"), limit=8))


def build_description_summary_messages(item: dict[str, Any], args: argparse.Namespace) -> list[dict[str, str]]:
    system = (
        "You summarize Amazon CDs/Vinyl item descriptions for recommendation prompts. "
        "Use only the provided fields. Do not use outside knowledge. "
        "Do not invent artist, genre, release year, awards, or style."
    )
    details = format_selected_details(item)
    user = (
        f"Write a factual summary of only the Description field in <= {args.summary_max_words} words.\n"
        "Use Title, Store/artist/format, Categories, and Details only to resolve references already present in Description.\n"
        "If Description is empty, return an empty string.\n"
        "Return only the summary text, without JSON, bullets, quotes, or explanations.\n\n"
        f"Title: {compact(item.get('title'), 300)}\n"
        f"Store/artist/format: {compact(item.get('store'), 300)}\n"
        f"Categories: {categories_text(item)}\n"
        f"Details: {details}\n"
        f"Description: {description_text(item)}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


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


def build_cot_messages(row: dict[str, Any], args: argparse.Namespace) -> list[dict[str, str]]:
    if args.cot_system:
        return [{"role": "system", "content": args.cot_system}, {"role": "user", "content": row["user_history"]}]
    return build_generation_messages(row["user_history"], row.get("category", ""))


def build_prompts(tokenizer, rows: list[dict[str, Any]], args: argparse.Namespace) -> list[str]:
    prompts = []
    for row in rows:
        if args.task == "description_summary":
            messages = build_description_summary_messages(row, args)
        elif args.task == "cot_generation":
            messages = build_cot_messages(row, args)
        else:
            raise ValueError(f"Unsupported task: {args.task}")
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(truncate_prompt(tokenizer, prompt, args.max_prompt_tokens))
    return prompts


def truncate_prompt(tokenizer, prompt: str, max_prompt_tokens: int) -> str:
    if max_prompt_tokens <= 0:
        return prompt
    old_side = getattr(tokenizer, "truncation_side", "right")
    tokenizer.truncation_side = "left"
    try:
        encoded = tokenizer(prompt, add_special_tokens=False, truncation=True, max_length=max_prompt_tokens)
        return tokenizer.decode(encoded["input_ids"], skip_special_tokens=False)
    finally:
        tokenizer.truncation_side = old_side


def cleanup_vllm(llm: Any) -> None:
    import torch

    try:
        executor = getattr(getattr(llm, "llm_engine", None), "model_executor", None)
        if executor is not None and hasattr(executor, "shutdown"):
            executor.shutdown()
    except Exception:
        pass
    try:
        from vllm.distributed.parallel_state import destroy_model_parallel

        destroy_model_parallel()
    except Exception:
        pass
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def init_llm(args: argparse.Namespace):
    patch_transformers_tokenizer_compat()
    from vllm import LLM
    from transformers import AutoTokenizer

    tokenizer_path = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs: dict[str, Any] = {
        "model": args.model,
        "tokenizer": tokenizer_path,
        "trust_remote_code": True,
        "tensor_parallel_size": args.tensor_parallel_size,
        "dtype": args.vllm_dtype,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.vllm_max_model_len,
        "max_num_seqs": args.vllm_max_num_seqs,
        "disable_log_stats": True,
        "seed": args.seed,
    }
    if args.enforce_eager:
        kwargs["enforce_eager"] = True
    if args.max_num_batched_tokens > 0:
        kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens
    return LLM(**kwargs), tokenizer


def output_row(src: dict[str, Any], raw: str, args: argparse.Namespace, elapsed: float) -> dict[str, Any]:
    if args.task == "description_summary":
        summary = truncate_words(clean_generation(raw), args.summary_max_words)
        return {
            "item_id": int(src["item_id"]),
            "title": src.get("title", ""),
            "description_summary": summary,
            "summary_word_count": word_count(summary),
            "summary_max_words": args.summary_max_words,
            "summary_model": args.model,
            "summary_task": args.task,
            "source_description_chars": len(description_text(src)),
            "generation_timing": {"seconds": round(elapsed, 6)},
        }

    key = example_key(src, args.task)
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
        "generation_mode": "vllm",
        "generation_timing": {"candidate_total_seconds": round(elapsed, 6)},
        "generation_meta": {
            "raw_output_chars": len(raw or ""),
            "answer_chars": len(answer or ""),
            "think_chars": len(think or ""),
        },
    }
    return {**src, "example_id": key, "candidate_count": 1, "candidates": [candidate]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generic vLLM JSONL batch inference for item summaries and CoT generation.")
    parser.add_argument("--task", choices=["description_summary", "cot_generation"], required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", default="")
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--generation-batch-size", type=int, default=64)
    parser.add_argument("--max-prompt-tokens", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--summary-max-words", type=int, default=60)
    parser.add_argument("--cot-system", default="")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--vllm-dtype", default="bfloat16")
    parser.add_argument("--vllm-max-model-len", type=int, default=4096)
    parser.add_argument("--vllm-max-num-seqs", type=int, default=64)
    parser.add_argument("--max-num-batched-tokens", type=int, default=0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")
    if args.generation_batch_size < 1:
        raise ValueError("--generation-batch-size must be >= 1")

    rows = selected_rows(list(read_jsonl(args.input, limit=args.max_examples)), args)
    rows = [row for row in rows if example_key(row, args.task)]
    output_path = Path(args.output)
    generated = load_existing(output_path, args.task) if args.resume else {}
    row_keys = {example_key(row, args.task) for row in rows}
    generated = {key: row for key, row in generated.items() if key in row_keys}
    pending = [row for row in rows if example_key(row, args.task) not in generated]
    print(
        f"task={args.task} selected={len(rows)} existing={len(generated)} pending={len(pending)} "
        f"shard={args.shard_index}/{args.num_shards} output={output_path}",
        flush=True,
    )
    if not pending:
        written = write_jsonl_ordered(output_path, rows, generated, args.task)
        print(f"no pending rows; wrote={written} output={output_path}", flush=True)
        return

    from vllm import SamplingParams

    llm, tokenizer = init_llm(args)
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
    )

    completed = 0
    try:
        for start in range(0, len(pending), args.generation_batch_size):
            batch = pending[start : start + args.generation_batch_size]
            prompts = build_prompts(tokenizer, batch, args)
            t0 = time.perf_counter()
            outputs = llm.generate(prompts, sampling_params=sampling, use_tqdm=False)
            elapsed = (time.perf_counter() - t0) / max(1, len(batch))
            for row, output in zip(batch, outputs):
                raw = output.outputs[0].text if output.outputs else ""
                generated[example_key(row, args.task)] = output_row(row, raw, args, elapsed)
                completed += 1
            if completed % args.save_every == 0 or completed >= len(pending):
                written = write_jsonl_ordered(output_path, rows, generated, args.task)
                print(f"completed={completed}/{len(pending)} written={written}/{len(rows)}", flush=True)
    finally:
        cleanup_vllm(llm)

    written = write_jsonl_ordered(output_path, rows, generated, args.task)
    print(f"done completed={completed} written={written}/{len(rows)} output={output_path}", flush=True)


if __name__ == "__main__":
    main()
