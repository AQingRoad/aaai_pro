#!/usr/bin/env python3
"""使用 QUERY-only LoRA checkpoint 生成 CoT，并与严格检索 query 对齐。"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
MANU_SCRIPTS_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = MANU_SCRIPTS_DIR.parents[1]
sys.path.insert(0, str(MANU_SCRIPTS_DIR / "prompts"))
sys.path.insert(0, str(MANU_SCRIPTS_DIR / "pre_datas"))
sys.path.insert(0, str(PROJECT_ROOT))

from build_sft_from_teacher_cot import (  # noqa: E402
    compose_query,
    extract_history_parts,
    remove_item_field,
)
from query_only_cot_student import (  # noqa: E402
    PROMPT_NAME,
    PROMPT_VERSION,
    build_messages,
)
from rubric_cot_pipeline.embeddings import append_recommendation_reasoning  # noqa: E402


THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL | re.IGNORECASE)
ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
RAW_ASIN_RE = re.compile(r"\bB0[A-Z0-9]{8}\b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-input", type=Path, required=True)
    parser.add_argument("--retrieval-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-prompt-tokens", type=int, default=3328)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--save-every", type=int, default=128)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def canonical_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    """使用两个测试文件都具备的字段对齐样本。"""
    return (
        str(row.get("split") or "test"),
        str(row.get("interaction_id") or ""),
        str(row.get("user_id") or ""),
        str(row.get("target_item_id") or row.get("item_id") or ""),
    )


def example_id(row: dict[str, Any]) -> str:
    if row.get("example_id"):
        return str(row["example_id"])
    split, interaction_id, user_id, _ = canonical_key(row)
    return f"CDs_and_Vinyl:{split}:{interaction_id}:{user_id}"


def align_rows(
    generation_rows: list[dict[str, Any]], retrieval_rows: list[dict[str, Any]]
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """按监督标识对齐富字段生成输入和严格检索输入。"""
    retrieval_map = {canonical_key(row): row for row in retrieval_rows}
    if len(retrieval_map) != len(retrieval_rows):
        raise ValueError("retrieval input 存在重复样本键")

    pairs = []
    for row in generation_rows:
        key = canonical_key(row)
        retrieval = retrieval_map.get(key)
        if retrieval is None:
            raise ValueError(f"generation input 在 retrieval input 中没有对应样本：{key}")
        pairs.append((row, retrieval))
    if len(pairs) != len(retrieval_rows):
        raise ValueError(
            f"两个测试文件行数不一致：generation={len(pairs)}, retrieval={len(retrieval_rows)}"
        )
    return pairs


def prompt_token_count(tokenizer: Any, query: str) -> int:
    messages = build_messages(query, "en")
    return len(
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
    )


def shorten_single_item(
    tokenizer: Any,
    header: str,
    item: tuple[int, str],
    max_prompt_tokens: int,
) -> tuple[str, int]:
    """单条历史仍超长时，保留其开头并二分查找最大 token 前缀。"""
    item_number, item_text = item
    item_tokens = tokenizer.encode(item_text, add_special_tokens=False)
    low, high = 1, len(item_tokens)
    best_query = ""
    best_kept = 0
    while low <= high:
        middle = (low + high) // 2
        shortened = tokenizer.decode(
            item_tokens[:middle], skip_special_tokens=True
        ).strip()
        candidate = compose_query(header, [(item_number, shortened)])
        if prompt_token_count(tokenizer, candidate) <= max_prompt_tokens:
            best_query = candidate
            best_kept = middle
            low = middle + 1
        else:
            high = middle - 1
    if not best_query:
        raise ValueError("system prompt 与最短历史已经超过输入 token 预算")
    return best_query, len(item_tokens) - best_kept


def fit_query(
    tokenizer: Any, query: str, max_prompt_tokens: int
) -> tuple[str, dict[str, Any]]:
    """沿用训练数据的字段级策略，从最早历史开始压缩输入。"""
    original_tokens = prompt_token_count(tokenizer, query)
    header, items = extract_history_parts(query)
    original_numbers = [number for number, _ in items]
    removed_numbers: list[int] = []
    details_removed_numbers: list[int] = []
    description_removed_numbers: list[int] = []

    def current_query() -> str:
        return compose_query(header, items)

    def fits() -> bool:
        return prompt_token_count(tokenizer, current_query()) <= max_prompt_tokens

    for item_number in original_numbers:
        if fits():
            break
        item_index = next(
            (index for index, (number, _) in enumerate(items) if number == item_number),
            None,
        )
        if item_index is None:
            continue

        for field, changed_numbers in (
            ("details", details_removed_numbers),
            ("description", description_removed_numbers),
        ):
            number, old_text = items[item_index]
            new_text = remove_item_field(old_text, field)
            if new_text != old_text:
                items[item_index] = (number, new_text)
                changed_numbers.append(number)
                if fits():
                    break
        if fits():
            break

        if len(items) > 1:
            removed_numbers.append(items.pop(item_index)[0])

    fitted_query = current_query()
    tail_tokens_removed = 0
    if prompt_token_count(tokenizer, fitted_query) > max_prompt_tokens:
        if len(items) != 1:
            raise ValueError("字段压缩后仍超长，且剩余历史物品数大于 1")
        fitted_query, tail_tokens_removed = shorten_single_item(
            tokenizer, header, items[0], max_prompt_tokens
        )

    final_tokens = prompt_token_count(tokenizer, fitted_query)
    if final_tokens > max_prompt_tokens:
        raise AssertionError("截断后的 prompt 仍超过 token 预算")
    audit = {
        "original_prompt_tokens": original_tokens,
        "final_prompt_tokens": final_tokens,
        "removed_history_item_numbers": removed_numbers,
        "details_removed_item_numbers": details_removed_numbers,
        "description_removed_item_numbers": description_removed_numbers,
        "oldest_retained_item_tail_tokens_removed": tail_tokens_removed,
    }
    return fitted_query, audit


def parse_tagged_cot(raw: str) -> tuple[str, str, str]:
    """优先规范标签；标签不完整时保留原始生成文本用于真实评测。"""
    think_match = THINK_RE.search(raw)
    answer_match = ANSWER_RE.search(raw)
    think = think_match.group(1).strip() if think_match else ""
    answer = answer_match.group(1).strip() if answer_match else ""
    if not think or not answer:
        return think, answer, raw.strip()
    cot = f"<think>\n{think}\n</think>\n<answer>\n{answer}\n</answer>"
    return think, answer, cot


def write_ordered(
    path: Path,
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    generated: dict[str, dict[str, Any]],
) -> int:
    """使用临时文件原子更新，保留断点续跑结果。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temp_path.open("w", encoding="utf-8") as file:
        for generation_row, _ in pairs:
            row = generated.get(example_id(generation_row))
            if row is not None:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
    temp_path.replace(path)
    return count


def load_existing(path: Path, adapter: str) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = read_jsonl(path)
    generated = {}
    for row in rows:
        if str(row.get("generator_adapter") or "") != adapter:
            raise ValueError(f"断点文件中的 adapter 与当前参数不同：{path}")
        candidates = row.get("candidates") or []
        if candidates and str(candidates[0].get("cot") or "").strip():
            generated[example_id(row)] = row
    return generated


def load_model(args: argparse.Namespace):
    from peft import PeftModel

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    model.to(args.device)
    model.eval()
    return model


def build_output_row(
    generation_row: dict[str, Any],
    retrieval_row: dict[str, Any],
    fitted_query: str,
    truncation: dict[str, Any],
    think: str,
    answer: str,
    cot: str,
    raw: str,
    elapsed_seconds: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """输出只保留评测监督标识、QUERY 和模型生成内容。"""
    strict_query = str(retrieval_row.get("query") or "").strip()
    result = {
        "example_id": example_id(generation_row),
        "category": retrieval_row.get("category") or generation_row.get("category"),
        "split": retrieval_row.get("split") or generation_row.get("split"),
        "user_id": retrieval_row.get("user_id") or generation_row.get("user_id"),
        "interaction_id": retrieval_row.get("interaction_id"),
        "target_item_id": int(retrieval_row["target_item_id"]),
        "target_item_title": retrieval_row.get("target_item_title", ""),
        "history_item_ids": retrieval_row.get("history_item_ids") or [],
        "user_history": strict_query,
        "query": strict_query,
        "query_with_cot": append_recommendation_reasoning(strict_query, cot),
        "generation_query": fitted_query,
        "generation_truncation": truncation,
        "generator_adapter": args.adapter,
        "generator_base_model": args.base_model,
        "prompt_name": PROMPT_NAME,
        "prompt_version": PROMPT_VERSION,
        "candidates": [
            {
                "candidate_index": 0,
                "think": think,
                "answer": answer,
                "cot": cot,
                "raw_output": raw,
                "generation_seconds": round(elapsed_seconds, 6),
            }
        ],
    }
    return result


def build_audit(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    truncations = [row["generation_truncation"] for row in rows]
    candidates = [(row.get("candidates") or [{}])[0] for row in rows]
    cots = [str(candidate.get("cot") or "") for candidate in candidates]
    return {
        "generation_input": str(args.generation_input),
        "generation_input_sha256": hashlib.sha256(
            args.generation_input.read_bytes()
        ).hexdigest(),
        "retrieval_input": str(args.retrieval_input),
        "adapter": args.adapter,
        "base_model": args.base_model,
        "prompt_name": PROMPT_NAME,
        "prompt_version": PROMPT_VERSION,
        "rows": len(rows),
        "batch_size": args.batch_size,
        "max_prompt_tokens": args.max_prompt_tokens,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "decode": "greedy",
        "empty_cot": sum(not cot.strip() for cot in cots),
        "missing_parsed_think": sum(
            not str(candidate.get("think") or "").strip() for candidate in candidates
        ),
        "missing_parsed_answer": sum(
            not str(candidate.get("answer") or "").strip() for candidate in candidates
        ),
        "raw_fallback_rows": sum(
            not str(candidate.get("think") or "").strip()
            or not str(candidate.get("answer") or "").strip()
            for candidate in candidates
        ),
        "raw_asin_in_generation_query_or_cot": sum(
            bool(RAW_ASIN_RE.search(f"{row['generation_query']}\n{cot}"))
            for row, cot in zip(rows, cots)
        ),
        "target_title_exact_in_cot": sum(
            bool(str(row.get("target_item_title") or "").strip())
            and str(row.get("target_item_title")).strip().lower() in cot.lower()
            for row, cot in zip(rows, cots)
        ),
        "prompt_truncated_rows": sum(
            item["original_prompt_tokens"] > item["final_prompt_tokens"]
            for item in truncations
        ),
        "history_item_removed_rows": sum(
            bool(item["removed_history_item_numbers"]) for item in truncations
        ),
        "max_original_prompt_tokens": max(
            (item["original_prompt_tokens"] for item in truncations), default=0
        ),
        "max_final_prompt_tokens": max(
            (item["final_prompt_tokens"] for item in truncations), default=0
        ),
    }


def main() -> None:
    args = parse_args()
    if args.seed != 42:
        raise ValueError("项目随机种子固定为 42")
    if args.batch_size < 1 or args.max_prompt_tokens < 1 or args.max_new_tokens < 1:
        raise ValueError("batch size 和 token 长度必须大于 0")
    if args.max_prompt_tokens + args.max_new_tokens != 4096:
        raise ValueError("本实验要求输入预算与生成预算之和严格等于 4096")
    if not torch.cuda.is_available():
        raise RuntimeError("评测需要 CUDA GPU")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    generation_rows = read_jsonl(args.generation_input)
    retrieval_rows = read_jsonl(args.retrieval_input)
    pairs = align_rows(generation_rows, retrieval_rows)
    if any(row.get("split") != "test" for row in generation_rows):
        raise ValueError("generation input 中存在非 test 样本")

    generated = load_existing(args.output, args.adapter) if args.resume else {}
    pending = [pair for pair in pairs if example_id(pair[0]) not in generated]
    print(
        f"selected={len(pairs)} existing={len(generated)} pending={len(pending)} "
        f"adapter={args.adapter}",
        flush=True,
    )
    if pending:
        tokenizer = AutoTokenizer.from_pretrained(
            args.base_model,
            trust_remote_code=True,
            use_fast=True,
            padding_side="left",
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = load_model(args)

        completed = 0
        for start in range(0, len(pending), args.batch_size):
            batch_pairs = pending[start : start + args.batch_size]
            fitted = [
                fit_query(tokenizer, str(generation_row.get("query") or "").strip(), args.max_prompt_tokens)
                for generation_row, _ in batch_pairs
            ]
            prompts = [
                tokenizer.apply_chat_template(
                    build_messages(query, "en"),
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for query, _ in fitted
            ]
            encoded = tokenizer(
                prompts,
                padding=True,
                truncation=False,
                return_tensors="pt",
            )
            encoded = {key: value.to(args.device) for key, value in encoded.items()}
            start_time = time.perf_counter()
            with torch.inference_mode():
                output_ids = model.generate(
                    **encoded,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            elapsed = (time.perf_counter() - start_time) / len(batch_pairs)
            prompt_width = encoded["input_ids"].shape[1]
            raw_outputs = tokenizer.batch_decode(
                output_ids[:, prompt_width:], skip_special_tokens=True
            )
            for (generation_row, retrieval_row), (fitted_query, truncation), raw in zip(
                batch_pairs, fitted, raw_outputs
            ):
                think, answer, cot = parse_tagged_cot(raw)
                generated[example_id(generation_row)] = build_output_row(
                    generation_row,
                    retrieval_row,
                    fitted_query,
                    truncation,
                    think,
                    answer,
                    cot,
                    raw,
                    elapsed,
                    args,
                )
                completed += 1
            if completed % args.save_every == 0 or completed == len(pending):
                written = write_ordered(args.output, pairs, generated)
                print(
                    f"completed={completed}/{len(pending)} written={written}/{len(pairs)}",
                    flush=True,
                )

        del model
        torch.cuda.empty_cache()

    ordered_rows = [generated[example_id(row)] for row, _ in pairs if example_id(row) in generated]
    if len(ordered_rows) != len(pairs):
        raise ValueError(f"生成结果不完整：{len(ordered_rows)}/{len(pairs)}")
    write_ordered(args.output, pairs, generated)
    audit = build_audit(ordered_rows, args)
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)
    if audit["empty_cot"]:
        raise ValueError("存在空生成结果，停止排序评测")


if __name__ == "__main__":
    main()
