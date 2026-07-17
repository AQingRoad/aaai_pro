#!/usr/bin/env python3
"""使用 vLLM 加载 Qwen2.5 LoRA，为指定 split 的 history 生成下一物品特征 CoT。"""

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


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from manu_src.scripts.prompts import build_general_recommendation_cot_messages  # noqa: E402


COT_SEPARATOR = "\n\nRecommendation reasoning:\n"
WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'-]*")
RAW_ASIN_RE = re.compile(r"(?<![A-Z0-9])B0[A-Z0-9]{8}(?![A-Z0-9])")
COT_RE = re.compile(
    r"^\s*<analysis>\s*(?P<analysis>.*?)\s*</analysis>\s*"
    r"<answer>\s*(?P<answer>.*?)\s*</answer>\s*$",
    re.DOTALL | re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--item-type", default="CD or vinyl release")
    parser.add_argument("--language", default="en")
    parser.add_argument("--generation-batch-size", type=int, default=32)
    parser.add_argument("--max-prompt-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-output-words", type=int, default=512)
    parser.add_argument("--vllm-max-model-len", type=int, default=6144)
    parser.add_argument("--vllm-max-num-seqs", type=int, default=32)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--allow-noncanonical-output",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="保留格式不完整但非空的原始 completion，并将其用于 retrieval query。",
    )
    parser.add_argument("--expected-split", default="test")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path} 第 {line_number} 行不是合法 JSON") from error
    return rows


def example_id(row: dict[str, Any]) -> str:
    return str(row.get("example_id") or "").strip()


def get_history(row: dict[str, Any]) -> str:
    return str(row.get("base_query") or row.get("user_history") or row.get("query") or "").strip()


def canonicalize_cot(raw: str, max_words: int) -> tuple[str, str, str, int]:
    cleaned = str(raw or "").strip()
    cleaned = re.sub(r"^```(?:text|xml)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE | re.DOTALL).strip()
    match = COT_RE.fullmatch(cleaned)
    if match is None:
        raise ValueError("输出未严格匹配 <analysis>/<answer>")
    analysis = match.group("analysis").strip()
    answer = match.group("answer").strip()
    if not analysis or not answer:
        raise ValueError("analysis 或 answer 为空")
    words = len(WORD_RE.findall(f"{analysis}\n{answer}"))
    if words > max_words:
        raise ValueError(f"analysis 与 answer 合计 {words} words，超过 {max_words}")
    cot = f"<analysis>\n{analysis}\n</analysis>\n<answer>\n{answer}\n</answer>"
    return analysis, answer, cot, words


def parse_generated_output(
    raw: str,
    max_words: int,
    allow_noncanonical: bool,
) -> tuple[str, str, str, int, bool]:
    """解析 completion；按需保留不完整格式的原始文本。"""
    raw_text = str(raw or "").strip()
    if not raw_text:
        raise ValueError("输出为空")
    try:
        analysis, answer, _, words = canonicalize_cot(raw_text, max_words)
        return analysis, answer, raw_text, words, True
    except ValueError:
        if not allow_noncanonical:
            raise
        words = len(WORD_RE.findall(raw_text))
        if words > max_words:
            raise ValueError(f"原始 completion 共 {words} words，超过 {max_words}")
        return "", "", raw_text, words, False


def truncate_prompt(tokenizer, prompt: str, max_prompt_tokens: int) -> tuple[str, int, int]:
    ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    raw_length = len(ids)
    if raw_length <= max_prompt_tokens:
        return prompt, raw_length, raw_length
    kept_ids = ids[-max_prompt_tokens:]
    return tokenizer.decode(kept_ids, skip_special_tokens=False), raw_length, len(kept_ids)


def build_prompt(tokenizer, row: dict[str, Any], args: argparse.Namespace) -> tuple[str, int, int]:
    history = get_history(row)
    if not history:
        raise ValueError(f"{example_id(row)} 缺少 history")
    messages = build_general_recommendation_cot_messages(
        history, args.item_type, language=args.language
    )
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return truncate_prompt(tokenizer, prompt, args.max_prompt_tokens)


def load_existing(
    path: Path,
    max_words: int,
    allow_noncanonical: bool,
) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    existing = {}
    for row in read_jsonl(path):
        key = example_id(row)
        cot = str(row.get("cot") or "")
        try:
            parse_generated_output(cot, max_words, allow_noncanonical)
        except ValueError:
            continue
        if key:
            existing[key] = row
    return existing


def write_ordered(path: Path, source: list[dict[str, Any]], generated: dict[str, dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    written = 0
    with temporary.open("w", encoding="utf-8") as file:
        for row in source:
            output = generated.get(example_id(row))
            if output is None:
                continue
            file.write(json.dumps(output, ensure_ascii=False) + "\n")
            written += 1
    temporary.replace(path)
    return written


def build_output(
    source: dict[str, Any],
    raw: str,
    prompt_raw_tokens: int,
    prompt_final_tokens: int,
    attempt: int,
    elapsed: float,
    finish_reason: str,
    generated_tokens: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    analysis, answer, cot, words, format_valid = parse_generated_output(
        raw,
        args.max_output_words,
        args.allow_noncanonical_output,
    )
    history = get_history(source)
    output = dict(source)
    output.update(
        {
            "base_query": history,
            "user_history": history,
            "query": f"{history}{COT_SEPARATOR}{cot}",
            "cot": cot,
            "analysis": analysis,
            "answer": answer,
            "cot_format_valid": format_valid,
            "cot_word_count": words,
            "generator_model": args.model,
            "generator_adapter": args.adapter,
            "generation_mode": "vllm_lora",
            "generation_attempt": attempt,
            "generation_seconds": round(elapsed, 6),
            "generation_finish_reason": finish_reason,
            "generation_tokens": generated_tokens,
            "prompt_raw_tokens": prompt_raw_tokens,
            "prompt_final_tokens": prompt_final_tokens,
            "prompt_truncated_left": prompt_raw_tokens > prompt_final_tokens,
            "candidates": [
                {
                    "candidate_index": 0,
                    "analysis": analysis,
                    "think": analysis,
                    "answer": answer,
                    "cot": cot,
                }
            ],
        }
    )
    return output


def audit(source: list[dict[str, Any]], output: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    source_ids = [example_id(row) for row in source]
    output_ids = [example_id(row) for row in output]
    stats: dict[str, Any] = {
        "source_rows": len(source),
        "output_rows": len(output),
        "missing_rows": len(set(source_ids) - set(output_ids)),
        "duplicate_output_ids": len(output_ids) - len(set(output_ids)),
        "order_matches_source": output_ids == source_ids,
        "invalid_format_rows": 0,
        "format_valid_rows": 0,
        "noncanonical_rows_used_for_retrieval": 0,
        "over_512_word_rows": 0,
        "empty_cot_rows": 0,
        "raw_asin_in_history_or_cot_rows": 0,
        "target_title_in_cot_rows": 0,
        "target_id_in_prompt_contract": False,
        "target_title_in_prompt_contract": False,
        "max_cot_words": 0,
        "mean_cot_words": 0.0,
        "max_generation_tokens": 0,
        "mean_generation_tokens": 0.0,
        "finish_reason_counts": {},
        "prompt_left_truncated_rows": 0,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "expected_split": args.expected_split,
        "allow_noncanonical_output": args.allow_noncanonical_output,
        "seed": args.seed,
    }
    cot_word_counts = []
    generation_token_counts = []
    for row in output:
        cot = str(row.get("cot") or "")
        history = str(row.get("user_history") or "")
        if not cot:
            stats["empty_cot_rows"] += 1
        words = int(row.get("cot_word_count") or len(WORD_RE.findall(cot)))
        cot_word_counts.append(words)
        stats["max_cot_words"] = max(stats["max_cot_words"], words)
        if words > args.max_output_words:
            stats["over_512_word_rows"] += 1
        format_valid = row.get("cot_format_valid")
        if format_valid is None:
            try:
                canonicalize_cot(cot, args.max_output_words)
                format_valid = True
            except ValueError:
                format_valid = False
        if format_valid:
            stats["format_valid_rows"] += 1
        else:
            stats["invalid_format_rows"] += 1
            stats["noncanonical_rows_used_for_retrieval"] += 1
        generation_tokens = int(row.get("generation_tokens") or 0)
        generation_token_counts.append(generation_tokens)
        stats["max_generation_tokens"] = max(
            stats["max_generation_tokens"], generation_tokens
        )
        finish_reason = str(row.get("generation_finish_reason") or "unknown")
        stats["finish_reason_counts"][finish_reason] = (
            stats["finish_reason_counts"].get(finish_reason, 0) + 1
        )
        if RAW_ASIN_RE.search(f"{history}\n{cot}"):
            stats["raw_asin_in_history_or_cot_rows"] += 1
        title = str(row.get("target_item_title") or "").strip()
        if title and title.casefold() in cot.casefold():
            stats["target_title_in_cot_rows"] += 1
        if row.get("prompt_truncated_left"):
            stats["prompt_left_truncated_rows"] += 1
    if cot_word_counts:
        stats["mean_cot_words"] = sum(cot_word_counts) / len(cot_word_counts)
    if generation_token_counts:
        stats["mean_generation_tokens"] = sum(generation_token_counts) / len(
            generation_token_counts
        )
    return stats


def cleanup_llm(llm: Any) -> None:
    try:
        executor = getattr(getattr(llm, "llm_engine", None), "model_executor", None)
        if executor is not None and hasattr(executor, "shutdown"):
            executor.shutdown()
    except Exception:
        pass
    del llm
    gc.collect()


def patch_transformers_tokenizer_compat() -> None:
    """兼容 vLLM 0.7.x 与较新 transformers tokenizer。"""
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
                if token is None or str(token) in seen:
                    continue
                seen.add(str(token))
                tokens.append(token)
        return tokens

    PreTrainedTokenizerBase.all_special_tokens_extended = all_special_tokens_extended


def main() -> None:
    args = parse_args()
    if args.seed != 42:
        raise ValueError("项目随机种子必须为 42")
    if args.vllm_max_model_len < args.max_prompt_tokens + args.max_new_tokens:
        raise ValueError("vllm-max-model-len 必须覆盖 max-prompt-tokens + max-new-tokens")
    if args.top_k == 0 or args.top_k < -1:
        raise ValueError("top-k 必须为 -1（关闭）或正整数")

    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    patch_transformers_tokenizer_compat()
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    source = read_jsonl(args.input)
    if not source or any(row.get("split") != args.expected_split for row in source):
        raise ValueError(f"输入必须是非空 {args.expected_split} JSONL")
    if len({example_id(row) for row in source}) != len(source):
        raise ValueError("输入 example_id 为空或重复")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=True)
    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=args.vllm_max_model_len,
        max_num_seqs=args.vllm_max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_lora=True,
        max_lora_rank=64,
        seed=args.seed,
        disable_log_stats=True,
    )
    lora_request = LoRARequest("sft_adapter", 1, args.adapter)
    generated = (
        load_existing(
            args.output,
            args.max_output_words,
            args.allow_noncanonical_output,
        )
        if args.resume
        else {}
    )
    pending = [row for row in source if example_id(row) not in generated]
    print(json.dumps({"source": len(source), "existing": len(generated), "pending": len(pending)}, ensure_ascii=False), flush=True)

    try:
        for attempt in range(1, args.max_attempts + 1):
            if not pending:
                break
            next_pending = []
            sampling = SamplingParams(
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                max_tokens=args.max_new_tokens,
                seed=args.seed + attempt - 1,
            )
            for start in range(0, len(pending), args.generation_batch_size):
                batch = pending[start : start + args.generation_batch_size]
                prompt_records = [build_prompt(tokenizer, row, args) for row in batch]
                prompts = [record[0] for record in prompt_records]
                started = time.perf_counter()
                results = llm.generate(
                    prompts,
                    sampling_params=sampling,
                    use_tqdm=False,
                    lora_request=lora_request,
                )
                elapsed = (time.perf_counter() - started) / len(batch)
                for row, prompt_record, result in zip(batch, prompt_records, results):
                    result_output = result.outputs[0] if result.outputs else None
                    raw = result_output.text if result_output is not None else ""
                    try:
                        generated[example_id(row)] = build_output(
                            row,
                            raw,
                            prompt_record[1],
                            prompt_record[2],
                            attempt,
                            elapsed,
                            str(getattr(result_output, "finish_reason", "") or "unknown"),
                            len(getattr(result_output, "token_ids", []) or []),
                            args,
                        )
                    except ValueError:
                        next_pending.append(row)
                if len(generated) % args.save_every < len(batch):
                    written = write_ordered(args.output, source, generated)
                    print(json.dumps({"attempt": attempt, "written": written, "total": len(source)}, ensure_ascii=False), flush=True)
            pending = next_pending
            print(json.dumps({"attempt": attempt, "retry_rows": len(pending)}, ensure_ascii=False), flush=True)
    finally:
        cleanup_llm(llm)

    written = write_ordered(args.output, source, generated)
    output_rows = read_jsonl(args.output)
    stats = audit(source, output_rows, args)
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)
    invalid_format_failure = (
        stats["invalid_format_rows"] and not args.allow_noncanonical_output
    )
    if written != len(source) or pending or invalid_format_failure or stats["over_512_word_rows"]:
        raise RuntimeError(f"vLLM 推理未完整通过审计，剩余 {len(pending)} 条")


if __name__ == "__main__":
    main()
