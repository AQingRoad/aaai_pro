#!/usr/bin/env python3
"""将 teacher CoT 结果转换为 QUERY-only SFT messages，并控制完整序列长度。"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROMPT_DIR = SCRIPT_DIR.parent / "prompts"
sys.path.insert(0, str(PROMPT_DIR))

from query_only_cot_student import PROMPT_NAME, PROMPT_VERSION, build_messages  # noqa: E402


# history 的每个物品以“编号. ”开头；删除物品时保留剩余物品的原编号。
ITEM_START_RE = re.compile(r"^\s*(\d+)\.\s+", re.MULTILINE)
COT_ITEM_REF_RE = re.compile(r"\bItem\s+(\d+)\b|物品\s*(\d+)", re.IGNORECASE)
DETAILS_RE = re.compile(r";\s*Details:.*$", re.DOTALL)
DESCRIPTION_RE = re.compile(r";\s*Description:.*?(?=;\s*Details:|$)", re.DOTALL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="teacher batch_results.jsonl")
    parser.add_argument("--output", type=Path, required=True, help="输出 SFT JSONL")
    parser.add_argument("--model-path", required=True, help="训练模型或同款 tokenizer 路径")
    parser.add_argument("--max-length", type=int, default=4096, help="完整 chat template 的最大 token 数")
    parser.add_argument("--language", choices=("en", "zh"), default="en", help="学生提示词语言")
    return parser.parse_args()


def extract_history_parts(query: str) -> tuple[str, list[tuple[int, str]]]:
    """拆分 query 头部和历史物品；每个物品可占多行。"""
    matches = list(ITEM_START_RE.finditer(query))
    if not matches:
        raise ValueError("query 中没有识别到历史物品编号")

    header = query[: matches[0].start()].rstrip()
    items: list[tuple[int, str]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(query)
        items.append((int(match.group(1)), query[match.start() : end].strip()))
    return header, items


def compose_query(header: str, items: list[tuple[int, str]]) -> str:
    """保持原始编号，重新拼接保留的历史物品。"""
    return "\n".join(part for part in (header, *(text for _, text in items)) if part).strip()


def format_cot(row: dict[str, Any]) -> str:
    """只读取解析后的 think/answer，确保教师输出标签完整。"""
    parsed = row.get("parsed_output") or {}
    think = str(parsed.get("think") or "").strip()
    answer = str(parsed.get("answer") or "").strip()
    if not think or not answer:
        raise ValueError("parsed_output 缺少 think 或 answer")
    return f"<think>\n{think}\n</think>\n<answer>\n{answer}\n</answer>"


def extract_cot_item_references(cot: str) -> list[int]:
    """提取 CoT 中显式引用的英文 Item 编号或中文物品编号。"""
    references = {
        int(first or second)
        for first, second in COT_ITEM_REF_RE.findall(cot)
        if first or second
    }
    return sorted(references)


def token_count(tokenizer: Any, messages: list[dict[str, str]]) -> int:
    """按训练时的 chat template 统计 system、user、assistant 的完整长度。"""
    token_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
    )
    return len(token_ids)


def make_messages(query: str, cot: str, language: str) -> list[dict[str, str]]:
    messages = build_messages(query, language)
    messages.append({"role": "assistant", "content": cot})
    return messages


def remove_item_field(item_text: str, field: str) -> str:
    """删除历史物品中信息较长的 Details 或 Description 字段。"""
    pattern = DETAILS_RE if field == "details" else DESCRIPTION_RE
    return pattern.sub("", item_text).rstrip("; ").strip()


def shorten_last_item(
    tokenizer: Any,
    header: str,
    item: tuple[int, str],
    cot: str,
    language: str,
    max_length: int,
) -> tuple[str, int]:
    """只有单个历史物品仍超长时，保留其开头并从尾部缩短。"""
    item_number, item_text = item
    item_tokens = tokenizer.encode(item_text, add_special_tokens=False)
    low, high = 1, len(item_tokens)
    best_query = ""
    best_kept = 0

    # 二分查找可放入完整 chat template 的最大物品前缀。
    while low <= high:
        middle = (low + high) // 2
        shortened = tokenizer.decode(item_tokens[:middle], skip_special_tokens=True).strip()
        candidate_query = compose_query(header, [(item_number, shortened)])
        candidate_messages = make_messages(candidate_query, cot, language)
        if token_count(tokenizer, candidate_messages) <= max_length:
            best_query = candidate_query
            best_kept = middle
            low = middle + 1
        else:
            high = middle - 1

    if not best_query:
        raise ValueError("system、最短 query 与完整 CoT 已超过 max_length")
    return best_query, len(item_tokens) - best_kept


def fit_messages(
    tokenizer: Any,
    query: str,
    cot: str,
    language: str,
    max_length: int,
    protected_item_numbers: set[int] | None = None,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """从最早物品开始缩短，并优先保留 CoT 明确引用的历史证据。"""
    header, items = extract_history_parts(query)
    original_numbers = [number for number, _ in items]
    original_messages = make_messages(query, cot, language)
    original_tokens = token_count(tokenizer, original_messages)
    protected_item_numbers = protected_item_numbers or set()
    removed_numbers: list[int] = []
    details_removed_numbers: list[int] = []
    description_removed_numbers: list[int] = []

    def current_messages() -> list[dict[str, str]]:
        return make_messages(compose_query(header, items), cot, language)

    def fits() -> bool:
        return token_count(tokenizer, current_messages()) <= max_length

    # 先处理未被 CoT 引用的物品；每组内部保持从最早到最近的顺序。
    processing_order = [
        number for number in original_numbers if number not in protected_item_numbers
    ] + [number for number in original_numbers if number in protected_item_numbers]

    for item_number in processing_order:
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

        # 引用物品只压缩长字段；整条删除仅用于未引用物品。
        if item_number not in protected_item_numbers and len(items) > 1:
            removed_numbers.append(items.pop(item_index)[0])

    fitted_query = compose_query(header, items)
    messages = make_messages(fitted_query, cot, language)
    tail_tokens_removed = 0
    if token_count(tokenizer, messages) > max_length:
        # 极端情况下保留最后一条历史的开头，教师 CoT 仍保持完整。
        if len(items) != 1:
            raise ValueError("压缩长字段后仍超长，且剩余多个 CoT 引用物品")
        fitted_query, tail_tokens_removed = shorten_last_item(
            tokenizer, header, items[0], cot, language, max_length
        )
        messages = make_messages(fitted_query, cot, language)

    final_tokens = token_count(tokenizer, messages)
    if final_tokens > max_length:
        raise AssertionError(f"截断后仍有 {final_tokens} tokens，超过 {max_length}")

    truncation = {
        "original_token_count": original_tokens,
        "final_token_count": final_tokens,
        "removed_history_item_count": len(removed_numbers),
        "removed_history_item_numbers": removed_numbers,
        "retained_history_item_numbers": [number for number in original_numbers if number not in removed_numbers],
        "details_removed_item_numbers": details_removed_numbers,
        "description_removed_item_numbers": description_removed_numbers,
        "oldest_retained_item_tail_tokens_removed": tail_tokens_removed,
    }
    return messages, truncation


def main() -> None:
    # transformers 只在执行转换时需要，便于在轻量环境中复用和测试解析函数。
    from transformers import AutoTokenizer

    args = parse_args()
    if args.max_length <= 0:
        raise ValueError("max_length 必须大于 0")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    row_count = 0
    shortened_count = 0
    removed_reference_count = 0
    modified_reference_count = 0
    with args.input.open("r", encoding="utf-8") as source, args.output.open(
        "w", encoding="utf-8"
    ) as target:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") != "success":
                raise ValueError(f"第 {line_number} 行 status 不是 success")

            cot = format_cot(row)
            cot_references = extract_cot_item_references(cot)
            messages, truncation = fit_messages(
                tokenizer,
                str(row.get("query") or "").strip(),
                cot,
                args.language,
                args.max_length,
                set(cot_references),
            )
            removed_references = sorted(
                set(cot_references) & set(truncation["removed_history_item_numbers"])
            )
            modified_item_numbers = set(truncation["details_removed_item_numbers"]) | set(
                truncation["description_removed_item_numbers"]
            )
            if truncation["oldest_retained_item_tail_tokens_removed"]:
                modified_item_numbers.add(truncation["retained_history_item_numbers"][0])
            modified_references = sorted(set(cot_references) & modified_item_numbers)
            output = {
                "example_id": row.get("example_id"),
                "source_line_index": row.get("source_line_index"),
                "messages": messages,
                "prompt_name": PROMPT_NAME,
                "prompt_version": PROMPT_VERSION,
                "language": args.language,
                "max_length": args.max_length,
                "truncation": truncation,
                "cot_referenced_item_numbers": cot_references,
                "removed_cot_referenced_item_numbers": removed_references,
                "modified_cot_referenced_item_numbers": modified_references,
            }
            target.write(json.dumps(output, ensure_ascii=False) + "\n")
            row_count += 1
            shortened_count += int(truncation["original_token_count"] > args.max_length)
            removed_reference_count += int(bool(removed_references))
            modified_reference_count += int(bool(modified_references))

    print(
        json.dumps(
            {
                "rows": row_count,
                "shortened_rows": shortened_count,
                "rows_with_removed_cot_references": removed_reference_count,
                "rows_with_modified_cot_references": modified_reference_count,
                "max_length": args.max_length,
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
