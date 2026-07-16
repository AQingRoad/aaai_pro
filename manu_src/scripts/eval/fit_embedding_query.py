#!/usr/bin/env python3
"""按 SFT 数据构建策略压缩超长 embedding query。"""

from __future__ import annotations

import re
from typing import Any, Callable


RECOMMENDATION_SEPARATOR = "\n\nRecommendation reasoning:\n"
TAGGED_COT_START_RE = re.compile(r"\n\n(?=<think>\s*)", re.IGNORECASE)
ITEM_START_RE = re.compile(r"^\s*(\d+)\.\s+", re.MULTILINE)
COT_ITEM_REF_RE = re.compile(r"\bItem\s+(\d+)\b|物品\s*(\d+)", re.IGNORECASE)
DETAILS_RE = re.compile(r";\s*Details:.*$", re.DOTALL)
DESCRIPTION_RE = re.compile(r";\s*Description:.*?(?=;\s*Details:|$)", re.DOTALL)


def extract_history_parts(query: str) -> tuple[str, list[tuple[int, str]]]:
    """拆分 query 头部和编号历史物品。"""
    matches = list(ITEM_START_RE.finditer(query))
    if not matches:
        raise ValueError("query 中没有识别到历史物品编号")
    header = query[: matches[0].start()].rstrip()
    items = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(query)
        items.append((int(match.group(1)), query[match.start() : end].strip()))
    return header, items


def compose_query(header: str, items: list[tuple[int, str]]) -> str:
    """保持原始编号并重新拼接历史。"""
    return "\n".join(part for part in (header, *(text for _, text in items)) if part).strip()


def extract_cot_item_references(cot: str) -> list[int]:
    """提取推理文本中显式引用的历史物品编号。"""
    return sorted(
        {
            int(first or second)
            for first, second in COT_ITEM_REF_RE.findall(cot)
            if first or second
        }
    )


def remove_item_field(item_text: str, field: str) -> str:
    """删除历史物品中的 Details 或 Description 字段。"""
    pattern = DETAILS_RE if field == "details" else DESCRIPTION_RE
    return pattern.sub("", item_text).rstrip("; ").strip()


def split_history_and_suffix(query: str) -> tuple[str, str]:
    """拆出编号历史和推理后缀，后续压缩不得修改推理文本。"""
    separator_index = query.find(RECOMMENDATION_SEPARATOR)
    if separator_index >= 0:
        return query[:separator_index].rstrip(), query[separator_index:]

    # 兼容直接在 history 后拼接 tagged CoT、没有 Recommendation reasoning 标题的旧数据。
    match = TAGGED_COT_START_RE.search(query)
    if match:
        return query[: match.start()].rstrip(), query[match.start() :]
    return query.strip(), ""


def join_history_and_suffix(history: str, suffix: str) -> str:
    """保持推理后缀原文及其分隔符。"""
    return history.rstrip() + suffix if suffix else history.strip()


def query_token_count(
    tokenizer: Any,
    query: str,
    format_query: Callable[[str], str],
) -> int:
    """使用 embedding 编码时的 instruction 和 special-token 口径计数。"""
    return len(
        tokenizer(
            format_query(query),
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]
    )


def shorten_single_item(
    tokenizer: Any,
    header: str,
    item: tuple[int, str],
    suffix: str,
    max_length: int,
    format_query: Callable[[str], str],
) -> tuple[str, int]:
    """只剩一条历史仍超长时，二分查找可保留的最大物品前缀。"""
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
        history = compose_query(header, [(item_number, shortened)])
        candidate = join_history_and_suffix(history, suffix)
        if query_token_count(tokenizer, candidate, format_query) <= max_length:
            best_query = candidate
            best_kept = middle
            low = middle + 1
        else:
            high = middle - 1

    if not best_query:
        raise ValueError("embedding instruction、最短历史与完整推理后缀已超过 max_length")
    return best_query, len(item_tokens) - best_kept


def fit_embedding_query(
    tokenizer: Any,
    query: str,
    max_length: int,
    format_query: Callable[[str], str],
    *,
    original_token_count: int | None = None,
) -> tuple[str, dict[str, Any]]:
    """完整保留推理后缀，按字段和物品粒度从最早历史开始压缩。"""
    original_token_count = original_token_count or query_token_count(
        tokenizer, query, format_query
    )
    history, suffix = split_history_and_suffix(query)

    empty_audit = {
        "compression_applied": False,
        "original_token_count": original_token_count,
        "final_token_count": original_token_count,
        "removed_history_item_count": 0,
        "removed_history_item_numbers": [],
        "retained_history_item_numbers": [],
        "details_removed_item_numbers": [],
        "description_removed_item_numbers": [],
        "oldest_retained_item_tail_tokens_removed": 0,
        "protected_cot_item_numbers": [],
        "reasoning_suffix_present": bool(suffix),
        "reasoning_suffix_preserved": True,
    }
    if original_token_count <= max_length:
        return query, empty_audit

    header, items = extract_history_parts(history)
    original_numbers = [number for number, _ in items]
    protected_numbers = set(extract_cot_item_references(suffix))
    removed_numbers: list[int] = []
    details_removed_numbers: list[int] = []
    description_removed_numbers: list[int] = []

    def current_query() -> str:
        current_history = compose_query(header, items)
        return join_history_and_suffix(current_history, suffix)

    def fits() -> bool:
        return query_token_count(tokenizer, current_query(), format_query) <= max_length

    # 与 SFT 一致：优先处理未被 CoT 引用的物品，每组内部从最早到最近。
    processing_order = [
        number for number in original_numbers if number not in protected_numbers
    ] + [number for number in original_numbers if number in protected_numbers]

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

        # 被推理显式引用的物品保留；未引用物品仍超长时删除整条。
        if item_number not in protected_numbers and len(items) > 1:
            removed_numbers.append(items.pop(item_index)[0])

    fitted_query = current_query()
    tail_tokens_removed = 0
    if query_token_count(tokenizer, fitted_query, format_query) > max_length:
        if len(items) != 1:
            raise ValueError("压缩长字段后仍超长，且剩余多个 CoT 引用物品")
        fitted_query, tail_tokens_removed = shorten_single_item(
            tokenizer,
            header,
            items[0],
            suffix,
            max_length,
            format_query,
        )

    final_token_count = query_token_count(tokenizer, fitted_query, format_query)
    if final_token_count > max_length:
        raise AssertionError(
            f"SFT 风格压缩后仍有 {final_token_count} tokens，超过 {max_length}"
        )
    if suffix and not fitted_query.endswith(suffix):
        raise AssertionError("query 压缩修改了推理后缀")

    audit = {
        "compression_applied": True,
        "original_token_count": original_token_count,
        "final_token_count": final_token_count,
        "removed_history_item_count": len(removed_numbers),
        "removed_history_item_numbers": removed_numbers,
        "retained_history_item_numbers": [
            number for number in original_numbers if number not in removed_numbers
        ],
        "details_removed_item_numbers": details_removed_numbers,
        "description_removed_item_numbers": description_removed_numbers,
        "oldest_retained_item_tail_tokens_removed": tail_tokens_removed,
        "protected_cot_item_numbers": sorted(protected_numbers),
        "reasoning_suffix_present": bool(suffix),
        "reasoning_suffix_preserved": True,
    }
    return fitted_query, audit
