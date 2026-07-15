#!/usr/bin/env python3
"""把完整测试集与 RRec API 输出整理成成对的 embedding 评测输入。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


ASIN_RE = re.compile(r"(?<![A-Z0-9])B0[A-Z0-9]{8}(?![A-Z0-9])", re.IGNORECASE)
REASONING_SEPARATOR = "\n\nRecommendation reasoning:\n"


def read_jsonl(path: Path) -> list[dict]:
    """逐行读取 JSONL；避免按 Unicode 换行符拆坏字段内容。"""
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


def sha256(path: Path) -> str:
    """记录输入和输出文件哈希，方便核对本地与服务器文件。"""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def request_text(row: dict) -> str:
    """提取实际发送给 API 的消息文本，仅用于检查监督文本是否误入请求。"""
    messages = row.get("request", {}).get("messages", [])
    return "\n".join(str(message.get("content", "")) for message in messages)


def build_full_query(query: str, think: str, answer: str) -> str:
    """保留 GLM 完整分析和答案，并用固定分隔符拼接到原始历史。"""
    return (
        query.rstrip()
        + REASONING_SEPARATOR
        + think.strip()
        + "\n\n<answer>"
        + answer.strip()
        + "</answer>"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="构建 history-only 与 query+RRec v1.3 完整输出的成对评测数据。"
    )
    parser.add_argument("--source", type=Path, required=True, help="严格口径 test.jsonl")
    parser.add_argument("--api-results", type=Path, required=True, help="API batch_results.jsonl")
    parser.add_argument("--output", type=Path, required=True, help="成对评测 JSONL")
    parser.add_argument("--audit-output", type=Path, required=True, help="输入与泄漏审计 JSON")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.seed != 42:
        parser.error("项目随机种子固定为 42")

    source_rows = read_jsonl(args.source)
    api_rows = read_jsonl(args.api_results)
    if len(source_rows) != len(api_rows):
        raise ValueError(f"source={len(source_rows)}，API success={len(api_rows)}，行数不一致")

    source_by_id = {str(row["example_id"]): row for row in source_rows}
    if len(source_by_id) != len(source_rows):
        raise ValueError("source 中 example_id 不唯一")
    api_by_id = {str(row["example_id"]): row for row in api_rows}
    if len(api_by_id) != len(api_rows):
        raise ValueError("API 输出中 example_id 不唯一")
    if source_by_id.keys() != api_by_id.keys():
        missing = sorted(source_by_id.keys() - api_by_id.keys())[:10]
        extra = sorted(api_by_id.keys() - source_by_id.keys())[:10]
        raise ValueError(f"example_id 集合不一致；missing={missing}；extra={extra}")

    failed_rows = []
    parse_error_rows = []
    query_mismatch_rows = []
    target_in_history_rows = []
    positive_in_request_rows = []
    positive_in_full_query_rows = []
    target_title_in_reasoning_rows = []
    raw_asin_rows = []
    truncated_rows = []
    format_attempt_histogram: dict[str, int] = {}
    output_rows = []

    for sample_index, source in enumerate(source_rows, 1):
        example_id = str(source["example_id"])
        api = api_by_id[example_id]
        status = str(api.get("status", ""))
        parsed = api.get("parsed_output") or {}
        think = str(parsed.get("think") or "").strip()
        answer = str(parsed.get("answer") or "").strip()
        parse_error = str(parsed.get("parse_error") or "").strip()

        if status != "success":
            failed_rows.append({"example_id": example_id, "status": status})
        if not think or not answer or parse_error:
            parse_error_rows.append(
                {"example_id": example_id, "think": bool(think), "answer": bool(answer), "parse_error": parse_error}
            )
        if str(api.get("query", "")) != str(source["query"]):
            query_mismatch_rows.append(example_id)

        attempts = len(api.get("response_attempts") or [])
        format_attempt_histogram[str(attempts)] = format_attempt_histogram.get(str(attempts), 0) + 1

        query = str(source["query"])
        positive = str(source.get("positive") or "")
        full_query = build_full_query(query, think, answer)
        reasoning = full_query[len(query.rstrip()) :]
        request = request_text(api)
        target_title = str(source.get("target_item_title") or "").strip()

        if int(source["target_item_id"]) in {int(value) for value in source.get("history_item_ids", [])}:
            target_in_history_rows.append(example_id)
        if positive and positive in request:
            positive_in_request_rows.append(example_id)
        if positive and positive in full_query:
            positive_in_full_query_rows.append(example_id)
        if target_title and target_title.casefold() in reasoning.casefold():
            target_title_in_reasoning_rows.append(
                {"example_id": example_id, "target_item_title": target_title, "answer": answer}
            )
        if ASIN_RE.search(query) or ASIN_RE.search(reasoning):
            raw_asin_rows.append(example_id)
        if "[TRUNCATED]" in query or "[TRUNCATED]" in reasoning:
            truncated_rows.append(example_id)

        common = {
            "example_id": example_id,
            "category": source.get("category", "CDs_and_Vinyl"),
            "split": "test",
            "user_id": source.get("user_id"),
            "interaction_id": source.get("interaction_id"),
            "target_item_id": int(source["target_item_id"]),
            "target_item_title": target_title,
            "history_item_ids": [int(value) for value in source.get("history_item_ids", [])],
            "history_item_count": int(source.get("history_item_count", len(source.get("history_item_ids", [])))),
            "sample_index": sample_index,
            "seed": args.seed,
            "query_scope": "title_store_categories",
            "prompt_name": api.get("prompt_name"),
            "prompt_version": api.get("prompt_version"),
        }
        output_rows.append({**common, "codex_mode": "history_only", "query": query})
        output_rows.append({**common, "codex_mode": "rrec_v1_3_full", "query": full_query})

    hard_errors = {
        "failed_rows": failed_rows,
        "parse_error_rows": parse_error_rows,
        "query_mismatch_rows": query_mismatch_rows,
    }
    if any(hard_errors.values()):
        raise ValueError(f"API 完整性门禁失败：{json.dumps(hard_errors, ensure_ascii=False)[:4000]}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        for row in output_rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    audit = {
        "seed": args.seed,
        "source": str(args.source.resolve()),
        "source_sha256": sha256(args.source),
        "api_results": str(args.api_results.resolve()),
        "api_results_sha256": sha256(args.api_results),
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
        "source_count": len(source_rows),
        "api_success_count": len(api_rows),
        "output_count": len(output_rows),
        "variant_counts": {"history_only": len(source_rows), "rrec_v1_3_full": len(source_rows)},
        "query_scope": "title_store_categories",
        "prompt_name": sorted({str(row.get("prompt_name")) for row in api_rows}),
        "prompt_version": sorted({str(row.get("prompt_version")) for row in api_rows}),
        "model": sorted({str(row.get("model")) for row in api_rows}),
        "format_attempt_histogram": format_attempt_histogram,
        "hard_error_counts": {name: len(rows) for name, rows in hard_errors.items()},
        "target_in_history_count": len(target_in_history_rows),
        "positive_exact_in_api_request_count": len(positive_in_request_rows),
        "positive_exact_in_full_query_count": len(positive_in_full_query_rows),
        "target_title_string_in_reasoning_count": len(target_title_in_reasoning_rows),
        "target_title_string_in_reasoning_rows": target_title_in_reasoning_rows,
        "raw_asin_count": len(raw_asin_rows),
        "truncated_marker_count": len(truncated_rows),
    }
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
