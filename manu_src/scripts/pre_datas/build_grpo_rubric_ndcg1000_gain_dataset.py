#!/usr/bin/env python3
"""Enrich target-hidden GRPO prompts with reward-only target/reference metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any


RAW_ASIN_RE = re.compile(r"(?<![A-Z0-9])B0[A-Z0-9]{8}(?![A-Z0-9])")
EXPECTED_QUERY_FIELDS = [
    "relative_time",
    "title",
    "rating",
    "store",
    "categories",
    "description",
    "details",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--messages-input", type=Path, required=True)
    parser.add_argument("--raw-input", type=Path, required=True)
    parser.add_argument("--reference-cot-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probe-output", type=Path)
    parser.add_argument("--expected-rows", type=int, default=8578)
    parser.add_argument("--probe-rows", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path} 第 {line_number} 行不是合法 JSON") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path} 第 {line_number} 行不是 JSON object")
            rows.append(row)
    return rows


def index_unique(rows: list[dict[str, Any]], path: Path) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for line_number, row in enumerate(rows, 1):
        example_id = str(row.get("example_id") or "").strip()
        if not example_id:
            raise ValueError(f"{path} 第 {line_number} 行缺少 example_id")
        if example_id in indexed:
            raise ValueError(f"{path} 出现重复 example_id: {example_id}")
        indexed[example_id] = row
    return indexed


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    manifest_path = path.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def message_text(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    return "\n".join(
        str(message.get("content") or "")
        for message in messages
        if isinstance(message, dict)
    )


def main() -> None:
    args = parse_args()
    if args.seed != 42:
        raise ValueError("项目随机种子必须为 42")
    if args.expected_rows <= 0 or args.probe_rows <= 0:
        raise ValueError("expected/probe rows 必须为正数")
    for path in (args.messages_input, args.raw_input, args.reference_cot_input):
        if not path.is_file():
            raise FileNotFoundError(path)
    output_paths = [args.output]
    if args.probe_output is not None:
        output_paths.append(args.probe_output)
    input_resolved = {
        args.messages_input.resolve(),
        args.raw_input.resolve(),
        args.reference_cot_input.resolve(),
    }
    if any(path.resolve() in input_resolved for path in output_paths):
        raise ValueError("输出文件不能覆盖任一输入文件")

    message_rows = read_jsonl(args.messages_input)
    raw_rows = read_jsonl(args.raw_input)
    cot_rows = read_jsonl(args.reference_cot_input)
    if len(message_rows) != args.expected_rows or len(raw_rows) != args.expected_rows:
        raise ValueError(
            "messages/raw 行数与预期不一致: "
            f"{len(message_rows)}/{len(raw_rows)}，预期 {args.expected_rows}"
        )
    raw_by_id = index_unique(raw_rows, args.raw_input)
    cot_by_id = index_unique(cot_rows, args.reference_cot_input)
    message_ids = [str(row.get("example_id") or "").strip() for row in message_rows]
    if len(set(message_ids)) != len(message_ids) or any(not value for value in message_ids):
        raise ValueError("messages-input 的 example_id 为空或重复")

    missing_raw = sorted(set(message_ids) - set(raw_by_id))
    missing_cot = sorted(set(message_ids) - set(cot_by_id))
    if missing_raw or missing_cot:
        raise ValueError(
            f"关联失败：raw 缺 {len(missing_raw)} 条，reference CoT 缺 {len(missing_cot)} 条"
        )

    output_rows: list[dict[str, Any]] = []
    target_title_exact_history_overlap = 0
    for line_number, base in enumerate(message_rows, 1):
        example_id = message_ids[line_number - 1]
        raw = raw_by_id[example_id]
        cot = cot_by_id[example_id]
        history = str(base.get("user_history") or "").strip()
        raw_history = str(raw.get("query") or "").strip()
        cot_history = str(cot.get("user_history") or "").strip()
        target_text = str(raw.get("positive") or "").strip()
        target_title = str(raw.get("target_item_title") or "").strip()
        reference_cot = str(cot.get("cot") or "").strip()
        target_id = raw.get("target_item_id")
        messages = base.get("messages")

        if base.get("split") != "train" or raw.get("split") != "train" or cot.get("split") != "train":
            raise ValueError(f"第 {line_number} 行 split 不一致或不是 train")
        if base.get("target_item_id") != target_id:
            raise ValueError(f"第 {line_number} 行 target_item_id 不一致")
        if history != raw_history or history != cot_history:
            raise ValueError(f"第 {line_number} 行三路 user history 不一致")
        if list(raw.get("query_fields") or []) != EXPECTED_QUERY_FIELDS:
            raise ValueError(f"第 {line_number} 行 raw query_fields 口径错误")
        if list(cot.get("query_fields") or []) != EXPECTED_QUERY_FIELDS:
            raise ValueError(f"第 {line_number} 行 CoT query_fields 口径错误")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"第 {line_number} 行 messages 为空或格式错误")
        if not history or not target_text or not target_title or not reference_cot:
            raise ValueError(f"第 {line_number} 行缺少 history、完整 target 或 reference CoT")
        if "[TRUNCATED]" in target_text or "[TRUNCATED]" in reference_cot:
            raise ValueError(f"第 {line_number} 行 target/reference CoT 含 [TRUNCATED]")
        if RAW_ASIN_RE.search(history):
            raise ValueError(f"第 {line_number} 行 history 含裸 ASIN")
        serialized_messages = json.dumps(messages, ensure_ascii=False)
        for forbidden_field in ("target_item_text", "target_item_title", "reference_cot"):
            if f'"{forbidden_field}"' in serialized_messages:
                raise ValueError(f"第 {line_number} 行 messages 含 reward-only 字段名")
        if history not in message_text(messages):
            raise ValueError(f"第 {line_number} 行 messages 未包含完整 user_history")
        target_title_exact_history_overlap += int(
            bool(target_title) and target_title.casefold() in history.casefold()
        )

        enriched = dict(base)
        enriched["target_item_text"] = target_text
        enriched["target_item_title"] = target_title
        enriched["reference_cot"] = reference_cot
        output_rows.append(enriched)

    write_jsonl(args.output, output_rows)
    common_manifest = {
        "messages_input": str(args.messages_input),
        "messages_input_sha256": sha256(args.messages_input),
        "raw_input": str(args.raw_input),
        "raw_input_sha256": sha256(args.raw_input),
        "reference_cot_input": str(args.reference_cot_input),
        "reference_cot_input_sha256": sha256(args.reference_cot_input),
        "seed": args.seed,
        "split": "train",
        "query_fields": EXPECTED_QUERY_FIELDS,
        "policy_input": "messages containing history only",
        "reward_target_field": "target_item_text (full positive, untruncated)",
        "reward_reference_field": "reference_cot (fixed API CoT, untruncated)",
        "target_available_to_policy": False,
        "reference_cot_available_to_policy": False,
        "target_or_reference_truncated": 0,
        "raw_asin_in_history": 0,
        "target_title_exact_history_overlap": target_title_exact_history_overlap,
    }
    manifest = {
        **common_manifest,
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "rows": len(output_rows),
        "expected_rows": args.expected_rows,
    }
    write_manifest(args.output, manifest)

    if args.probe_output is not None:
        if args.probe_rows > len(output_rows):
            raise ValueError("probe rows 超过完整数据行数")
        selected_indices = sorted(
            random.Random(args.seed).sample(range(len(output_rows)), args.probe_rows)
        )
        probe_rows = [output_rows[index] for index in selected_indices]
        write_jsonl(args.probe_output, probe_rows)
        probe_manifest = {
            **common_manifest,
            "source_output": str(args.output),
            "output": str(args.probe_output),
            "output_sha256": sha256(args.probe_output),
            "rows": len(probe_rows),
            "selection": "random.sample without replacement, then restore source order",
            "selected_source_indices_zero_based": selected_indices,
        }
        write_manifest(args.probe_output, probe_manifest)
        manifest["probe_output"] = str(args.probe_output)
        manifest["probe_rows"] = len(probe_rows)
        manifest["probe_selected_source_indices_zero_based"] = selected_indices
        write_manifest(args.output, manifest)

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
