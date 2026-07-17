#!/usr/bin/env python3
"""Build target-hidden ms-swift GRPO prompts from the reserved 80% split."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from manu_src.scripts.prompts import build_general_recommendation_cot_messages


RAW_ASIN_RE = re.compile(r"(?<![A-Z0-9])B0[A-Z0-9]{8}(?![A-Z0-9])")
TARGET_FIELDS = {"target_item_id", "target_item_title", "positive"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sft-example-ids", type=Path, required=True)
    parser.add_argument("--item-type", default="CD or vinyl release")
    parser.add_argument("--language", default="en")
    parser.add_argument("--expected-rows", type=int, default=8578)
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_ids(path: Path) -> set[str]:
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.seed != 42:
        raise ValueError("项目随机种子必须为 42")
    if args.expected_rows <= 0:
        raise ValueError("--expected-rows 必须为正数")
    for path in (args.input, args.sft_example_ids):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output.resolve() == args.input.resolve():
        raise ValueError("输出不能覆盖输入")

    source_rows = read_jsonl(args.input)
    if len(source_rows) != args.expected_rows:
        raise ValueError(
            f"GRPO reserved split 应为 {args.expected_rows} 条，当前为 {len(source_rows)}"
        )
    sft_ids = read_ids(args.sft_example_ids)

    output_rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    raw_asin_rows = 0
    for line_number, row in enumerate(source_rows, 1):
        example_id = str(row.get("example_id") or "").strip()
        history = str(row.get("query") or "").strip()
        target_item_id = row.get("target_item_id")
        history_item_ids = row.get("history_item_ids")
        if not example_id or example_id in seen_ids:
            raise ValueError(f"第 {line_number} 行 example_id 为空或重复: {example_id!r}")
        seen_ids.add(example_id)
        if example_id in sft_ids:
            raise ValueError(f"第 {line_number} 行与 SFT 20% 重叠: {example_id}")
        if row.get("split") != "train":
            raise ValueError(f"第 {line_number} 行 split 必须为 train")
        if not history or target_item_id is None or not isinstance(history_item_ids, list):
            raise ValueError(
                f"第 {line_number} 行缺少 query、target_item_id 或 history_item_ids"
            )
        if RAW_ASIN_RE.search(history):
            raw_asin_rows += 1
            raise ValueError(f"第 {line_number} 行 history 含裸 ASIN")
        if "[TRUNCATED]" in str(row.get("positive") or ""):
            raise ValueError(f"第 {line_number} 行 positive 含 [TRUNCATED]")

        messages = build_general_recommendation_cot_messages(
            history,
            args.item_type,
            language=args.language,
        )
        serialized_messages = json.dumps(messages, ensure_ascii=False)
        leaked_field_names = {
            field for field in TARGET_FIELDS if f'"{field}"' in serialized_messages
        }
        if leaked_field_names:
            raise ValueError(
                f"第 {line_number} 行 messages 含 target 字段名: {sorted(leaked_field_names)}"
            )
        output_rows.append(
            {
                "example_id": example_id,
                "user_id": str(row.get("user_id") or ""),
                "interaction_id": row.get("interaction_id"),
                "split": "train",
                "prompt_type": "general_non_target",
                "prompt_language": args.language,
                "item_type": args.item_type,
                "user_history": history,
                "target_item_id": int(target_item_id),
                "history_item_ids": [int(item_id) for item_id in history_item_ids],
                "messages": messages,
            }
        )

    if seen_ids & sft_ids:
        raise AssertionError("SFT/GRPO example_id overlap is not zero")
    write_jsonl(args.output, output_rows)
    manifest = {
        "input": str(args.input),
        "input_sha256": sha256(args.input),
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "rows": len(output_rows),
        "expected_rows": args.expected_rows,
        "seed": args.seed,
        "split": "train",
        "split_unit": "example_id",
        "sft_example_ids": str(args.sft_example_ids),
        "sft_example_count": len(sft_ids),
        "sft_grpo_example_overlap": len(seen_ids & sft_ids),
        "prompt_type": "general_non_target",
        "prompt_language": args.language,
        "target_available_to_policy": False,
        "target_item_id_reward_metadata_only": True,
        "positive_in_output": False,
        "target_title_field_in_output": False,
        "target_title_overlap_policy": "允许真实历史中自然出现的同名物品，不注入 target 字段",
        "raw_asin_rows": raw_asin_rows,
        "history_truncated_during_build": False,
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
