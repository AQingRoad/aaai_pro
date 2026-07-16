#!/usr/bin/env python3
"""把 history-only API CoT 转成 ms-swift messages 格式的全量 SFT 数据。"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from manu_src.scripts.prompts import (  # noqa: E402
    build_general_recommendation_cot_messages,
)


RAW_ASIN_RE = re.compile(r"(?<![A-Z0-9])B0[A-Z0-9]{8}(?![A-Z0-9])")
COT_RE = re.compile(
    r"^\s*<analysis>\s*(?P<analysis>.*?)\s*</analysis>\s*"
    r"<answer>\s*(?P<answer>.*?)\s*</answer>\s*$",
    re.DOTALL,
)
FORBIDDEN_KEYS = {"target_item_id", "target_item_title", "positive"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--item-type", default="CD or vinyl release")
    parser.add_argument("--language", default="en")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path} 第 {line_number} 行不是合法 JSON") from error
    return rows


def validate_source(row: dict[str, Any], line_number: int) -> tuple[str, str]:
    leaked_keys = FORBIDDEN_KEYS & set(row)
    if leaked_keys:
        raise ValueError(f"第 {line_number} 行包含 target/positive 字段: {sorted(leaked_keys)}")
    history = str(row.get("user_history") or "").strip()
    cot = str(row.get("cot") or "").strip()
    if not history or not cot:
        raise ValueError(f"第 {line_number} 行缺少 user_history 或 cot")
    match = COT_RE.fullmatch(cot)
    if match is None or not match.group("analysis").strip() or not match.group("answer").strip():
        raise ValueError(f"第 {line_number} 行 CoT 未严格匹配 <analysis>/<answer>")
    if RAW_ASIN_RE.search(history) or RAW_ASIN_RE.search(cot):
        raise ValueError(f"第 {line_number} 行含裸 ASIN")
    return history, cot


def main() -> None:
    args = parse_args()
    if args.seed != 42:
        raise ValueError("项目随机种子必须为 42")
    source_rows = read_jsonl(args.input)
    output_rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for line_number, row in enumerate(source_rows, 1):
        history, cot = validate_source(row, line_number)
        example_id = str(row.get("example_id") or "").strip()
        if not example_id or example_id in seen_ids:
            raise ValueError(f"第 {line_number} 行 example_id 为空或重复: {example_id!r}")
        seen_ids.add(example_id)
        messages = build_general_recommendation_cot_messages(
            history,
            args.item_type,
            language=args.language,
        )
        messages.append({"role": "assistant", "content": cot})
        serialized = json.dumps(messages, ensure_ascii=False)
        if any(f'"{key}"' in serialized for key in FORBIDDEN_KEYS):
            raise ValueError(f"第 {line_number} 行 messages 含 target/positive 字段名")
        output_rows.append(
            {
                "example_id": example_id,
                "source_line_index": int(row.get("source_line_index", line_number - 1)),
                "split": "train",
                "prompt_type": "general_non_target",
                "prompt_language": args.language,
                "item_type": args.item_type,
                "messages": messages,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        for row in output_rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "input": str(args.input),
                "output": str(args.output),
                "rows": len(output_rows),
                "seed": args.seed,
                "prompt_language": args.language,
                "item_type": args.item_type,
                "target_fields_in_messages": False,
                "raw_asin_rows": 0,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
