#!/usr/bin/env python3
"""把原始 train pair 与 history-only CoT 对齐为结构化 embedding 训练数据。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


COT_SEPARATOR = "\n\nRecommendation reasoning:\n"
COT_RE = re.compile(
    r"^\s*<analysis>\s*(?P<analysis>.*?)\s*</analysis>\s*"
    r"<answer>\s*(?P<answer>.*?)\s*</answer>\s*$",
    re.DOTALL,
)
RAW_ASIN_RE = re.compile(r"(?<![A-Z0-9])B0[A-Z0-9]{8}(?![A-Z0-9])")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--cot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.seed != 42:
        raise ValueError("项目随机种子必须为 42")

    pair_rows = read_jsonl(args.pairs)
    cot_rows = read_jsonl(args.cot)
    cot_by_id: dict[str, dict[str, Any]] = {}
    for line_number, row in enumerate(cot_rows, 1):
        example_id = str(row.get("example_id") or "").strip()
        if not example_id or example_id in cot_by_id:
            raise ValueError(f"CoT 第 {line_number} 行 example_id 为空或重复: {example_id!r}")
        cot_by_id[example_id] = row

    output_rows: list[dict[str, Any]] = []
    seen_pair_ids: set[str] = set()
    for line_number, pair in enumerate(pair_rows, 1):
        example_id = str(pair.get("example_id") or "").strip()
        if not example_id or example_id in seen_pair_ids:
            raise ValueError(f"pair 第 {line_number} 行 example_id 为空或重复: {example_id!r}")
        seen_pair_ids.add(example_id)
        if pair.get("split") != "train":
            raise ValueError(f"pair 第 {line_number} 行 split 必须为 train")
        cot_row = cot_by_id.get(example_id)
        if cot_row is None:
            raise ValueError(f"pair 第 {line_number} 行缺少对应 CoT: {example_id}")

        history = str(pair.get("query") or "").strip()
        cot_history = str(cot_row.get("user_history") or "").strip()
        cot = str(cot_row.get("cot") or "").strip()
        positive = str(pair.get("positive") or "").strip()
        target_item_id = pair.get("target_item_id")
        if not history or not cot or not positive or target_item_id is None:
            raise ValueError(f"pair 第 {line_number} 行缺少 history、CoT、positive 或 target_item_id")
        if history != cot_history:
            raise ValueError(f"pair 第 {line_number} 行 history 与 CoT 生成输入不一致")
        match = COT_RE.fullmatch(cot)
        if match is None or not match.group("analysis").strip() or not match.group("answer").strip():
            raise ValueError(f"pair 第 {line_number} 行 CoT 未严格匹配 <analysis>/<answer>")
        if RAW_ASIN_RE.search(history) or RAW_ASIN_RE.search(cot):
            raise ValueError(f"pair 第 {line_number} 行 history 或 CoT 含裸 ASIN")
        if "[TRUNCATED]" in positive:
            raise ValueError(f"pair 第 {line_number} 行 positive 含 [TRUNCATED]")

        output_rows.append(
            {
                "example_id": example_id,
                "split": "train",
                "query": f"{history}{COT_SEPARATOR}{cot}",
                "history": history,
                "cot": cot,
                "positive": positive,
                "target_item_id": int(target_item_id),
                "history_item_ids": pair.get("history_item_ids", []),
                "query_fields": pair.get("query_fields", []),
                "cot_prompt_type": cot_row.get("prompt_type", "general_non_target"),
            }
        )

    extra_cot_ids = set(cot_by_id) - seen_pair_ids
    if extra_cot_ids:
        raise ValueError(f"存在 {len(extra_cot_ids)} 条未匹配的 CoT")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        for row in output_rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(args.output)

    manifest = {
        "rows": len(output_rows),
        "seed": args.seed,
        "pairs": str(args.pairs),
        "pairs_sha256": sha256(args.pairs),
        "cot": str(args.cot),
        "cot_sha256": sha256(args.cot),
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "query_composition": "history + Recommendation reasoning + full <analysis>/<answer> CoT",
        "history_truncated_during_build": False,
        "cot_truncated_during_build": False,
        "positive_truncated_during_build": False,
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
