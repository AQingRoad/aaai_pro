#!/usr/bin/env python3
"""把原始 test pair 与非 target API CoT 对齐成 embedding 评测输入。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from manu_src.scripts.inference.vllm_lora_non_target_cot import (
    COT_SEPARATOR,
    RAW_ASIN_RE,
    canonicalize_cot,
)


TARGET_FIELDS = {"target_item_id", "target_item_title", "positive"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--cot", type=Path, required=True)
    parser.add_argument("--cot-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=1341)
    parser.add_argument("--max-output-words", type=int, default=512)
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.seed != 42:
        raise ValueError("项目随机种子固定为 42")

    pairs = read_jsonl(args.pairs)
    cot_rows = read_jsonl(args.cot)
    cot_manifest = json.loads(args.cot_manifest.read_text(encoding="utf-8"))
    if len(pairs) != args.expected_rows or len(cot_rows) != args.expected_rows:
        raise ValueError(
            f"pairs 与 CoT 都应为 {args.expected_rows} 条，"
            f"当前分别为 {len(pairs)}、{len(cot_rows)}"
        )
    if cot_manifest.get("success_rows") != args.expected_rows:
        raise ValueError("API CoT manifest 的 success_rows 不完整")
    if cot_manifest.get("failure_rows") != 0:
        raise ValueError("API CoT manifest 仍包含失败行")
    if cot_manifest.get("seed") != args.seed:
        raise ValueError("API CoT seed 与评测 seed 不一致")
    if cot_manifest.get("target_fields_in_messages") is not False:
        raise ValueError("API CoT manifest 未证明生成消息排除 target 字段")

    output_rows = []
    providers: Counter[str] = Counter()
    models: Counter[str] = Counter()
    target_title_matches = 0
    max_words = 0
    seen_ids = set()

    for line_number, (pair, cot_row) in enumerate(zip(pairs, cot_rows), 1):
        pair_id = str(pair.get("example_id") or "").strip()
        cot_id = str(cot_row.get("example_id") or "").strip()
        if not pair_id or pair_id in seen_ids:
            raise ValueError(f"pair 第 {line_number} 行 example_id 为空或重复")
        seen_ids.add(pair_id)
        if pair_id != cot_id:
            raise ValueError(f"第 {line_number} 行 pair 与 CoT example_id 不一致")
        if pair.get("split") != "test" or cot_row.get("split") != "test":
            raise ValueError(f"第 {line_number} 行 split 必须同时为 test")
        if TARGET_FIELDS & set(cot_row):
            raise ValueError(f"API CoT 第 {line_number} 行含 target 或 positive 字段")

        history = str(pair.get("query") or "").strip()
        cot_history = str(cot_row.get("user_history") or "").strip()
        raw_cot = str(cot_row.get("cot") or "").strip()
        if not history or history != cot_history:
            raise ValueError(f"第 {line_number} 行生成 history 与 test query 不一致")
        analysis, answer, canonical_cot, words = canonicalize_cot(
            raw_cot, args.max_output_words
        )
        if canonical_cot != raw_cot:
            raise ValueError(f"第 {line_number} 行 CoT 未使用规范格式")
        if RAW_ASIN_RE.search(f"{history}\n{canonical_cot}"):
            raise ValueError(f"第 {line_number} 行 history 或 CoT 含裸 ASIN")
        if pair.get("target_item_id") is None or not pair.get("history_item_ids"):
            raise ValueError(f"第 {line_number} 行缺少 target_item_id 或 history_item_ids")
        if "[TRUNCATED]" in str(pair.get("positive") or ""):
            raise ValueError(f"第 {line_number} 行 positive 含 [TRUNCATED]")

        title = str(pair.get("target_item_title") or "").strip()
        if title and title.casefold() in canonical_cot.casefold():
            target_title_matches += 1
        provider = str(cot_row.get("provider") or "unknown")
        model = str(cot_row.get("model") or "unknown")
        providers[provider] += 1
        models[model] += 1
        max_words = max(max_words, words)

        output = dict(pair)
        output.update(
            {
                "base_query": history,
                "user_history": history,
                "query": f"{history}{COT_SEPARATOR}{canonical_cot}",
                "cot": canonical_cot,
                "analysis": analysis,
                "answer": answer,
                "cot_word_count": words,
                "cot_prompt_type": cot_row.get("prompt_type", "general_non_target"),
                "cot_provider": provider,
                "cot_model": model,
                "cot_source_line_index": cot_row.get("source_line_index"),
            }
        )
        output_rows.append(output)

    write_jsonl_atomic(args.output, output_rows)
    manifest = {
        "rows": len(output_rows),
        "split": "test",
        "seed": args.seed,
        "pairs": str(args.pairs),
        "pairs_sha256": sha256(args.pairs),
        "cot": str(args.cot),
        "cot_sha256": sha256(args.cot),
        "cot_manifest": str(args.cot_manifest),
        "cot_manifest_sha256": sha256(args.cot_manifest),
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "providers": dict(providers),
        "models": dict(models),
        "max_cot_words": max_words,
        "target_fields_in_api_cot": False,
        "target_fields_in_generation_messages": False,
        "target_title_exact_match_rows": target_title_matches,
        "history_order_and_content_match": True,
        "raw_asin_rows": 0,
        "positive_truncated_rows": 0,
        "cot_separator": COT_SEPARATOR,
    }
    write_json_atomic(args.manifest_output, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
