#!/usr/bin/env python3
"""使用 embedding tokenizer 统计 query、positive 和候选 item 的真实长度分布。"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from transformers import AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "pre_datas"))
from format_positive import format_positive  # noqa: E402


QUERY_INSTRUCTION = (
    "Given a user's past item interactions and optional recommendation reasoning, "
    "retrieve items the user is likely to prefer next."
)
THRESHOLDS = (512, 1024, 2048, 4096)
PERCENTILES = (50, 75, 90, 95, 99)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def format_query(query: str) -> str:
    return f"Instruct: {QUERY_INSTRUCTION}\nQuery: {query}"


def token_lengths(tokenizer, texts: list[str], batch_size: int) -> list[int]:
    lengths = []
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            texts[start : start + batch_size],
            truncation=False,
            padding=False,
            add_special_tokens=True,
        )["input_ids"]
        lengths.extend(len(ids) for ids in encoded)
    return lengths


def percentile(sorted_values: list[int], percent: int) -> float:
    position = (len(sorted_values) - 1) * percent / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def summarize(values: list[int]) -> dict:
    ordered = sorted(values)
    result = {
        "count": len(ordered),
        "min": ordered[0],
        "mean": sum(ordered) / len(ordered),
        "max": ordered[-1],
    }
    result.update({f"p{p}": percentile(ordered, p) for p in PERCENTILES})
    result["over_threshold"] = {
        str(threshold): {
            "count": sum(value > threshold for value in ordered),
            "rate": sum(value > threshold for value in ordered) / len(ordered),
        }
        for threshold in THRESHOLDS
    }
    return result


def split_report(tokenizer, base_path: Path, rated_path: Path, batch_size: int) -> dict:
    base_rows = read_jsonl(base_path)
    rated_rows = read_jsonl(rated_path)
    if len(base_rows) != len(rated_rows):
        raise ValueError(f"行数不一致: {base_path} vs {rated_path}")
    for index, (base, rated) in enumerate(zip(base_rows, rated_rows), 1):
        if base["example_id"] != rated["example_id"]:
            raise ValueError(f"第 {index} 行 example_id 不一致")
        if base["positive"] != rated["positive"]:
            raise ValueError(f"第 {index} 行 positive 不一致")

    base_raw = token_lengths(tokenizer, [row["query"] for row in base_rows], batch_size)
    rated_raw = token_lengths(tokenizer, [row["query"] for row in rated_rows], batch_size)
    base_actual = token_lengths(
        tokenizer, [format_query(row["query"]) for row in base_rows], batch_size
    )
    rated_actual = token_lengths(
        tokenizer, [format_query(row["query"]) for row in rated_rows], batch_size
    )
    positive = token_lengths(tokenizer, [row["positive"] for row in base_rows], batch_size)
    return {
        "rows": len(base_rows),
        "base_raw_query": summarize(base_raw),
        "rated_raw_query": summarize(rated_raw),
        "base_actual_query_with_instruction": summarize(base_actual),
        "rated_actual_query_with_instruction": summarize(rated_actual),
        "rating_token_delta": summarize(
            [rated - base for base, rated in zip(base_actual, rated_actual)]
        ),
        "positive": summarize(positive),
    }


def candidate_report(tokenizer, item_info_path: Path, batch_size: int) -> dict:
    rows = sorted(read_jsonl(item_info_path), key=lambda row: int(row["item_id"]))
    texts = []
    for row in rows:
        item_id = int(row["item_id"])
        if item_id == 0:
            continue
        fallback = str(row.get("title") or "").strip() or f"item_{item_id}"
        if not str(row.get("title") or "").strip():
            row = {**row, "title": fallback}
        texts.append(format_positive(row, fallback))
    return summarize(token_lengths(tokenizer, texts, batch_size))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--rated-dir", type=Path, required=True)
    parser.add_argument("--item-info", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.seed != 42:
        parser.error("项目随机种子固定为 42")

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=True,
        use_fast=True,
        padding_side="left",
    )
    report = {
        "tokenizer": args.tokenizer,
        "tokenizer_class": tokenizer.__class__.__name__,
        "add_special_tokens": True,
        "truncation": False,
        "seed": args.seed,
        "splits": {
            split: split_report(
                tokenizer,
                args.base_dir / filename,
                args.rated_dir / filename,
                args.batch_size,
            )
            for split, filename in (("train", "train.jsonl"), ("valid", "val.jsonl"), ("test", "test.jsonl"))
        },
        "candidate_items": candidate_report(tokenizer, args.item_info, args.batch_size),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
