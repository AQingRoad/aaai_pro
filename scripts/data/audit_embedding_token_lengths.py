#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from statistics import median

from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rubric_cot_pipeline.embeddings import (
    DEFAULT_RECOMMENDATION_QUERY_INSTRUCTION,
    format_qwen3_query,
)
from rubric_cot_pipeline.item_metadata import build_item_text


def percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(len(ordered) * fraction) - 1))]


def summary(values: list[int], limit: int) -> dict[str, int | float]:
    return {
        "min": min(values),
        "median": median(values),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values),
        "over_limit": sum(value > limit for value in values),
        "at_limit": sum(value == limit for value in values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit untruncated Qwen3 query and positive token lengths before embedding training."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--item-info",
        type=Path,
        default=None,
        help="Optionally audit every full candidate item text used by evaluation.",
    )
    parser.add_argument(
        "--query-instruction", default=DEFAULT_RECOMMENDATION_QUERY_INSTRUCTION
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=True, use_fast=True
    )
    queries: list[str] = []
    positives: list[str] = []
    with args.dataset.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            queries.append(
                format_qwen3_query(str(row.get("query") or ""), args.query_instruction)
            )
            positives.append(str(row.get("positive") or ""))
    if not queries:
        raise ValueError(f"No rows in {args.dataset}")

    def lengths(texts: list[str]) -> list[int]:
        return [
            len(tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])
            for text in texts
        ]

    query_lengths = lengths(queries)
    positive_lengths = lengths(positives)
    report = {
        "dataset": str(args.dataset.resolve()),
        "tokenizer": args.tokenizer,
        "rows": len(queries),
        "max_length": args.max_length,
        "query_includes_instruction": True,
        "query_tokens": summary(query_lengths, args.max_length),
        "positive_tokens": summary(positive_lengths, args.max_length),
    }
    if args.item_info is not None:
        item_texts: list[str] = []
        with args.item_info.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                text = build_item_text(item, str(item.get("title") or ""), max_chars=0)
                if "[TRUNCATED]" in text:
                    raise ValueError("Candidate item text contains [TRUNCATED]")
                item_texts.append(text)
        if not item_texts:
            raise ValueError(f"No rows in {args.item_info}")
        report["item_info"] = str(args.item_info.resolve())
        report["candidate_items"] = len(item_texts)
        report["candidate_item_tokens"] = summary(lengths(item_texts), args.max_length)
    report["safe_without_token_truncation"] = (
        report["query_tokens"]["over_limit"] == 0
        and report["positive_tokens"]["over_limit"] == 0
        and report.get("candidate_item_tokens", {}).get("over_limit", 0) == 0
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["safe_without_token_truncation"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
