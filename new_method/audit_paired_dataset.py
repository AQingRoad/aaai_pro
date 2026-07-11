#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from new_method.core import (
    RAW_ASIN_RE,
    RANK_BUCKETS,
    append_think,
    as_int_set,
    normalize_space,
    read_jsonl,
)
from rubric_cot_pipeline.embeddings import (
    DEFAULT_RECOMMENDATION_QUERY_INSTRUCTION,
    format_qwen3_query,
)


def percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def summary(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"min": 0, "median": 0.0, "p95": 0, "max": 0}
    return {
        "min": min(values),
        "median": median(values),
        "p95": percentile(values, 0.95),
        "max": max(values),
    }


def normalized_for_overlap(value: Any) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", str(value or "").lower()).strip()


def tokenize_lengths(tokenizer, texts: list[str]) -> list[int]:
    encoded = tokenizer(
        texts,
        add_special_tokens=True,
        padding=False,
        truncation=False,
        return_length=True,
    )
    if "length" in encoded:
        return [int(value) for value in encoded["length"]]
    return [len(ids) for ids in encoded["input_ids"]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a GDR paired retriever JSONL.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-split", default="")
    parser.add_argument("--tokenizer", default="")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--query-instruction", default=DEFAULT_RECOMMENDATION_QUERY_INSTRUCTION)
    parser.add_argument("--min-paired-rate", type=float, default=0.0)
    parser.add_argument("--min-target-title-chars", type=int, default=8)
    parser.add_argument("--fail-on-target-title-overlap", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if not 0.0 <= args.min_paired_rate <= 1.0:
        raise ValueError("--min-paired-rate must be in [0, 1]")

    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True, use_fast=True)

    critical = Counter()
    warnings = Counter()
    modes = Counter()
    buckets = Counter()
    target_counts = Counter()
    pair_keys = Counter()
    scorer_checkpoints = Counter()
    query_modes = Counter()
    query_instructions = Counter()
    history_chars: list[int] = []
    good_chars: list[int] = []
    bad_chars: list[int] = []
    positive_chars: list[int] = []
    negative_counts: list[int] = []
    history_tokens: list[int] = []
    good_tokens: list[int] = []
    bad_tokens: list[int] = []
    positive_tokens: list[int] = []
    rows = 0

    for row in read_jsonl(args.dataset):
        rows += 1
        history = str(row.get("history") or "").strip()
        positive = str(row.get("positive") or "").strip()
        good = str(row.get("good_cot") or "").strip()
        bad = str(row.get("bad_cot") or "").strip()
        target_id = row.get("target_item_id")
        target_key = "" if target_id is None else str(target_id)
        history_ids = as_int_set(row.get("history_item_ids"))
        negatives = row.get("negatives") or []
        if not isinstance(negatives, list):
            negatives = [negatives]

        mode = str(row.get("training_mode") or "")
        bucket = str(row.get("baseline_rank_bucket") or "")
        modes[mode] += 1
        buckets[bucket] += 1
        scorer_checkpoints[str(row.get("scorer_checkpoint") or "")] += 1
        query_modes[str(row.get("query_mode") or "")] += 1
        query_instructions[str(row.get("query_instruction") or "")] += 1

        if not history:
            critical["empty_history"] += 1
        if not positive:
            critical["empty_positive"] += 1
        if not target_key:
            critical["missing_target_item_id"] += 1
        else:
            target_counts[target_key] += 1
        if args.expected_split and str(row.get("split") or "") != args.expected_split:
            critical["split_mismatch"] += 1
        if target_id is not None and int(target_id) in history_ids:
            critical["target_in_history_ids"] += 1
        if row.get("query_mode") != "history_plus_think_only":
            critical["query_mode_mismatch"] += 1
        if bool(row.get("has_good_cot")) != bool(good):
            critical["good_presence_mismatch"] += 1
        if bool(row.get("has_bad_cot")) != bool(bad):
            critical["bad_presence_mismatch"] += 1
        if mode == "paired" and (not good or not bad):
            critical["paired_mode_missing_view"] += 1
        if mode == "history_only" and (good or bad):
            critical["history_only_has_cot"] += 1
        if good and bad and normalize_space(good) == normalize_space(bad):
            critical["identical_good_bad_cot"] += 1
        if good and row.get("good_delta_log_rank") is not None:
            if float(row["good_delta_log_rank"]) <= 0:
                critical["nonpositive_good_log_rank"] += 1
        if bad and row.get("bad_delta_log_rank") is not None:
            if float(row["bad_delta_log_rank"]) >= 0:
                critical["nonnegative_bad_log_rank"] += 1

        for field, text in (
            ("history", history),
            ("positive", positive),
            ("good_cot", good),
            ("bad_cot", bad),
        ):
            if "[TRUNCATED]" in text:
                critical[f"{field}_truncated_marker"] += 1
            if RAW_ASIN_RE.search(text):
                critical[f"raw_asin_in_{field}"] += 1

        title = normalized_for_overlap(row.get("target_item_title"))
        if len(title) >= args.min_target_title_chars:
            for field, cot in (("good", good), ("bad", bad)):
                if cot and title in normalized_for_overlap(cot):
                    warnings[f"target_title_overlap_{field}"] += 1
                    if args.fail_on_target_title_overlap:
                        critical[f"target_title_overlap_{field}"] += 1

        if not negatives:
            warnings["no_explicit_negatives"] += 1
        if any(not str(text or "").strip() for text in negatives):
            critical["empty_negative_text"] += 1
        if any("[TRUNCATED]" in str(text or "") for text in negatives):
            critical["negative_truncated_marker"] += 1

        history_chars.append(len(history))
        good_chars.append(len(good))
        bad_chars.append(len(bad))
        positive_chars.append(len(positive))
        negative_counts.append(len(negatives))
        pair_keys[(history, positive)] += 1

        if tokenizer is not None:
            formatted_history = format_qwen3_query(history, args.query_instruction)
            history_length = tokenize_lengths(tokenizer, [formatted_history])[0]
            history_tokens.append(history_length)
            if history_length > args.max_length:
                critical["history_over_max_length"] += 1
            if good:
                good_query = format_qwen3_query(append_think(history, good), args.query_instruction)
                good_length = tokenize_lengths(tokenizer, [good_query])[0]
                good_tokens.append(good_length)
                if good_length > args.max_length:
                    critical["history_good_over_max_length"] += 1
            if bad:
                bad_query = format_qwen3_query(append_think(history, bad), args.query_instruction)
                bad_length = tokenize_lengths(tokenizer, [bad_query])[0]
                bad_tokens.append(bad_length)
                if bad_length > args.max_length:
                    critical["history_bad_over_max_length"] += 1
            positive_length = tokenize_lengths(tokenizer, [positive])[0]
            positive_tokens.append(positive_length)
            if positive_length > args.max_length:
                critical["positive_over_max_length"] += 1

    if rows == 0:
        raise ValueError(f"No rows found in {args.dataset}")

    paired_rows = modes.get("paired", 0)
    paired_rate = paired_rows / rows
    if paired_rate < args.min_paired_rate:
        critical["paired_rate_below_minimum"] += 1
    for bucket in RANK_BUCKETS:
        if buckets.get(bucket, 0) == 0:
            warnings[f"empty_rank_bucket_{bucket}"] += 1
    if len(scorer_checkpoints) != 1:
        critical["multiple_scorer_checkpoints"] += 1
    if len(query_modes) != 1:
        critical["multiple_query_modes"] += 1
    if len(query_instructions) != 1:
        critical["multiple_query_instructions"] += 1

    duplicate_pairs = sum(count - 1 for count in pair_keys.values() if count > 1)
    repeated_target_rows = sum(count for count in target_counts.values() if count > 1)
    report = {
        "dataset": str(Path(args.dataset).resolve()),
        "rows": rows,
        "critical_failures": dict(sorted(critical.items())),
        "warnings": dict(sorted(warnings.items())),
        "training_modes": dict(sorted(modes.items())),
        "rank_buckets": dict(sorted(buckets.items())),
        "paired_rate": paired_rate,
        "unique_targets": len(target_counts),
        "repeated_target_rows": repeated_target_rows,
        "duplicate_history_positive_rows": duplicate_pairs,
        "scorer_checkpoints": dict(scorer_checkpoints),
        "query_modes": dict(query_modes),
        "query_instructions": dict(query_instructions),
        "history_chars": summary(history_chars),
        "good_cot_chars": summary(good_chars),
        "bad_cot_chars": summary(bad_chars),
        "positive_chars": summary(positive_chars),
        "negative_count": summary(negative_counts),
        "tokenizer": args.tokenizer or None,
        "max_length": args.max_length,
        "history_tokens": summary(history_tokens),
        "history_good_tokens": summary(good_tokens),
        "history_bad_tokens": summary(bad_tokens),
        "positive_tokens": summary(positive_tokens),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.strict and critical:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
