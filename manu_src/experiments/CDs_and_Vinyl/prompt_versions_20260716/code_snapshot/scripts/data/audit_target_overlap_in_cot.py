#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


RAW_ASIN_RE = re.compile(r"\bB0[A-Z0-9]{8}\b", re.IGNORECASE)
EXPLICIT_RATING_RE = re.compile(
    r"\b(?:rating|rated|reviews?|feedback|high-rated|low-rated|avg_rating|rating_count)\b",
    re.IGNORECASE,
)
AMBIGUOUS_STAR_RE = re.compile(r"\bstars?\b", re.IGNORECASE)
TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def text(value: Any) -> str:
    return "" if value is None else str(value)


def key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        text(row.get("split")),
        text(row.get("interaction_id")),
        text(row.get("user_id")),
        text(row.get("target_item_id")),
    )


def normalized_tokens(value: Any) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text(value))]


def contains_token_sequence(haystack: Any, needle: Any) -> bool:
    haystack_tokens = normalized_tokens(haystack)
    needle_tokens = normalized_tokens(needle)
    if not needle_tokens:
        return False
    width = len(needle_tokens)
    return any(
        haystack_tokens[index : index + width] == needle_tokens
        for index in range(max(0, len(haystack_tokens) - width + 1))
    )


def cot_text(row: dict[str, Any]) -> str:
    direct = text(row.get("cot")).strip()
    if direct:
        return direct
    candidates = row.get("candidates")
    if isinstance(candidates, list) and candidates:
        candidate = candidates[0]
        if isinstance(candidate, dict):
            return text(candidate.get("cot")).strip()
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Report accidental held-out target-title overlap in target-free CoT. "
            "This audit never filters or selects CoT using the target."
        )
    )
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--cot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-split", required=True)
    args = parser.parse_args()

    history_rows = read_jsonl(args.history)
    cot_rows = read_jsonl(args.cot)
    history_by_key = {key(row): row for row in history_rows}
    failures: Counter[str] = Counter()
    overlaps: list[dict[str, Any]] = []
    normalized_cots: list[str] = []
    raw_asin_rows = 0
    explicit_rating_term_rows = 0
    ambiguous_star_term_rows = 0

    if len(history_by_key) != len(history_rows):
        failures["duplicate_history_key"] += len(history_rows) - len(history_by_key)

    seen_cot_keys: set[tuple[str, str, str, str]] = set()
    for row in cot_rows:
        row_key = key(row)
        if row_key in seen_cot_keys:
            failures["duplicate_cot_key"] += 1
        seen_cot_keys.add(row_key)
        history = history_by_key.get(row_key)
        if history is None:
            failures["cot_key_missing_from_history"] += 1
            continue
        if text(row.get("split")) != args.expected_split:
            failures["split_mismatch"] += 1
        cot = cot_text(row)
        if not cot:
            failures["empty_cot"] += 1
            continue
        if not all(tag in cot for tag in ("<think>", "</think>", "<answer>", "</answer>")):
            failures["missing_tagged_cot"] += 1
        normalized_cots.append(" ".join(normalized_tokens(cot)))
        raw_asin_rows += int(bool(RAW_ASIN_RE.search(cot)))
        explicit_rating_term_rows += int(bool(EXPLICIT_RATING_RE.search(cot)))
        ambiguous_star_term_rows += int(bool(AMBIGUOUS_STAR_RE.search(cot)))

        title = text(history.get("target_item_title")).strip()
        if not title or not contains_token_sequence(cot, title):
            continue
        base_query = text(history.get("query"))
        title_tokens = normalized_tokens(title)
        overlaps.append(
            {
                "interaction_id": history.get("interaction_id"),
                "user_id": history.get("user_id"),
                "target_item_id": history.get("target_item_id"),
                "target_item_title": title,
                "title_token_count": len(title_tokens),
                "title_in_base_history": contains_token_sequence(base_query, title),
                "informative_title": len(title_tokens) >= 2 or len(" ".join(title_tokens)) >= 12,
            }
        )

    missing_cot_keys = set(history_by_key) - seen_cot_keys
    if missing_cot_keys:
        failures["history_key_missing_from_cot"] += len(missing_cot_keys)

    target_only = [row for row in overlaps if not row["title_in_base_history"]]
    informative_target_only = [row for row in target_only if row["informative_title"]]
    report = {
        "history": str(args.history.resolve()),
        "cot": str(args.cot.resolve()),
        "expected_split": args.expected_split,
        "history_rows": len(history_rows),
        "cot_rows": len(cot_rows),
        "matched_rows": len(seen_cot_keys & set(history_by_key)),
        "unique_cot_texts": len(set(normalized_cots)),
        "duplicate_cot_texts": len(normalized_cots) - len(set(normalized_cots)),
        "raw_asin_rows": raw_asin_rows,
        "explicit_rating_term_rows": explicit_rating_term_rows,
        "ambiguous_star_term_rows": ambiguous_star_term_rows,
        "rating_term_interpretation": (
            "Explicit rating terms exclude standalone star/stars because music titles, artist names, "
            "and soundtrack franchises frequently contain those tokens. Ambiguous star rows remain "
            "reported separately for manual context review."
        ),
        "exact_target_title_in_cot": len(overlaps),
        "target_title_also_in_base_history": len(overlaps) - len(target_only),
        "target_title_absent_from_base_history": len(target_only),
        "informative_target_title_absent_from_base_history": len(informative_target_only),
        "target_only_overlap_examples": target_only,
        "interpretation": (
            "Diagnostic only. Target title/text was not rendered in the generation prompt. "
            "Do not filter or regenerate CoT from this target-based audit."
        ),
        "failures": dict(failures),
        "structurally_complete": not failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
