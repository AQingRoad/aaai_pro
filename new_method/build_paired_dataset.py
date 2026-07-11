#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from new_method.core import (
    RANK_BUCKETS,
    as_int_set,
    classify_gain,
    cot_from_row,
    history_from_row,
    rank_bucket,
    read_jsonl,
    stable_row_key,
    write_jsonl,
)


def as_text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def item_text(row: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("title", "main_category", "store"):
        value = str(row.get(key) or "").strip()
        if value:
            parts.append(value)
    categories = " > ".join(as_text_list(row.get("categories")))
    if categories:
        parts.append(f"Categories: {categories}")
    features = "; ".join(as_text_list(row.get("features")))
    if features:
        parts.append(f"Features: {features}")
    description = " ".join(as_text_list(row.get("description")))
    if description:
        parts.append(f"Description: {description}")
    return " ".join(parts).strip()


def load_item_map(path: str) -> dict[int, str]:
    output: dict[int, str] = {}
    for row in read_jsonl(path):
        try:
            item_id = int(row["item_id"])
        except (KeyError, TypeError, ValueError):
            continue
        text = item_text(row)
        if text:
            output[item_id] = text
    if not output:
        raise ValueError(f"No usable item text found in {path}")
    return output


def candidate_identity(row: dict[str, Any]) -> str:
    return str(
        row.get("candidate_id")
        or row.get("candidate_index")
        or row.get("cot_source")
        or row.get("source_model")
        or ""
    )


def select_hard_negatives(
    rows: list[dict[str, Any]],
    *,
    item_map: dict[int, str],
    target_id: int,
    history_item_ids: set[int],
    limit: int,
) -> tuple[list[int], list[str]]:
    candidate_ids: list[int] = []
    for row in rows:
        for field in ("cot_hard_negative_item_ids", "baseline_hard_negative_item_ids"):
            for item_id in row.get(field) or []:
                try:
                    candidate_ids.append(int(item_id))
                except (TypeError, ValueError):
                    continue
    excluded = set(history_item_ids)
    excluded.update({0, target_id})
    selected_ids: list[int] = []
    seen: set[int] = set()
    for item_id in candidate_ids:
        if item_id in excluded or item_id in seen or item_id not in item_map:
            continue
        selected_ids.append(item_id)
        seen.add(item_id)
        if len(selected_ids) >= limit:
            break
    return selected_ids, [item_map[item_id] for item_id in selected_ids]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build history/good-CoT/bad-CoT paired retriever data from gain-scored candidates."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--item-info", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rejected-output", required=True)
    parser.add_argument("--audit-output", required=True)
    parser.add_argument("--expected-split", default="")
    parser.add_argument("--min-good-log-rank", type=float, default=math.log(1.5))
    parser.add_argument("--min-good-margin", type=float, default=0.0)
    parser.add_argument("--min-bad-log-rank", type=float, default=-math.log(1.5))
    parser.add_argument("--min-bad-margin", type=float, default=0.0)
    parser.add_argument("--max-unsupported-claims", type=int, default=0)
    parser.add_argument("--require-quality-fields", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--hard-negative-count", type=int, default=16)
    parser.add_argument("--include-history-only", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    item_map = load_item_map(args.item_info)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_rows = 0
    for row in read_jsonl(args.input):
        source_rows += 1
        if args.expected_split and str(row.get("split") or "") != args.expected_split:
            continue
        groups[stable_row_key(row)].append(row)

    output_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    candidate_labels = Counter()
    output_modes = Counter()
    bucket_counts: dict[str, Counter[str]] = {bucket: Counter() for bucket in RANK_BUCKETS}
    group_failures = Counter()
    candidates_missing_quality_fields = 0

    for example_id, rows in groups.items():
        histories = {history_from_row(row) for row in rows if history_from_row(row)}
        target_ids = {
            int(row["target_item_id"])
            for row in rows
            if row.get("target_item_id") is not None
        }
        baseline_ranks = {
            int(row["baseline_rank"])
            for row in rows
            if row.get("baseline_rank") is not None
        }
        history_id_sets = {
            tuple(sorted(as_int_set(row.get("history_item_ids") or row.get("history_item_id"))))
            for row in rows
        }
        scorer_checkpoints = {str(row.get("scorer_checkpoint") or "") for row in rows}
        query_instructions = {str(row.get("query_instruction") or "") for row in rows}
        query_modes = {str(row.get("query_mode") or "") for row in rows}
        if len(histories) != 1:
            group_failures["history_conflict_or_empty"] += 1
            continue
        if len(target_ids) != 1:
            group_failures["target_conflict_or_empty"] += 1
            continue
        if len(baseline_ranks) != 1:
            group_failures["baseline_rank_conflict_or_empty"] += 1
            continue
        if len(history_id_sets) != 1:
            group_failures["history_item_ids_conflict"] += 1
            continue
        if len(scorer_checkpoints) != 1:
            group_failures["scorer_checkpoint_conflict"] += 1
            continue
        if len(query_instructions) != 1:
            group_failures["query_instruction_conflict"] += 1
            continue
        if query_modes != {"history_plus_think_only"}:
            group_failures["query_mode_conflict"] += 1
            continue
        history = next(iter(histories))
        target_id = next(iter(target_ids))
        positive = item_map.get(target_id, "")
        if not positive:
            group_failures["target_missing_from_item_info"] += 1
            continue

        labeled: list[tuple[str, dict[str, Any]]] = []
        for row in rows:
            if not bool(row.get("quality_fields_present")):
                candidates_missing_quality_fields += 1
            if args.require_quality_fields and not bool(row.get("quality_fields_present")):
                label, failures = "rejected", ["missing_quality_fields"]
            else:
                label, failures = classify_gain(
                    row,
                    min_good_log_rank=args.min_good_log_rank,
                    min_good_margin=args.min_good_margin,
                    min_bad_log_rank=args.min_bad_log_rank,
                    min_bad_margin=args.min_bad_margin,
                    max_unsupported_claims=args.max_unsupported_claims,
                )
            candidate_labels[label] += 1
            enriched = {**row, "gain_label": label, "label_failures": failures}
            labeled.append((label, enriched))
            if label == "rejected":
                rejected_rows.append(enriched)

        good_rows = [row for label, row in labeled if label == "good"]
        bad_rows = [row for label, row in labeled if label == "bad"]
        good_rows.sort(
            key=lambda row: (
                float(row.get("delta_log_rank") or 0.0),
                float(row.get("delta_margin") or 0.0),
                -int(row.get("cot_rank") or 10**9),
            ),
            reverse=True,
        )
        bad_rows.sort(
            key=lambda row: (
                float(row.get("delta_log_rank") or 0.0),
                float(row.get("delta_margin") or 0.0),
                -int(row.get("cot_rank") or 0),
            )
        )
        best_good = good_rows[0] if good_rows else None
        worst_bad = bad_rows[0] if bad_rows else None

        if not best_good and not worst_bad and not args.include_history_only:
            group_failures["no_paired_candidate"] += 1
            continue

        good_cot = cot_from_row(best_good)[0] if best_good else ""
        bad_cot = cot_from_row(worst_bad)[0] if worst_bad else ""
        history_item_ids = set(next(iter(history_id_sets)))
        hard_ids, hard_texts = select_hard_negatives(
            [row for _, row in labeled],
            item_map=item_map,
            target_id=target_id,
            history_item_ids=history_item_ids,
            limit=args.hard_negative_count,
        )
        baseline_rank = next(iter(baseline_ranks))
        bucket = str(rows[0].get("baseline_rank_bucket") or rank_bucket(baseline_rank))
        mode = "paired" if best_good and worst_bad else (
            "history_good" if best_good else "history_bad" if worst_bad else "history_only"
        )
        output_modes[mode] += 1
        bucket_counts.setdefault(bucket, Counter())[mode] += 1

        base = rows[0]
        output_rows.append(
            {
                "example_id": example_id,
                "user_id": base.get("user_id"),
                "interaction_id": base.get("interaction_id"),
                "dataset": base.get("dataset"),
                "category": base.get("category"),
                "split": base.get("split"),
                "history": history,
                "positive": positive,
                "target_item_id": target_id,
                "target_item_title": base.get("target_item_title", ""),
                "history_item_ids": sorted(history_item_ids),
                "good_cot": good_cot,
                "bad_cot": bad_cot,
                "has_good_cot": bool(best_good),
                "has_bad_cot": bool(worst_bad),
                "training_mode": mode,
                "baseline_rank": baseline_rank,
                "baseline_rank_bucket": bucket,
                "good_candidate_id": candidate_identity(best_good or {}),
                "bad_candidate_id": candidate_identity(worst_bad or {}),
                "good_delta_log_rank": (
                    float(best_good["delta_log_rank"]) if best_good else None
                ),
                "good_delta_margin": (
                    float(best_good["delta_margin"]) if best_good else None
                ),
                "bad_delta_log_rank": (
                    float(worst_bad["delta_log_rank"]) if worst_bad else None
                ),
                "bad_delta_margin": (
                    float(worst_bad["delta_margin"]) if worst_bad else None
                ),
                "hard_negative_item_ids": hard_ids,
                "negatives": hard_texts,
                "query_mode": "history_plus_think_only",
                "scorer_checkpoint": base.get("scorer_checkpoint"),
                "query_instruction": base.get("query_instruction"),
                "max_length": base.get("max_length"),
            }
        )

    written = write_jsonl(args.output, output_rows)
    rejected_written = write_jsonl(args.rejected_output, rejected_rows)
    audit = {
        "input": str(Path(args.input).resolve()),
        "output": str(Path(args.output).resolve()),
        "source_rows": source_rows,
        "source_groups": len(groups),
        "written_rows": written,
        "rejected_candidates_written": rejected_written,
        "candidate_labels": dict(sorted(candidate_labels.items())),
        "training_modes": dict(sorted(output_modes.items())),
        "rank_bucket_training_modes": {
            bucket: dict(sorted(bucket_counts[bucket].items())) for bucket in RANK_BUCKETS
        },
        "group_failures": dict(sorted(group_failures.items())),
        "candidates_missing_quality_fields": candidates_missing_quality_fields,
        "thresholds": {
            "min_good_log_rank": args.min_good_log_rank,
            "min_good_margin": args.min_good_margin,
            "min_bad_log_rank": args.min_bad_log_rank,
            "min_bad_margin": args.min_bad_margin,
            "max_unsupported_claims": args.max_unsupported_claims,
            "require_quality_fields": args.require_quality_fields,
        },
        "hard_negative_count": args.hard_negative_count,
    }
    audit_path = Path(args.audit_output)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
