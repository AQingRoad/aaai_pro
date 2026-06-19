#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rubric_cot_pipeline.io import ensure_parent, read_jsonl, write_jsonl
from rubric_cot_pipeline.prompts import ANSWER_TAG, REASONING_TAG


def task_key(example_id: str, candidate_index: Any) -> tuple[str, int] | None:
    if not example_id:
        return None
    try:
        return str(example_id), int(candidate_index)
    except (TypeError, ValueError):
        return None


def build_cot(candidate: dict[str, Any]) -> str:
    think = str(candidate.get("think") or "").strip()
    answer = str(candidate.get("answer") or "").strip()
    return f"<{REASONING_TAG}>\n{think}\n</{REASONING_TAG}>\n<{ANSWER_TAG}>\n{answer}\n</{ANSWER_TAG}>"


def normalize_score(score: Any) -> dict[str, Any] | None:
    if not isinstance(score, dict):
        return None
    out = dict(score)
    if "total" not in out:
        return None
    try:
        total = float(out["total"])
    except (TypeError, ValueError):
        return None
    if "score_norm" not in out:
        out["score_norm"] = total / 25.0
    return out


def load_scores(path: Path) -> tuple[dict[tuple[str, int], dict[str, Any]], int]:
    scores: dict[tuple[str, int], dict[str, Any]] = {}
    duplicates = 0
    for row in read_jsonl(path):
        key = task_key(str(row.get("example_id") or ""), row.get("candidate_index"))
        if key is None:
            continue
        score = normalize_score(row.get("rubric_score"))
        if score is None:
            continue
        if key in scores:
            duplicates += 1
        scores[key] = {**row, "rubric_score": score}
    return scores, duplicates


def merge_rows(candidate_lists_path: Path, rubric_path: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    scores, duplicate_scores = load_scores(rubric_path)
    rows: list[dict[str, Any]] = []
    stats = {
        "rubric_rows": len(scores),
        "duplicate_score_keys": duplicate_scores,
        "candidate_rows": 0,
        "candidate_items": 0,
        "matched_candidates": 0,
        "missing_score_candidates": 0,
        "score_without_candidate": 0,
    }
    seen_score_keys: set[tuple[str, int]] = set()

    for row in read_jsonl(candidate_lists_path):
        stats["candidate_rows"] += 1
        example_id = str(row.get("example_id") or row.get("user_id") or row.get("id") or "")
        candidates = row.get("candidates")
        if not isinstance(candidates, list):
            continue
        for candidate in candidates:
            stats["candidate_items"] += 1
            key = task_key(example_id, candidate.get("candidate_index"))
            if key is None:
                continue
            score_row = scores.get(key)
            if score_row is None:
                stats["missing_score_candidates"] += 1
                continue
            seen_score_keys.add(key)
            score = score_row["rubric_score"]
            candidate_id = candidate.get("candidate_id") or score_row.get("candidate_id") or f"{example_id}-{key[1]}"
            out = {
                **row,
                **candidate,
                "example_id": example_id,
                "candidate_id": candidate_id,
                "candidate_index": key[1],
                "cot": build_cot(candidate),
                "source_answer": candidate.get("answer", ""),
                "rubric": score,
                "rubric_total": score["total"],
                "rubric_score_norm": score["score_norm"],
                "judge_mode": f"api_{score_row.get('judge_provider', '')}".rstrip("_"),
                "judge_api_provider": score_row.get("judge_provider", ""),
                "judge_model": score_row.get("judge_model", ""),
                "judge_used_target": score_row.get("judge_used_target", False),
                "judge_meta": score_row.get("judge_meta", {}),
                "judge_raw": score_row.get("judge_raw", ""),
            }
            out.pop("candidates", None)
            out.pop("candidate_count", None)
            out.pop("list_generation_timing", None)
            rows.append(out)
            stats["matched_candidates"] += 1

    stats["score_without_candidate"] = len(set(scores) - seen_score_keys)
    return rows, stats


def write_scored_examples(rows: list[dict[str, Any]], output: Path) -> int:
    seen: set[str] = set()
    examples: list[dict[str, Any]] = []
    for row in rows:
        example_id = str(row.get("example_id") or "")
        if not example_id or example_id in seen:
            continue
        seen.add(example_id)
        example = dict(row)
        for key in [
            "candidate_id",
            "candidate_index",
            "temperature",
            "think",
            "answer",
            "cot",
            "source_answer",
            "rubric",
            "rubric_total",
            "rubric_score_norm",
            "judge_mode",
            "judge_api_provider",
            "judge_model",
            "judge_used_target",
            "judge_meta",
            "judge_raw",
            "generation_timing",
            "generation_api_meta",
        ]:
            example.pop(key, None)
        examples.append(example)
    return write_jsonl(output, examples)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge candidate-list JSONL and candidate-level rubric scores into cot_judged JSONL."
    )
    parser.add_argument("--candidate-lists", required=True)
    parser.add_argument("--rubric-scores", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scored-examples-output", default="")
    args = parser.parse_args()

    rows, stats = merge_rows(Path(args.candidate_lists), Path(args.rubric_scores))
    count = write_jsonl(args.output, rows)
    example_count = 0
    if args.scored_examples_output:
        example_count = write_scored_examples(rows, ensure_parent(args.scored_examples_output))
    print(
        json.dumps(
            {
                **stats,
                "output_rows": count,
                "scored_examples": example_count,
                "output": args.output,
                "scored_examples_output": args.scored_examples_output,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
