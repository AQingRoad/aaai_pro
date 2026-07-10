#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.IGNORECASE | re.DOTALL)
ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
IM_END_RE = re.compile(r"(?:<\|im_end\|>\s*)+$")


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield line_number, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc


def stable_bucket(value: str, buckets: int) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % buckets


def normalize_completion(text: Any) -> str:
    return IM_END_RE.sub("", str(text or "").strip()).strip()


def extract_blocks(text: str) -> tuple[str, str]:
    think_match = THINK_RE.search(text)
    answer_match = ANSWER_RE.search(text)
    return (
        think_match.group(1).strip() if think_match else "",
        answer_match.group(1).strip() if answer_match else "",
    )


def first_sentence(text: str) -> str:
    pieces = re.split(r"(?<=[.!?])\s+", text.strip(), maxsplit=1)
    return pieces[0].strip() if pieces else ""


def build_corruption(example_id: str, source_cot: str) -> tuple[str, str]:
    think, answer = extract_blocks(source_cot)
    mode = stable_bucket(example_id + ":corruption", 4)
    if mode == 0:
        return (
            "generic_profile",
            "<think>\nThe history shows broad interest in music and physical media.\n</think>\n"
            "<answer>\nA broadly appealing music release in a familiar physical format.\n</answer>",
        )
    if mode == 1:
        short_think = first_sentence(think) or "The history contains a music interaction."
        return (
            "truncated_reasoning",
            f"<think>\n{short_think}\n</think>\n<answer>\nA related music item.\n</answer>",
        )
    if mode == 2:
        short_answer = answer or "A music release."
        return (
            "missing_evidence",
            f"<think>\nThe available evidence is not analyzed.\n</think>\n<answer>\n{short_answer}\n</answer>",
        )
    return (
        "unsupported_keyword_stack",
        "<think>\nThe next item should combine award-winning limited editions, rare imports, deluxe remasters, "
        "collector box sets, and popular chart releases regardless of the observed history.\n</think>\n"
        "<answer>\nA rare award-winning deluxe collector edition from a popular creator.\n</answer>",
    )


def source_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for _, row in read_jsonl(path):
        example_id = str(row.get("example_id") or "")
        if example_id:
            rows[example_id] = row
    if not rows:
        raise ValueError(f"No rows with example_id in {path}")
    return rows


def policy_candidate(
    completion_row: dict[str, Any],
    item: dict[str, Any],
    item_index: int,
    run_id: str,
) -> dict[str, Any]:
    completions = completion_row.get("completion") or []
    advantages = completion_row.get("advantages") or []
    logged_rewards = completion_row.get("ReferenceSoftNdcgReward") or []
    text = normalize_completion(completions[item_index] if item_index < len(completions) else "")
    think, answer = extract_blocks(text)
    step_values = completion_row.get("step") or []
    step = str(step_values[item_index] if item_index < len(step_values) else "")
    return {
        "candidate_id": f"policy:{run_id}:{step}:{item_index}",
        "source": "policy_rollout",
        "text": text,
        "think": think,
        "answer": answer,
        "format_valid": bool(think and answer),
        "new_rank": item.get("new_rank"),
        "reference_rank": item.get("reference_rank"),
        "q_new": item.get("q_new"),
        "q_ref": item.get("q_ref"),
        "logged_reward": logged_rewards[item_index] if item_index < len(logged_rewards) else item.get("reward"),
        "advantage": advantages[item_index] if item_index < len(advantages) else None,
    }


def external_candidate(row: dict[str, Any]) -> dict[str, Any]:
    text = normalize_completion(row.get("cot"))
    think, answer = extract_blocks(text)
    rubric = row.get("rubric") if isinstance(row.get("rubric"), dict) else None
    return {
        "candidate_id": f"external_glm:{row.get('candidate_id') or row.get('example_id')}",
        "source": "external_glm",
        "text": text,
        "think": think,
        "answer": answer,
        "format_valid": bool(think and answer),
        "new_rank": row.get("cot_rank"),
        "reference_rank": row.get("baseline_rank"),
        "q_new": None,
        "q_ref": None,
        "logged_reward": None,
        "advantage": None,
        "source_rubric": rubric if not row.get("judge_used_target") else None,
        "source_rubric_score_norm": row.get("rubric_score_norm") if not row.get("judge_used_target") else None,
    }


def corruption_candidate(example_id: str, source_cot: str) -> dict[str, Any]:
    corruption_type, text = build_corruption(example_id, source_cot)
    think, answer = extract_blocks(text)
    return {
        "candidate_id": f"corruption:{example_id}:{corruption_type}",
        "source": "synthetic_corruption",
        "corruption_type": corruption_type,
        "text": text,
        "think": think,
        "answer": answer,
        "format_valid": bool(think and answer),
        "new_rank": None,
        "reference_rank": None,
        "q_new": None,
        "q_ref": None,
        "logged_reward": None,
        "advantage": None,
    }


def deduplicate_candidates(candidates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    seen: set[str] = set()
    output = []
    removed = 0
    for candidate in candidates:
        key = re.sub(r"\s+", " ", str(candidate.get("text") or "")).strip().lower()
        if not key or key in seen:
            removed += 1
            continue
        seen.add(key)
        output.append(candidate)
    return output, removed


def main() -> None:
    parser = argparse.ArgumentParser(description="Build same-history CoT preference groups from GRPO rollouts.")
    parser.add_argument("--completions", required=True)
    parser.add_argument("--components", required=True)
    parser.add_argument("--source-scored", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", default="")
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--max-groups", type=int, default=0)
    parser.add_argument("--min-candidates", type=int, default=4)
    args = parser.parse_args()

    if args.num_folds < 2:
        raise ValueError("--num-folds must be at least 2")
    completions_path = Path(args.completions)
    components_path = Path(args.components)
    source_path = Path(args.source_scored)
    for path in (completions_path, components_path, source_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    sources = source_rows(source_path)
    run_id = completions_path.parent.name
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats = Counter()
    folds = Counter()
    candidate_sources = Counter()
    corruption_types = Counter()
    seen_examples: set[str] = set()

    with completions_path.open("r", encoding="utf-8") as completion_handle, components_path.open(
        "r", encoding="utf-8"
    ) as component_handle, output_path.open("w", encoding="utf-8") as output_handle:
        paired_lines = itertools.zip_longest(completion_handle, component_handle)
        for line_index, pair in enumerate(paired_lines, start=1):
            completion_line, component_line = pair
            if completion_line is None or component_line is None:
                raise ValueError("Completion and component JSONL files have different line counts")
            if not completion_line.strip() or not component_line.strip():
                raise ValueError(f"Blank aligned row at line {line_index}")
            completion_row = json.loads(completion_line)
            component_row = json.loads(component_line)
            items = component_row.get("items") or []
            completions = completion_row.get("completion") or []
            if len(items) != len(completions):
                raise ValueError(
                    f"Width mismatch at line {line_index}: components={len(items)} completions={len(completions)}"
                )

            grouped: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
            for item_index, item in enumerate(items):
                grouped[str(item.get("example_id") or "")].append((item_index, item))

            for example_id, group_items in grouped.items():
                stats["input_groups"] += 1
                if not example_id or example_id in seen_examples:
                    stats["skipped_missing_or_duplicate_example"] += 1
                    continue
                source = sources.get(example_id)
                if source is None:
                    stats["skipped_missing_source"] += 1
                    continue
                candidates = [
                    policy_candidate(completion_row, item, item_index, run_id)
                    for item_index, item in group_items
                ]
                external = external_candidate(source)
                if external["text"]:
                    candidates.append(external)
                    candidates.append(corruption_candidate(example_id, external["text"]))
                candidates, removed = deduplicate_candidates(candidates)
                stats["deduplicated_candidates"] += removed
                if len(candidates) < args.min_candidates:
                    stats["skipped_too_few_candidates"] += 1
                    continue

                fold = stable_bucket(example_id, args.num_folds)
                row = {
                    "example_id": example_id,
                    "fold": fold,
                    "num_folds": args.num_folds,
                    "split": source.get("split"),
                    "category": source.get("category"),
                    "user_history": source.get("user_history"),
                    "history_item_ids": source.get("history_item_ids"),
                    "history_item_count": source.get("history_item_count"),
                    "target_item_id": source.get("target_item_id"),
                    "baseline_rank": source.get("baseline_rank"),
                    "external_cot_rank": source.get("cot_rank"),
                    "rollout_step": completion_row.get("step", [None])[0],
                    "reward_call_index": component_row.get("call_index"),
                    "candidates": candidates,
                }
                output_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                seen_examples.add(example_id)
                stats["written_groups"] += 1
                stats["written_candidates"] += len(candidates)
                folds[fold] += 1
                candidate_sources.update(candidate["source"] for candidate in candidates)
                corruption_types.update(
                    candidate.get("corruption_type")
                    for candidate in candidates
                    if candidate.get("corruption_type")
                )
                if args.max_groups and stats["written_groups"] >= args.max_groups:
                    break
            if args.max_groups and stats["written_groups"] >= args.max_groups:
                break

    summary = {
        "completions": str(completions_path.resolve()),
        "components": str(components_path.resolve()),
        "source_scored": str(source_path.resolve()),
        "output": str(output_path.resolve()),
        "stats": dict(stats),
        "fold_groups": {str(key): folds[key] for key in sorted(folds)},
        "candidate_sources": dict(candidate_sources),
        "corruption_types": dict(corruption_types),
    }
    rendered = json.dumps(summary, ensure_ascii=False, indent=2)
    print(rendered)
    summary_path = Path(args.summary_output) if args.summary_output else output_path.with_suffix(".summary.json")
    summary_path.write_text(rendered + "\n", encoding="utf-8")
    if stats["written_groups"] == 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
