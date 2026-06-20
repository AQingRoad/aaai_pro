#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rubric_cot_pipeline.io import read_jsonl, write_jsonl
from rubric_cot_pipeline.prompts import ANSWER_TAG, COT_SYSTEM, REASONING_TAG, build_user_prompt
from rubric_cot_pipeline.rubric import extract_blocks


INNER_TAG_RE = re.compile(r"</?(?:analysis|recommendation|think|thinking|thoughts|answer)>", re.IGNORECASE)
DIAGNOSTIC_FIELDS = (
    "gain_mode",
    "embedder_mode",
    "baseline_sim",
    "cot_sim",
    "sim_gain",
    "baseline_rank",
    "cot_rank",
    "baseline_ndcg",
    "cot_ndcg",
    "ndcg_k",
    "masked_history_items",
    "masked_pad_item",
    "selection_rank",
    "fallback_selected",
)


def ensure_tagged_assistant(cot: str, fallback_answer: str = "") -> str:
    cot = (cot or "").strip()
    think, answer, has_tags = extract_blocks(cot)
    if has_tags:
        clean_think = INNER_TAG_RE.sub("", think).strip()
        clean_answer = INNER_TAG_RE.sub("", answer).strip()
        if clean_think and clean_answer:
            return f"<{REASONING_TAG}>\n{clean_think}\n</{REASONING_TAG}>\n<{ANSWER_TAG}>\n{clean_answer}\n</{ANSWER_TAG}>"
    answer = fallback_answer.strip() or "Use the grounded reasoning above as a concise user preference profile."
    clean_cot = INNER_TAG_RE.sub("", cot).strip()
    clean_answer = INNER_TAG_RE.sub("", answer).strip()
    return f"<{REASONING_TAG}>\n{clean_cot}\n</{REASONING_TAG}>\n<{ANSWER_TAG}>\n{clean_answer}\n</{ANSWER_TAG}>"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/filtered_high_quality_cot.jsonl")
    parser.add_argument("--output", default="outputs/filtered_high_quality_cot_sft.jsonl")
    parser.add_argument("--max-examples", type=int, default=0)
    args = parser.parse_args()

    rows = []
    for row in read_jsonl(args.input, limit=args.max_examples):
        _, answer, _ = extract_blocks(row.get("cot", ""))
        assistant = ensure_tagged_assistant(row.get("cot", ""), answer or row.get("source_answer", ""))
        diagnostics = {key: row.get(key) for key in DIAGNOSTIC_FIELDS if key in row}
        rows.append(
            {
                "example_id": row.get("example_id"),
                "user_id": row["user_id"],
                "interaction_id": row.get("interaction_id"),
                "candidate_id": row.get("candidate_id"),
                "dataset": row.get("dataset"),
                "category": row.get("category"),
                "target_item_id": row.get("target_item_id"),
                "history_item_ids": row.get("history_item_ids") or row.get("history_item_id") or [],
                "rubric_total": row.get("rubric_total"),
                "cot_gain": row.get("cot_gain"),
                "selection_score": row.get("selection_score"),
                **diagnostics,
                "messages": [
                    {"role": "system", "content": COT_SYSTEM},
                    {"role": "user", "content": build_user_prompt(row["user_history"], row.get("category", ""))},
                    {"role": "assistant", "content": assistant},
                ],
            }
        )
    count = write_jsonl(args.output, rows)
    print(f"Wrote {count} SFT rows to {args.output}")


if __name__ == "__main__":
    main()
