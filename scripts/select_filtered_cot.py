#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rubric_cot_pipeline.io import read_jsonl, write_jsonl


def selection_score(row: dict) -> float:
    return float(row.get("rubric_score_norm", 0.0)) * max(float(row.get("cot_gain", 0.0)), 0.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/cot_scored.jsonl")
    parser.add_argument("--output", default="outputs/filtered_high_quality_cot.jsonl")
    parser.add_argument("--rejected-output", default="outputs/rejected_cot.jsonl")
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--min-rubric", type=float, default=0.0)
    parser.add_argument("--min-gain", type=float, default=0.0)
    parser.add_argument("--fallback-when-empty", action="store_true")
    args = parser.parse_args()

    groups: dict[str, list[dict]] = defaultdict(list)
    for row in read_jsonl(args.input):
        row["selection_score"] = selection_score(row)
        group_id = str(row.get("example_id") or row["user_id"])
        groups[group_id].append(row)

    selected = []
    rejected = []
    for user_id, rows in groups.items():
        rows = sorted(
            rows,
            key=lambda r: (
                float(r.get("selection_score", 0.0)),
                float(r.get("rubric_score_norm", 0.0)),
                float(r.get("cot_gain", 0.0)),
            ),
            reverse=True,
        )
        passing = [
            r
            for r in rows
            if float(r.get("rubric_score_norm", 0.0)) >= args.min_rubric
            and float(r.get("cot_gain", 0.0)) >= args.min_gain
        ]
        if not passing and args.fallback_when_empty and rows:
            passing = rows[:1]
            passing[0]["fallback_selected"] = True
        keep_ids = {id(r) for r in passing[: args.top_k]}
        for rank, row in enumerate(rows, start=1):
            row["selection_rank"] = rank
            if id(row) in keep_ids:
                selected.append(row)
            else:
                rejected.append(row)

    selected_count = write_jsonl(args.output, selected)
    rejected_count = write_jsonl(args.rejected_output, rejected)
    print(f"selected={selected_count} rejected={rejected_count} users={len(groups)} output={args.output}")


if __name__ == "__main__":
    main()
