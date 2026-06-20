#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rubric_cot_pipeline.io import read_jsonl, write_jsonl


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(out) or math.isinf(out):
        return default
    return out


def default_selection_score(row: dict[str, Any]) -> float:
    rubric = as_float(row.get("rubric_score_norm"))
    gain = max(as_float(row.get("cot_gain")), 0.0)
    return rubric * gain


def score_row(row: dict[str, Any], score_field: str) -> float:
    if score_field == "selection_score":
        value = row.get("selection_score")
        if value is None:
            return default_selection_score(row)
    else:
        value = row.get(score_field)
    return as_float(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Select global top-percent CoT rows for SFT.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rejected-output", default="")
    parser.add_argument("--summary-output", default="")
    parser.add_argument("--top-percent", type=float, default=0.2)
    parser.add_argument("--score-field", default="selection_score")
    parser.add_argument("--min-rubric", type=float, default=0.0)
    parser.add_argument("--min-gain", type=float, default=0.0)
    args = parser.parse_args()

    if not 0 < args.top_percent <= 1:
        raise ValueError("--top-percent must be in (0, 1]")

    rows = []
    rejected_by_filter = []
    for row in read_jsonl(args.input):
        row = dict(row)
        row["selection_score"] = score_row(row, args.score_field)
        rubric = as_float(row.get("rubric_score_norm"))
        gain = as_float(row.get("cot_gain"))
        if rubric >= args.min_rubric and gain >= args.min_gain:
            rows.append(row)
        else:
            row["rejected_reason"] = "below_min_rubric_or_gain"
            rejected_by_filter.append(row)

    rows = sorted(
        rows,
        key=lambda row: (
            as_float(row.get("selection_score")),
            as_float(row.get("rubric_score_norm")),
            as_float(row.get("cot_gain")),
            as_float(row.get("cot_ndcg")),
        ),
        reverse=True,
    )
    keep_n = math.ceil(len(rows) * args.top_percent)
    selected = []
    rejected = rejected_by_filter[:]
    for rank, row in enumerate(rows, start=1):
        row["global_selection_rank"] = rank
        row["global_top_percent"] = args.top_percent
        if rank <= keep_n:
            selected.append(row)
        else:
            row["rejected_reason"] = "below_top_percent"
            rejected.append(row)

    selected_count = write_jsonl(args.output, selected)
    rejected_count = 0
    if args.rejected_output:
        rejected_count = write_jsonl(args.rejected_output, rejected)

    summary = {
        "input": args.input,
        "output": args.output,
        "rejected_output": args.rejected_output,
        "top_percent": args.top_percent,
        "score_field": args.score_field,
        "min_rubric": args.min_rubric,
        "min_gain": args.min_gain,
        "eligible_rows": len(rows),
        "selected_rows": selected_count,
        "rejected_rows": rejected_count,
        "cutoff_score": selected[-1]["selection_score"] if selected else None,
    }
    if args.summary_output:
        Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_output).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
