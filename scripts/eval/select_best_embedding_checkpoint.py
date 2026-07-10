#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


CHECKPOINT_RE = re.compile(r"checkpoint-(\d+)")


def checkpoint_step(path: Path, payload: dict) -> int:
    candidates = [path.name, str(payload.get("embedding_model") or "")]
    for candidate in candidates:
        match = CHECKPOINT_RE.search(candidate)
        if match:
            return int(match.group(1))
    raise ValueError(f"Cannot infer checkpoint step from {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Select an embedding checkpoint using validation metrics only.")
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--pattern", default="checkpoint-*_valid.json")
    parser.add_argument("--metric", default="NDCG@20")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    paths = sorted(eval_dir.glob(args.pattern))
    if not paths:
        raise FileNotFoundError(f"No files matched {eval_dir / args.pattern}")

    candidates = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        split = payload.get("split")
        if split != "valid":
            raise ValueError(f"Refusing non-valid result {path}: split={split!r}")
        metrics = payload.get("metrics") or {}
        if args.metric not in metrics:
            raise KeyError(f"Metric {args.metric!r} missing from {path}")
        candidates.append(
            {
                "checkpoint_step": checkpoint_step(path, payload),
                "checkpoint": payload.get("embedding_model"),
                "result_file": str(path.resolve()),
                "metric_value": float(metrics[args.metric]),
                "metrics": metrics,
                "mean_rank": payload.get("mean_rank"),
                "median_rank": payload.get("median_rank"),
            }
        )

    candidates.sort(key=lambda row: (-row["metric_value"], row["checkpoint_step"]))
    summary = {
        "selection_split": "valid",
        "selection_metric": args.metric,
        "best_checkpoint": candidates[0]["checkpoint"],
        "best_checkpoint_step": candidates[0]["checkpoint_step"],
        "best_metric_value": candidates[0]["metric_value"],
        "candidates": candidates,
    }
    rendered = json.dumps(summary, ensure_ascii=False, indent=2)
    print(rendered)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
