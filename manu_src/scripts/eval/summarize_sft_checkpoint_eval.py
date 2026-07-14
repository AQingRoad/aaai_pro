#!/usr/bin/env python3
"""汇总三个 SFT checkpoint 的全量检索指标和相对参考结果。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_ranks(path: Path) -> list[int]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line)["rank"] for line in file if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--checkpoints", default="134,268,402")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reference = read_json(args.reference)
    reference_metrics = dict(reference.get("metrics") or {})
    results = []
    for step in [value.strip() for value in args.checkpoints.split(",") if value.strip()]:
        run_dir = args.output_dir / f"checkpoint-{step}"
        metric_path = run_dir / "embedding_eval.json"
        ranks_path = run_dir / "ranks.jsonl"
        metric = read_json(metric_path)
        ranks = read_ranks(ranks_path)
        if len(ranks) != int(metric["evaluated"]):
            raise ValueError(f"checkpoint-{step} 的 rank 行数与 evaluated 不一致")
        metrics = dict(metric["metrics"])
        metrics["MRR"] = sum(1.0 / rank for rank in ranks) / len(ranks)
        comparison = {
            name: metrics[name] - reference_metrics[name]
            for name in reference_metrics
            if name in metrics
        }
        results.append(
            {
                "checkpoint": f"checkpoint-{step}",
                "epoch": {"134": 1, "268": 2, "402": 3}.get(step),
                "evaluated": metric["evaluated"],
                "metrics": metrics,
                "mean_rank": metric["mean_rank"],
                "median_rank": metric["median_rank"],
                "delta_vs_glm_teacher_cot": comparison,
                "generation_audit": read_json(run_dir / "generated_cot.audit.json"),
            }
        )

    best = {
        metric: max(results, key=lambda row: row["metrics"][metric])["checkpoint"]
        for metric in ("NDCG@20", "HR@20", "MRR")
    }
    summary = {
        "reference_glm_teacher_cot": {
            "path": str(args.reference),
            "metrics": reference_metrics,
            "mean_rank": reference.get("mean_rank"),
            "median_rank": reference.get("median_rank"),
        },
        "results": results,
        "best_checkpoint": best,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
