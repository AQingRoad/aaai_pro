#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def update_metrics(totals: dict[str, float], prefix: str, rank: int, ks: list[int]) -> None:
    for k in ks:
        totals[f"{prefix}_HR@{k}"] += 1.0 if rank <= k else 0.0
        totals[f"{prefix}_NDCG@{k}"] += 1.0 / math.log2(rank + 1) if rank <= k else 0.0


def row_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("category"),
        row.get("split"),
        row.get("global_index", row.get("index")),
        row.get("user_id"),
        row.get("target_item_id"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", nargs="+", required=True)
    parser.add_argument("--ks", default="5,10,20")
    parser.add_argument("--output", required=True)
    parser.add_argument("--combined-predictions-output", default="")
    parser.add_argument("--category", default="")
    parser.add_argument("--split", default="")
    parser.add_argument("--adapter", default="")
    parser.add_argument("--adapter-name", default="")
    parser.add_argument("--scorer", default="")
    parser.add_argument("--embedding-model", default="")
    args = parser.parse_args()

    ks = [int(x.strip()) for x in args.ks.split(",") if x.strip()]
    rows_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    loaded_files = []
    for raw_path in args.predictions:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(path)
        loaded_files.append(str(path))
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                rows_by_key.setdefault(row_key(row), row)

    rows = sorted(rows_by_key.values(), key=lambda x: (int(x.get("global_index", x.get("index", 0))), str(x.get("user_id", ""))))
    metric_keys = [f"{prefix}_{metric}@{k}" for prefix in ("baseline", "reasoner") for metric in ("HR", "NDCG") for k in ks]
    totals = {key: 0.0 for key in metric_keys}
    for row in rows:
        update_metrics(totals, "baseline", int(row["baseline_rank"]), ks)
        update_metrics(totals, "reasoner", int(row["reasoner_rank"]), ks)

    n = max(1, len(rows))
    metrics = {key: value / n for key, value in totals.items()}
    for k in ks:
        metrics[f"delta_NDCG@{k}"] = metrics[f"reasoner_NDCG@{k}"] - metrics[f"baseline_NDCG@{k}"]
        metrics[f"delta_HR@{k}"] = metrics[f"reasoner_HR@{k}"] - metrics[f"baseline_HR@{k}"]

    result = {
        "category": args.category or (rows[0].get("category") if rows else ""),
        "split": args.split or (rows[0].get("split") if rows else ""),
        "adapter": args.adapter,
        "adapter_name": args.adapter_name,
        "evaluated": len(rows),
        "num_prediction_files": len(loaded_files),
        "prediction_files": loaded_files,
        "metrics": metrics,
        "scorer": args.scorer,
        "embedding_model": args.embedding_model or None,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if args.combined_predictions_output:
        pred_path = Path(args.combined_predictions_output)
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        with pred_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
