#!/usr/bin/env python3
"""Precompute fixed reference-CoT rank and NDCG for GRPO reward metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from manu_src.scripts.train.cot_rubric_ndcg1000_gain_reward import (  # noqa: E402
    CotRetrievalRewardState,
)


ENV_PREFIX = "COT_RUBRIC_NDCG_GAIN"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=8578)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--ndcg-k", type=int, default=1000)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path} 第 {line_number} 行不是合法 JSON") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path} 第 {line_number} 行不是 JSON object")
            rows.append(row)
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.seed != 42:
        raise ValueError("项目随机种子必须为 42")
    if args.batch_size <= 0 or args.ndcg_k <= 0 or args.temperature <= 0:
        raise ValueError("batch size、NDCG K 和 temperature 必须为正数")
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if args.input.resolve() == args.output.resolve():
        raise ValueError("输出文件不能覆盖输入文件")

    rows = read_jsonl(args.input)
    if len(rows) != args.expected_rows:
        raise ValueError(f"输入应为 {args.expected_rows} 条，当前为 {len(rows)}")
    for line_number, row in enumerate(rows, 1):
        if not str(row.get("user_history") or "").strip():
            raise ValueError(f"第 {line_number} 行缺少 user_history")
        if not str(row.get("reference_cot") or "").strip():
            raise ValueError(f"第 {line_number} 行缺少 reference_cot")
        if row.get("target_item_id") is None:
            raise ValueError(f"第 {line_number} 行缺少 target_item_id")
        if not isinstance(row.get("history_item_ids"), list):
            raise ValueError(f"第 {line_number} 行缺少 history_item_ids")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial_path = args.output.with_suffix(args.output.suffix + ".partial")
    progress_path = args.output.with_suffix(".progress.json")
    completed_rows: list[dict[str, Any]] = []
    if partial_path.is_file():
        completed_rows = read_jsonl(partial_path)
        if len(completed_rows) > len(rows):
            raise RuntimeError("partial 行数超过输入行数")
        for index, completed in enumerate(completed_rows):
            if completed.get("example_id") != rows[index].get("example_id"):
                raise RuntimeError("partial 与输入的 example_id 前缀不一致")
            if "reference_ndcg" not in completed or "reference_rank" not in completed:
                raise RuntimeError("partial 缺少 reference_ndcg/reference_rank")

    started = time.time()
    state = CotRetrievalRewardState(env_prefix=ENV_PREFIX)
    mode = "a" if completed_rows else "w"
    with partial_path.open(mode, encoding="utf-8") as output_handle:
        for start in range(len(completed_rows), len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            queries = [
                f"{str(row['user_history']).strip()}\n\n"
                f"Recommendation reasoning:\n{str(row['reference_cot']).strip()}"
                for row in batch
            ]
            target_ids = [int(row["target_item_id"]) for row in batch]
            history_sets = [
                {int(item_id) for item_id in row["history_item_ids"]}
                for row in batch
            ]
            embeddings = state.encode_queries(queries)
            _, _, ndcgs, ranks, masked_counts = state.score_queries(
                embeddings,
                target_ids,
                history_sets,
                temperature=args.temperature,
                ndcg_k=args.ndcg_k,
            )
            for row, ndcg, rank, masked_count in zip(
                batch, ndcgs, ranks, masked_counts
            ):
                enriched = dict(row)
                enriched["reference_rank"] = int(rank)
                enriched["reference_ndcg"] = float(ndcg)
                enriched["reference_ndcg_k"] = args.ndcg_k
                enriched["reference_seen_items_masked"] = int(masked_count)
                output_handle.write(json.dumps(enriched, ensure_ascii=False) + "\n")
            output_handle.flush()
            completed = min(start + len(batch), len(rows))
            elapsed = max(time.time() - started, 1e-9)
            atomic_json(
                progress_path,
                {
                    "input": str(args.input),
                    "output": str(args.output),
                    "completed": completed,
                    "total": len(rows),
                    "remaining": len(rows) - completed,
                    "rows_per_minute_since_resume": (completed - len(completed_rows))
                    / elapsed
                    * 60.0,
                    "ndcg_k": args.ndcg_k,
                    "temperature": args.temperature,
                    "seed": args.seed,
                },
            )
            print(
                f"reference NDCG: {completed}/{len(rows)} "
                f"({(completed - len(completed_rows)) / elapsed * 60.0:.1f} rows/min)",
                flush=True,
            )

    partial_path.replace(args.output)
    output_rows = read_jsonl(args.output)
    if len(output_rows) != len(rows):
        raise RuntimeError("reference NDCG 输出行数不完整")
    rank_values = [int(row["reference_rank"]) for row in output_rows]
    ndcg_values = [float(row["reference_ndcg"]) for row in output_rows]
    manifest = {
        "input": str(args.input),
        "input_sha256": sha256(args.input),
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "rows": len(output_rows),
        "embedding_model": os.getenv(f"{ENV_PREFIX}_EMBEDDING_MODEL", ""),
        "item_info": os.getenv(f"{ENV_PREFIX}_ITEM_INFO", ""),
        "candidate_items": len(state.item_ids),
        "query_max_length": state.max_length,
        "query_batch_size": args.batch_size,
        "temperature": args.temperature,
        "ndcg_k": args.ndcg_k,
        "seen_item_mask": True,
        "seed": args.seed,
        "reference_rank_min": min(rank_values),
        "reference_rank_max": max(rank_values),
        "reference_rank_mean": sum(rank_values) / len(rank_values),
        "reference_ndcg_mean": sum(ndcg_values) / len(ndcg_values),
        "reference_ndcg_nonzero": sum(value > 0.0 for value in ndcg_values),
    }
    atomic_json(args.output.with_suffix(".manifest.json"), manifest)
    atomic_json(progress_path, {**manifest, "completed": len(rows), "remaining": 0})
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
