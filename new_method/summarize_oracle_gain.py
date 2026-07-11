#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any


BUCKET_ORDER = ("1-20", "21-100", "101-1000", "1000+")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
    if not rows:
        raise ValueError(f"No scored rows found in {path}")
    return rows


def mean(rows: list[dict[str, Any]], field: str) -> float:
    return fmean(float(row[field]) for row in rows)


def signs(rows: list[dict[str, Any]], field: str, eps: float = 1e-12) -> dict[str, Any]:
    values = [float(row[field]) for row in rows]
    positive = sum(value > eps for value in values)
    negative = sum(value < -eps for value in values)
    zero = len(values) - positive - negative
    return {
        "positive": positive,
        "negative": negative,
        "zero": zero,
        "positive_rate": positive / len(values),
        "negative_rate": negative / len(values),
        "zero_rate": zero / len(values),
        "mean": fmean(values),
    }


def budget_curve(rows: list[dict[str, Any]], rates: tuple[float, ...]) -> list[dict[str, Any]]:
    baseline_total = sum(float(row["baseline_ndcg"]) for row in rows)
    positive_gains = sorted(
        (float(row["delta_ndcg"]) for row in rows if float(row["delta_ndcg"]) > 0),
        reverse=True,
    )
    curve: list[dict[str, Any]] = []
    for rate in rates:
        budget = min(len(rows), math.floor(len(rows) * rate))
        selected = positive_gains[:budget]
        ndcg = (baseline_total + sum(selected)) / len(rows)
        curve.append(
            {
                "budget_rate": rate,
                "budget_samples": budget,
                "cot_selected": len(selected),
                "oracle_ndcg": ndcg,
                "absolute_gain": ndcg - baseline_total / len(rows),
            }
        )
    return curve


def summarize(rows: list[dict[str, Any]], metadata: dict[str, Any] | None) -> dict[str, Any]:
    n = len(rows)
    baseline_ndcg = mean(rows, "baseline_ndcg")
    cot_ndcg = mean(rows, "cot_ndcg")
    oracle_ndcg = fmean(
        max(float(row["baseline_ndcg"]), float(row["cot_ndcg"])) for row in rows
    )
    baseline_rank = mean(rows, "baseline_rank")
    cot_rank = mean(rows, "cot_rank")
    oracle_rank = fmean(
        min(int(row["baseline_rank"]), int(row["cot_rank"])) for row in rows
    )

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["baseline_rank_bucket"])].append(row)
    buckets: list[dict[str, Any]] = []
    for bucket in BUCKET_ORDER:
        subset = grouped.get(bucket, [])
        if not subset:
            continue
        log_rank_signs = signs(subset, "delta_log_rank")
        bucket_baseline = mean(subset, "baseline_ndcg")
        bucket_cot = mean(subset, "cot_ndcg")
        bucket_oracle = fmean(
            max(float(row["baseline_ndcg"]), float(row["cot_ndcg"])) for row in subset
        )
        buckets.append(
            {
                "bucket": bucket,
                "count": len(subset),
                "fraction": len(subset) / n,
                "baseline_ndcg": bucket_baseline,
                "cot_ndcg": bucket_cot,
                "oracle_ndcg": bucket_oracle,
                "oracle_absolute_gain": bucket_oracle - bucket_baseline,
                "mean_delta_log_rank": log_rank_signs["mean"],
                "positive_log_rank_rate": log_rank_signs["positive_rate"],
                "negative_log_rank_rate": log_rank_signs["negative_rate"],
            }
        )

    overlength_items = None
    if metadata is not None:
        overlength_items = metadata.get("overlength_item_count")
    return {
        "sample_count": n,
        "ndcg_k": int(rows[0]["ndcg_k"]),
        "baseline": {"mean_ndcg": baseline_ndcg, "mean_rank": baseline_rank},
        "cot": {"mean_ndcg": cot_ndcg, "mean_rank": cot_rank},
        "oracle": {
            "mean_ndcg": oracle_ndcg,
            "absolute_ndcg_gain": oracle_ndcg - baseline_ndcg,
            "relative_ndcg_gain": (
                (oracle_ndcg - baseline_ndcg) / baseline_ndcg if baseline_ndcg else None
            ),
            "mean_rank": oracle_rank,
            "mean_rank_reduction": baseline_rank - oracle_rank,
            "cot_route_count": sum(float(row["delta_ndcg"]) > 1e-12 for row in rows),
        },
        "delta_ndcg": signs(rows, "delta_ndcg"),
        "delta_log_rank": signs(rows, "delta_log_rank"),
        "delta_margin": signs(rows, "delta_margin"),
        "topk_transitions": {
            "entered": sum(bool(row["target_entered_topk"]) for row in rows),
            "left": sum(bool(row["target_left_topk"]) for row in rows),
        },
        "truncation": {
            "history_count": sum(int(row["history_truncated_tokens"]) > 0 for row in rows),
            "history_cot_count": sum(
                int(row["history_cot_truncated_tokens"]) > 0 for row in rows
            ),
            "overlength_item_count": overlength_items,
        },
        "rank_buckets": buckets,
        "oracle_budget_curve": budget_curve(rows, (0.01, 0.05, 0.10, 0.20, 0.50, 1.0)),
    }


def pct(value: float | None) -> str:
    return "NA" if value is None else f"{100 * value:.3f}%"


def markdown(summary: dict[str, Any], input_path: Path) -> str:
    baseline = summary["baseline"]
    cot = summary["cot"]
    oracle = summary["oracle"]
    ndcg_delta = summary["delta_ndcg"]
    log_delta = summary["delta_log_rank"]
    truncation = summary["truncation"]
    lines = [
        "# CoT 双路 Oracle 汇总",
        "",
        f"输入：`{input_path}`",
        "",
        "## 总体结果",
        "",
        f"样本数为 {summary['sample_count']}，评价指标为 NDCG@{summary['ndcg_k']}。",
        "",
        "| 路径 | 平均 NDCG | 平均 rank |",
        "|---|---:|---:|",
        f"| history | {baseline['mean_ndcg']:.6f} | {baseline['mean_rank']:.3f} |",
        f"| history+CoT | {cot['mean_ndcg']:.6f} | {cot['mean_rank']:.3f} |",
        f"| 逐样本 Oracle | {oracle['mean_ndcg']:.6f} | {oracle['mean_rank']:.3f} |",
        "",
        f"Oracle 的 NDCG 绝对增量为 {oracle['absolute_ndcg_gain']:.6f}，相对增量为 {pct(oracle['relative_ndcg_gain'])}。"
        f" Oracle 在 {oracle['cot_route_count']} 个样本上选择 CoT 路径。",
        "",
        "## 样本增益分布",
        "",
        f"按 NDCG 判断，CoT 改善 {ndcg_delta['positive']} 个样本，损害 {ndcg_delta['negative']} 个样本，"
        f"不变 {ndcg_delta['zero']} 个样本。按 log-rank 判断，改善比例为 {pct(log_delta['positive_rate'])}，"
        f"损害比例为 {pct(log_delta['negative_rate'])}。",
        "",
        f"进入 Top-{summary['ndcg_k']} 的样本有 {summary['topk_transitions']['entered']} 个，"
        f"离开 Top-{summary['ndcg_k']} 的样本有 {summary['topk_transitions']['left']} 个。",
        "",
        "## 初始排名分桶",
        "",
        "| history rank | 样本数 | CoT 改善 log-rank 比例 | CoT 损害 log-rank 比例 | history NDCG | CoT NDCG | Oracle NDCG |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for bucket in summary["rank_buckets"]:
        lines.append(
            f"| {bucket['bucket']} | {bucket['count']} | {pct(bucket['positive_log_rank_rate'])} | "
            f"{pct(bucket['negative_log_rank_rate'])} | {bucket['baseline_ndcg']:.6f} | "
            f"{bucket['cot_ndcg']:.6f} | {bucket['oracle_ndcg']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Oracle 触发预算",
            "",
            "该表按真实正 NDCG 增量从高到低分配 CoT 预算，表示路由器在对应预算下的理论上限。",
            "",
            "| CoT 预算 | 预算样本数 | 实际选择 CoT | Oracle NDCG | 绝对增量 |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for point in summary["oracle_budget_curve"]:
        lines.append(
            f"| {pct(point['budget_rate'])} | {point['budget_samples']} | {point['cot_selected']} | "
            f"{point['oracle_ndcg']:.6f} | {point['absolute_gain']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## 截断统计",
            "",
            f"history 超过 4096 token 的样本为 {truncation['history_count']} 个，history+CoT 超过 4096 token 的样本为 "
            f"{truncation['history_cot_count']} 个，候选 item 超长数量为 {truncation['overlength_item_count']}。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize paired retrieval gains and Oracle routing bounds.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--metadata")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    input_path = Path(args.input)
    rows = read_jsonl(input_path)
    metadata = None
    if args.metadata:
        metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
    summary = summarize(rows, metadata)

    json_path = Path(args.output_json)
    md_path = Path(args.output_md)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(markdown(summary, input_path), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
