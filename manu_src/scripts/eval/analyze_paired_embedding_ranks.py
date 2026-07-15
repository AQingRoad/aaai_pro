#!/usr/bin/env python3
"""汇总同一批样本在两种 query 输入下的检索 rank 与置信区间。"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path


BASE_MODE = "history_only"
RREC_MODE = "rrec_v1_3_full"
MODES = (BASE_MODE, RREC_MODE)
METRIC_NAMES = ("MRR", "HR@5", "NDCG@5", "HR@10", "NDCG@10", "HR@20", "NDCG@20")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def metrics(ranks: list[int]) -> dict[str, float]:
    result = {
        "count": len(ranks),
        "mean_rank": statistics.fmean(ranks),
        "median_rank": statistics.median(ranks),
        "MRR": statistics.fmean(1.0 / rank for rank in ranks),
    }
    for k in (5, 10, 20):
        result[f"HR@{k}"] = statistics.fmean(rank <= k for rank in ranks)
        result[f"NDCG@{k}"] = statistics.fmean(
            1.0 / math.log2(rank + 1) if rank <= k else 0.0 for rank in ranks
        )
    return result


def metric_deltas(rows: list[dict]) -> dict[str, float]:
    base = metrics([row[f"rank_{BASE_MODE}"] for row in rows])
    rrec = metrics([row[f"rank_{RREC_MODE}"] for row in rows])
    deltas = {name: rrec[name] - base[name] for name in METRIC_NAMES}
    deltas["mean_rank_gain"] = statistics.fmean(
        row[f"rank_{BASE_MODE}"] - row[f"rank_{RREC_MODE}"] for row in rows
    )
    return deltas


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    left = math.floor(position)
    right = math.ceil(position)
    if left == right:
        return ordered[left]
    return ordered[left] + (ordered[right] - ordered[left]) * (position - left)


def summarize_bootstrap(samples: dict[str, list[float]], seed: int, repetitions: int) -> dict:
    return {
        "seed": seed,
        "repetitions": repetitions,
        "confidence_intervals_95": {
            name: [percentile(values, 0.025), percentile(values, 0.975)]
            for name, values in samples.items()
        },
        "positive_probability": {
            name: statistics.fmean(value > 0 for value in values)
            for name, values in samples.items()
        },
    }


def sample_bootstrap(rows: list[dict], seed: int, repetitions: int) -> dict:
    """按交互样本重采样，保留两种输入的配对关系。"""
    rng = random.Random(seed)
    names = (*METRIC_NAMES, "mean_rank_gain")
    samples = {name: [] for name in names}
    for _ in range(repetitions):
        batch = [rows[rng.randrange(len(rows))] for _ in range(len(rows))]
        deltas = metric_deltas(batch)
        for name in names:
            samples[name].append(deltas[name])
    return summarize_bootstrap(samples, seed, repetitions)


def user_cluster_bootstrap(rows: list[dict], seed: int, repetitions: int) -> dict:
    """按 user_id 重采样，避免同一用户多条交互被当成完全独立样本。"""
    clusters: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        clusters[str(row["user_id"])].append(row)
    user_ids = sorted(clusters)
    rng = random.Random(seed)
    names = (*METRIC_NAMES, "mean_rank_gain")
    samples = {name: [] for name in names}
    for _ in range(repetitions):
        batch = []
        for _ in user_ids:
            batch.extend(clusters[user_ids[rng.randrange(len(user_ids))]])
        deltas = metric_deltas(batch)
        for name in names:
            samples[name].append(deltas[name])
    result = summarize_bootstrap(samples, seed, repetitions)
    result["cluster_count"] = len(user_ids)
    return result


def paired_summary(rows: list[dict]) -> dict:
    gains = [row[f"rank_{BASE_MODE}"] - row[f"rank_{RREC_MODE}"] for row in rows]
    summary = {
        "count": len(rows),
        "improved": sum(gain > 0 for gain in gains),
        "tied": sum(gain == 0 for gain in gains),
        "regressed": sum(gain < 0 for gain in gains),
        "win_rate": statistics.fmean(gain > 0 for gain in gains),
        "mean_rank_gain": statistics.fmean(gains),
        "median_rank_gain": statistics.median(gains),
    }
    for k in (5, 10, 20):
        summary[f"top{k}_transition"] = {
            "miss_to_hit": sum(
                row[f"rank_{BASE_MODE}"] > k and row[f"rank_{RREC_MODE}"] <= k for row in rows
            ),
            "hit_to_miss": sum(
                row[f"rank_{BASE_MODE}"] <= k and row[f"rank_{RREC_MODE}"] > k for row in rows
            ),
            "both_hit": sum(
                row[f"rank_{BASE_MODE}"] <= k and row[f"rank_{RREC_MODE}"] <= k for row in rows
            ),
            "both_miss": sum(
                row[f"rank_{BASE_MODE}"] > k and row[f"rank_{RREC_MODE}"] > k for row in rows
            ),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="分析 history-only 与完整 RRec 输出的配对 rank。")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--ranks", type=Path, required=True)
    parser.add_argument("--eval-summary", type=Path, required=True)
    parser.add_argument("--input-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    args = parser.parse_args()

    if args.seed != 42:
        parser.error("项目随机种子固定为 42")
    inputs = read_jsonl(args.input)
    ranks = read_jsonl(args.ranks)
    if len(inputs) != len(ranks):
        raise ValueError(f"评测输入 {len(inputs)} 行，rank {len(ranks)} 行")

    paired: dict[int, dict] = {}
    for source, result in zip(inputs, ranks):
        for field in ("example_id", "target_item_id"):
            if source[field] != result[field]:
                raise ValueError(f"{field} 顺序不一致：{source[field]!r} != {result[field]!r}")
        mode = str(source["codex_mode"])
        if mode not in MODES:
            raise ValueError(f"未知 codex_mode={mode}")
        sample_index = int(source["sample_index"])
        row = paired.setdefault(
            sample_index,
            {
                "sample_index": sample_index,
                "example_id": source["example_id"],
                "user_id": source.get("user_id"),
                "target_item_id": int(source["target_item_id"]),
                "target_item_title": source.get("target_item_title", ""),
            },
        )
        row[f"rank_{mode}"] = int(result["rank"])
        row[f"query_compressed_{mode}"] = bool(result.get("query_compressed", False))
        row[f"query_original_tokens_{mode}"] = int(result.get("query_original_tokens", 0))
        row[f"query_final_tokens_{mode}"] = int(result.get("query_final_tokens", 0))
        row[f"removed_history_item_numbers_{mode}"] = result.get("removed_history_item_numbers", [])

    rows = [paired[index] for index in sorted(paired)]
    for row in rows:
        missing = [mode for mode in MODES if f"rank_{mode}" not in row]
        if missing:
            raise ValueError(f"sample_index={row['sample_index']} 缺少 {missing}")
        gain = row[f"rank_{BASE_MODE}"] - row[f"rank_{RREC_MODE}"]
        row["rank_gain_history_minus_rrec"] = gain
        row["outcome"] = "improved" if gain > 0 else "regressed" if gain < 0 else "tied"

    by_mode = {mode: metrics([row[f"rank_{mode}"] for row in rows]) for mode in MODES}
    delta = {
        name: by_mode[RREC_MODE][name] - by_mode[BASE_MODE][name]
        for name in ("mean_rank", "median_rank", *METRIC_NAMES)
    }
    eval_summary = json.loads(args.eval_summary.read_text(encoding="utf-8"))
    input_audit = json.loads(args.input_audit.read_text(encoding="utf-8"))
    result = {
        "seed": args.seed,
        "sample_count": len(rows),
        "user_count": len({row["user_id"] for row in rows}),
        "candidate_count": int(eval_summary["num_candidates"]),
        "max_length": int(eval_summary["max_length"]),
        "metrics": by_mode,
        "rrec_minus_history": delta,
        "paired": paired_summary(rows),
        "sample_paired_bootstrap": sample_bootstrap(rows, args.seed, args.bootstrap_repetitions),
        "user_cluster_bootstrap": user_cluster_bootstrap(rows, args.seed, args.bootstrap_repetitions),
        "compression": {
            mode: {
                "count": sum(row[f"query_compressed_{mode}"] for row in rows),
                "max_original_tokens": max(row[f"query_original_tokens_{mode}"] for row in rows),
                "max_final_tokens": max(row[f"query_final_tokens_{mode}"] for row in rows),
            }
            for mode in MODES
        },
        "input_audit": input_audit,
        "eval_summary": eval_summary,
        "largest_improvements": sorted(rows, key=lambda row: row["rank_gain_history_minus_rrec"], reverse=True)[:20],
        "largest_regressions": sorted(rows, key=lambda row: row["rank_gain_history_minus_rrec"])[:20],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "rank_comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "per_sample_rank_changes.jsonl").open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    base = by_mode[BASE_MODE]
    rrec = by_mode[RREC_MODE]
    pair = result["paired"]
    cluster_ci = result["user_cluster_bootstrap"]["confidence_intervals_95"]
    top20 = pair["top20_transition"]
    report = f"""# RRec v1.3 完整输出的全测试集检索对比

## 结论

在 {len(rows)} 条严格 `title_store_categories` 测试样本上，拼接 RRec v1.3 完整输出后，NDCG@20 从 {base['NDCG@20']:.6f} 变为 {rrec['NDCG@20']:.6f}，差值 {delta['NDCG@20']:+.6f}；HR@20 从 {base['HR@20']:.6f} 变为 {rrec['HR@20']:.6f}，差值 {delta['HR@20']:+.6f}。平均 rank 从 {base['mean_rank']:.2f} 变为 {rrec['mean_rank']:.2f}，rank 越小越好。

完整输出改善 {pair['improved']} 条、持平 {pair['tied']} 条、退化 {pair['regressed']} 条，逐样本胜率 {pair['win_rate']:.2%}。平均 rank gain 为 {pair['mean_rank_gain']:+.2f}。按 {result['user_cluster_bootstrap']['cluster_count']} 个用户聚类重采样后，NDCG@20 差值的 95% 区间为 [{cluster_ci['NDCG@20'][0]:+.6f}, {cluster_ci['NDCG@20'][1]:+.6f}]，HR@20 差值区间为 [{cluster_ci['HR@20'][0]:+.6f}, {cluster_ci['HR@20'][1]:+.6f}]。

## 指标

| 输入 | MRR | Mean rank | Median rank | HR@10 | NDCG@10 | HR@20 | NDCG@20 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 仅 query | {base['MRR']:.6f} | {base['mean_rank']:.2f} | {base['median_rank']:.1f} | {base['HR@10']:.6f} | {base['NDCG@10']:.6f} | {base['HR@20']:.6f} | {base['NDCG@20']:.6f} |
| query + RRec v1.3 完整输出 | {rrec['MRR']:.6f} | {rrec['mean_rank']:.2f} | {rrec['median_rank']:.1f} | {rrec['HR@10']:.6f} | {rrec['NDCG@10']:.6f} | {rrec['HR@20']:.6f} | {rrec['NDCG@20']:.6f} |
| 差值 | {delta['MRR']:+.6f} | {delta['mean_rank']:+.2f} | {delta['median_rank']:+.1f} | {delta['HR@10']:+.6f} | {delta['NDCG@10']:+.6f} | {delta['HR@20']:+.6f} | {delta['NDCG@20']:+.6f} |

Top-20 状态迁移：未命中转为命中 {top20['miss_to_hit']} 条，命中转为未命中 {top20['hit_to_miss']} 条，两者均命中 {top20['both_hit']} 条，两者均未命中 {top20['both_miss']} 条。

## 长度处理与输入审计

仅 query 压缩 {result['compression'][BASE_MODE]['count']} 条，RRec 完整输出压缩 {result['compression'][RREC_MODE]['count']} 条。压缩脚本完整保留推理后缀，并从最早历史开始按物品粒度缩短输入。输入中 target 位于 history、positive 全文进入 API request、positive 全文进入评测 query、裸 ASIN 和 `[TRUNCATED]` 的计数依次为 {input_audit['target_in_history_count']}、{input_audit['positive_exact_in_api_request_count']}、{input_audit['positive_exact_in_full_query_count']}、{input_audit['raw_asin_count']}、{input_audit['truncated_marker_count']}。

目标标题字符串在生成推理中命中 {input_audit['target_title_string_in_reasoning_count']} 条。该计数包含通用短标题和模型先验生成的同名文本，需要结合审计明细判断；API 请求只读取 query，未传入 positive。

## 评测口径

- checkpoint：`title_store_categories_no_trunc_plus_cot/checkpoint-83`
- 候选物品：{eval_summary['num_candidates']} 个，屏蔽 history 中已见物品
- max_length：{eval_summary['max_length']}；query、item、score batch size 均为 128
- seed：{args.seed}
"""
    (args.output_dir / "rank_analysis.md").write_text(report, encoding="utf-8")
    print(json.dumps({"metrics": by_mode, "delta": delta, "paired": pair}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
