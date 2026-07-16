#!/usr/bin/env python3
"""在完整 item 集合上评测 manu_src 训练的 Qwen3 Embedding checkpoint。"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer


SCRIPTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS_DIR / "models"))
sys.path.insert(0, str(SCRIPTS_DIR / "pre_datas"))

from format_positive import format_positive  # noqa: E402
from train_embedding import QUERY_TRUNCATION, encode, format_query, read_jsonl  # noqa: E402


def set_seed(seed: int) -> None:
    """固定评测中使用的随机状态。"""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_test_rows(path: Path) -> list[dict]:
    """读取 test pair，并检查全量评测需要的字段。"""
    rows = read_jsonl(path)
    for line_number, row in enumerate(rows, 1):
        if row.get("split") != "test":
            raise ValueError(f"{path} 第 {line_number} 行 split 不是 test")
        for field in ("example_id", "query", "target_item_id", "history_item_ids"):
            if field not in row:
                raise ValueError(f"{path} 第 {line_number} 行缺少 {field}")
    return rows


def load_candidates(path: Path) -> tuple[list[int], list[str]]:
    """使用训练 positive 的相同 formatter 构造12000个真实候选物品。"""
    item_rows = sorted(read_jsonl(path), key=lambda row: int(row["item_id"]))
    item_ids = []
    item_texts = []
    for row in item_rows:
        item_id = int(row["item_id"])
        if item_id == 0:
            continue
        fallback_title = str(row.get("title") or "").strip() or f"item_{item_id}"
        # item_id=2134 的原始标题只有空格；覆盖为空白标题后再调用统一 formatter。
        if not str(row.get("title") or "").strip():
            row = {**row, "title": fallback_title}
        item_ids.append(item_id)
        item_texts.append(format_positive(row, fallback_title))
    if len(item_ids) != 12000 or len(set(item_ids)) != 12000:
        raise ValueError(f"候选集合应包含12000个唯一真实物品，当前为 {len(set(item_ids))}")
    return item_ids, item_texts


def token_length_audit(tokenizer, texts: list[str], max_length: int, name: str) -> dict:
    """记录原始 token 长度；候选物品超限时停止评测。"""
    lengths = []
    for start in range(0, len(texts), 256):
        encoded = tokenizer(texts[start : start + 256], truncation=False, padding=False)[
            "input_ids"
        ]
        lengths.extend(len(ids) for ids in encoded)
    audit = {
        "name": name,
        "count": len(texts),
        "max_tokens": max(lengths),
        "over_limit": sum(length > max_length for length in lengths),
        "max_length": max_length,
    }
    if name == "items" and audit["over_limit"]:
        raise ValueError(f"候选物品文本超过 max_length，禁止截断：{audit}")
    return audit


@torch.inference_mode()
def encode_batches(
    model,
    tokenizer,
    texts: list[str],
    *,
    batch_size: int,
    max_length: int,
    device: torch.device,
    is_query: bool,
) -> torch.Tensor:
    """分批编码并保留归一化后的 CPU embedding。"""
    outputs = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        outputs.append(
            encode(model, tokenizer, batch_texts, device, max_length, is_query).cpu()
        )
    return torch.cat(outputs, dim=0)


def metrics_from_ranks(ranks: list[int], ks: list[int]) -> dict[str, float]:
    """由唯一相关物品的 rank 计算 HR、NDCG 和 MRR。"""
    count = len(ranks)
    metrics = {
        "MRR": sum(1.0 / rank for rank in ranks) / count,
        "mean_rank": sum(ranks) / count,
        "median_rank": float(torch.tensor(ranks).median().item()),
    }
    for k in ks:
        metrics[f"HR@{k}"] = sum(rank <= k for rank in ranks) / count
        metrics[f"NDCG@{k}"] = sum(
            1.0 / math.log2(rank + 1) if rank <= k else 0.0 for rank in ranks
        ) / count
    return metrics


@torch.inference_mode()
def rank_queries(
    query_embeddings: torch.Tensor,
    rows: list[dict],
    item_embeddings: torch.Tensor,
    item_ids: list[int],
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[list[int], list[dict], dict]:
    """屏蔽 seen items 后计算每条 test query 的目标排名。"""
    item_index = {item_id: index for index, item_id in enumerate(item_ids)}
    item_embeddings = item_embeddings.to(device)
    ranks = []
    rank_rows = []
    masked_total = 0
    target_in_history = 0

    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        batch_queries = query_embeddings[start : start + batch_size].to(device)
        scores = batch_queries @ item_embeddings.T

        target_indices = []
        for offset, row in enumerate(batch_rows):
            target_id = int(row["target_item_id"])
            if target_id not in item_index:
                raise ValueError(f"target_item_id={target_id} 不在候选集合")
            target_indices.append(item_index[target_id])

            history_ids = {int(item_id) for item_id in row.get("history_item_ids", [])}
            if target_id in history_ids:
                target_in_history += 1
            # 即使目标物品曾出现，也保留本次监督目标，只屏蔽其它 seen items。
            history_ids.discard(target_id)
            masked = [item_index[item_id] for item_id in history_ids if item_id in item_index]
            if masked:
                scores[offset, masked] = -torch.inf
            masked_total += len(masked)

        target_indices_tensor = torch.tensor(target_indices, device=device)
        target_scores = scores[torch.arange(len(batch_rows), device=device), target_indices_tensor]
        batch_ranks = 1 + scores.gt(target_scores[:, None]).sum(dim=1)

        for row, rank_tensor in zip(batch_rows, batch_ranks):
            rank = int(rank_tensor.item())
            ranks.append(rank)
            rank_rows.append(
                {
                    "example_id": row["example_id"],
                    "target_item_id": int(row["target_item_id"]),
                    "target_item_title": row.get("target_item_title", ""),
                    "rank": rank,
                }
            )

    audit = {
        "masked_score_total": masked_total,
        "masked_score_mean": masked_total / len(rows),
        "target_in_history_count": target_in_history,
    }
    return ranks, rank_rows, audit


def main() -> None:
    parser = argparse.ArgumentParser(description="在12000个候选物品上评测 embedding checkpoint。")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--item-info", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ranks-output", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--item-batch-size", type=int, default=128)
    parser.add_argument("--query-batch-size", type=int, default=128)
    parser.add_argument("--score-batch-size", type=int, default=128)
    parser.add_argument("--ks", default="5,10,20")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    args = parser.parse_args()

    if args.seed != 42:
        parser.error("项目随机种子固定为 42")
    if not torch.cuda.is_available():
        parser.error("该评测脚本要求 CUDA GPU")
    set_seed(args.seed)
    device = torch.device("cuda:0")

    rows = load_test_rows(args.test_file)
    item_ids, item_texts = load_candidates(args.item_info)
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint,
        trust_remote_code=True,
        use_fast=True,
        padding_side="left",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    query_texts = [format_query(str(row["query"])) for row in rows]
    token_audit = [
        token_length_audit(tokenizer, item_texts, args.max_length, "items"),
        token_length_audit(tokenizer, query_texts, args.max_length, "queries"),
    ]
    model = AutoModel.from_pretrained(
        args.checkpoint,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
    ).to(device)
    model.eval()

    item_embeddings = encode_batches(
        model,
        tokenizer,
        item_texts,
        batch_size=args.item_batch_size,
        max_length=args.max_length,
        device=device,
        is_query=False,
    )
    query_embeddings = encode_batches(
        model,
        tokenizer,
        query_texts,
        batch_size=args.query_batch_size,
        max_length=args.max_length,
        device=device,
        is_query=True,
    )
    ranks, rank_rows, mask_audit = rank_queries(
        query_embeddings,
        rows,
        item_embeddings,
        item_ids,
        batch_size=args.score_batch_size,
        device=device,
    )

    ks = [int(value) for value in args.ks.split(",") if value.strip()]
    result = {
        "checkpoint": str(args.checkpoint),
        "test_file": str(args.test_file),
        "item_info": str(args.item_info),
        "evaluated": len(rows),
        "num_candidates": len(item_ids),
        "max_length": args.max_length,
        "query_truncation": QUERY_TRUNCATION,
        "item_text": "format_positive_desc256_details256",
        "mask_history_items": True,
        "seed": args.seed,
        "token_audit": token_audit,
        "metrics": metrics_from_ranks(ranks, ks),
        **mask_audit,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.ranks_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.ranks_output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rank_rows),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
