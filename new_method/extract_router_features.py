#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from new_method.core import as_int_set, file_sha256, history_from_row, read_jsonl
from new_method.score_paired_gain import check_lengths, load_items
from rubric_cot_pipeline.embeddings import (
    DEFAULT_RECOMMENDATION_QUERY_INSTRUCTION,
    Qwen3TextEmbedder,
    format_qwen3_query,
)


STAT_FEATURE_NAMES = [
    "log1p_history_item_count",
    "log1p_history_tokens",
    "top1_score",
    "top2_score",
    "top1_top2_gap",
    "top5_mean",
    "top5_std",
    "top20_mean",
    "top20_std",
    "top1_top20_gap",
    "top20_entropy",
    "top20_top1_probability",
]


def parse_dataset(value: list[str]) -> tuple[str, Path, Path]:
    if len(value) != 3:
        raise ValueError("--dataset requires NAME INPUT_GAIN_JSONL OUTPUT_PT")
    return value[0], Path(value[1]), Path(value[2])


def router_statistics(
    scores: torch.Tensor,
    *,
    history_item_count: int,
    history_tokens: int,
    temperature: float,
) -> torch.Tensor:
    if scores.numel() < 20:
        raise ValueError("At least 20 unmasked candidates are required")
    top20 = torch.topk(scores, k=20).values.float()
    top5 = top20[:5]
    probabilities = torch.softmax(top20 / temperature, dim=0)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
    values = [
        math.log1p(history_item_count),
        math.log1p(history_tokens),
        float(top20[0]),
        float(top20[1]),
        float(top20[0] - top20[1]),
        float(top5.mean()),
        float(top5.std(unbiased=False)),
        float(top20.mean()),
        float(top20.std(unbiased=False)),
        float(top20[0] - top20[-1]),
        float(entropy),
        float(probabilities[0]),
    ]
    return torch.tensor(values, dtype=torch.float32)


def extract_one(
    *,
    name: str,
    input_path: Path,
    output_path: Path,
    embedder: Qwen3TextEmbedder,
    item_embeddings: torch.Tensor,
    item_index: dict[int, int],
    query_max_length: int,
    allow_query_truncation: bool,
    batch_size: int,
    score_temperature: float,
    mask_history_items: bool,
    mask_pad_item: bool,
    device: torch.device,
    embedding_model: str,
    item_info: str,
) -> dict[str, Any]:
    rows = list(read_jsonl(input_path))
    if not rows:
        raise ValueError(f"No rows found in {input_path}")

    features: list[torch.Tensor] = []
    labels: list[int] = []
    baseline_ndcg: list[float] = []
    cot_ndcg: list[float] = []
    baseline_rank: list[int] = []
    cot_rank: list[int] = []
    example_ids: list[str] = []
    user_ids: list[str] = []
    query_truncated_count = 0

    embedder.max_length = query_max_length
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        histories = [history_from_row(row) for row in batch]
        if any(not history for history in histories):
            raise ValueError(f"Empty history in {input_path} batch starting at {start}")
        formatted = [format_qwen3_query(text, embedder.query_instruction) for text in histories]
        lengths = check_lengths(
            embedder.tokenizer,
            formatted,
            max_length=query_max_length,
            label=f"{name} history queries",
            allow_truncation=allow_query_truncation,
        )
        query_truncated_count += sum(length > query_max_length for length in lengths)
        embeddings = embedder.encode_queries(histories).to(device)
        score_matrix = embeddings @ item_embeddings.T

        for index, row in enumerate(batch):
            scores = score_matrix[index].clone()
            masked_ids: set[int] = set()
            if mask_history_items:
                masked_ids.update(
                    as_int_set(row.get("history_item_ids") or row.get("history_item_id"))
                )
            if mask_pad_item:
                masked_ids.add(0)
            masked_indices = [item_index[item_id] for item_id in masked_ids if item_id in item_index]
            if masked_indices:
                scores[masked_indices] = -torch.inf

            stats = router_statistics(
                scores,
                history_item_count=int(row.get("history_item_count") or len(masked_ids)),
                history_tokens=lengths[index],
                temperature=score_temperature,
            )
            features.append(torch.cat([embeddings[index].float().cpu(), stats]))
            delta_ndcg = float(row["delta_ndcg"])
            labels.append(int(delta_ndcg > 1e-12))
            baseline_ndcg.append(float(row["baseline_ndcg"]))
            cot_ndcg.append(float(row["cot_ndcg"]))
            baseline_rank.append(int(row["baseline_rank"]))
            cot_rank.append(int(row["cot_rank"]))
            example_ids.append(str(row["example_id"]))
            user_ids.append(str(row.get("user_id") or row["example_id"]))

        print(json.dumps({"dataset": name, "encoded": min(start + len(batch), len(rows))}), flush=True)

    feature_tensor = torch.stack(features)
    embedding_dim = feature_tensor.shape[1] - len(STAT_FEATURE_NAMES)
    payload = {
        "features": feature_tensor,
        "labels": torch.tensor(labels, dtype=torch.float32),
        "baseline_ndcg": torch.tensor(baseline_ndcg, dtype=torch.float32),
        "cot_ndcg": torch.tensor(cot_ndcg, dtype=torch.float32),
        "baseline_rank": torch.tensor(baseline_rank, dtype=torch.int64),
        "cot_rank": torch.tensor(cot_rank, dtype=torch.int64),
        "example_ids": example_ids,
        "user_ids": user_ids,
        "feature_names": [f"history_embedding_{index}" for index in range(embedding_dim)]
        + STAT_FEATURE_NAMES,
        "metadata": {
            "dataset": name,
            "source": str(input_path.resolve()),
            "source_sha256": file_sha256(input_path),
            "rows": len(rows),
            "positive_labels": sum(labels),
            "feature_dim": int(feature_tensor.shape[1]),
            "history_embedding_dim": embedding_dim,
            "query_truncated_count": query_truncated_count,
            "target_dependent_feature_fields": [],
            "label_definition": "delta_ndcg > 0",
            "embedding_model": str(Path(embedding_model).resolve()),
            "item_info": str(Path(item_info).resolve()),
            "item_info_sha256": file_sha256(item_info),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    meta_path.write_text(json.dumps(payload["metadata"], ensure_ascii=False, indent=2) + "\n")
    return payload["metadata"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract target-free history and retrieval-confidence features for the gain router."
    )
    parser.add_argument(
        "--dataset",
        nargs=3,
        action="append",
        metavar=("NAME", "INPUT_GAIN_JSONL", "OUTPUT_PT"),
        required=True,
    )
    parser.add_argument("--item-info", required=True)
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--query-max-length", type=int, default=4096)
    parser.add_argument("--item-max-length", type=int, default=4096)
    parser.add_argument("--allow-query-truncation", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--allow-item-truncation", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--item-batch-size", type=int, default=32)
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--query-instruction", default=DEFAULT_RECOMMENDATION_QUERY_INSTRUCTION)
    parser.add_argument("--score-temperature", type=float, default=0.05)
    parser.add_argument("--mask-history-items", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mask-pad-item", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if args.score_temperature <= 0:
        raise ValueError("--score-temperature must be positive")
    datasets = [parse_dataset(value) for value in args.dataset]
    item_ids, item_texts, item_index = load_items(args.item_info)
    del item_ids
    embedder = Qwen3TextEmbedder(
        args.embedding_model,
        max_length=args.item_max_length,
        batch_size=args.item_batch_size,
        torch_dtype=args.torch_dtype,
        device=args.device,
        query_instruction=args.query_instruction,
    )
    item_lengths = check_lengths(
        embedder.tokenizer,
        item_texts,
        max_length=args.item_max_length,
        label="item-info",
        allow_truncation=args.allow_item_truncation,
    )
    device = torch.device(args.device)
    item_embeddings = embedder.encode_documents(item_texts).to(device)
    embedder.batch_size = args.batch_size

    summaries = []
    for name, input_path, output_path in datasets:
        summaries.append(
            extract_one(
                name=name,
                input_path=input_path,
                output_path=output_path,
                embedder=embedder,
                item_embeddings=item_embeddings,
                item_index=item_index,
                query_max_length=args.query_max_length,
                allow_query_truncation=args.allow_query_truncation,
                batch_size=args.batch_size,
                score_temperature=args.score_temperature,
                mask_history_items=args.mask_history_items,
                mask_pad_item=args.mask_pad_item,
                device=device,
                embedding_model=args.embedding_model,
                item_info=args.item_info,
            )
        )
    print(
        json.dumps(
            {
                "datasets": summaries,
                "item_count": len(item_texts),
                "item_token_max": max(item_lengths),
                "overlength_item_count": sum(length > args.item_max_length for length in item_lengths),
                "parameters": vars(args),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
