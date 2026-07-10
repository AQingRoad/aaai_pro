#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

from rubric_cot_pipeline.embeddings import (
    DEFAULT_RECOMMENDATION_QUERY_INSTRUCTION,
    append_recommendation_reasoning,
    format_qwen3_query,
    last_token_pool,
    resolve_torch_dtype,
)
from rubric_cot_pipeline.preference_scorer_v2 import (
    PairwisePreferenceHead,
    PreferenceScorerConfig,
    build_preference_features,
)


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield line_number, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc


def candidate_reasoning(candidate: dict[str, Any]) -> str:
    think = str(candidate.get("think") or "").strip()
    if think:
        return f"<think>\n{think}\n</think>"
    return "[INVALID_OR_MISSING_THINK]\n" + str(candidate.get("text") or "").strip()


def load_rows(groups_path: Path, judgments_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    groups = {str(row.get("example_id")): row for _, row in read_jsonl(groups_path) if row.get("example_id")}
    judgments = {
        str(row.get("example_id")): row
        for _, row in read_jsonl(judgments_path)
        if row.get("example_id") and row.get("ranking") and not row.get("judge_used_target")
    }
    rows = []
    missing_groups = 0
    missing_candidates = 0
    for example_id, judgment in judgments.items():
        group = groups.get(example_id)
        if group is None:
            missing_groups += 1
            continue
        candidate_map = {str(candidate.get("candidate_id")): candidate for candidate in group.get("candidates") or []}
        ranking = [str(candidate_id) for candidate_id in judgment.get("ranking") or []]
        if not ranking or any(candidate_id not in candidate_map for candidate_id in ranking):
            missing_candidates += 1
            continue
        score_map = {
            str(item.get("candidate_id")): item
            for item in judgment.get("candidate_scores") or []
            if item.get("candidate_id")
        }
        rows.append(
            {
                "example_id": example_id,
                "fold": int(group.get("fold")),
                "history": str(group.get("user_history") or ""),
                "ranking": ranking,
                "candidates": candidate_map,
                "score_map": score_map,
            }
        )
    if not rows:
        raise ValueError("No joined preference groups")
    return rows, {
        "group_rows": len(groups),
        "judgment_rows": len(judgments),
        "joined_rows": len(rows),
        "missing_groups": missing_groups,
        "missing_candidates": missing_candidates,
    }


def flatten_candidates(
    rows: list[dict[str, Any]],
) -> tuple[list[str], list[str], list[str], list[int], list[float], dict[str, list[int]]]:
    histories = []
    joint_texts = []
    candidate_ids = []
    folds = []
    utilities = []
    group_indices: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        for candidate_id in row["ranking"]:
            candidate = row["candidates"][candidate_id]
            reasoning = candidate_reasoning(candidate)
            histories.append(str(row["history"]))
            joint_texts.append(append_recommendation_reasoning(str(row["history"]), reasoning))
            candidate_ids.append(candidate_id)
            folds.append(int(row["fold"]))
            score = row["score_map"].get(candidate_id) or {}
            utilities.append(float(score.get("utility", 0.0)))
            group_indices[str(row["example_id"])].append(len(candidate_ids) - 1)
    return histories, joint_texts, candidate_ids, folds, utilities, group_indices


def feature_fingerprint(candidate_ids: list[str], histories: list[str], joint_texts: list[str]) -> str:
    digest = hashlib.sha256()
    for values in zip(candidate_ids, histories, joint_texts):
        for value in values:
            digest.update(value.encode("utf-8"))
            digest.update(b"\0")
    return digest.hexdigest()


def encode_texts(
    model,
    tokenizer,
    texts: list[str],
    batch_size: int,
    max_length: int,
    query_instruction: str,
    device: str,
    stage: str,
) -> torch.Tensor:
    outputs = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = [format_qwen3_query(text, query_instruction) for text in texts[start : start + batch_size]]
            batch = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            batch = {key: value.to(device) for key, value in batch.items()}
            model_output = model(**batch)
            embeddings = last_token_pool(model_output.last_hidden_state, batch["attention_mask"])
            outputs.append(F.normalize(embeddings.float(), p=2, dim=1).cpu())
            print(
                json.dumps(
                    {"stage": stage, "encoded": min(start + batch_size, len(texts)), "total": len(texts)},
                    ensure_ascii=False,
                ),
                flush=True,
            )
    return torch.cat(outputs)


def load_or_encode_features(
    args: argparse.Namespace,
    histories: list[str],
    joint_texts: list[str],
    candidate_ids: list[str],
) -> tuple[torch.Tensor, dict[str, Any]]:
    fingerprint = feature_fingerprint(candidate_ids, histories, joint_texts)
    cache_path = Path(args.embedding_cache) if args.embedding_cache else None
    metadata = {
        "embedding_model": str(Path(args.embedding_model).resolve()),
        "max_length": args.max_length,
        "query_instruction": args.query_instruction,
        "candidate_count": len(candidate_ids),
        "fingerprint": fingerprint,
        "feature_mode": args.feature_mode,
    }
    if cache_path and cache_path.exists() and not args.force_reencode:
        cached = torch.load(cache_path, map_location="cpu")
        if cached.get("metadata") == metadata:
            return cached["features"].float(), {"cache_hit": True, **metadata}

    tokenizer = AutoTokenizer.from_pretrained(
        args.embedding_model,
        trust_remote_code=True,
        padding_side="left",
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": resolve_torch_dtype(args.torch_dtype),
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModel.from_pretrained(args.embedding_model, **model_kwargs).to(args.device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    unique_histories = list(dict.fromkeys(histories))
    history_to_index = {text: index for index, text in enumerate(unique_histories)}
    unique_history_embeddings = encode_texts(
        model,
        tokenizer,
        unique_histories,
        args.encode_batch_size,
        args.max_length,
        args.query_instruction,
        args.device,
        "encode_history",
    )
    history_embeddings = unique_history_embeddings[
        torch.tensor([history_to_index[text] for text in histories], dtype=torch.long)
    ]
    joint_embeddings = encode_texts(
        model,
        tokenizer,
        joint_texts,
        args.encode_batch_size,
        args.max_length,
        args.query_instruction,
        args.device,
        "encode_history_think",
    )
    features = build_preference_features(history_embeddings, joint_embeddings, args.feature_mode).cpu()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"features": features, "metadata": metadata}, cache_path)
    return features, {"cache_hit": False, **metadata}


def build_pairs(
    rows: list[dict[str, Any]],
    group_indices: dict[str, list[int]],
    valid_fold: int,
) -> tuple[torch.Tensor, torch.Tensor, list[str], list[str]]:
    train_pairs = []
    valid_pairs = []
    train_groups = []
    valid_groups = []
    for row in rows:
        indices = group_indices[row["example_id"]]
        destination = valid_pairs if int(row["fold"]) == valid_fold else train_pairs
        group_destination = valid_groups if int(row["fold"]) == valid_fold else train_groups
        group_destination.append(str(row["example_id"]))
        for winner_position in range(len(indices)):
            for loser_position in range(winner_position + 1, len(indices)):
                rank_gap = loser_position - winner_position
                destination.append((indices[winner_position], indices[loser_position], 1.0 + 0.1 * rank_gap))
    if not train_pairs or not valid_pairs:
        raise ValueError(f"Empty train or valid pair split for valid_fold={valid_fold}")
    return (
        torch.tensor(train_pairs, dtype=torch.float32),
        torch.tensor(valid_pairs, dtype=torch.float32),
        train_groups,
        valid_groups,
    )


def make_pair_loader(pairs: torch.Tensor, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    dataset = TensorDataset(pairs[:, 0].long(), pairs[:, 1].long(), pairs[:, 2].float())
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def pair_loss(scores: torch.Tensor, winners: torch.Tensor, losers: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    margins = scores[winners] - scores[losers]
    return (F.softplus(-margins) * weights).mean()


def rank_correlation(predicted: list[float]) -> float:
    n = len(predicted)
    if n < 2:
        return 0.0
    predicted_order = sorted(range(n), key=lambda index: (-predicted[index], index))
    predicted_rank = [0] * n
    for rank, index in enumerate(predicted_order):
        predicted_rank[index] = rank
    squared = sum((index - predicted_rank[index]) ** 2 for index in range(n))
    return 1.0 - 6.0 * squared / (n * (n * n - 1))


@torch.no_grad()
def evaluate(
    model: PairwisePreferenceHead,
    features: torch.Tensor,
    pairs: torch.Tensor,
    group_ids: list[str],
    group_indices: dict[str, list[int]],
    device: str,
) -> dict[str, float]:
    model.eval()
    scores = model(features.to(device)).cpu()
    winners = pairs[:, 0].long()
    losers = pairs[:, 1].long()
    margins = scores[winners] - scores[losers]
    pair_accuracy = float((margins > 0).float().mean().item())
    group_top1 = []
    correlations = []
    for group_id in group_ids:
        indices = group_indices[group_id]
        group_scores = [float(scores[index].item()) for index in indices]
        group_top1.append(int(max(range(len(indices)), key=lambda index: group_scores[index]) == 0))
        correlations.append(rank_correlation(group_scores))
    return {
        "pair_accuracy": pair_accuracy,
        "pair_margin_mean": float(margins.mean().item()),
        "group_top1_accuracy": sum(group_top1) / len(group_top1),
        "group_spearman_mean": sum(correlations) / len(correlations),
        "score_mean": float(scores.mean().item()),
        "score_std": float(scores.std().item()),
    }


def save_checkpoint(
    model: PairwisePreferenceHead,
    config: PreferenceScorerConfig,
    output_dir: Path,
    name: str,
    step: int,
    metrics: dict[str, float],
) -> None:
    checkpoint = output_dir / name
    checkpoint.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint / "preference_scorer_head.pt")
    (checkpoint / "preference_scorer_config.json").write_text(
        json.dumps(config.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (checkpoint / "metrics.json").write_text(
        json.dumps({"step": step, **metrics}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a fold-disjoint pairwise CoT preference scorer.")
    parser.add_argument("--groups", required=True)
    parser.add_argument("--judgments", required=True)
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--embedding-cache", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--valid-fold", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--encode-batch-size", type=int, default=64)
    parser.add_argument("--train-batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--feature-mode", default="joint_delta_product")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--query-instruction", default=DEFAULT_RECOMMENDATION_QUERY_INSTRUCTION)
    parser.add_argument("--force-reencode", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_args.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    rows, join_stats = load_rows(Path(args.groups), Path(args.judgments))
    histories, joint_texts, candidate_ids, folds, utilities, group_indices = flatten_candidates(rows)
    features, feature_meta = load_or_encode_features(args, histories, joint_texts, candidate_ids)
    train_pairs, valid_pairs, train_groups, valid_groups = build_pairs(rows, group_indices, args.valid_fold)
    train_loader = make_pair_loader(train_pairs, args.train_batch_size, True, args.seed)

    embedding_dim = features.shape[1] // 3
    config = PreferenceScorerConfig(
        embedding_dim=int(embedding_dim),
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        encoder_model=args.embedding_model,
        encoder_max_length=args.max_length,
        query_instruction=args.query_instruction,
        text_mode="think_only",
        feature_mode=args.feature_mode,
    )
    model = PairwisePreferenceHead(config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = max(1, args.epochs * len(train_loader))
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    metadata = {
        "join_stats": join_stats,
        "feature_meta": feature_meta,
        "candidates": len(candidate_ids),
        "train_groups": len(train_groups),
        "valid_groups": len(valid_groups),
        "train_pairs": len(train_pairs),
        "valid_pairs": len(valid_pairs),
        "valid_fold": args.valid_fold,
        "total_steps": total_steps,
        "utility_mean": sum(utilities) / max(1, len(utilities)),
    }
    (output_dir / "data_meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"stage": "train_start", **metadata}, ensure_ascii=False), flush=True)

    best_pair_accuracy = -1.0
    metrics_path = output_dir / "metrics.jsonl"
    step = 0
    features_device = features.to(args.device)
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        for winners, losers, weights in train_loader:
            step += 1
            winners = winners.to(args.device)
            losers = losers.to(args.device)
            weights = weights.to(args.device)
            winner_scores = model(features_device[winners])
            loser_scores = model(features_device[losers])
            loss = (F.softplus(-(winner_scores - loser_scores)) * weights).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            epoch_losses.append(float(loss.item()))

        valid_metrics = evaluate(model, features_device, valid_pairs, valid_groups, group_indices, args.device)
        record = {
            "epoch": epoch,
            "step": step,
            "train_loss": sum(epoch_losses) / len(epoch_losses),
            **{f"valid_{key}": value for key, value in valid_metrics.items()},
            "lr": scheduler.get_last_lr()[0],
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps(record, ensure_ascii=False), flush=True)
        save_checkpoint(model, config, output_dir, f"epoch-{epoch}", step, record)
        if valid_metrics["pair_accuracy"] > best_pair_accuracy:
            best_pair_accuracy = valid_metrics["pair_accuracy"]
            save_checkpoint(model, config, output_dir, "best", step, record)


if __name__ == "__main__":
    main()
