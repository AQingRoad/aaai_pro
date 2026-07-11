#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from new_method.core import (
    RAW_ASIN_RE,
    append_think,
    as_int_set,
    cot_from_row,
    delta_log_rank,
    file_sha256,
    history_from_row,
    ndcg_at_rank,
    rank_bucket,
    read_jsonl,
    stable_row_key,
)
from rubric_cot_pipeline.embeddings import (
    DEFAULT_RECOMMENDATION_QUERY_INSTRUCTION,
    Qwen3TextEmbedder,
    format_qwen3_query,
)


def as_text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def item_text(row: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("title", "main_category", "store"):
        value = str(row.get(key) or "").strip()
        if value:
            parts.append(value)
    categories = " > ".join(as_text_list(row.get("categories")))
    if categories:
        parts.append(f"Categories: {categories}")
    features = "; ".join(as_text_list(row.get("features")))
    if features:
        parts.append(f"Features: {features}")
    description = " ".join(as_text_list(row.get("description")))
    if description:
        parts.append(f"Description: {description}")
    return " ".join(parts).strip()


def load_items(path: str) -> tuple[list[int], list[str], dict[int, int]]:
    records: list[tuple[int, str]] = []
    for row in read_jsonl(path):
        try:
            item_id = int(row["item_id"])
        except (KeyError, TypeError, ValueError):
            continue
        text = item_text(row)
        if not text:
            raise ValueError(f"Empty item text for item_id={item_id}")
        if "[TRUNCATED]" in text:
            raise ValueError(f"Truncated item text for item_id={item_id}")
        records.append((item_id, text))
    records.sort(key=lambda pair: pair[0])
    if not records:
        raise ValueError(f"No usable items found in {path}")
    item_ids = [pair[0] for pair in records]
    item_texts = [pair[1] for pair in records]
    return item_ids, item_texts, {item_id: index for index, item_id in enumerate(item_ids)}


def iter_candidates(path: str, limit: int = 0) -> Iterator[dict[str, Any]]:
    emitted = 0
    for source_row in read_jsonl(path):
        candidates = source_row.get("candidates")
        if isinstance(candidates, list):
            base = {key: value for key, value in source_row.items() if key != "candidates"}
            for candidate_index, candidate in enumerate(candidates):
                if not isinstance(candidate, dict):
                    continue
                row = {**base, **candidate}
                row.setdefault("candidate_index", candidate_index)
                yield row
                emitted += 1
                if limit and emitted >= limit:
                    return
        else:
            yield source_row
            emitted += 1
            if limit and emitted >= limit:
                return


def token_lengths(tokenizer, texts: list[str]) -> list[int]:
    if not texts:
        return []
    encoded = tokenizer(
        texts,
        add_special_tokens=True,
        padding=False,
        truncation=False,
        return_length=True,
    )
    lengths = encoded.get("length")
    if lengths is not None:
        return [int(length) for length in lengths]
    return [len(ids) for ids in encoded["input_ids"]]


def check_lengths(
    tokenizer,
    texts: list[str],
    *,
    max_length: int,
    label: str,
) -> list[int]:
    lengths = token_lengths(tokenizer, texts)
    over = [(index, length) for index, length in enumerate(lengths) if length > max_length]
    if over:
        first_index, first_length = over[0]
        raise ValueError(
            f"{label} contains {len(over)} texts longer than max_length={max_length}; "
            f"first_index={first_index} first_length={first_length}"
        )
    return lengths


def score_scores(
    scores: torch.Tensor,
    *,
    row: dict[str, Any],
    item_ids: list[int],
    item_index: dict[int, int],
    ndcg_k: int,
    margin_temperature: float,
    hard_negative_count: int,
    mask_history_items: bool,
    mask_pad_item: bool,
) -> dict[str, Any]:
    target_id = int(row["target_item_id"])
    if target_id not in item_index:
        raise ValueError(f"target_item_id={target_id} is absent from item-info")
    target_index = item_index[target_id]
    masked_item_ids: set[int] = set()
    if mask_history_items:
        masked_item_ids.update(as_int_set(row.get("history_item_ids") or row.get("history_item_id")))
    if mask_pad_item:
        masked_item_ids.add(0)
    masked_item_ids.discard(target_id)
    masked_indices = [item_index[item_id] for item_id in masked_item_ids if item_id in item_index]

    usable_scores = scores.clone()
    if masked_indices:
        usable_scores[masked_indices] = -torch.inf
    target_score = usable_scores[target_index]
    rank = int((usable_scores > target_score).sum().item()) + 1

    negative_mask = torch.isfinite(usable_scores)
    negative_mask[target_index] = False
    negative_scores = usable_scores[negative_mask]
    if negative_scores.numel() == 0:
        raise ValueError("No unmasked negative items remain")
    margin = float(
        (
            target_score / margin_temperature
            - torch.logsumexp(negative_scores / margin_temperature, dim=0)
        ).item()
    )

    hard_count = min(hard_negative_count, int(negative_scores.numel()))
    hard_negative_ids: list[int] = []
    if hard_count > 0:
        candidate_scores = usable_scores.clone()
        candidate_scores[target_index] = -torch.inf
        top_indices = torch.topk(candidate_scores, k=hard_count).indices.tolist()
        hard_negative_ids = [item_ids[index] for index in top_indices]

    return {
        "score": float(target_score.item()),
        "rank": rank,
        "ndcg": ndcg_at_rank(rank, ndcg_k),
        "margin": margin,
        "hard_negative_item_ids": hard_negative_ids,
        "masked_item_count": len(masked_indices),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score paired history and history+think queries with a frozen embedding retriever."
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Candidate JSONL. Repeat this argument to score multiple CoT sources.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--item-info", required=True)
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--query-max-length", type=int, default=4096)
    parser.add_argument("--item-max-length", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--item-batch-size", type=int, default=32)
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--query-instruction", default=DEFAULT_RECOMMENDATION_QUERY_INSTRUCTION)
    parser.add_argument("--output-dim", type=int, default=0)
    parser.add_argument("--ndcg-k", type=int, default=20)
    parser.add_argument("--margin-temperature", type=float, default=0.05)
    parser.add_argument("--hard-negative-count", type=int, default=32)
    parser.add_argument("--mask-history-items", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mask-pad-item", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if args.query_max_length <= 0 or args.item_max_length <= 0:
        raise ValueError("--query-max-length and --item-max-length must be positive")
    if args.ndcg_k <= 0:
        raise ValueError("--ndcg-k must be positive")
    if args.margin_temperature <= 0:
        raise ValueError("--margin-temperature must be positive")

    item_ids, item_texts, item_index = load_items(args.item_info)
    embedder = Qwen3TextEmbedder(
        args.embedding_model,
        max_length=args.item_max_length,
        batch_size=args.item_batch_size,
        torch_dtype=args.torch_dtype,
        device=args.device,
        query_instruction=args.query_instruction,
        output_dim=args.output_dim,
    )
    item_lengths = check_lengths(
        embedder.tokenizer,
        item_texts,
        max_length=args.item_max_length,
        label="item-info",
    )
    score_device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    item_embeddings = embedder.encode_documents(item_texts).to(score_device)
    embedder.max_length = args.query_max_length

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("", encoding="utf-8")

    rows: list[dict[str, Any]] = []
    written = 0
    candidate_seen = 0

    def process_batch(batch: list[dict[str, Any]]) -> int:
        histories: list[str] = []
        cot_queries: list[str] = []
        thinks: list[str] = []
        has_tags: list[bool] = []
        for row in batch:
            history = history_from_row(row)
            think, tagged = cot_from_row(row)
            if not history:
                raise ValueError(f"Empty history for example_id={stable_row_key(row)}")
            if not think:
                raise ValueError(f"Empty CoT think for example_id={stable_row_key(row)}")
            if "[TRUNCATED]" in history or "[TRUNCATED]" in think:
                raise ValueError(f"Truncation marker for example_id={stable_row_key(row)}")
            histories.append(history)
            cot_queries.append(append_think(history, think))
            thinks.append(think)
            has_tags.append(tagged)

        formatted_histories = [format_qwen3_query(text, args.query_instruction) for text in histories]
        formatted_cot_queries = [format_qwen3_query(text, args.query_instruction) for text in cot_queries]
        history_lengths = check_lengths(
            embedder.tokenizer,
            formatted_histories,
            max_length=args.query_max_length,
            label="history queries",
        )
        cot_lengths = check_lengths(
            embedder.tokenizer,
            formatted_cot_queries,
            max_length=args.query_max_length,
            label="history+think queries",
        )
        history_embeddings = embedder.encode_queries(histories).to(score_device)
        cot_embeddings = embedder.encode_queries(cot_queries).to(score_device)
        baseline_score_matrix = history_embeddings @ item_embeddings.T
        cot_score_matrix = cot_embeddings @ item_embeddings.T

        count = 0
        for index, row in enumerate(batch):
            baseline = score_scores(
                baseline_score_matrix[index],
                row=row,
                item_ids=item_ids,
                item_index=item_index,
                ndcg_k=args.ndcg_k,
                margin_temperature=args.margin_temperature,
                hard_negative_count=args.hard_negative_count,
                mask_history_items=args.mask_history_items,
                mask_pad_item=args.mask_pad_item,
            )
            cot = score_scores(
                cot_score_matrix[index],
                row=row,
                item_ids=item_ids,
                item_index=item_index,
                ndcg_k=args.ndcg_k,
                margin_temperature=args.margin_temperature,
                hard_negative_count=args.hard_negative_count,
                mask_history_items=args.mask_history_items,
                mask_pad_item=args.mask_pad_item,
            )
            log_gain = delta_log_rank(baseline["rank"], cot["rank"])
            out = {
                **row,
                "example_id": stable_row_key(row),
                "user_history": histories[index],
                "cot_think": thinks[index],
                "has_tags": has_tags[index],
                "format_ok": bool(row.get("format_ok", has_tags[index])),
                "quality_fields_present": any(
                    field in row
                    for field in (
                        "leakage",
                        "target_leakage",
                        "unsupported_claim_count",
                        "metadata_contradiction_count",
                    )
                ),
                "raw_asin_in_cot": bool(RAW_ASIN_RE.search(thinks[index])),
                "baseline_score": baseline["score"],
                "cot_score": cot["score"],
                "baseline_rank": baseline["rank"],
                "cot_rank": cot["rank"],
                "baseline_ndcg": baseline["ndcg"],
                "cot_ndcg": cot["ndcg"],
                "delta_ndcg": cot["ndcg"] - baseline["ndcg"],
                "baseline_margin": baseline["margin"],
                "cot_margin": cot["margin"],
                "delta_margin": cot["margin"] - baseline["margin"],
                "delta_log_rank": log_gain,
                "baseline_rank_bucket": rank_bucket(baseline["rank"]),
                "target_entered_topk": baseline["rank"] > args.ndcg_k >= cot["rank"],
                "target_left_topk": baseline["rank"] <= args.ndcg_k < cot["rank"],
                "baseline_hard_negative_item_ids": baseline["hard_negative_item_ids"],
                "cot_hard_negative_item_ids": cot["hard_negative_item_ids"],
                "history_tokens": history_lengths[index],
                "history_cot_tokens": cot_lengths[index],
                "history_truncated_tokens": 0,
                "ndcg_k": args.ndcg_k,
                "margin_temperature": args.margin_temperature,
                "scorer_checkpoint": args.embedding_model,
                "query_mode": "history_plus_think_only",
                "query_instruction": args.query_instruction,
                "query_max_length": args.query_max_length,
                "item_max_length": args.item_max_length,
                "masked_history_items": args.mask_history_items,
                "masked_pad_item": args.mask_pad_item,
            }
            with output_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(out, ensure_ascii=False) + "\n")
            count += 1
        return count

    for input_path in args.input:
        if args.max_examples and candidate_seen >= args.max_examples:
            break
        for row in iter_candidates(input_path):
            if args.max_examples and candidate_seen >= args.max_examples:
                break
            candidate_seen += 1
            row.setdefault("cot_source", Path(input_path).stem)
            row.setdefault("source_file", str(Path(input_path).resolve()))
            rows.append(row)
            if len(rows) >= args.batch_size:
                written += process_batch(rows)
                rows = []
                print(json.dumps({"scored": written}, ensure_ascii=False), flush=True)
    if rows:
        written += process_batch(rows)

    metadata = {
        "inputs": [str(Path(path).resolve()) for path in args.input],
        "input_sha256": {str(Path(path).resolve()): file_sha256(path) for path in args.input},
        "output": str(output_path.resolve()),
        "item_info": str(Path(args.item_info).resolve()),
        "item_info_sha256": file_sha256(args.item_info),
        "item_count": len(item_ids),
        "item_token_max": max(item_lengths),
        "score_device": str(score_device),
        "scored_rows": written,
        "parameters": vars(args),
    }
    output_path.with_suffix(output_path.suffix + ".meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
