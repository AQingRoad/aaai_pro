#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from new_method.core import append_think, as_text_list, read_jsonl
from new_method.paired_loss import paired_cot_loss
from rubric_cot_pipeline.embeddings import (
    DEFAULT_RECOMMENDATION_QUERY_INSTRUCTION,
    format_qwen3_query,
    last_token_pool,
    resolve_torch_dtype,
)


class PairedRetrieverDataset(Dataset):
    def __init__(self, path: str, limit: int = 0):
        self.rows: list[dict[str, Any]] = []
        for row_index, row in enumerate(read_jsonl(path)):
            if limit and len(self.rows) >= limit:
                break
            history = str(row.get("history") or "").strip()
            positive = str(row.get("positive") or "").strip()
            target_id = row.get("target_item_id")
            if not history or not positive or target_id is None:
                raise ValueError(
                    f"Row {row_index + 1} requires non-empty history, positive and target_item_id"
                )
            self.rows.append(
                {
                    "example_id": str(row.get("example_id") or row_index),
                    "history": history,
                    "positive": positive,
                    "target_item_id": int(target_id),
                    "good_cot": str(row.get("good_cot") or "").strip(),
                    "bad_cot": str(row.get("bad_cot") or "").strip(),
                    "negatives": as_text_list(row.get("negatives")),
                }
            )
        if not self.rows:
            raise ValueError(f"No usable rows found in {path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def collate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "example_ids": [row["example_id"] for row in rows],
        "histories": [row["history"] for row in rows],
        "positives": [row["positive"] for row in rows],
        "target_item_ids": [row["target_item_id"] for row in rows],
        "good_cots": [row["good_cot"] for row in rows],
        "bad_cots": [row["bad_cot"] for row in rows],
        "negatives": [row["negatives"] for row in rows],
    }


def encode_texts(model, tokenizer, texts: list[str], device: torch.device) -> torch.Tensor:
    batch = tokenizer(
        texts,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )
    batch = {key: value.to(device) for key, value in batch.items()}
    outputs = model(**batch)
    embeddings = last_token_pool(outputs.last_hidden_state, batch["attention_mask"])
    return F.normalize(embeddings.float(), p=2, dim=1)


def token_length(tokenizer, text: str) -> int:
    encoded = tokenizer(
        text,
        add_special_tokens=True,
        truncation=False,
        padding=False,
    )
    return len(encoded["input_ids"])


def audit_lengths(
    dataset: PairedRetrieverDataset,
    tokenizer,
    *,
    query_instruction: str,
    max_length: int,
) -> dict[str, int]:
    maxima = {
        "history": 0,
        "history_good": 0,
        "history_bad": 0,
        "positive": 0,
        "negative": 0,
    }
    failures: list[dict[str, Any]] = []
    for row in dataset.rows:
        texts: list[tuple[str, str]] = [
            ("history", format_qwen3_query(row["history"], query_instruction)),
            ("positive", row["positive"]),
        ]
        if row["good_cot"]:
            texts.append(
                (
                    "history_good",
                    format_qwen3_query(
                        append_think(row["history"], row["good_cot"]),
                        query_instruction,
                    ),
                )
            )
        if row["bad_cot"]:
            texts.append(
                (
                    "history_bad",
                    format_qwen3_query(
                        append_think(row["history"], row["bad_cot"]),
                        query_instruction,
                    ),
                )
            )
        texts.extend(("negative", text) for text in row["negatives"])
        for kind, text in texts:
            if "[TRUNCATED]" in text:
                failures.append(
                    {
                        "example_id": row["example_id"],
                        "kind": kind,
                        "reason": "truncated_marker",
                    }
                )
                continue
            length = token_length(tokenizer, text)
            maxima[kind] = max(maxima[kind], length)
            if length > max_length:
                failures.append(
                    {
                        "example_id": row["example_id"],
                        "kind": kind,
                        "reason": "over_max_length",
                        "tokens": length,
                    }
                )
    if failures:
        raise ValueError(
            "Length audit failed. "
            + json.dumps(
                {"failure_count": len(failures), "examples": failures[:10]},
                ensure_ascii=False,
            )
        )
    return maxima


def save_checkpoint(model, tokenizer, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)


def scalar(value: torch.Tensor) -> float:
    return round(float(value.detach().item()), 6)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the first-stage GDR paired retriever.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=0.2)
    parser.add_argument("--good-order-margin", type=float, default=0.0)
    parser.add_argument("--bad-order-margin", type=float, default=0.0)
    parser.add_argument("--negatives-per-row", type=int, default=2)
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--query-instruction", default=DEFAULT_RECOMMENDATION_QUERY_INSTRUCTION)
    parser.add_argument("--save-steps", type=int, default=0)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--length-audit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preview-cases", type=int, default=2)
    args = parser.parse_args()

    from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

    if args.max_length <= 0:
        raise ValueError("--max-length must be positive")
    if args.batch_size <= 0 or args.grad_accum <= 0:
        raise ValueError("--batch-size and --grad-accum must be positive")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")
    if args.negatives_per_row < 0:
        raise ValueError("--negatives-per-row must be non-negative")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    dataset = PairedRetrieverDataset(args.dataset, limit=args.max_rows)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        padding_side="left",
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    length_maxima = (
        audit_lengths(
            dataset,
            tokenizer,
            query_instruction=args.query_instruction,
            max_length=args.max_length,
        )
        if args.length_audit
        else {}
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for index in range(min(args.preview_cases, len(dataset))):
        row = dataset[index]
        print(
            json.dumps(
                {
                    "preview_type": "paired_retriever_case",
                    "case_index": index + 1,
                    "example_id": row["example_id"],
                    "history": row["history"],
                    "good_cot": row["good_cot"],
                    "bad_cot": row["bad_cot"],
                    "positive": row["positive"],
                    "negative_count": len(row["negatives"]),
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        drop_last=True,
    )
    if len(loader) == 0:
        raise ValueError("Dataset must contain at least one full batch")

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": resolve_torch_dtype(args.torch_dtype),
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModel.from_pretrained(args.model, **model_kwargs).to(device)
    model.train()
    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            model.gradient_checkpointing_enable()

    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight", "norm.weight")
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [
                    parameter
                    for name, parameter in model.named_parameters()
                    if parameter.requires_grad and not any(key in name for key in no_decay)
                ],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [
                    parameter
                    for name, parameter in model.named_parameters()
                    if parameter.requires_grad and any(key in name for key in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ],
        lr=args.learning_rate,
    )
    usable_batches_per_epoch = len(loader) // args.grad_accum * args.grad_accum
    updates_per_epoch = usable_batches_per_epoch // args.grad_accum
    if updates_per_epoch == 0:
        raise ValueError(
            "Need at least grad_accum full batches; "
            f"loader_batches={len(loader)} grad_accum={args.grad_accum}"
        )
    total_steps = (
        args.max_steps
        if args.max_steps > 0
        else max(1, math.ceil(args.epochs * updates_per_epoch))
    )
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    metadata = {
        **vars(args),
        "dataset_rows": len(dataset),
        "steps_per_epoch": updates_per_epoch,
        "usable_batches_per_epoch": usable_batches_per_epoch,
        "total_steps": total_steps,
        "effective_history_batch": args.batch_size * args.grad_accum,
        "length_maxima": length_maxima,
        "training_objective": "paired_cot",
    }
    (output_dir / "paired_args.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    epoch_index = 0
    while global_step < total_steps:
        epoch_index += 1
        for batch_index, batch in enumerate(loader, start=1):
            if batch_index > usable_batches_per_epoch:
                break
            histories = [
                format_qwen3_query(text, args.query_instruction)
                for text in batch["histories"]
            ]
            good_mask = torch.tensor(
                [bool(text) for text in batch["good_cots"]],
                dtype=torch.bool,
                device=device,
            )
            bad_mask = torch.tensor(
                [bool(text) for text in batch["bad_cots"]],
                dtype=torch.bool,
                device=device,
            )
            good_queries = [
                format_qwen3_query(
                    append_think(history, cot),
                    args.query_instruction,
                )
                for history, cot in zip(batch["histories"], batch["good_cots"])
                if cot
            ]
            bad_queries = [
                format_qwen3_query(
                    append_think(history, cot),
                    args.query_instruction,
                )
                for history, cot in zip(batch["histories"], batch["bad_cots"])
                if cot
            ]
            explicit_negatives = [
                text
                for row_negatives in batch["negatives"]
                for text in row_negatives[: args.negatives_per_row]
            ]
            documents = list(batch["positives"]) + explicit_negatives

            history_embeddings = encode_texts(model, tokenizer, histories, device)
            good_embeddings = (
                encode_texts(model, tokenizer, good_queries, device)
                if good_queries
                else None
            )
            bad_embeddings = (
                encode_texts(model, tokenizer, bad_queries, device)
                if bad_queries
                else None
            )
            document_embeddings = encode_texts(model, tokenizer, documents, device)
            query_target_ids = torch.tensor(
                batch["target_item_ids"],
                dtype=torch.long,
                device=device,
            )
            document_target_ids = torch.cat(
                [
                    query_target_ids,
                    torch.full(
                        (len(explicit_negatives),),
                        -1,
                        dtype=torch.long,
                        device=device,
                    ),
                ]
            )
            history_logits = history_embeddings @ document_embeddings.T / args.temperature
            good_logits = (
                good_embeddings @ document_embeddings.T / args.temperature
                if good_embeddings is not None
                else None
            )
            bad_logits = (
                bad_embeddings @ document_embeddings.T / args.temperature
                if bad_embeddings is not None
                else None
            )
            output = paired_cot_loss(
                history_logits=history_logits,
                query_target_ids=query_target_ids,
                document_target_ids=document_target_ids,
                good_logits=good_logits,
                good_mask=good_mask,
                bad_logits=bad_logits,
                bad_mask=bad_mask,
                alpha=args.alpha,
                beta=args.beta,
                gamma=args.gamma,
                good_order_margin=args.good_order_margin,
                bad_order_margin=args.bad_order_margin,
            )
            (output.loss / args.grad_accum).backward()

            if batch_index % args.grad_accum != 0:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            log = {
                "step": global_step,
                "total_steps": total_steps,
                "epoch": epoch_index,
                "loss": scalar(output.loss),
                "loss_history": scalar(output.loss_history),
                "loss_good_retrieval": scalar(output.loss_good_retrieval),
                "loss_good_order": scalar(output.loss_good_order),
                "loss_bad_order": scalar(output.loss_bad_order),
                "history_accuracy": scalar(output.history_accuracy),
                "good_accuracy": scalar(output.good_accuracy),
                "good_order_accuracy": scalar(output.good_order_accuracy),
                "bad_order_accuracy": scalar(output.bad_order_accuracy),
                "history_margin_mean": scalar(output.history_margin_mean),
                "good_margin_mean": scalar(output.good_margin_mean),
                "bad_margin_mean": scalar(output.bad_margin_mean),
                "good_rows": int(good_mask.sum().item()),
                "bad_rows": int(bad_mask.sum().item()),
                "explicit_negatives": len(explicit_negatives),
                "candidate_documents": len(documents),
                "lr": scheduler.get_last_lr()[0],
            }
            print(json.dumps(log, ensure_ascii=False), flush=True)

            if args.save_steps > 0 and global_step % args.save_steps == 0:
                save_checkpoint(model, tokenizer, output_dir / f"checkpoint-{global_step}")
            if global_step >= total_steps:
                break

    final_dir = output_dir / f"checkpoint-{global_step}"
    save_checkpoint(model, tokenizer, final_dir)
    print(
        json.dumps(
            {"checkpoint": str(final_dir), "steps": global_step},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
