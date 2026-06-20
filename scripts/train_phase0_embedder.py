#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

from rubric_cot_pipeline.embeddings import (
    DEFAULT_RECOMMENDATION_QUERY_INSTRUCTION,
    format_qwen3_query,
    last_token_pool,
    resolve_torch_dtype,
)
from rubric_cot_pipeline.io import read_jsonl


class PairDataset(Dataset):
    def __init__(self, path: str, limit: int = 0):
        self.rows = [
            {"query": str(row.get("query") or ""), "positive": str(row.get("positive") or "")}
            for row in read_jsonl(path, limit=limit)
            if row.get("query") and row.get("positive")
        ]
        if not self.rows:
            raise ValueError(f"No usable query/positive rows found in {path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, str]:
        return self.rows[idx]


def collate(rows: list[dict[str, str]]) -> dict[str, list[str]]:
    return {
        "queries": [row["query"] for row in rows],
        "positives": [row["positive"] for row in rows],
    }


def encode_texts(model, tokenizer, texts: list[str], max_length: int):
    batch = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    device = next(model.parameters()).device
    batch = {key: value.to(device) for key, value in batch.items()}
    outputs = model(**batch)
    embeddings = last_token_pool(outputs.last_hidden_state, batch["attention_mask"])
    return F.normalize(embeddings.float(), p=2, dim=1)


def init_distributed() -> tuple[bool, int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    return distributed, rank, local_rank, world_size, device


def is_main_process(rank: int) -> bool:
    return rank == 0


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def save_checkpoint(model, tokenizer, path: Path, rank: int) -> None:
    if not is_main_process(rank):
        return
    path.mkdir(parents=True, exist_ok=True)
    unwrap_model(model).save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/root/autodl-tmp/modelscope_cache/models/Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--query-instruction", default=DEFAULT_RECOMMENDATION_QUERY_INSTRUCTION)
    parser.add_argument("--save-steps", type=int, default=0)
    parser.add_argument("--gradient-checkpointing", choices=["auto", "on", "off"], default="auto")
    args = parser.parse_args()

    distributed, rank, local_rank, world_size, device = init_distributed()

    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + rank)
    if is_main_process(rank):
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()

    dataset = PairDataset(args.dataset, limit=args.max_rows)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed, drop_last=True) if distributed else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        collate_fn=collate,
        drop_last=True,
    )
    if len(loader) == 0:
        raise ValueError("Need at least one full batch for in-batch negative InfoNCE")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        padding_side="left",
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModel.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=resolve_torch_dtype(args.torch_dtype),
    ).to(device)
    model.train()
    use_gradient_checkpointing = args.gradient_checkpointing == "on" or (
        args.gradient_checkpointing == "auto" and not distributed
    )
    if use_gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    elif hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    if distributed:
        model = DDP(model, device_ids=[local_rank] if torch.cuda.is_available() else None)

    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight", "norm.weight"]
    grouped = [
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(grouped, lr=args.learning_rate)

    steps_per_epoch = math.ceil(len(loader) / max(1, args.grad_accum))
    total_steps = args.max_steps if args.max_steps > 0 else max(1, int(math.ceil(args.epochs * steps_per_epoch)))
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    args_path = Path(args.output_dir) / "phase0_args.json"
    if is_main_process(rank):
        metadata = {
            **vars(args),
            "distributed": distributed,
            "world_size": world_size,
            "per_device_batch_size": args.batch_size,
            "global_batch_size": args.batch_size * world_size * max(1, args.grad_accum),
            "steps_per_epoch_per_rank": steps_per_epoch,
            "gradient_checkpointing_enabled": use_gradient_checkpointing,
        }
        args_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    if distributed:
        dist.barrier()

    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    while global_step < total_steps:
        if sampler is not None:
            sampler.set_epoch(global_step)
        for batch_idx, batch in enumerate(loader, start=1):
            query_texts = [format_qwen3_query(text, args.query_instruction) for text in batch["queries"]]
            doc_texts = batch["positives"]

            query_emb = encode_texts(model, tokenizer, query_texts, args.max_length)
            doc_emb = encode_texts(model, tokenizer, doc_texts, args.max_length)
            logits = query_emb @ doc_emb.T / args.temperature
            labels = torch.arange(logits.shape[0], device=logits.device)
            loss = F.cross_entropy(logits, labels)
            (loss / args.grad_accum).backward()

            if batch_idx % args.grad_accum != 0:
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            with torch.no_grad():
                acc = (logits.argmax(dim=1) == labels).float().mean().item()
            if is_main_process(rank):
                print(
                    json.dumps(
                        {
                            "step": global_step,
                            "max_steps": total_steps,
                            "loss": round(float(loss.item()), 6),
                            "batch_acc": round(acc, 4),
                            "lr": scheduler.get_last_lr()[0],
                            "world_size": world_size,
                            "global_batch_size": args.batch_size * world_size * max(1, args.grad_accum),
                        }
                    ),
                    flush=True,
                )

            if args.save_steps > 0 and global_step % args.save_steps == 0:
                ckpt_dir = Path(args.output_dir) / f"checkpoint-{global_step}"
                save_checkpoint(model, tokenizer, ckpt_dir, rank)
                if distributed:
                    dist.barrier()

            if global_step >= total_steps:
                break

    final_dir = Path(args.output_dir) / f"checkpoint-{global_step}"
    save_checkpoint(model, tokenizer, final_dir, rank)
    if distributed:
        dist.barrier()
    if is_main_process(rank):
        print(
            json.dumps(
                {
                    "checkpoint": str(final_dir),
                    "steps": global_step,
                    "distributed": distributed,
                    "world_size": world_size,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
