#!/usr/bin/env python3
"""用 query-positive pair 微调 Qwen3 Embedding，保存每轮模型并在末轮后测试。"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup


QUERY_INSTRUCTION = (
    "Given a user's past item interactions and optional recommendation reasoning, "
    "retrieve items the user is likely to prefer next."
)


def read_jsonl(path: Path) -> list[dict]:
    """读取 JSONL，并忽略空行。"""
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


class PairDataset(Dataset):
    """只保留 embedding 训练需要的 query、positive 和 target_item_id。"""

    def __init__(self, path: Path, expected_split: str):
        source_rows = read_jsonl(path)
        self.rows = []
        for line_number, row in enumerate(source_rows, 1):
            query = str(row.get("query") or "").strip()
            positive = str(row.get("positive") or "").strip()
            target_item_id = row.get("target_item_id")
            if not query or not positive or target_item_id is None:
                raise ValueError(f"{path} 第 {line_number} 行缺少 query、positive 或 target_item_id")
            if row.get("split") != expected_split:
                raise ValueError(f"{path} 第 {line_number} 行 split={row.get('split')!r}，预期 {expected_split!r}")
            self.rows.append(
                {
                    "query": query,
                    "positive": positive,
                    "target_item_id": int(target_item_id),
                    "example_id": str(row.get("example_id") or ""),
                }
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return self.rows[index]


def collate(rows: list[dict]) -> dict:
    """保留原始字符串，tokenize 放到训练设备侧执行。"""
    return {
        "queries": [row["query"] for row in rows],
        "positives": [row["positive"] for row in rows],
        "target_item_ids": [row["target_item_id"] for row in rows],
    }


def format_query(query: str) -> str:
    """加入与 Qwen3 Embedding 训练和评测一致的 query instruction。"""
    return f"Instruct: {QUERY_INSTRUCTION}\nQuery: {query}"


def last_token_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Qwen3 Embedding 使用每个序列最后一个有效 token 的向量。"""
    if bool((attention_mask[:, -1] == 1).all()):
        return hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    return hidden_states[torch.arange(hidden_states.shape[0], device=hidden_states.device), sequence_lengths]


def encode(model, tokenizer, texts: list[str], device: torch.device) -> torch.Tensor:
    """数据已在训练前完成长度审计，因此此处禁止静默截断。"""
    batch = tokenizer(texts, padding=True, truncation=False, return_tensors="pt")
    batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
    outputs = model(**batch)
    embeddings = last_token_pool(outputs.last_hidden_state, batch["attention_mask"])
    return F.normalize(embeddings.float(), p=2, dim=1)


def multi_positive_info_nce(
    query_embeddings: torch.Tensor,
    document_embeddings: torch.Tensor,
    target_item_ids: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, float]:
    """同一 target_item_id 的文档都作为正例，避免 batch 内假负例。"""
    logits = query_embeddings @ document_embeddings.T / temperature
    positive_mask = target_item_ids[:, None].eq(target_item_ids[None, :])
    positive_logits = logits.masked_fill(~positive_mask, -torch.inf)
    loss = (torch.logsumexp(logits, dim=1) - torch.logsumexp(positive_logits, dim=1)).mean()

    predicted_ids = target_item_ids[logits.argmax(dim=1)]
    accuracy = predicted_ids.eq(target_item_ids).float().mean().item()
    return loss, accuracy


def token_length_audit(tokenizer, dataset: PairDataset, max_length: int, split: str) -> dict:
    """训练前测量真实长度，任一 query 或 positive 超限时直接停止。"""
    query_lengths = []
    positive_lengths = []
    longest_query = (0, "")
    longest_positive = (0, "")

    for start in range(0, len(dataset), 256):
        rows = dataset.rows[start : start + 256]
        queries = [format_query(row["query"]) for row in rows]
        positives = [row["positive"] for row in rows]
        query_ids = tokenizer(queries, truncation=False, padding=False)["input_ids"]
        positive_ids = tokenizer(positives, truncation=False, padding=False)["input_ids"]

        for row, ids in zip(rows, query_ids):
            length = len(ids)
            query_lengths.append(length)
            if length > longest_query[0]:
                longest_query = (length, row["example_id"])
        for row, ids in zip(rows, positive_ids):
            length = len(ids)
            positive_lengths.append(length)
            if length > longest_positive[0]:
                longest_positive = (length, row["example_id"])

    audit = {
        "split": split,
        "rows": len(dataset),
        "query_max_tokens": longest_query[0],
        "query_max_example_id": longest_query[1],
        "positive_max_tokens": longest_positive[0],
        "positive_max_example_id": longest_positive[1],
        "query_over_limit": sum(length > max_length for length in query_lengths),
        "positive_over_limit": sum(length > max_length for length in positive_lengths),
        "max_length": max_length,
    }
    if audit["query_over_limit"] or audit["positive_over_limit"]:
        raise ValueError(f"token 长度超过 max_length：{json.dumps(audit, ensure_ascii=False)}")
    return audit


@torch.no_grad()
def evaluate(model, tokenizer, loader: DataLoader, device: torch.device, temperature: float) -> dict:
    """计算给定数据集上的 multi-positive loss 和 batch accuracy。"""
    model.eval()
    loss_sum = 0.0
    accuracy_sum = 0.0
    examples = 0

    for batch in loader:
        query_texts = [format_query(text) for text in batch["queries"]]
        target_ids = torch.tensor(batch["target_item_ids"], dtype=torch.long, device=device)
        query_embeddings = encode(model, tokenizer, query_texts, device)
        document_embeddings = encode(model, tokenizer, batch["positives"], device)
        loss, accuracy = multi_positive_info_nce(query_embeddings, document_embeddings, target_ids, temperature)
        batch_size = len(query_texts)
        loss_sum += loss.item() * batch_size
        accuracy_sum += accuracy * batch_size
        examples += batch_size

    model.train()
    return {"test_loss": loss_sum / examples, "test_batch_accuracy": accuracy_sum / examples}


def save_checkpoint(model, tokenizer, output_dir: Path, name: str) -> Path:
    """保存模型和完整 tokenizer，保证 checkpoint 可直接用于评测。"""
    path = output_dir / name
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)
    return path


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="微调 Qwen3 Embedding 检索模型。")
    parser.add_argument("--model", default="/home/user/models_hf/Qwen3-Embedding-0.6B")
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    args = parser.parse_args()

    if args.seed != 42:
        parser.error("项目随机种子固定为 42")
    if args.batch_size < 1 or args.grad_accum < 1 or args.epochs < 1:
        parser.error("batch-size、grad-accum 和 epochs 必须大于 0")
    if not torch.cuda.is_available():
        parser.error("该训练脚本要求 CUDA GPU")

    set_seed(args.seed)
    device = torch.device("cuda:0")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = PairDataset(args.train_file, expected_split="train")
    test_dataset = PairDataset(args.test_file, expected_split="test")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        use_fast=True,
        padding_side="left",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 测试集不参与训练前选模；这里只先审计训练集长度。
    audits = [token_length_audit(tokenizer, train_dataset, args.max_length, "train")]
    (args.output_dir / "token_audit.json").write_text(
        json.dumps(audits, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"token_audit": audits}, ensure_ascii=False, indent=2), flush=True)

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collate,
        num_workers=0,
    )
    model = AutoModel.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
    ).to(device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    updates_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_updates = updates_per_epoch * args.epochs
    warmup_steps = int(total_updates * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_updates)

    run_config = {
        **vars(args),
        "train_rows": len(train_dataset),
        "test_rows": len(test_dataset),
        "global_batch_size": args.batch_size * args.grad_accum,
        "updates_per_epoch": updates_per_epoch,
        "total_updates": total_updates,
        "query_instruction": QUERY_INSTRUCTION,
        "loss": "multi_positive_info_nce",
        "token_audit": audits,
    }
    run_config["train_file"] = str(args.train_file)
    run_config["test_file"] = str(args.test_file)
    run_config["output_dir"] = str(args.output_dir)
    (args.output_dir / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    metrics_path = args.output_dir / "train_metrics.jsonl"
    global_step = 0
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        epoch_examples = 0

        for batch_index, batch in enumerate(train_loader):
            # 最后一个累积组不足 grad_accum 时，按实际组大小缩放 loss。
            group_start = (batch_index // args.grad_accum) * args.grad_accum
            group_size = min(args.grad_accum, len(train_loader) - group_start)
            query_texts = [format_query(text) for text in batch["queries"]]
            target_ids = torch.tensor(batch["target_item_ids"], dtype=torch.long, device=device)

            query_embeddings = encode(model, tokenizer, query_texts, device)
            document_embeddings = encode(model, tokenizer, batch["positives"], device)
            loss, accuracy = multi_positive_info_nce(
                query_embeddings,
                document_embeddings,
                target_ids,
                args.temperature,
            )
            (loss / group_size).backward()

            batch_size = len(query_texts)
            epoch_loss += loss.item() * batch_size
            epoch_accuracy += accuracy * batch_size
            epoch_examples += batch_size

            is_group_end = (batch_index + 1) % args.grad_accum == 0 or batch_index + 1 == len(train_loader)
            if is_group_end:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

        train_metrics = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": epoch_loss / epoch_examples,
            "train_batch_accuracy": epoch_accuracy / epoch_examples,
            "learning_rate": scheduler.get_last_lr()[0],
        }
        print(json.dumps(train_metrics, ensure_ascii=False), flush=True)
        with metrics_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(train_metrics, ensure_ascii=False) + "\n")

        save_checkpoint(model, tokenizer, args.output_dir, f"checkpoint-epoch-{epoch:02d}")

    # 所有 epoch 完成后才审计并读取测试集，不用测试指标选择 checkpoint。
    test_audit = token_length_audit(tokenizer, test_dataset, args.max_length, "test")
    audits.append(test_audit)
    (args.output_dir / "token_audit.json").write_text(
        json.dumps(audits, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=0,
    )
    test_metrics = evaluate(model, tokenizer, test_loader, device, args.temperature)
    test_metrics.update({"epoch": args.epochs, "checkpoint": f"checkpoint-epoch-{args.epochs:02d}"})
    (args.output_dir / "test_metrics.json").write_text(
        json.dumps(test_metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(test_metrics, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
