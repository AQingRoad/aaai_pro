#!/usr/bin/env python3
"""评测 10 个 embedding checkpoint 与 5 份 SFT CoT 的完整组合矩阵。"""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer


EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR))

from evaluate_embedding_fullset import (  # noqa: E402
    QUERY_TRUNCATION,
    encode_batches,
    format_query,
    load_candidates,
    load_test_rows,
    metrics_from_ranks,
    rank_queries,
    set_seed,
    token_length_audit,
)


def natural_key(path: Path) -> tuple[int, ...]:
    """按路径中的数字自然排序 checkpoint。"""
    return tuple(int(value) for value in re.findall(r"\d+", path.name))


def parse_embedding_runs(values: list[str]) -> list[tuple[str, Path]]:
    """解析重复传入的 tag=/absolute/run/path。"""
    runs = []
    seen_tags = set()
    for value in values:
        if "=" not in value:
            raise ValueError(f"--embedding-run 必须为 tag=path，当前为：{value}")
        tag, raw_path = value.split("=", 1)
        tag = tag.strip()
        path = Path(raw_path).expanduser().resolve()
        if not re.fullmatch(r"[a-z0-9][a-z0-9_]*", tag):
            raise ValueError(f"embedding tag 只允许小写字母、数字和下划线：{tag}")
        if tag in seen_tags:
            raise ValueError(f"embedding tag 重复：{tag}")
        if not path.is_dir():
            raise FileNotFoundError(f"embedding 输出目录不存在：{path}")
        seen_tags.add(tag)
        runs.append((tag, path))
    return runs


def checkpoint_dirs(run_dir: Path, expected: int) -> list[Path]:
    """查找并严格检查每轮保存的 embedding checkpoint。"""
    checkpoints = sorted(run_dir.glob("checkpoint-epoch-*"), key=natural_key)
    if len(checkpoints) != expected:
        raise ValueError(
            f"{run_dir} 应包含 {expected} 个 embedding checkpoint，当前为 {len(checkpoints)}"
        )
    expected_names = [f"checkpoint-epoch-{epoch:02d}" for epoch in range(1, expected + 1)]
    names = [path.name for path in checkpoints]
    if names != expected_names:
        raise ValueError(f"embedding checkpoint 轮次不连续：{names}")
    return checkpoints


def sft_test_files(eval_root: Path, expected: int) -> list[tuple[str, Path]]:
    """读取 5 个 SFT checkpoint 已生成的 test CoT。"""
    paths = sorted(eval_root.glob("checkpoint-*/test_generated_cot.jsonl"), key=lambda p: natural_key(p.parent))
    if len(paths) != expected:
        raise ValueError(
            f"{eval_root} 应包含 {expected} 份 SFT test CoT，当前为 {len(paths)}"
        )
    return [(path.parent.name, path.resolve()) for path in paths]


def write_json_atomic(path: Path, value: object) -> None:
    """原子写入 JSON，避免中断后留下半文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    """原子写入 JSONL。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def existing_result(
    metrics_path: Path,
    ranks_path: Path,
    *,
    embedding_checkpoint: Path,
    test_file: Path,
    expected_rows: int,
) -> dict | None:
    """校验已完成组合；匹配时用于断点续跑。"""
    if not metrics_path.is_file() or not ranks_path.is_file():
        return None
    try:
        result = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if Path(result.get("checkpoint", "")).resolve() != embedding_checkpoint.resolve():
        return None
    if Path(result.get("test_file", "")).resolve() != test_file.resolve():
        return None
    if int(result.get("evaluated", -1)) != expected_rows:
        return None
    with ranks_path.open(encoding="utf-8") as handle:
        if sum(1 for _ in handle) != expected_rows:
            return None
    return result


def summary_row(result: dict) -> dict:
    """提取便于排序和汇总的矩阵指标。"""
    return {
        "embedding_variant": result["embedding_variant"],
        "embedding_checkpoint": result["embedding_checkpoint"],
        "embedding_epoch": result["embedding_epoch"],
        "sft_checkpoint": result["sft_checkpoint"],
        **result["metrics"],
    }


def write_summary(output_root: Path) -> list[dict]:
    """从已落盘的组合指标重建全局汇总，支持断点续跑。"""
    rows = []
    for path in output_root.glob(
        "*/checkpoint-epoch-*/checkpoint-*/retrieval_metrics.json"
    ):
        result = json.loads(path.read_text(encoding="utf-8"))
        rows.append(summary_row(result))
    rows.sort(
        key=lambda row: (
            row["embedding_variant"],
            int(row["embedding_epoch"]),
            natural_key(Path(row["sft_checkpoint"])),
        )
    )
    write_json_atomic(output_root / "all_metrics.json", rows)
    write_jsonl_atomic(output_root / "all_metrics.jsonl", rows)
    return rows


def validate_sft_rows(
    test_sets: list[tuple[str, Path]], expected_rows: int
) -> list[tuple[str, Path, list[dict]]]:
    """检查 5 份生成结果共享相同 test 样本和监督目标。"""
    loaded = []
    reference = None
    for checkpoint_name, path in test_sets:
        rows = load_test_rows(path)
        if len(rows) != expected_rows:
            raise ValueError(f"{path} 应为 {expected_rows} 条，当前为 {len(rows)}")
        identity = [
            (
                row["example_id"],
                int(row["target_item_id"]),
                tuple(int(value) for value in row["history_item_ids"]),
            )
            for row in rows
        ]
        if reference is None:
            reference = identity
        elif identity != reference:
            raise ValueError(f"{path} 与其它 SFT test 文件的样本顺序或监督目标不一致")
        missing_cot = sum(not str(row.get("cot") or "").strip() for row in rows)
        if missing_cot:
            raise ValueError(f"{path} 有 {missing_cot} 条缺少 cot")
        loaded.append((checkpoint_name, path, rows))
    return loaded


def main() -> None:
    parser = argparse.ArgumentParser(
        description="复用候选 embedding，评测 embedding checkpoint × SFT checkpoint 完整矩阵。"
    )
    parser.add_argument("--embedding-run", action="append", required=True)
    parser.add_argument("--sft-eval-root", type=Path, required=True)
    parser.add_argument("--item-info", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-embedding-checkpoints", type=int, default=5)
    parser.add_argument("--expected-sft-checkpoints", type=int, default=5)
    parser.add_argument("--expected-test-rows", type=int, default=1341)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--item-batch-size", type=int, default=128)
    parser.add_argument("--query-batch-size", type=int, default=64)
    parser.add_argument("--score-batch-size", type=int, default=128)
    parser.add_argument("--ks", default="5,10,20")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    args = parser.parse_args()

    if args.seed != 42:
        parser.error("项目随机种子固定为 42")
    if not torch.cuda.is_available():
        parser.error("矩阵评测要求 CUDA GPU")

    embedding_runs = parse_embedding_runs(args.embedding_run)
    if len(embedding_runs) != 2:
        parser.error(f"本实验应传入 2 个 embedding run，当前为 {len(embedding_runs)}")
    checkpoints_by_run = {
        tag: checkpoint_dirs(path, args.expected_embedding_checkpoints)
        for tag, path in embedding_runs
    }
    test_sets = validate_sft_rows(
        sft_test_files(args.sft_eval_root.resolve(), args.expected_sft_checkpoints),
        args.expected_test_rows,
    )
    item_ids, item_texts = load_candidates(args.item_info.resolve())
    ks = [int(value) for value in args.ks.split(",") if value.strip()]
    expected_total = (
        len(embedding_runs)
        * args.expected_embedding_checkpoints
        * args.expected_sft_checkpoints
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda:0")
    started = time.time()

    for embedding_variant, _ in embedding_runs:
        for embedding_epoch, checkpoint in enumerate(
            checkpoints_by_run[embedding_variant], start=1
        ):
            pending = []
            for sft_checkpoint, test_file, rows in test_sets:
                combo_dir = (
                    args.output_root
                    / embedding_variant
                    / checkpoint.name
                    / sft_checkpoint
                )
                metrics_path = combo_dir / "retrieval_metrics.json"
                ranks_path = combo_dir / "retrieval_ranks.jsonl"
                result = existing_result(
                    metrics_path,
                    ranks_path,
                    embedding_checkpoint=checkpoint,
                    test_file=test_file,
                    expected_rows=args.expected_test_rows,
                )
                if result is None:
                    pending.append(
                        (sft_checkpoint, test_file, rows, metrics_path, ranks_path)
                    )

            if not pending:
                rows_done = write_summary(args.output_root)
                print(
                    json.dumps(
                        {
                            "event": "embedding_checkpoint_already_complete",
                            "embedding_variant": embedding_variant,
                            "embedding_checkpoint": checkpoint.name,
                            "completed_combinations": len(rows_done),
                            "expected_combinations": expected_total,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                continue

            tokenizer = AutoTokenizer.from_pretrained(
                checkpoint,
                trust_remote_code=True,
                use_fast=True,
                padding_side="left",
            )
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            item_audit = token_length_audit(
                tokenizer, item_texts, args.max_length, "items"
            )
            model = AutoModel.from_pretrained(
                checkpoint,
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

            for sft_checkpoint, test_file, rows, metrics_path, ranks_path in pending:
                query_texts = [format_query(str(row["query"])) for row in rows]
                query_audit = token_length_audit(
                    tokenizer, query_texts, args.max_length, "queries"
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
                result = {
                    "embedding_variant": embedding_variant,
                    "embedding_checkpoint": checkpoint.name,
                    "embedding_epoch": embedding_epoch,
                    "sft_checkpoint": sft_checkpoint,
                    "checkpoint": str(checkpoint.resolve()),
                    "test_file": str(test_file.resolve()),
                    "item_info": str(args.item_info.resolve()),
                    "evaluated": len(rows),
                    "num_candidates": len(item_ids),
                    "max_length": args.max_length,
                    "query_truncation": QUERY_TRUNCATION,
                    "item_text": "format_positive_desc256_details256",
                    "mask_history_items": True,
                    "seed": args.seed,
                    "token_audit": [item_audit, query_audit],
                    "metrics": metrics_from_ranks(ranks, ks),
                    **mask_audit,
                }
                write_json_atomic(metrics_path, result)
                write_jsonl_atomic(ranks_path, rank_rows)
                rows_done = write_summary(args.output_root)
                print(
                    json.dumps(
                        {
                            "event": "combination_complete",
                            **summary_row(result),
                            "completed_combinations": len(rows_done),
                            "expected_combinations": expected_total,
                            "elapsed_seconds": round(time.time() - started, 1),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                del query_embeddings

            del item_embeddings, model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()

    final_rows = write_summary(args.output_root)
    if len(final_rows) != expected_total:
        raise RuntimeError(
            f"矩阵评测应完成 {expected_total} 个组合，当前汇总为 {len(final_rows)}"
        )
    print(
        json.dumps(
            {
                "event": "matrix_complete",
                "completed_combinations": len(final_rows),
                "expected_combinations": expected_total,
                "output": str((args.output_root / "all_metrics.json").resolve()),
                "elapsed_seconds": round(time.time() - started, 1),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
