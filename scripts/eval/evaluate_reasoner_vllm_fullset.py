#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from transformers import AutoTokenizer

from rubric_cot_pipeline.embeddings import (
    DEFAULT_RECOMMENDATION_QUERY_INSTRUCTION,
    Qwen3TextEmbedder,
    append_recommendation_reasoning,
)
from rubric_cot_pipeline.io import read_jsonl
from rubric_cot_pipeline.prompts import COT_SYSTEM, build_user_prompt
from scripts.eval.evaluate_reasoner_fullset_proxy import (
    build_item_text,
    counts,
    history_text,
    norm,
    rank_target,
    rank_target_embedding_from_emb,
    update_metrics,
)


def parse_csv_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def truncate_prompt(tokenizer, prompt: str, max_prompt_tokens: int) -> str:
    if max_prompt_tokens <= 0:
        return prompt
    old_side = getattr(tokenizer, "truncation_side", "right")
    tokenizer.truncation_side = "left"
    try:
        encoded = tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=max_prompt_tokens,
        )
        return tokenizer.decode(encoded["input_ids"], skip_special_tokens=False)
    finally:
        tokenizer.truncation_side = old_side


def build_prompts(tokenizer, user_histories: list[str], category: str, max_prompt_tokens: int) -> list[str]:
    prompts = []
    for user_history in user_histories:
        messages = [
            {"role": "system", "content": COT_SYSTEM},
            {"role": "user", "content": build_user_prompt(user_history, category)},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(truncate_prompt(tokenizer, prompt, max_prompt_tokens))
    return prompts


def cleanup_vllm(llm: Any) -> None:
    try:
        executor = getattr(getattr(llm, "llm_engine", None), "model_executor", None)
        if executor is not None and hasattr(executor, "shutdown"):
            executor.shutdown()
    except Exception:
        pass
    try:
        from vllm.distributed.parallel_state import destroy_model_parallel

        destroy_model_parallel()
    except Exception:
        pass
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def generate_with_vllm(args, user_histories: list[str]) -> list[str]:
    from vllm import LLM, SamplingParams

    tokenizer_path = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompts = build_prompts(tokenizer, user_histories, args.category, args.max_prompt_tokens)

    llm_kwargs: dict[str, Any] = {
        "model": args.model,
        "tokenizer": tokenizer_path,
        "trust_remote_code": True,
        "tensor_parallel_size": args.tensor_parallel_size,
        "dtype": args.vllm_dtype,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.vllm_max_model_len,
        "max_num_seqs": args.vllm_max_num_seqs,
        "disable_log_stats": True,
    }
    if args.enforce_eager:
        llm_kwargs["enforce_eager"] = True
    if args.max_num_batched_tokens > 0:
        llm_kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens

    llm = LLM(**llm_kwargs)
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        stop=args.stop or None,
    )

    cots: list[str] = []
    try:
        for start in range(0, len(prompts), args.generation_batch_size):
            batch_prompts = prompts[start : start + args.generation_batch_size]
            outputs = llm.generate(batch_prompts, sampling_params=sampling, use_tqdm=False)
            for output in outputs:
                text = output.outputs[0].text if output.outputs else ""
                cots.append(text.strip())
            print(f"generated {min(start + len(batch_prompts), len(prompts))}/{len(prompts)}", flush=True)
    finally:
        cleanup_vllm(llm)
    return cots


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", default="")
    parser.add_argument("--item-info", default="")
    parser.add_argument("--category", required=True)
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", default="")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--max-history-items", type=int, default=20)
    parser.add_argument("--max-prompt-tokens", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--generation-batch-size", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--stop", action="append", default=[])
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--vllm-dtype", default="bfloat16")
    parser.add_argument("--vllm-max-model-len", type=int, default=4096)
    parser.add_argument("--vllm-max-num-seqs", type=int, default=64)
    parser.add_argument("--max-num-batched-tokens", type=int, default=0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--ks", default="5,10,20,100")
    parser.add_argument("--scorer", choices=["lexical", "qwen3_embedding"], default="qwen3_embedding")
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--embedding-max-length", type=int, default=2048)
    parser.add_argument("--embedding-batch-size", type=int, default=128)
    parser.add_argument("--embedding-output-dim", type=int, default=0)
    parser.add_argument("--embedding-device", default=os.getenv("QWEN3_EMBEDDING_DEVICE", "cuda:0"))
    parser.add_argument("--embedding-torch-dtype", default="bfloat16")
    parser.add_argument("--query-instruction", default=DEFAULT_RECOMMENDATION_QUERY_INSTRUCTION)
    parser.add_argument("--output", required=True)
    parser.add_argument("--predictions-output", required=True)
    args = parser.parse_args()

    if args.generation_batch_size < 1:
        raise ValueError("--generation-batch-size must be >= 1")
    if args.tensor_parallel_size < 1:
        raise ValueError("--tensor-parallel-size must be >= 1")

    examples_path = Path(args.examples) if args.examples else Path("github_artifacts") / args.category / "rrec_eval" / f"{args.split}.jsonl"
    item_info_path = Path(args.item_info) if args.item_info else Path("github_artifacts") / args.category / "rrec_eval" / "item_info.jsonl"
    if not examples_path.exists():
        raise FileNotFoundError(f"Examples JSONL not found: {examples_path}")
    if not item_info_path.exists():
        raise FileNotFoundError(f"Item info JSONL not found: {item_info_path}")

    all_rows = list(read_jsonl(examples_path))
    if args.max_examples > 0:
        all_rows = all_rows[: args.max_examples]
    if not all_rows:
        raise ValueError(f"No examples loaded from {examples_path}")

    item_map = {int(row["item_id"]): row for row in read_jsonl(item_info_path)}
    item_ids: list[int] = []
    item_id_to_index: dict[int, int] = {}
    item_texts: list[str] = []
    item_vecs = []
    item_norms = []
    for item_id, item in sorted(item_map.items()):
        item_text = build_item_text(item, str(item.get("title", "")), max_chars=1200)
        item_id_to_index[item_id] = len(item_ids)
        item_ids.append(item_id)
        item_texts.append(item_text)
        if args.scorer == "lexical":
            vec = counts(item_text)
            item_vecs.append(vec)
            item_norms.append(norm(vec))

    user_histories = [
        history_text(
            args.category,
            [str(x) for x in row.get("history_item_title", [])],
            [float(x) for x in row.get("history_rating", [])],
            args.max_history_items,
        )
        for row in all_rows
    ]
    target_ids = [int(row["item_id"]) for row in all_rows]
    cots = generate_with_vllm(args, user_histories)
    if len(cots) != len(all_rows):
        raise RuntimeError(f"Generated CoT count mismatch: cots={len(cots)} rows={len(all_rows)}")
    reasoner_queries = [append_recommendation_reasoning(history, cot) for history, cot in zip(user_histories, cots)]

    if args.scorer == "qwen3_embedding":
        embedder = Qwen3TextEmbedder(
            args.embedding_model,
            max_length=args.embedding_max_length,
            batch_size=args.embedding_batch_size,
            torch_dtype=args.embedding_torch_dtype,
            device=args.embedding_device,
            query_instruction=args.query_instruction,
            output_dim=args.embedding_output_dim,
        )
        item_embs = embedder.encode_documents(item_texts)
        baseline_query_embs = embedder.encode_queries(user_histories)
        reasoner_query_embs = embedder.encode_queries(reasoner_queries)
    else:
        embedder = None
        item_embs = None
        baseline_query_embs = None
        reasoner_query_embs = None

    ks = parse_csv_ints(args.ks)
    metric_keys = [f"{prefix}_{metric}@{k}" for prefix in ("baseline", "reasoner") for metric in ("HR", "NDCG") for k in ks]
    totals = {key: 0.0 for key in metric_keys}

    pred_path = Path(args.predictions_output)
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    with pred_path.open("w", encoding="utf-8") as f:
        for index, row in enumerate(all_rows, start=1):
            target_id = target_ids[index - 1]
            if args.scorer == "lexical":
                baseline_rank = rank_target(user_histories[index - 1], item_ids, item_vecs, item_norms, target_id)
                reasoner_rank = rank_target(reasoner_queries[index - 1], item_ids, item_vecs, item_norms, target_id)
            else:
                target_index = item_id_to_index.get(target_id)
                if target_index is None:
                    baseline_rank = len(item_ids) + 1
                    reasoner_rank = len(item_ids) + 1
                else:
                    baseline_rank = rank_target_embedding_from_emb(baseline_query_embs[index - 1], item_embs, target_index)  # type: ignore[arg-type]
                    reasoner_rank = rank_target_embedding_from_emb(reasoner_query_embs[index - 1], item_embs, target_index)  # type: ignore[arg-type]

            update_metrics(totals, "baseline", baseline_rank, ks)
            update_metrics(totals, "reasoner", reasoner_rank, ks)
            f.write(
                json.dumps(
                    {
                        "category": args.category,
                        "split": args.split,
                        "index": index,
                        "global_index": index,
                        "user_id": row.get("user_id"),
                        "target_item_id": target_id,
                        "target_item_title": row.get("item_title", ""),
                        "baseline_rank": baseline_rank,
                        "reasoner_rank": reasoner_rank,
                        "cot": cots[index - 1],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            if index % 100 == 0:
                print(f"ranked {index}/{len(all_rows)}", flush=True)

    n = max(1, len(all_rows))
    metrics = {key: value / n for key, value in totals.items()}
    for k in ks:
        metrics[f"delta_NDCG@{k}"] = metrics[f"reasoner_NDCG@{k}"] - metrics[f"baseline_NDCG@{k}"]
        metrics[f"delta_HR@{k}"] = metrics[f"reasoner_HR@{k}"] - metrics[f"baseline_HR@{k}"]

    result = {
        "category": args.category,
        "split": args.split,
        "examples": str(examples_path),
        "item_info": str(item_info_path),
        "model": args.model,
        "tokenizer": args.tokenizer or args.model,
        "run_name": args.run_name or Path(args.model).name,
        "evaluated": len(all_rows),
        "num_items": len(item_ids),
        "metrics": metrics,
        "scorer": args.scorer,
        "embedding_model": args.embedding_model if args.scorer == "qwen3_embedding" else None,
        "vllm": {
            "tensor_parallel_size": args.tensor_parallel_size,
            "dtype": args.vllm_dtype,
            "max_model_len": args.vllm_max_model_len,
            "max_num_seqs": args.vllm_max_num_seqs,
            "generation_batch_size": args.generation_batch_size,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
        },
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
