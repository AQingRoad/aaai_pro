#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures as futures
import http.client
import json
import os
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rubric_cot_pipeline.io import ensure_parent, read_jsonl
from rubric_cot_pipeline.prompts import ANSWER_TAG, REASONING_TAG, build_generation_messages, normalize_cot_tags


OPENAI_COMPATIBLE_PROVIDERS = {"openai", "openai_compatible", "chat_completions"}
GLM_CODEPLAN_PROVIDERS = {"glm_codeplan", "bigmodel_codeplan", "zhipu_codeplan", "zai_codeplan"}
GLM_CODEPLAN_BASE_URL = "https://open.bigmodel.cn/api/coding/paas/v4"
GLM_CODEPLAN_MODEL = "glm-5.2"
API_KEY_ENV_NAMES = (
    "COT_GENERATION_API_KEY",
    "BIGMODEL_API_KEY",
    "ZAI_API_KEY",
    "ZHIPUAI_API_KEY",
    "ZHIPU_API_KEY",
    "GLM_API_KEY",
)


def first_env(*names: str) -> str:
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return ""


def cli_arg_supplied(name: str) -> bool:
    return name in sys.argv[1:]


def chat_completions_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"


def resolve_api_args(args: argparse.Namespace) -> None:
    args.api_provider = args.api_provider.strip().lower()
    if args.api_provider in GLM_CODEPLAN_PROVIDERS:
        if not args.api_base_url:
            args.api_base_url = GLM_CODEPLAN_BASE_URL
        if not cli_arg_supplied("--api-model") and not os.getenv("COT_GENERATION_API_MODEL"):
            args.api_model = GLM_CODEPLAN_MODEL
        if not args.api_key:
            args.api_key = first_env(*API_KEY_ENV_NAMES)
    elif not args.api_key:
        args.api_key = first_env("COT_GENERATION_API_KEY")


def throttle_api(args: argparse.Namespace) -> None:
    if args.api_min_interval <= 0:
        return
    with args._api_request_lock:
        now = time.time()
        wait = args.api_min_interval - (now - args._api_last_request_ts)
        if wait > 0:
            time.sleep(wait)
        args._api_last_request_ts = time.time()


def parse_temperatures(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def example_key(row: dict[str, Any]) -> str:
    return str(row.get("example_id") or row.get("user_id") or row.get("id") or "")


def candidate_checkpoint_path(output: str | Path) -> Path:
    path = Path(output)
    return path.with_name(path.stem + ".candidates" + path.suffix)


def failures_path(output: str | Path) -> Path:
    path = Path(output)
    return path.with_name(path.stem + ".failures" + path.suffix)


def load_candidate_checkpoint(path: Path) -> dict[str, dict[int, dict[str, Any]]]:
    done: dict[str, dict[int, dict[str, Any]]] = {}
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = str(row.get("example_id") or "")
            try:
                cand_idx = int(row.get("candidate_index"))
            except (TypeError, ValueError):
                continue
            if not key or not str(row.get("answer") or "").strip():
                continue
            done.setdefault(key, {})[cand_idx] = row
    return done


def load_existing_output_candidates(path: Path, num_candidates: int) -> dict[str, dict[int, dict[str, Any]]]:
    done: dict[str, dict[int, dict[str, Any]]] = {}
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = example_key(row)
            candidates = row.get("candidates")
            if not key or not isinstance(candidates, list):
                continue
            for c in candidates:
                try:
                    cand_idx = int(c.get("candidate_index"))
                except (AttributeError, TypeError, ValueError):
                    continue
                if 0 <= cand_idx < num_candidates and str(c.get("answer") or "").strip():
                    item = {**row, **c, "example_id": key}
                    item.pop("candidates", None)
                    item.pop("candidate_count", None)
                    item.pop("list_generation_timing", None)
                    done.setdefault(key, {})[cand_idx] = item
    return done


def merge_candidate_maps(*maps: dict[str, dict[int, dict[str, Any]]]) -> dict[str, dict[int, dict[str, Any]]]:
    merged: dict[str, dict[int, dict[str, Any]]] = {}
    for mp in maps:
        for key, candidates in mp.items():
            merged.setdefault(key, {}).update(candidates)
    return merged


def sort_and_rewrite_checkpoint(
    path: Path,
    candidate_map: dict[str, dict[int, dict[str, Any]]],
    input_rows: list[dict[str, Any]],
    num_candidates: int,
) -> int:
    """Keep candidate checkpoint deterministic: input row order, then candidate_index.

    The checkpoint is append-only during generation for crash safety. At the beginning
    of each run, rewrite it into a sorted, de-duplicated form so future resume reads
    are deterministic and easy to inspect.
    """
    ensure_parent(path)
    written = 0
    seen_keys = {example_key(row) for row in input_rows}
    ordered_keys = [example_key(row) for row in input_rows if example_key(row)]
    extra_keys = sorted(key for key in candidate_map if key and key not in seen_keys)
    with path.open("w", encoding="utf-8") as f:
        for key in ordered_keys + extra_keys:
            by_idx = candidate_map.get(key, {})
            for cand_idx in range(num_candidates):
                item = by_idx.get(cand_idx)
                if item is None:
                    continue
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                written += 1
    return written


def extract_recommendation(content: str) -> str:
    normalized = normalize_cot_tags(content)
    reasoning_block_re = re.compile(
        r"<\s*(?:hidden_reasoning|reasoning|analysis|think|thinking|thoughts)\s*>[\s\S]*?<\s*/\s*(?:hidden_reasoning|reasoning|analysis|think|thinking|thoughts)\s*>",
        re.IGNORECASE,
    )
    matches = list(reasoning_block_re.finditer(normalized))
    if matches:
        answer = normalized[matches[-1].end() :]
    else:
        lower = normalized.lower()
        answer_start = lower.rfind(f"<{ANSWER_TAG}>")
        if answer_start >= 0:
            answer = normalized[answer_start + len(ANSWER_TAG) + 2 :]
            answer_end = answer.lower().find(f"</{ANSWER_TAG}>")
            if answer_end >= 0:
                answer = answer[:answer_end]
        else:
            answer = normalized
    answer = reasoning_block_re.sub("", answer)
    answer = re.sub(
        rf"</?(?:{REASONING_TAG}|{ANSWER_TAG}|think|thinking|thoughts|answer|hidden_reasoning|reasoning|analysis)>|</?tool_call>|```[\s\S]*?```",
        "",
        answer,
        flags=re.IGNORECASE,
    ).strip()
    answer = re.sub(r"^(?:recommendation|answer|final answer|final)\s*[:：]\s*", "", answer, flags=re.IGNORECASE).strip()
    return answer


def split_api_output(content: str, reasoning: str) -> tuple[str, str]:
    think = reasoning.strip()
    answer = extract_recommendation(content)
    if not answer:
        raise ValueError("API returned empty content/answer; increase --max-new-tokens or retry")
    return think, answer


def call_api(messages: list[dict[str, str]], args: argparse.Namespace, temperature: float) -> tuple[str, str, dict[str, Any]]:
    stage_start = time.perf_counter()
    url = chat_completions_url(args.api_base_url)
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"
    payload: dict[str, Any] = {
        "model": args.api_model,
        "messages": messages,
        "temperature": temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_new_tokens,
    }
    if args.api_thinking:
        payload["thinking"] = {"type": args.api_thinking}
    if args.api_reasoning_effort:
        payload["reasoning_effort"] = args.api_reasoning_effort
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request_start = time.perf_counter()
    last_error: Exception | None = None
    retryable = (
        urllib.error.URLError,
        TimeoutError,
        http.client.RemoteDisconnected,
        KeyError,
        IndexError,
        json.JSONDecodeError,
        ValueError,
    )
    for attempt in range(args.api_max_retries + 1):
        attempt_start = time.perf_counter()
        try:
            throttle_api(args)
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=args.api_timeout) as resp:
                raw = resp.read().decode("utf-8")
            response_received = time.perf_counter()
            obj = json.loads(raw)
            parsed_json = time.perf_counter()
            choice = obj["choices"][0]
            message = choice["message"]
            content = str(message.get("content") or "").strip()
            reasoning = str(message.get("reasoning_content") or "").strip()
            think, answer = split_api_output(content, reasoning)
            parsed_cot = time.perf_counter()
            meta = {
                "timing": {
                    "stage_total_seconds": round(parsed_cot - stage_start, 6),
                    "api_request_seconds": round(response_received - attempt_start, 6),
                    "api_total_with_retries_seconds": round(response_received - request_start, 6),
                    "json_parse_seconds": round(parsed_json - response_received, 6),
                    "cot_parse_seconds": round(parsed_cot - parsed_json, 6),
                    "attempts": attempt + 1,
                },
                "api_finish_reason": str(choice.get("finish_reason", "")),
                "api_has_reasoning_content": bool(reasoning),
                "api_content_chars": len(content),
                "api_reasoning_chars": len(reasoning),
                "api_usage": obj.get("usage", {}),
            }
            return think, answer, meta
        except urllib.error.HTTPError as exc:
            try:
                body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                body = ""
            last_error = RuntimeError(f"HTTPError {exc.code}: {body[:1000]}")
            if attempt < args.api_max_retries:
                if exc.code == 429:
                    sleep_s = max(20.0, 10.0 * (attempt + 1))
                elif exc.code in {500, 502, 503, 504}:
                    sleep_s = max(10.0, 5.0 * (attempt + 1))
                else:
                    sleep_s = min(2**attempt, 8)
                time.sleep(sleep_s)
        except retryable as exc:
            last_error = exc
            if attempt < args.api_max_retries:
                time.sleep(min(2**attempt, 8))
    raise RuntimeError(f"API failed after retries: {last_error}")


def build_candidate_task(row: dict[str, Any], cand_idx: int, args: argparse.Namespace, temperatures: list[float]) -> dict[str, Any]:
    key = example_key(row)
    if not key:
        raise ValueError("row must contain example_id, user_id, or id")
    temp = temperatures[cand_idx % len(temperatures)]
    messages = build_generation_messages(row["user_history"], row.get("category", ""))
    cand_start = time.perf_counter()
    think, answer, meta = call_api(messages, args, temp)
    cand_end = time.perf_counter()
    meta.setdefault("timing", {})["candidate_total_seconds"] = round(cand_end - cand_start, 6)
    return {
        "example_id": key,
        "candidate_id": f"{key}-{cand_idx}",
        "candidate_index": cand_idx,
        "temperature": temp,
        "think": think,
        "answer": answer,
        "generator_model": args.api_model,
        "generation_mode": "api",
        "generation_timing": meta.get("timing", {}),
        "generation_api_meta": {k: v for k, v in meta.items() if k != "timing"},
    }


def append_jsonl_locked(path: Path, row: dict[str, Any], lock: threading.Lock) -> None:
    text = json.dumps(row, ensure_ascii=False) + "\n"
    with lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(text)
            f.flush()


def aggregate_output(input_rows: list[dict[str, Any]], candidate_map: dict[str, dict[int, dict[str, Any]]], output: Path, num_candidates: int) -> int:
    output = ensure_parent(output)
    written = 0
    with output.open("w", encoding="utf-8") as f:
        for row in input_rows:
            key = example_key(row)
            by_idx = candidate_map.get(key, {})
            candidates = [by_idx[i] for i in range(num_candidates) if i in by_idx]
            if not candidates:
                continue
            out = {
                **row,
                "example_id": key,
                "candidate_count": len(candidates),
                "candidates": candidates,
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            written += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate per-example CoT candidate lists with candidate-level checkpoint and resume support.")
    parser.add_argument("--input", required=True, help="Examples JSONL in standard pipeline schema.")
    parser.add_argument("--output", required=True, help="Final output JSONL; each row contains candidates list with split think/answer fields.")
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--num-candidates", type=int, default=4)
    parser.add_argument("--temperatures", default="0.6,0.8,1.0,1.1")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--aggregate-every", type=int, default=100, help="Rewrite final output every N successful candidates; 0 disables periodic aggregation.")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--api-provider", default=os.getenv("COT_GENERATION_API_PROVIDER", "openai_compatible"))
    parser.add_argument("--api-base-url", default=os.getenv("COT_GENERATION_API_BASE_URL", ""))
    parser.add_argument("--api-key", default=os.getenv("COT_GENERATION_API_KEY", ""))
    parser.add_argument("--api-model", default=os.getenv("COT_GENERATION_API_MODEL", "glm-5-1"))
    parser.add_argument("--api-timeout", type=float, default=float(os.getenv("COT_GENERATION_API_TIMEOUT", "180")))
    parser.add_argument("--api-max-retries", type=int, default=int(os.getenv("COT_GENERATION_API_MAX_RETRIES", "3")))
    parser.add_argument("--api-min-interval", type=float, default=float(os.getenv("COT_GENERATION_API_MIN_INTERVAL", "0")))
    parser.add_argument("--api-thinking", default=os.getenv("COT_GENERATION_API_THINKING", "enabled"))
    parser.add_argument("--api-reasoning-effort", default=os.getenv("COT_GENERATION_API_REASONING_EFFORT", ""))
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-prompt-tokens", type=int, default=2048)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    resolve_api_args(args)
    args._api_request_lock = threading.Lock()
    args._api_last_request_ts = 0.0

    if args.api_provider not in OPENAI_COMPATIBLE_PROVIDERS | GLM_CODEPLAN_PROVIDERS:
        raise ValueError(f"Unsupported API provider: {args.api_provider}")
    if not args.api_base_url:
        raise ValueError("--api-base-url is required")
    if not args.api_model:
        raise ValueError("--api-model is required")

    random.seed(args.seed)
    temperatures = parse_temperatures(args.temperatures) or [0.7]
    output_path = ensure_parent(args.output)
    checkpoint_path = candidate_checkpoint_path(output_path)
    failure_path = failures_path(output_path)
    input_rows = list(read_jsonl(args.input, limit=args.max_examples))

    existing_from_output = load_existing_output_candidates(output_path, args.num_candidates) if args.resume else {}
    existing_from_checkpoint = load_candidate_checkpoint(checkpoint_path) if args.resume else {}
    candidate_map = merge_candidate_maps(existing_from_output, existing_from_checkpoint)
    sorted_checkpoint_count = sort_and_rewrite_checkpoint(checkpoint_path, candidate_map, input_rows, args.num_candidates)

    tasks: list[tuple[dict[str, Any], int]] = []
    for row in input_rows:
        key = example_key(row)
        have = candidate_map.get(key, {})
        for cand_idx in range(args.num_candidates):
            if cand_idx not in have:
                tasks.append((row, cand_idx))

    print(
        f"loaded_examples={len(input_rows)} existing_candidates={sum(len(v) for v in candidate_map.values())} "
        f"sorted_checkpoint_candidates={sorted_checkpoint_count} pending_candidates={len(tasks)} "
        f"checkpoint={checkpoint_path} output={output_path}",
        flush=True,
    )
    if not tasks:
        written = aggregate_output(input_rows, candidate_map, output_path, args.num_candidates)
        print(f"no pending candidates; aggregated_rows={written} output={output_path}", flush=True)
        return

    lock = threading.Lock()
    completed = 0
    failed = 0
    with futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        future_map = {pool.submit(build_candidate_task, row, cand_idx, args, temperatures): (row, cand_idx) for row, cand_idx in tasks}
        for fut in futures.as_completed(future_map):
            src, cand_idx = future_map[fut]
            key = example_key(src)
            try:
                item = fut.result()
                append_jsonl_locked(checkpoint_path, item, lock)
                candidate_map.setdefault(key, {})[cand_idx] = item
                completed += 1
                if completed % 10 == 0 or completed == 1:
                    print(f"completed_candidates={completed}/{len(tasks)} example={key} cand={cand_idx}", flush=True)
                if args.aggregate_every > 0 and completed % args.aggregate_every == 0:
                    written = aggregate_output(input_rows, candidate_map, output_path, args.num_candidates)
                    complete_examples = sum(1 for row in input_rows if len(candidate_map.get(example_key(row), {})) >= args.num_candidates)
                    print(
                        f"periodic_aggregate completed_candidates={completed} aggregated_rows={written} "
                        f"complete_examples={complete_examples}/{len(input_rows)} output={output_path}",
                        flush=True,
                    )
            except Exception as exc:
                failed += 1
                failure = {
                    "example_id": key,
                    "candidate_index": cand_idx,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "time": time.time(),
                }
                append_jsonl_locked(failure_path, failure, lock)
                print(f"failed example={key} cand={cand_idx}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)

    written = aggregate_output(input_rows, candidate_map, output_path, args.num_candidates)
    complete_examples = sum(1 for row in input_rows if len(candidate_map.get(example_key(row), {})) >= args.num_candidates)
    print(
        f"done completed_candidates={completed} failed_candidates={failed} "
        f"aggregated_rows={written} complete_examples={complete_examples}/{len(input_rows)} output={output_path}",
        flush=True,
    )
    if failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
