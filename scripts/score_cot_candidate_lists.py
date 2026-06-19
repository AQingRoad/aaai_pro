#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures as futures
import http.client
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rubric_cot_pipeline.io import ensure_parent
from rubric_cot_pipeline.judge_api import _extract_score_from_chat_response
from rubric_cot_pipeline.prompts import ANSWER_TAG, REASONING_TAG, build_judge_messages


def candidate_key(row: dict[str, Any], candidate: dict[str, Any]) -> tuple[str, int]:
    example_id = str(row.get("example_id") or row.get("user_id") or row.get("id") or "")
    cand_idx = int(candidate.get("candidate_index"))
    return example_id, cand_idx


def build_cot(candidate: dict[str, Any]) -> str:
    think = str(candidate.get("think") or "").strip()
    answer = str(candidate.get("answer") or "").strip()
    return f"<{REASONING_TAG}>\n{think}\n</{REASONING_TAG}>\n<{ANSWER_TAG}>\n{answer}\n</{ANSWER_TAG}>"


def build_target_item(row: dict[str, Any], use_target: bool = True) -> str:
    if not use_target:
        return ""
    return (
        f"Target title: {row.get('target_item_title', '')}\n"
        f"Target item text: {row.get('target_item_text', '')}"
    ).strip()


def read_candidate_tasks(input_path: Path, max_examples: int = 0, max_candidates: int = 0) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    tasks: list[tuple[dict[str, Any], dict[str, Any]]] = []
    examples_seen = 0
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            examples_seen += 1
            candidates = row.get("candidates")
            if not isinstance(candidates, list):
                continue
            for candidate in candidates:
                if str(candidate.get("answer") or "").strip():
                    tasks.append((row, candidate))
                    if max_candidates and len(tasks) >= max_candidates:
                        return tasks
            if max_examples and examples_seen >= max_examples:
                break
    return tasks


def load_completed(path: Path) -> set[tuple[str, int]]:
    done: set[tuple[str, int]] = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                example_id = str(row.get("example_id") or "")
                cand_idx = int(row.get("candidate_index"))
            except Exception:
                continue
            if example_id and row.get("rubric_score"):
                done.add((example_id, cand_idx))
    return done


def append_jsonl_locked(path: Path, row: dict[str, Any], lock: threading.Lock) -> None:
    text = json.dumps(row, ensure_ascii=False) + "\n"
    with lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(text)
            f.flush()


def score_with_rule(row: dict[str, Any], candidate: dict[str, Any]) -> tuple[dict[str, Any], str, dict[str, Any]]:
    from rubric_cot_pipeline.rubric import rule_score

    score = rule_score(str(row.get("user_history") or ""), build_cot(candidate))
    return score, json.dumps(score, ensure_ascii=False), {}


def score_with_api(row: dict[str, Any], candidate: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, Any], str, dict[str, Any]]:
    payload: dict[str, Any] = {
        "model": args.judge_model,
        "messages": build_judge_messages(
            str(row.get("user_history") or ""),
            build_cot(candidate),
            build_target_item(row, args.use_target),
        ),
        "temperature": args.judge_temperature,
        "top_p": args.judge_top_p,
        "max_tokens": args.judge_max_tokens,
        "response_format": {"type": "json_object"},
    }
    if args.judge_thinking:
        payload["thinking"] = {"type": args.judge_thinking}
    if args.judge_reasoning_effort:
        payload["reasoning_effort"] = args.judge_reasoning_effort

    url = args.judge_base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if args.judge_api_key:
        headers["Authorization"] = f"Bearer {args.judge_api_key}"
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    last_error = ""
    for attempt in range(args.judge_max_retries + 1):
        try:
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            start = time.perf_counter()
            with urllib.request.urlopen(req, timeout=args.judge_timeout) as resp:
                raw = resp.read().decode("utf-8")
            elapsed = time.perf_counter() - start
            score = _extract_score_from_chat_response(raw)
            if not score:
                raise ValueError("judge response did not contain a valid rubric JSON")
            obj = json.loads(raw)
            meta = {
                "judge_elapsed_seconds": round(elapsed, 6),
                "judge_finish_reason": obj.get("choices", [{}])[0].get("finish_reason"),
                "judge_usage": obj.get("usage", {}),
            }
            return score, raw, meta
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            last_error = f"HTTPError {exc.code}: {body[:1000]}"
            if attempt < args.judge_max_retries:
                if exc.code == 429:
                    sleep_s = max(args.retry_sleep_seconds, 20.0 * (attempt + 1))
                elif exc.code in {500, 502, 503, 504}:
                    sleep_s = max(args.retry_sleep_seconds, 10.0 * (attempt + 1))
                else:
                    sleep_s = args.retry_sleep_seconds
                time.sleep(sleep_s)
        except (urllib.error.URLError, TimeoutError, http.client.RemoteDisconnected, json.JSONDecodeError, KeyError, IndexError, ValueError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < args.judge_max_retries:
                time.sleep(args.retry_sleep_seconds)
    raise RuntimeError(f"judge API failed after retries: {last_error}")


def score_task(row: dict[str, Any], candidate: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    example_id, cand_idx = candidate_key(row, candidate)
    if args.judge_provider == "rule":
        score, raw, meta = score_with_rule(row, candidate)
        provider = "rule"
    else:
        score, raw, meta = score_with_api(row, candidate, args)
        provider = args.judge_provider
    return {
        "example_id": example_id,
        "candidate_id": candidate.get("candidate_id") or f"{example_id}-{cand_idx}",
        "candidate_index": cand_idx,
        "temperature": candidate.get("temperature"),
        "target_item_title": row.get("target_item_title", ""),
        "judge_used_target": bool(args.use_target),
        "judge_provider": provider,
        "judge_model": args.judge_model if provider != "rule" else "rule_score",
        "rubric_score": score,
        "judge_meta": meta,
        "judge_raw": raw if args.save_raw else "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score split think/answer CoT candidate lists with target-aware rubric judge.")
    parser.add_argument("--input", required=True, help="Input candidate-list JSONL generated by generate_cot_candidate_lists.py")
    parser.add_argument("--output", required=True, help="Output candidate-level rubric score JSONL")
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--max-candidates", type=int, default=0)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-target", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-raw", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--judge-provider", choices=["rule", "openai_compatible", "openai", "chat_completions"], default="openai_compatible")
    parser.add_argument("--judge-base-url", default=os.getenv("RUBRIC_JUDGE_API_BASE_URL", ""))
    parser.add_argument("--judge-api-key", default=os.getenv("RUBRIC_JUDGE_API_KEY", ""))
    parser.add_argument("--judge-model", default=os.getenv("RUBRIC_JUDGE_API_MODEL", "deepseek-v4-pro"))
    parser.add_argument("--judge-timeout", type=float, default=float(os.getenv("RUBRIC_JUDGE_API_TIMEOUT", "300")))
    parser.add_argument("--judge-max-retries", type=int, default=int(os.getenv("RUBRIC_JUDGE_API_MAX_RETRIES", "5")))
    parser.add_argument("--judge-temperature", type=float, default=0.0)
    parser.add_argument("--judge-top-p", type=float, default=0.9)
    parser.add_argument("--judge-max-tokens", type=int, default=int(os.getenv("RUBRIC_JUDGE_API_MAX_TOKENS", "1024")))
    parser.add_argument("--judge-thinking", default=os.getenv("RUBRIC_JUDGE_API_THINKING", "disabled"))
    parser.add_argument("--judge-reasoning-effort", default=os.getenv("RUBRIC_JUDGE_API_REASONING_EFFORT", ""))
    parser.add_argument("--min-interval", type=float, default=2.5, help="Sleep seconds after each completed/failed score in the main loop.")
    parser.add_argument("--retry-sleep-seconds", type=float, default=10.0)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = ensure_parent(args.output)
    failure_path = output_path.with_name(output_path.stem + ".failures" + output_path.suffix)
    if args.judge_provider != "rule" and not args.judge_base_url:
        raise ValueError("--judge-base-url is required for API judge")
    if args.judge_provider != "rule" and not args.judge_model:
        raise ValueError("--judge-model is required for API judge")

    tasks_all = read_candidate_tasks(input_path, args.max_examples, args.max_candidates)
    done = load_completed(output_path) if args.resume else set()
    tasks = [(row, cand) for row, cand in tasks_all if candidate_key(row, cand) not in done]
    print(
        f"loaded_candidates={len(tasks_all)} skipped_completed={len(done)} pending={len(tasks)} "
        f"output={output_path} failures={failure_path}",
        flush=True,
    )
    if not tasks:
        return

    lock = threading.Lock()
    completed = 0
    failed = 0
    with futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        future_map = {pool.submit(score_task, row, cand, args): (row, cand) for row, cand in tasks}
        for fut in futures.as_completed(future_map):
            row, cand = future_map[fut]
            example_id, cand_idx = candidate_key(row, cand)
            try:
                out = fut.result()
                append_jsonl_locked(output_path, out, lock)
                completed += 1
                score = out.get("rubric_score") or {}
                print(
                    f"completed {completed}/{len(tasks)} example={example_id} cand={cand_idx} "
                    f"total={score.get('total')} norm={score.get('score_norm')}",
                    flush=True,
                )
            except Exception as exc:
                failed += 1
                failure = {
                    "example_id": example_id,
                    "candidate_index": cand_idx,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "time": time.time(),
                }
                append_jsonl_locked(failure_path, failure, lock)
                print(f"failed example={example_id} cand={cand_idx}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            if args.min_interval > 0:
                time.sleep(args.min_interval)

    print(f"done completed={completed} failed={failed} output={output_path}", flush=True)
    if failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
