#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import random
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


PROMPT_VERSION = "cot_preference_listwise_v2_history_only"
DIMENSIONS = (
    "history_grounding",
    "preference_specificity",
    "transition_reasoning",
    "discriminative_constraints",
    "factual_support",
    "conciseness",
)
UTILITY_WEIGHTS = {
    "history_grounding": 0.30,
    "preference_specificity": 0.15,
    "transition_reasoning": 0.15,
    "discriminative_constraints": 0.25,
    "factual_support": 0.10,
    "conciseness": 0.05,
}


SYSTEM_PROMPT = (
    "You are a strict listwise evaluator of recommendation reasoning. "
    "Use only the observed user history. Return one compact JSON object and no prose."
)


USER_TEMPLATE = """\
Rank the candidate reasoning texts for the same user history.

User history:
<history>
{history}
</history>

Candidates:
{candidates}

Rules:
- Do not infer or reward any held-out target item. No target is provided.
- Prefer reasoning that cites concrete recurring evidence from the history and converts it into a transferable next-item profile.
- Prefer narrow inclusion and exclusion constraints that would separate suitable from unsuitable catalog items.
- Penalize unsupported creators, labels, eras, regions, awards, popularity, rarity, collector status, or exact-item guesses.
- Penalize keyword stacking, copied title lists, repeated claims, missing analysis, and invalid <think>/<answer> structure.
- Judge content and evidence. Do not reward polished wording by itself.
- Every score is an integer from 1 to 5, where 5 is best.
- ranking must contain every candidate label exactly once, best first.

Score dimensions:
- history_grounding: uses concrete evidence and repeated patterns from the observed history.
- preference_specificity: forms a specific transferable profile rather than a broad category.
- transition_reasoning: explains how history evidence leads to next-item attributes.
- discriminative_constraints: states useful inclusion or exclusion boundaries for retrieval.
- factual_support: avoids unsupported claims and metadata hallucinations.
- conciseness: covers the needed evidence without repetition or filler.

Return JSON only:
{{
  "ranking": ["C1", "C2"],
  "scores": {{
    "C1": {{
      "history_grounding": 1,
      "preference_specificity": 1,
      "transition_reasoning": 1,
      "discriminative_constraints": 1,
      "factual_support": 1,
      "conciseness": 1,
      "format_valid": true
    }}
  }}
}}
"""


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield line_number, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc


def stable_seed(value: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def candidate_payload(row: dict[str, Any], seed: int) -> tuple[str, dict[str, str]]:
    candidates = list(row.get("candidates") or [])
    random.Random(stable_seed(str(row.get("example_id")), seed)).shuffle(candidates)
    token_to_id: dict[str, str] = {}
    blocks = []
    for index, candidate in enumerate(candidates, start=1):
        token = f"C{index}"
        token_to_id[token] = str(candidate.get("candidate_id") or "")
        blocks.append(f"<{token}>\n{str(candidate.get('text') or '').strip()}\n</{token}>")
    return "\n\n".join(blocks), token_to_id


def messages_for_row(row: dict[str, Any], seed: int) -> tuple[list[dict[str, str]], dict[str, str]]:
    rendered_candidates, token_to_id = candidate_payload(row, seed)
    return (
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_TEMPLATE.format(
                    history=str(row.get("user_history") or "").strip(),
                    candidates=rendered_candidates,
                ),
            },
        ],
        token_to_id,
    )


def extract_json_object(response: dict[str, Any]) -> dict[str, Any]:
    content = response.get("choices", [{}])[0].get("message", {}).get("content")
    if isinstance(content, list):
        content = "".join(str(part.get("text") or "") if isinstance(part, dict) else str(part) for part in content)
    text = str(content or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lstrip().startswith("json"):
            text = text.lstrip()[4:].lstrip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("judge response does not contain a JSON object")
    parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("judge response JSON must be an object")
    return parsed


def normalized_judgment(raw: dict[str, Any], token_to_id: dict[str, str]) -> dict[str, Any]:
    ranking = raw.get("ranking")
    scores = raw.get("scores")
    expected = set(token_to_id)
    if not isinstance(ranking, list) or len(ranking) != len(expected) or set(map(str, ranking)) != expected:
        raise ValueError(f"ranking must contain {sorted(expected)} exactly once")
    if not isinstance(scores, dict):
        raise ValueError("scores must be an object")

    candidate_scores = []
    for token in ranking:
        token = str(token)
        values = scores.get(token)
        if not isinstance(values, dict):
            raise ValueError(f"missing score object for {token}")
        normalized: dict[str, Any] = {}
        for dimension in DIMENSIONS:
            value = values.get(dimension)
            if not isinstance(value, (int, float)) or int(value) != value or not 1 <= int(value) <= 5:
                raise ValueError(f"{token}.{dimension} must be an integer from 1 to 5")
            normalized[dimension] = int(value)
        normalized["format_valid"] = bool(values.get("format_valid"))
        normalized["utility"] = sum(normalized[name] * weight for name, weight in UTILITY_WEIGHTS.items()) / 5.0
        candidate_scores.append(
            {
                "candidate_id": token_to_id[token],
                "rank": len(candidate_scores) + 1,
                **normalized,
            }
        )
    return {
        "ranking": [token_to_id[str(token)] for token in ranking],
        "candidate_scores": candidate_scores,
    }


def request_judgment(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    messages, token_to_id = messages_for_row(row, args.seed)
    payload: dict[str, Any] = {
        "model": args.judge_model,
        "messages": messages,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "response_format": {"type": "json_object"},
    }
    if args.thinking:
        payload["thinking"] = {"type": args.thinking}
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"
    url = args.base_url.rstrip("/") + "/chat/completions"
    last_error = ""
    for attempt in range(args.max_retries + 1):
        try:
            request = urllib.request.Request(url, data=data, headers=headers, method="POST")
            started = time.perf_counter()
            with urllib.request.urlopen(request, timeout=args.timeout) as response:
                response_text = response.read().decode("utf-8")
            elapsed = time.perf_counter() - started
            response_obj = json.loads(response_text)
            judgment = normalized_judgment(extract_json_object(response_obj), token_to_id)
            return {
                "example_id": row.get("example_id"),
                "fold": row.get("fold"),
                "judge_model": args.judge_model,
                "judge_prompt_version": PROMPT_VERSION,
                "judge_used_target": False,
                "elapsed_seconds": round(elapsed, 6),
                "usage": response_obj.get("usage", {}),
                **judgment,
                "judge_raw": response_text if args.save_raw else "",
            }
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            last_error = f"HTTPError {exc.code}: {body[:1000]}"
            if attempt < args.max_retries:
                time.sleep(args.retry_sleep_seconds * (attempt + 1))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, IndexError, ValueError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < args.max_retries:
                time.sleep(args.retry_sleep_seconds * (attempt + 1))
    raise RuntimeError(last_error)


def completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        str(row.get("example_id"))
        for _, row in read_jsonl(path)
        if row.get("example_id") and row.get("ranking")
    }


def append_locked(path: Path, row: dict[str, Any], lock: threading.Lock) -> None:
    with lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Listwise-judge same-history CoT preference groups.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-url", default=os.getenv("RUBRIC_JUDGE_API_BASE_URL", ""))
    parser.add_argument("--api-key", default=os.getenv("RUBRIC_JUDGE_API_KEY", ""))
    parser.add_argument("--judge-model", default=os.getenv("RUBRIC_JUDGE_API_MODEL", "glm-5.2"))
    parser.add_argument("--max-groups", type=int, default=0)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-sleep-seconds", type=float, default=5.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--thinking", default="disabled")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-raw", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if not args.base_url:
        raise ValueError("--base-url is required")
    input_path = Path(args.input)
    output_path = Path(args.output)
    failure_path = output_path.with_name(output_path.stem + ".failures" + output_path.suffix)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    done = completed_ids(output_path) if args.resume else set()
    rows = [row for _, row in read_jsonl(input_path) if str(row.get("example_id")) not in done]
    random.Random(args.seed).shuffle(rows)
    if args.max_groups:
        rows = rows[: max(0, args.max_groups - len(done))]
    print(
        json.dumps(
            {
                "input": str(input_path),
                "already_completed": len(done),
                "pending_groups": len(rows),
                "judge_model": args.judge_model,
                "judge_prompt_version": PROMPT_VERSION,
                "judge_used_target": False,
                "max_workers": args.max_workers,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    lock = threading.Lock()
    completed = 0
    failed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        future_map = {pool.submit(request_judgment, row, args): row for row in rows}
        for future in concurrent.futures.as_completed(future_map):
            row = future_map[future]
            try:
                result = future.result()
                append_locked(output_path, result, lock)
                completed += 1
            except Exception as exc:
                append_locked(
                    failure_path,
                    {"example_id": row.get("example_id"), "error": f"{type(exc).__name__}: {exc}"},
                    lock,
                )
                failed += 1
            if (completed + failed) % 20 == 0 or completed + failed == len(rows):
                print(
                    json.dumps(
                        {"processed": completed + failed, "pending": len(rows), "completed": completed, "failed": failed},
                        ensure_ascii=False,
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
