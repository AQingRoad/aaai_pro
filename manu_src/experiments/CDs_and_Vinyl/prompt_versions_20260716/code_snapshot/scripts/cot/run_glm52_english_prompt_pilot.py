#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from pathlib import Path


INDEXES_BY_STRATUM = {
    "1-2": [899, 166, 45, 1041, 370],
    "3-4": [526, 476, 281, 1335, 238],
    "5-7": [1039, 1107, 1337, 865, 160],
    "8+": [594, 463, 22, 17, 69],
}
INDEXES = [index for indexes in INDEXES_BY_STRATUM.values() for index in indexes]
ASIN_RE = re.compile(r"\bB0[A-Z0-9]{8}\b")
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*")
FORBIDDEN_HISTORY_MARKERS = (" stars)", "Description:", "Details:", "Catalog stats:", "[TRUNCATED]")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def api_key() -> str:
    for key in ("BIGMODEL_API_KEY", "ZHIPUAI_API_KEY", "ZHIPU_API_KEY", "GLM_API_KEY"):
        value = os.getenv(key, "").strip()
        if value:
            return value
    raise RuntimeError("GLM API key is not configured")


def parse_output(
    content: str,
    *,
    min_think_words: int,
    max_think_words: int,
    min_answer_words: int,
    max_answer_words: int,
) -> tuple[str, str]:
    think_match = re.search(r"<think>\s*(.*?)\s*</think>", content, re.I | re.S)
    answer_match = re.search(r"<answer>\s*(.*?)\s*</answer>", content, re.I | re.S)
    if not think_match or not answer_match:
        raise ValueError("missing literal <think>/<answer> blocks")
    think = think_match.group(1).strip()
    answer = answer_match.group(1).strip()
    if not think or not answer:
        raise ValueError("empty think or answer block")
    think_words = len(WORD_RE.findall(think))
    answer_words = len(WORD_RE.findall(answer))
    if not min_think_words <= think_words <= max_think_words:
        raise ValueError(f"think length outside tolerant range: {think_words}")
    if not min_answer_words <= answer_words <= max_answer_words:
        raise ValueError(f"answer length outside tolerant range: {answer_words}")
    return think, answer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--model", default="glm-5.2")
    parser.add_argument("--base-url", default="https://open.bigmodel.cn/api/coding/paas/v4/chat/completions")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.85)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-think-words", type=int, default=55)
    parser.add_argument("--max-think-words", type=int, default=165)
    parser.add_argument("--min-answer-words", type=int, default=8)
    parser.add_argument("--max-answer-words", type=int, default=45)
    parser.add_argument("--generation-mode", default="api_target_free_user_english_prompt")
    args = parser.parse_args()

    if args.seed != 42:
        raise ValueError("Project seed must remain 42")
    random.seed(args.seed)
    prompt = args.prompt.read_text(encoding="utf-8").strip()
    rows = {int(row["index"]): row for row in read_jsonl(args.source)}
    if len(INDEXES) != 20 or len(set(INDEXES)) != 20:
        raise ValueError("Expected 20 unique pilot indexes")

    selected = []
    for stratum, indexes in INDEXES_BY_STRATUM.items():
        for index in indexes:
            row = rows[index]
            history = str(row["base_query"]).strip()
            if any(marker in history for marker in FORBIDDEN_HISTORY_MARKERS):
                raise ValueError(f"Forbidden history marker at index {index}")
            if ASIN_RE.search(history):
                raise ValueError(f"ASIN found in history at index {index}")
            selected.append(
                {
                    "index": index,
                    "stratum": stratum,
                    "history_item_count": int(row["history_item_count"]),
                    "interaction_id": row["interaction_id"],
                    "base_query": history,
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_path = args.output_dir / "selected_target_free_inputs.jsonl"
    selected_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected),
        encoding="utf-8",
    )
    audit = {
        "seed": args.seed,
        "rows": len(selected),
        "indexes": INDEXES,
        "strata": {key: len(value) for key, value in INDEXES_BY_STRATUM.items()},
        "generator_fields": ["index", "stratum", "history_item_count", "interaction_id", "base_query"],
        "target_fields_in_generator_input": 0,
        "history_truncated": 0,
        "history_forbidden_metadata": 0,
        "history_asin": 0,
    }
    (args.output_dir / "input_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)
    if args.prepare_only:
        return

    key = api_key()
    output_path = args.output_dir / "glm52_english_prompt_cots.jsonl"
    existing = {int(row["index"]): row for row in read_jsonl(output_path)} if output_path.exists() else {}
    results = []
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
    for position, selected_row in enumerate(selected, 1):
        index = int(selected_row["index"])
        if index in existing:
            results.append(existing[index])
            print(f"[{position}/20] index={index} resumed", flush=True)
            continue
        source_row = rows[index]
        user_prompt = (
            "Domain or item category:\nAmazon CDs and Vinyl\n\n"
            "Available history fields:\n"
            "- item title\n"
            "- store, creator, artist, or format text\n"
            "- hierarchical categories\n"
            "Some fields may be missing. Use only fields actually present in the history.\n\n"
            f"User interaction history:\n{selected_row['base_query']}\n\n"
            "Generate one evidence-supported next-item preference direction in the required tagged format."
        )
        payload = {
            "model": args.model,
            "messages": [{"role": "system", "content": prompt}, {"role": "user", "content": user_prompt}],
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "thinking": {"type": "disabled"},
        }
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        last_error: Exception | None = None
        for attempt in range(1, args.max_retries + 2):
            started = time.time()
            try:
                request = urllib.request.Request(args.base_url, data=encoded, headers=headers, method="POST")
                with urllib.request.urlopen(request, timeout=args.timeout) as response:
                    response_obj = json.loads(response.read().decode("utf-8"))
                content = str(response_obj["choices"][0]["message"].get("content") or "").strip()
                think, answer = parse_output(
                    content,
                    min_think_words=args.min_think_words,
                    max_think_words=args.max_think_words,
                    min_answer_words=args.min_answer_words,
                    max_answer_words=args.max_answer_words,
                )
                break
            except (urllib.error.URLError, TimeoutError, KeyError, IndexError, ValueError, json.JSONDecodeError) as error:
                last_error = error
                if attempt > args.max_retries:
                    raise RuntimeError(f"index={index} failed after {attempt} attempts: {error}") from error
                time.sleep(2 * attempt)
        record = {
            "index": index,
            "stratum": selected_row["stratum"],
            "history_item_count": selected_row["history_item_count"],
            "user_id": source_row["user_id"],
            "interaction_id": source_row["interaction_id"],
            "target_item_id": source_row["target_item_id"],
            "target_item_title": source_row["target_item_title"],
            "base_query": selected_row["base_query"],
            "glm47_cot": source_row["glm47_cot"],
            "glm47_rank": source_row["glm47_rank"],
            "new_think": think,
            "new_answer": answer,
            "new_cot": f"<think>\n{think}\n</think>\n<answer>\n{answer}\n</answer>",
            "generator_model": args.model,
            "generation_mode": args.generation_mode,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
            "api_attempts": attempt,
            "api_seconds_last_attempt": round(time.time() - started, 3),
            "api_usage": response_obj.get("usage", {}),
        }
        results.append(record)
        output_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results), encoding="utf-8"
        )
        print(f"[{position}/20] index={index} attempts={attempt} answer={answer}", flush=True)
    print(f"OUTPUT={output_path}", flush=True)


if __name__ == "__main__":
    main()
