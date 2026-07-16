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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rubric_cot_pipeline.io import ensure_parent, read_jsonl
from rubric_cot_pipeline.prompts import (
    API_REASONING_TAG,
    ANSWER_TAG,
    REASONING_TAG,
    build_generation_messages,
    build_target_aware_messages,
    normalize_cot_tags,
    normalize_rating_context,
)


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
WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'-]*|[\u4e00-\u9fff]")
LEAK_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?|[\u4e00-\u9fff]")
ASIN_RE = re.compile(r"\bB0[A-Z0-9]{8}\b", flags=re.IGNORECASE)
LEAK_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by",
    "for", "from", "had", "has", "have", "in", "into", "is", "it", "its", "of",
    "on", "or", "that", "the", "their", "them", "these", "this", "those", "to",
    "was", "were", "with", "without", "within", "user", "history", "item", "audio",
    "cd", "cds", "vinyl", "format", "category", "categories", "music", "album",
    "albums", "release", "released", "edition", "recording", "recordings", "production",
    "content", "style", "features", "featuring", "original",
}
ANSWER_BANNED_PATTERNS = (
    # User/history self-reference rather than item features.
    r"\b(?:the|this)\s+user\b|\buser['’]s\b|\b(?:user|interaction)\s+history\b|\bhistorical\s+items?\b",
    # Advice, policy, or future-action wording rather than a neutral item profile.
    r"\b(?:recommend\w*|suggest\w*|should|must|need(?:s)?\s+to|future\s+\w+|focus(?:es|ed|ing)?\s+on|avoid\w*|instead)\b",
    # Audience-targeting copy rather than transferable item attributes.
    r"\b(?:suitable|appeal\w*|fans?|enthusiasts?)\b",
    # External appraisal, popularity, review, sales, rating, or catalog-stat signals.
    r"\b(?:rating\w*|review\w*|critic\w*|acclaim\w*|award\w*|popular\w*|popularity|best[- ]selling|customer\w*|catalog\w*|avg_rating|rating_count)\b",
)
NO_RATING_BANNED_PATTERNS = (
    r"\b(?:rating\w*|rated|feedback|review\w*|critic\w*|acclaim\w*|award\w*|popular\w*|popularity|customer\w*|catalog\w*|avg_rating|rating_count)\b",
    r"\brating\s+scores?\b|\bstar\s+ratings?\b|\b[1-5](?:\.\d+)?\s*(?:/|out\s+of)\s*5\b|\b[1-5](?:\.\d+)?\s*stars?\b",
    r"\b(?:positive|negative)\s+(?:interaction|evidence|feedback|signal|signals?|item|items?|example|examples?)\b",
    r"\b(?:high|low)[- ]rated\b|\b(?:high|low)\s+rating\b|\b(?:liked|disliked)\b",
)


def first_env(*names: str) -> str:
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return ""


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"", "0", "false", "off", "no", "none", "null"}


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


def strip_output_markup(text: str) -> str:
    text = re.sub(
        rf"</?(?:{REASONING_TAG}|{ANSWER_TAG}|think|thinking|thoughts|answer|hidden_reasoning|reasoning|analysis)>|</?tool_call>|```[\s\S]*?```",
        "",
        text or "",
        flags=re.IGNORECASE,
    )
    return re.sub(r"\s+", " ", text).strip()


def extract_tagged_output(content: str, normalize_legacy_tags: bool = True) -> tuple[str, str, bool]:
    normalized = normalize_cot_tags(content) if normalize_legacy_tags else (content or "").strip()
    reasoning_tag = REASONING_TAG if normalize_legacy_tags else API_REASONING_TAG
    think_match = re.search(rf"<\s*{reasoning_tag}\s*>([\s\S]*?)<\s*/\s*{reasoning_tag}\s*>", normalized, flags=re.IGNORECASE)
    answer_match = re.search(rf"<\s*{ANSWER_TAG}\s*>([\s\S]*?)<\s*/\s*{ANSWER_TAG}\s*>", normalized, flags=re.IGNORECASE)
    think = strip_output_markup(think_match.group(1)) if think_match else ""
    answer = strip_output_markup(answer_match.group(1)) if answer_match else ""
    return think, answer, bool(think_match or answer_match)


def detect_block_tag(content: str, tags: tuple[str, ...]) -> str:
    for tag in tags:
        if re.search(rf"<\s*{tag}\s*>", content or "", flags=re.IGNORECASE):
            return tag
    return ""


def output_word_count(*parts: str) -> int:
    return len(WORD_RE.findall(" ".join(part for part in parts if part)))


def normalized_match_words(text: str) -> list[str]:
    return [token.lower() for token in WORD_RE.findall(text or "")]


def normalized_leak_words(text: str) -> list[str]:
    """Normalize punctuation and split hyphenated compounds for leakage checks."""
    return [token.lower() for token in LEAK_WORD_RE.findall(text or "")]


def word_ngrams(words: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(words[index : index + n]) for index in range(max(0, len(words) - n + 1))}


def informative_ngram(ngram: tuple[str, ...]) -> bool:
    required = max(1, len(ngram) - 1)
    return sum(token not in LEAK_STOP_WORDS and len(token) > 2 for token in ngram) >= required


def normalized_phrase_in_text(phrase: str, text_words: list[str]) -> bool:
    phrase_words = normalized_leak_words(phrase)
    if not phrase_words:
        return False
    width = len(phrase_words)
    return any(text_words[index : index + width] == phrase_words for index in range(max(0, len(text_words) - width + 1)))


def target_leakage_audit(row: dict[str, Any], think: str, answer: str) -> dict[str, Any]:
    history = str(row.get("user_history") or row.get("query") or "")
    target_title = str(row.get("target_item_title") or "")
    target_text = str(row.get("target_item_text") or row.get("positive") or "")
    output = f"{think}\n{answer}"
    history_words = normalized_leak_words(history)
    history_set = set(history_words)
    output_words = normalized_leak_words(output)
    output_set = set(output_words)
    target_words = normalized_leak_words(f"{target_title} {target_text}")
    title_words = normalized_leak_words(target_title)

    normalized_output = " ".join(output_words)
    normalized_title = " ".join(title_words)
    exact_title = bool(
        normalized_title
        and (len(title_words) >= 2 or len(normalized_title) >= 12)
        and normalized_title in normalized_output
    )

    target_only_by_n: dict[int, list[str]] = {}
    informative_by_n: dict[int, list[str]] = {}
    for n in (2, 3, 4):
        common = word_ngrams(target_words, n) & word_ngrams(output_words, n)
        target_only = {ngram for ngram in common if not all(token in history_set for token in ngram)}
        informative = {ngram for ngram in target_only if informative_ngram(ngram)}
        target_only_by_n[n] = sorted(" ".join(ngram) for ngram in target_only)[:50]
        informative_by_n[n] = sorted(" ".join(ngram) for ngram in informative)[:50]

    target_only_tokens = sorted((set(target_words) - history_set) & output_set)
    target_asins = {match.upper() for match in ASIN_RE.findall(f"{target_title} {target_text}")}
    output_asins = {match.upper() for match in ASIN_RE.findall(output)}
    copied_asins = sorted(target_asins & output_asins)
    target_id = str(row.get("target_item_id") or "").strip()
    copied_target_id = bool(target_id and len(target_id) >= 4 and re.search(rf"(?<!\d){re.escape(target_id)}(?!\d)", output))

    target_quoted_phrases = []
    for phrase in re.findall(r'["“”]([^"“”]{2,120})["“”]', target_text):
        if len(normalized_leak_words(phrase)) >= 2 and normalized_phrase_in_text(phrase, output_words) and not normalized_phrase_in_text(phrase, history_words):
            target_quoted_phrases.append(re.sub(r"\s+", " ", phrase).strip())

    target_artist = ""
    text_after_title = target_text
    if target_title and target_text.lower().startswith(target_title.lower()):
        text_after_title = target_text[len(target_title) :].lstrip()
    artist_match = re.match(r"(?:Digital Music\s+)?(.+?)\s+Format\s*:", text_after_title, flags=re.IGNORECASE)
    if artist_match:
        target_artist = artist_match.group(1).strip(" ;,")
    copied_target_artist = bool(
        target_artist
        and normalized_phrase_in_text(target_artist, output_words)
        and not normalized_phrase_in_text(target_artist, history_words)
    )

    label_match = re.search(r"\bLabel\s*=\s*([^;\n]+)", target_text, flags=re.IGNORECASE)
    target_label = label_match.group(1).strip() if label_match else ""
    copied_target_label = bool(
        target_label
        and normalized_phrase_in_text(target_label, output_words)
        and not normalized_phrase_in_text(target_label, history_words)
    )

    target_only_numbers = sorted(
        token for token in (set(target_words) - history_set) & output_set if token.isdigit() and len(token) >= 2
    )
    # A single informative target-only bigram is sufficient to expose an attribute
    # that cannot be recovered from history (for example, "soft rock").
    strong_ngram_leakage = bool(informative_by_n[2]) or bool(informative_by_n[3])
    severe = any(
        (
            exact_title,
            strong_ngram_leakage,
            bool(informative_by_n[4]),
            bool(copied_asins),
            copied_target_id,
            copied_target_artist,
            copied_target_label,
            bool(target_quoted_phrases),
            bool(target_only_numbers),
        )
    )
    return {
        "target_visible": True,
        "severe_leakage": severe,
        "exact_target_title": exact_title,
        "target_only_bigrams": target_only_by_n[2],
        "target_only_trigrams": target_only_by_n[3],
        "target_only_four_grams": target_only_by_n[4],
        "informative_target_only_bigrams": informative_by_n[2],
        "informative_target_only_trigrams": informative_by_n[3],
        "informative_target_only_four_grams": informative_by_n[4],
        "strong_ngram_leakage": strong_ngram_leakage,
        "copied_target_asins": copied_asins,
        "copied_target_id": copied_target_id,
        "copied_target_artist": target_artist if copied_target_artist else "",
        "copied_target_label": target_label if copied_target_label else "",
        "copied_target_quoted_phrases": sorted(set(target_quoted_phrases))[:20],
        "target_only_numbers": target_only_numbers,
        "target_only_output_tokens": target_only_tokens[:50],
        "target_only_output_token_count": len(target_only_tokens),
    }


def prompt_length_meta(messages: list[dict[str, str]]) -> dict[str, int]:
    text = "\n".join(str(message.get("content") or "") for message in messages)
    return {
        "api_prompt_message_count": len(messages),
        "api_prompt_chars": len(text),
        "api_prompt_est_tokens": output_word_count(text),
    }


def validate_answer_constraints(
    answer: str,
    min_answer_words: int = 0,
    max_answer_words: int = 0,
    rating_context: str = "rating",
) -> int:
    normalized = re.sub(r"\s+", " ", answer or "").strip().lower()
    answer_words = output_word_count(answer)
    if min_answer_words > 0 and answer_words < min_answer_words:
        raise ValueError(f"API answer has {answer_words} words, below --min-answer-words={min_answer_words}")
    if max_answer_words > 0 and answer_words > max_answer_words:
        raise ValueError(f"API answer has {answer_words} words, above --max-answer-words={max_answer_words}")
    for pattern in ANSWER_BANNED_PATTERNS:
        if re.search(pattern, normalized):
            raise ValueError(f"API answer contains banned wording pattern: {pattern}")
    if normalize_rating_context(rating_context) == "no_rating":
        for pattern in NO_RATING_BANNED_PATTERNS:
            if re.search(pattern, normalized):
                raise ValueError(f"API answer contains no-rating banned wording pattern: {pattern}")
    return answer_words


def validate_reasoning_constraints(think: str, rating_context: str = "rating") -> None:
    if normalize_rating_context(rating_context) != "no_rating":
        return
    normalized = re.sub(r"\s+", " ", think or "").strip().lower()
    for pattern in NO_RATING_BANNED_PATTERNS:
        if re.search(pattern, normalized):
            raise ValueError(f"API analysis contains no-rating banned wording pattern: {pattern}")


def split_api_output(
    content: str,
    reasoning: str,
    max_output_words: int = 0,
    min_answer_words: int = 0,
    max_answer_words: int = 0,
    rating_context: str = "rating",
    require_content_tags: bool = False,
    require_literal_tags: bool = False,
) -> tuple[str, str, dict[str, Any]]:
    rating_context = normalize_rating_context(rating_context)
    content_think, content_answer, has_content_tags = extract_tagged_output(
        content,
        normalize_legacy_tags=not (require_content_tags and require_literal_tags),
    )
    if require_content_tags and (not content_think or not content_answer):
        raise ValueError(f"API output missing required <{API_REASONING_TAG}> or <{ANSWER_TAG}> block in tagged mode")
    think = content_think or reasoning.strip()
    answer = content_answer or extract_recommendation(content)
    if not answer:
        raise ValueError("API returned empty content/answer; increase --max-new-tokens or retry")
    validate_output = require_content_tags or min_answer_words > 0 or max_answer_words > 0 or rating_context == "no_rating"
    if validate_output:
        answer_words = validate_answer_constraints(answer, min_answer_words, max_answer_words, rating_context)
        validate_reasoning_constraints(think, rating_context)
    else:
        answer_words = output_word_count(answer)
    reasoning_words = output_word_count(think)
    total_words = output_word_count(think, answer)
    if max_output_words > 0 and total_words > max_output_words:
        raise ValueError(f"API output has {total_words} words, above --max-output-words={max_output_words}")
    return think, answer, {
        "api_has_content_tags": has_content_tags,
        "api_has_content_analysis": detect_block_tag(content, (API_REASONING_TAG,)) == API_REASONING_TAG,
        "api_has_content_think": bool(content_think),
        "api_has_content_answer": bool(content_answer),
        "api_content_reasoning_tag": detect_block_tag(content, (API_REASONING_TAG, REASONING_TAG, "reasoning", "thinking", "thoughts")),
        "api_content_answer_tag": detect_block_tag(content, ("answer", "recommendation")),
        "api_reasoning_word_count": reasoning_words,
        "api_answer_word_count": answer_words,
        "api_output_word_count": total_words,
    }


def call_api(messages: list[dict[str, str]], args: argparse.Namespace, temperature: float) -> tuple[str, str, dict[str, Any]]:
    stage_start = time.perf_counter()
    url = chat_completions_url(args.api_base_url)
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"
    prompt_meta = prompt_length_meta(messages)
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
            think, answer, parse_meta = split_api_output(
                content,
                reasoning,
                args.max_output_words,
                min_answer_words=args.min_answer_words,
                max_answer_words=args.max_answer_words,
                rating_context=args.rating_context,
                require_content_tags=args.cot_output_format == "tagged",
                require_literal_tags=args.require_literal_tags,
            )
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
                "api_output_format": args.cot_output_format,
                "api_rating_context": args.rating_context,
                "api_min_answer_words": args.min_answer_words,
                "api_max_answer_words": args.max_answer_words,
                **prompt_meta,
                **parse_meta,
            }
            if args.record_api_raw:
                meta.update(
                    {
                        "api_request_payload": payload,
                        "api_raw_response": raw,
                        "api_raw_content": content,
                        "api_raw_reasoning_content": reasoning,
                    }
                )
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
    if args.target_aware:
        messages = build_target_aware_messages(
            row["user_history"],
            str(row.get("target_item_title") or ""),
            str(row.get("target_item_text") or row.get("positive") or ""),
            row.get("category", ""),
        )
    else:
        messages = build_generation_messages(
            row["user_history"],
            row.get("category", ""),
            args.cot_output_format,
            args.rating_context,
        )
    cand_start = time.perf_counter()
    target_aware_attempt = 0
    while True:
        target_aware_attempt += 1
        attempt_temperature = min(1.0, temp + 0.1 * (target_aware_attempt - 1)) if args.target_aware else temp
        think, answer, meta = call_api(messages, args, attempt_temperature)
        leakage_audit = target_leakage_audit(row, think, answer) if args.target_aware else {"target_visible": False}
        if not leakage_audit.get("severe_leakage"):
            break
        if target_aware_attempt > args.target_leakage_retries:
            raise ValueError(f"Target-aware output failed leakage audit: {json.dumps(leakage_audit, ensure_ascii=False)}")
        forbidden_fragments = list(leakage_audit.get("informative_target_only_trigrams") or [])
        forbidden_fragments.extend(leakage_audit.get("informative_target_only_bigrams") or [])
        forbidden_fragments.extend(leakage_audit.get("copied_target_quoted_phrases") or [])
        forbidden_fragments.extend(leakage_audit.get("target_only_numbers") or [])
        forbidden_fragments.extend(
            fragment
            for fragment in (leakage_audit.get("copied_target_artist"), leakage_audit.get("copied_target_label"))
            if fragment
        )
        if leakage_audit.get("exact_target_title"):
            forbidden_fragments.append(str(row.get("target_item_title") or ""))
        if forbidden_fragments:
            fragment_text = "; ".join(dict.fromkeys(fragment for fragment in forbidden_fragments if fragment))
            messages = [
                *messages,
                {
                    "role": "user",
                    "content": (
                        "Regenerate both blocks from scratch. The previous response copied next-interaction item wording. "
                        f"Do not use or paraphrase these forbidden fragments: {fragment_text}. "
                        "Use only independently supported history evidence and a broader transferable profile. "
                        "In both blocks, do not mention the next item, next interaction, target, hidden information, "
                        "auxiliary information, leakage, or the correction rules."
                    ),
                },
            ]
    cand_end = time.perf_counter()
    meta.setdefault("timing", {})["candidate_total_seconds"] = round(cand_end - cand_start, 6)
    return {
        "example_id": key,
        "candidate_id": f"{key}-{cand_idx}",
        "candidate_index": cand_idx,
        "temperature": temp,
        "final_generation_temperature": attempt_temperature,
        "think": think,
        "answer": answer,
        "cot": f"<{REASONING_TAG}>\n{think}\n</{REASONING_TAG}>\n<{ANSWER_TAG}>\n{answer}\n</{ANSWER_TAG}>",
        "generator_model": args.api_model,
        "generation_mode": "api_target_aware" if args.target_aware else "api",
        "target_aware_generation_attempts": target_aware_attempt,
        "target_leakage_audit": leakage_audit,
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
    parser.add_argument(
        "--random-sample",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="When --max-examples is positive, select that many rows with --seed instead of taking the prefix.",
    )
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
    parser.add_argument("--cot-output-format", choices=["answer_only", "tagged"], default=os.getenv("COT_GENERATION_OUTPUT_FORMAT", "answer_only"))
    parser.add_argument("--max-output-words", type=int, default=int(os.getenv("COT_GENERATION_MAX_OUTPUT_WORDS", "0")))
    parser.add_argument("--rating-context", default=os.getenv("COT_GENERATION_RATING_CONTEXT", "rating"))
    parser.add_argument("--min-answer-words", type=int, default=int(os.getenv("COT_GENERATION_MIN_ANSWER_WORDS", "0")))
    parser.add_argument("--max-answer-words", type=int, default=int(os.getenv("COT_GENERATION_MAX_ANSWER_WORDS", "0")))
    parser.add_argument("--record-api-raw", action=argparse.BooleanOptionalAction, default=env_bool("COT_GENERATION_RECORD_API_RAW", False))
    parser.add_argument("--require-literal-tags", action=argparse.BooleanOptionalAction, default=env_bool("COT_GENERATION_REQUIRE_LITERAL_TAGS", False))
    parser.add_argument(
        "--target-aware",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Expose target title/text to the generator and reject target-only output content.",
    )
    parser.add_argument(
        "--target-leakage-retries",
        type=int,
        default=3,
        help="Regenerate a target-aware candidate this many times after leakage audit failures.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=0,
        help="Deprecated compatibility option. API generation does not truncate prompts; use 0.",
    )
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    resolve_api_args(args)
    args.rating_context = normalize_rating_context(args.rating_context)
    args._api_request_lock = threading.Lock()
    args._api_last_request_ts = 0.0

    if args.api_provider not in OPENAI_COMPATIBLE_PROVIDERS | GLM_CODEPLAN_PROVIDERS:
        raise ValueError(f"Unsupported API provider: {args.api_provider}")
    if not args.api_base_url:
        raise ValueError("--api-base-url is required")
    if not args.api_model:
        raise ValueError("--api-model is required")
    if args.cot_output_format == "tagged" and args.max_output_words <= 0:
        args.max_output_words = 1024
    if args.min_answer_words < 0 or args.max_answer_words < 0:
        raise ValueError("--min-answer-words and --max-answer-words must be non-negative")
    if args.max_answer_words > 0 and args.min_answer_words > args.max_answer_words:
        raise ValueError("--min-answer-words cannot exceed --max-answer-words")

    random.seed(args.seed)
    temperatures = parse_temperatures(args.temperatures) or [0.7]
    output_path = ensure_parent(args.output)
    checkpoint_path = candidate_checkpoint_path(output_path)
    failure_path = failures_path(output_path)
    if args.random_sample and args.max_examples > 0:
        all_input_rows = list(read_jsonl(args.input))
        input_rows = random.sample(all_input_rows, min(args.max_examples, len(all_input_rows)))
    else:
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
