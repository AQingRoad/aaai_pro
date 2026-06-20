"""ms-swift reward plugin for Rubric-Gated RL.

Registers:
- rubric_format: exact <think>...</think><answer>...</answer> structure.
- rubric_quality: LLM/rule/classifier rubric score.
- rubric_gated_gain: rubric-quality gate multiplied by recommendation gain.
  Set RUBRIC_GAIN_MODE=ndcg to use online NDCG@K gain over item_info.jsonl.
"""

from __future__ import annotations

import os
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    from swift.plugin import ORM, orms
except Exception:  # Allows local py_compile without ms-swift installed.
    class ORM:  # type: ignore[no-redef]
        pass

    orms = {}

from rubric_cot_pipeline.embeddings import (
    DEFAULT_RECOMMENDATION_QUERY_INSTRUCTION,
    Qwen3TextEmbedder,
    append_recommendation_reasoning,
)
from rubric_cot_pipeline.judge_api import build_judge_api_client
from rubric_cot_pipeline.io import read_jsonl
from rubric_cot_pipeline.rubric import extract_blocks, hashed_cosine, rule_score


_QWEN3_EMBEDDER: Qwen3TextEmbedder | None = None
_RUBRIC_API_CLIENT = None
_RUBRIC_SCORE_CACHE: dict[tuple[str, str, str, str], float] = {}
_RUBRIC_SOURCE_CACHE: dict[tuple[str, str, str, str], str] = {}
_RUBRIC_SOURCE_COUNTS = {
    "api_success": 0,
    "fallback_rules": 0,
    "cache_api_success": 0,
    "cache_fallback_rules": 0,
}
_RUBRIC_SOURCE_LOG_COUNT = 0
_NDCG_ITEM_CACHE = None
_BASELINE_NDCG_CACHE: dict[tuple[str, int, tuple[int, ...], int, str], tuple[float, int, float]] = {}


def _log_rubric_source(source: str, provider: str, model: str, score: float, raw: str = "") -> None:
    global _RUBRIC_SOURCE_LOG_COUNT
    if source in _RUBRIC_SOURCE_COUNTS:
        _RUBRIC_SOURCE_COUNTS[source] += 1
    _RUBRIC_SOURCE_LOG_COUNT += 1

    log_path = os.getenv("RUBRIC_REWARD_SOURCE_LOG", "").strip()
    if log_path:
        try:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            with Path(log_path).open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "source": source,
                            "provider": provider,
                            "model": model,
                            "score": score,
                            "rank": os.getenv("RANK", ""),
                            "local_rank": os.getenv("LOCAL_RANK", ""),
                            "api_error": raw[:500] if raw else "",
                            "counts": dict(_RUBRIC_SOURCE_COUNTS),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        except Exception as exc:
            print(f"[rubric_reward_source_log_error] {type(exc).__name__}: {exc}", flush=True)

    log_every = int(os.getenv("RUBRIC_REWARD_SOURCE_LOG_EVERY", "20"))
    if log_every > 0 and _RUBRIC_SOURCE_LOG_COUNT % log_every == 0:
        print(
            "[rubric_reward_source] "
            f"api_success={_RUBRIC_SOURCE_COUNTS['api_success']} "
            f"fallback_rules={_RUBRIC_SOURCE_COUNTS['fallback_rules']} "
            f"cache_api_success={_RUBRIC_SOURCE_COUNTS['cache_api_success']} "
            f"cache_fallback_rules={_RUBRIC_SOURCE_COUNTS['cache_fallback_rules']}",
            flush=True,
        )


def _as_list(value: Any, n: int) -> list[str]:
    if value is None:
        return [""] * n
    if isinstance(value, str):
        return [value] * n
    try:
        seq = list(value)
    except TypeError:
        return [str(value)] * n
    if len(seq) < n:
        seq.extend([""] * (n - len(seq)))
    return [str(x or "") for x in seq[:n]]


def _as_float_list(value: Any, n: int) -> list[float | None]:
    raw = _as_list(value, n)
    out: list[float | None] = []
    for item in raw:
        try:
            out.append(float(item))
        except Exception:
            out.append(None)
    return out


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_int_list(value: Any, n: int) -> list[int | None]:
    if value is None:
        return [None] * n
    if isinstance(value, (str, int, float)):
        return [_as_int(value)] * n
    try:
        seq = list(value)
    except TypeError:
        return [_as_int(value)] * n
    if len(seq) == n:
        return [_as_int(item) for item in seq]
    if len(seq) == 1:
        return [_as_int(seq[0])] * n
    out = [_as_int(item) for item in seq[:n]]
    if len(out) < n:
        out.extend([None] * (n - len(out)))
    return out


def _parse_int_set(value: Any) -> set[int]:
    if value is None:
        return set()
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return set()
        if text.startswith("["):
            try:
                return _parse_int_set(json.loads(text))
            except json.JSONDecodeError:
                pass
        return {int(x) for x in re.findall(r"-?\d+", text)}
    if isinstance(value, (int, float)):
        parsed = _as_int(value)
        return {parsed} if parsed is not None else set()
    try:
        seq = list(value)
    except TypeError:
        parsed = _as_int(value)
        return {parsed} if parsed is not None else set()
    out: set[int] = set()
    for item in seq:
        parsed = _as_int(item)
        if parsed is not None:
            out.add(parsed)
    return out


def _as_int_set_list(value: Any, n: int) -> list[set[int]]:
    if value is None:
        return [set() for _ in range(n)]
    if isinstance(value, str):
        parsed = _parse_int_set(value)
        return [set(parsed) for _ in range(n)]
    try:
        seq = list(value)
    except TypeError:
        parsed = _parse_int_set(value)
        return [set(parsed) for _ in range(n)]
    if n == 1:
        return [_parse_int_set(seq)]
    if len(seq) == n and any(isinstance(item, (list, tuple, set, dict)) or (isinstance(item, str) and item.strip().startswith("[")) for item in seq):
        return [_parse_int_set(item) for item in seq]
    parsed = _parse_int_set(seq)
    return [set(parsed) for _ in range(n)]


def _env_bool(name: str, default: bool = True) -> bool:
    value = os.getenv(name, "")
    if not value:
        return default
    return value.strip().lower() not in {"0", "false", "off", "no", "none", "null"}


def _compact(text: Any, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max_chars - 15].rstrip() + " [TRUNCATED]"


def _as_text_list(value: Any, limit: int = 8) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            text = _compact(item, 500)
            if text:
                out.append(text)
            if len(out) >= limit:
                break
        return out
    return [_compact(value, 500)]


def _build_item_text(item: dict[str, Any] | None, title: str, max_chars: int) -> str:
    if not item:
        return _compact(title, max_chars)

    parts: list[str] = []
    for key in ("title", "main_category", "store"):
        value = _compact(item.get(key), 300)
        if value:
            parts.append(value)
    categories = " > ".join(_as_text_list(item.get("categories"), limit=6))
    if categories:
        parts.append(f"Categories: {categories}")
    features = "; ".join(_as_text_list(item.get("features"), limit=8))
    if features:
        parts.append(f"Features: {features}")
    description = " ".join(_as_text_list(item.get("description"), limit=2))
    if description:
        parts.append(f"Description: {description}")
    if not parts:
        parts.append(title)
    return _compact(" ".join(parts), max_chars)


def _ndcg_at_rank(rank: int, k: int) -> float:
    if rank <= 0 or rank > k:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def _get_qwen3_embedder() -> Qwen3TextEmbedder:
    global _QWEN3_EMBEDDER
    if _QWEN3_EMBEDDER is None:
        default_device = f"cuda:{os.getenv('LOCAL_RANK', '0')}" if os.getenv("LOCAL_RANK") is not None else "cuda:0"
        _QWEN3_EMBEDDER = Qwen3TextEmbedder(
            os.getenv("QWEN3_EMBEDDING_MODEL", "/root/autodl-tmp/modelscope_cache/models/Qwen/Qwen3-Embedding-0.6B"),
            max_length=int(os.getenv("QWEN3_EMBEDDING_MAX_LENGTH", "4096")),
            batch_size=int(os.getenv("QWEN3_EMBEDDING_BATCH_SIZE", "4")),
            torch_dtype=os.getenv("QWEN3_EMBEDDING_TORCH_DTYPE", "bfloat16"),
            device=os.getenv("QWEN3_EMBEDDING_DEVICE", default_device),
            query_instruction=os.getenv("QWEN3_EMBEDDING_QUERY_INSTRUCTION", DEFAULT_RECOMMENDATION_QUERY_INSTRUCTION),
            output_dim=int(os.getenv("QWEN3_EMBEDDING_OUTPUT_DIM", "0")),
        )
    return _QWEN3_EMBEDDER


def _get_ndcg_item_cache():
    global _NDCG_ITEM_CACHE
    item_info = os.getenv("RUBRIC_NDCG_ITEM_INFO") or os.getenv("RUBRIC_GAIN_ITEM_INFO") or ""
    if not item_info:
        raise RuntimeError("RUBRIC_GAIN_MODE=ndcg requires RUBRIC_NDCG_ITEM_INFO or RUBRIC_GAIN_ITEM_INFO.")
    item_max_chars = int(os.getenv("RUBRIC_NDCG_ITEM_MAX_CHARS", "1200"))
    cache_key = (
        item_info,
        item_max_chars,
        os.getenv("QWEN3_EMBEDDING_MODEL", ""),
        os.getenv("QWEN3_EMBEDDING_OUTPUT_DIM", "0"),
    )
    if _NDCG_ITEM_CACHE is not None and _NDCG_ITEM_CACHE["cache_key"] == cache_key:
        return _NDCG_ITEM_CACHE

    item_ids: list[int] = []
    item_texts: list[str] = []
    for row in read_jsonl(item_info):
        item_id = int(row["item_id"])
        item_ids.append(item_id)
        item_texts.append(_build_item_text(row, str(row.get("title", "")), item_max_chars))
    order = sorted(range(len(item_ids)), key=lambda idx: item_ids[idx])
    item_ids = [item_ids[idx] for idx in order]
    item_texts = [item_texts[idx] for idx in order]
    item_index = {item_id: idx for idx, item_id in enumerate(item_ids)}
    item_embs = _get_qwen3_embedder().encode_documents(item_texts)
    _NDCG_ITEM_CACHE = {
        "cache_key": cache_key,
        "item_ids": item_ids,
        "item_index": item_index,
        "item_embs": item_embs,
    }
    return _NDCG_ITEM_CACHE


def _masked_indices(item_index: dict[int, int], target_id: int, history_item_ids: set[int]) -> set[int]:
    masked_item_ids = set()
    if _env_bool("RUBRIC_NDCG_MASK_HISTORY_ITEMS", True):
        masked_item_ids.update(history_item_ids)
    if _env_bool("RUBRIC_NDCG_MASK_PAD_ITEM", True):
        masked_item_ids.add(0)
    masked_item_ids.discard(target_id)
    return {item_index[item_id] for item_id in masked_item_ids if item_id in item_index}


def _score_query_embedding_ndcg(query_emb, target_id: int | None, history_item_ids: set[int], k: int) -> tuple[float, int, float]:
    cache = _get_ndcg_item_cache()
    item_index = cache["item_index"]
    item_embs = cache["item_embs"]
    if target_id is None or target_id not in item_index:
        return 0.0, len(cache["item_ids"]) + 1, 0.0
    target_index = item_index[target_id]
    scores = item_embs @ query_emb
    mask = _masked_indices(item_index, target_id, history_item_ids)
    if mask:
        scores = scores.clone()
        scores[list(mask)] = -float("inf")
    target_score = float(scores[target_index].item())
    rank = int((scores > scores[target_index]).sum().item()) + 1
    return target_score, rank, _ndcg_at_rank(rank, k)


def _get_api_client():
    global _RUBRIC_API_CLIENT
    if _RUBRIC_API_CLIENT is None:
        _RUBRIC_API_CLIENT = build_judge_api_client(
            provider=os.getenv("RUBRIC_REWARD_API_PROVIDER") or os.getenv("RUBRIC_JUDGE_API_PROVIDER"),
            base_url=os.getenv("RUBRIC_REWARD_API_BASE_URL") or os.getenv("RUBRIC_JUDGE_API_BASE_URL"),
            api_key=os.getenv("RUBRIC_REWARD_API_KEY") or os.getenv("RUBRIC_JUDGE_API_KEY"),
            model=os.getenv("RUBRIC_REWARD_API_MODEL") or os.getenv("RUBRIC_JUDGE_API_MODEL"),
            timeout=float(os.getenv("RUBRIC_REWARD_API_TIMEOUT") or os.getenv("RUBRIC_JUDGE_API_TIMEOUT", "60")),
            max_retries=int(os.getenv("RUBRIC_REWARD_API_MAX_RETRIES") or os.getenv("RUBRIC_JUDGE_API_MAX_RETRIES", "2")),
        )
    return _RUBRIC_API_CLIENT


def _score_with_rules(user_history: str, completion: str, target_item: str = "") -> float:
    return float(rule_score(user_history, completion)["score_norm"])


def _score_with_api(user_history: str, completion: str, target_item: str = "") -> float:
    provider = os.getenv("RUBRIC_REWARD_API_PROVIDER") or os.getenv("RUBRIC_JUDGE_API_PROVIDER", "")
    model = os.getenv("RUBRIC_REWARD_API_MODEL") or os.getenv("RUBRIC_JUDGE_API_MODEL", "")
    cache_key = (provider, model, user_history, completion, target_item)
    if cache_key in _RUBRIC_SCORE_CACHE:
        score = _RUBRIC_SCORE_CACHE[cache_key]
        source = _RUBRIC_SOURCE_CACHE.get(cache_key, "api_success")
        cache_source = "cache_fallback_rules" if source == "fallback_rules" else "cache_api_success"
        _log_rubric_source(cache_source, provider, model, score)
        return score

    result = _get_api_client().score(user_history, completion, target_item)
    if result.score is not None:
        score = float(result.score["score_norm"])
        _RUBRIC_SCORE_CACHE[cache_key] = score
        _RUBRIC_SOURCE_CACHE[cache_key] = "api_success"
        _log_rubric_source("api_success", provider, model, score)
        return score

    fallback = os.getenv("RUBRIC_REWARD_API_FALLBACK", "rules").strip().lower()
    if fallback in {"rules", "rule", "local"}:
        score = _score_with_rules(user_history, completion, target_item)
        _RUBRIC_SCORE_CACHE[cache_key] = score
        _RUBRIC_SOURCE_CACHE[cache_key] = "fallback_rules"
        _log_rubric_source("fallback_rules", provider, model, score, result.raw)
        return score
    raise RuntimeError(f"Rubric API scorer failed and fallback is disabled: {result.raw}")


def _score_with_classifier(user_history: str, completion: str, target_item: str = "") -> float:
    checkpoint = os.getenv("RUBRIC_CLASSIFIER_CHECKPOINT", "").strip()
    if not checkpoint:
        raise RuntimeError(
            "RUBRIC_REWARD_SCORER=classifier requires RUBRIC_CLASSIFIER_CHECKPOINT. "
            "Train or provide a lightweight rubric classifier checkpoint first."
        )
    raise NotImplementedError(
        "Lightweight rubric classifier inference is reserved but not implemented in this pipeline yet. "
        "Use RUBRIC_REWARD_SCORER=api for best quality or RUBRIC_REWARD_SCORER=rules for offline runs."
    )


def rubric_quality_score(user_history: str, completion: str, target_item: str = "") -> float:
    scorer = os.getenv("RUBRIC_REWARD_SCORER", "api").strip().lower()
    if scorer in {"api", "llm", "judge", "llm_judge"}:
        return _score_with_api(user_history, completion, target_item)
    if scorer in {"rules", "rule", "local", "proxy"}:
        return _score_with_rules(user_history, completion, target_item)
    if scorer in {"classifier", "clf", "rubric_classifier"}:
        return _score_with_classifier(user_history, completion, target_item)
    raise ValueError(f"Unsupported RUBRIC_REWARD_SCORER: {scorer}")


class RubricFormatReward(ORM):
    def __call__(self, completions, **kwargs) -> list[float]:
        rewards = []
        for text in completions:
            think, answer, ok = extract_blocks(text)
            rewards.append(1.0 if ok and think and answer else 0.0)
        return rewards


class RubricQualityReward(ORM):
    def __call__(self, completions, user_history=None, source_prompt=None, target_item_text=None, target_item_title=None, **kwargs) -> list[float]:
        prompts = _as_list(user_history if user_history is not None else source_prompt, len(completions))
        targets = _as_list(target_item_text if target_item_text is not None else target_item_title, len(completions))
        return [rubric_quality_score(prompt, text, target) for text, prompt, target in zip(completions, prompts, targets)]


class RubricGatedGainReward(ORM):
    def __call__(
        self,
        completions,
        user_history=None,
        source_prompt=None,
        target_item_id=None,
        target_item_text=None,
        target_item_title=None,
        history_item_ids=None,
        history_item_id=None,
        baseline_sim=None,
        baseline_ndcg=None,
        baseline_embedder_mode=None,
        **kwargs,
    ) -> list[float]:
        n = len(completions)
        histories = _as_list(user_history if user_history is not None else source_prompt, n)
        target_ids = _as_int_list(target_item_id, n)
        targets = _as_list(target_item_text if target_item_text is not None else target_item_title, n)
        history_sets = _as_int_set_list(history_item_ids if history_item_ids is not None else history_item_id, n)
        baselines = _as_float_list(baseline_sim, n)
        baseline_ndcgs = _as_float_list(baseline_ndcg, n)
        baseline_modes = _as_list(baseline_embedder_mode, n)
        reward_gain_mode = os.getenv("RUBRIC_GAIN_MODE", "ndcg").strip().lower()
        gain_mode = os.getenv("RUBRIC_GAIN_EMBEDDER_MODE", "lexical")
        threshold = float(os.getenv("RUBRIC_GATE_THRESHOLD", "0.45"))
        penalty = float(os.getenv("RUBRIC_GATE_PENALTY", "-0.05"))
        qualities = [
            rubric_quality_score(history, text, target)
            for text, history, target in zip(completions, histories, targets)
        ]

        if reward_gain_mode in {"ndcg", "ndcg@k", "rank_ndcg"}:
            k = int(os.getenv("RUBRIC_NDCG_K") or os.getenv("RUBRIC_GAIN_NDCG_K", "100"))
            if k <= 0:
                raise RuntimeError("RUBRIC_NDCG_K must be positive.")
            embedder = _get_qwen3_embedder()

            missing_base_queries: list[str] = []
            missing_base_indices: list[int] = []
            base_values: list[tuple[float, int, float] | None] = [None] * n
            for idx, (history, target_id, history_ids, maybe_ndcg) in enumerate(
                zip(histories, target_ids, history_sets, baseline_ndcgs)
            ):
                cache_key = (
                    history,
                    target_id if target_id is not None else -1,
                    tuple(sorted(history_ids)),
                    k,
                    os.getenv("QWEN3_EMBEDDING_MODEL", ""),
                )
                if maybe_ndcg is not None:
                    base_values[idx] = (0.0, 0, maybe_ndcg)
                elif cache_key in _BASELINE_NDCG_CACHE:
                    base_values[idx] = _BASELINE_NDCG_CACHE[cache_key]
                else:
                    missing_base_queries.append(history)
                    missing_base_indices.append(idx)

            if missing_base_queries:
                base_embs = embedder.encode_queries(missing_base_queries)
                for query_emb, idx in zip(base_embs, missing_base_indices):
                    value = _score_query_embedding_ndcg(query_emb, target_ids[idx], history_sets[idx], k)
                    base_values[idx] = value
                    cache_key = (
                        histories[idx],
                        target_ids[idx] if target_ids[idx] is not None else -1,
                        tuple(sorted(history_sets[idx])),
                        k,
                        os.getenv("QWEN3_EMBEDDING_MODEL", ""),
                    )
                    _BASELINE_NDCG_CACHE[cache_key] = value

            cot_queries = [
                append_recommendation_reasoning(history, text)
                for text, history in zip(completions, histories)
            ]
            cot_embs = embedder.encode_queries(cot_queries)
            rewards = []
            for idx, (quality, target_id, target) in enumerate(zip(qualities, target_ids, targets)):
                if quality < threshold or target_id is None or not target:
                    rewards.append(penalty)
                    continue
                base_value = base_values[idx]
                if base_value is None:
                    rewards.append(penalty)
                    continue
                _, _, base_ndcg = base_value
                _, _, cot_ndcg = _score_query_embedding_ndcg(cot_embs[idx], target_id, history_sets[idx], k)
                gain = cot_ndcg - base_ndcg
                rewards.append(quality * gain if gain > 0 else penalty)
            return rewards

        if gain_mode == "qwen3_embedding":
            embedder = _get_qwen3_embedder()
            cot_queries = [
                append_recommendation_reasoning(history, text)
                for text, history in zip(completions, histories)
            ]
            cot_sims = embedder.pairwise_cosine(cot_queries, targets)
            missing_base = [
                base is None or mode != "qwen3_embedding"
                for base, mode in zip(baselines, baseline_modes)
            ]
            computed_base = [None] * n
            if any(missing_base):
                base_queries = [history for history, missing in zip(histories, missing_base) if missing]
                base_targets = [target for target, missing in zip(targets, missing_base) if missing]
                base_sims = embedder.pairwise_cosine(base_queries, base_targets)
                it = iter(base_sims)
                computed_base = [next(it) if missing else None for missing in missing_base]

            rewards = []
            for quality, target, base, new_base, cot_sim in zip(qualities, targets, baselines, computed_base, cot_sims):
                if quality < threshold or not target:
                    rewards.append(penalty)
                    continue
                base_sim = new_base if new_base is not None else float(base)
                gain = cot_sim - base_sim
                rewards.append(quality * gain if gain > 0 else penalty)
            return rewards

        rewards = []
        for text, history, target, base, quality in zip(completions, histories, targets, baselines, qualities):
            if quality < threshold or not target:
                rewards.append(penalty)
                continue
            base_sim = hashed_cosine(history, target) if base is None else base
            cot_sim = hashed_cosine(append_recommendation_reasoning(history, text), target)
            gain = cot_sim - base_sim
            rewards.append(quality * gain if gain > 0 else penalty)
        return rewards


orms["rubric_format"] = RubricFormatReward
orms["rubric_quality"] = RubricQualityReward
orms["rubric_gated_gain"] = RubricGatedGainReward
