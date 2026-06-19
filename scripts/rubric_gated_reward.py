"""ms-swift reward plugin for Rubric-Gated RL.

Registers:
- rubric_format: exact <think>...</think><answer>...</answer> structure.
- rubric_quality: local five-dimension recommendation CoT rubric proxy.
- rubric_gated_gain: rubric-quality gate multiplied by recommendation gain.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
from rubric_cot_pipeline.rubric import extract_blocks, hashed_cosine, rule_score


_QWEN3_EMBEDDER: Qwen3TextEmbedder | None = None
_RUBRIC_API_CLIENT = None
_RUBRIC_SCORE_CACHE: dict[tuple[str, str, str, str], float] = {}


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


def _get_qwen3_embedder() -> Qwen3TextEmbedder:
    global _QWEN3_EMBEDDER
    if _QWEN3_EMBEDDER is None:
        _QWEN3_EMBEDDER = Qwen3TextEmbedder(
            os.getenv("QWEN3_EMBEDDING_MODEL", "/root/autodl-tmp/modelscope_cache/models/Qwen/Qwen3-Embedding-0.6B"),
            max_length=int(os.getenv("QWEN3_EMBEDDING_MAX_LENGTH", "4096")),
            batch_size=int(os.getenv("QWEN3_EMBEDDING_BATCH_SIZE", "4")),
            torch_dtype=os.getenv("QWEN3_EMBEDDING_TORCH_DTYPE", "bfloat16"),
            device=os.getenv("QWEN3_EMBEDDING_DEVICE", "auto"),
            query_instruction=os.getenv("QWEN3_EMBEDDING_QUERY_INSTRUCTION", DEFAULT_RECOMMENDATION_QUERY_INSTRUCTION),
            output_dim=int(os.getenv("QWEN3_EMBEDDING_OUTPUT_DIM", "0")),
        )
    return _QWEN3_EMBEDDER


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
        return _RUBRIC_SCORE_CACHE[cache_key]

    result = _get_api_client().score(user_history, completion, target_item)
    if result.score is not None:
        score = float(result.score["score_norm"])
        _RUBRIC_SCORE_CACHE[cache_key] = score
        return score

    fallback = os.getenv("RUBRIC_REWARD_API_FALLBACK", "rules").strip().lower()
    if fallback in {"rules", "rule", "local"}:
        score = _score_with_rules(user_history, completion, target_item)
        _RUBRIC_SCORE_CACHE[cache_key] = score
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
        target_item_text=None,
        target_item_title=None,
        baseline_sim=None,
        baseline_embedder_mode=None,
        **kwargs,
    ) -> list[float]:
        n = len(completions)
        histories = _as_list(user_history if user_history is not None else source_prompt, n)
        targets = _as_list(target_item_text if target_item_text is not None else target_item_title, n)
        baselines = _as_float_list(baseline_sim, n)
        baseline_modes = _as_list(baseline_embedder_mode, n)
        gain_mode = os.getenv("RUBRIC_GAIN_EMBEDDER_MODE", "lexical")
        threshold = float(os.getenv("RUBRIC_GATE_THRESHOLD", "0.45"))
        penalty = float(os.getenv("RUBRIC_GATE_PENALTY", "-0.05"))
        qualities = [
            rubric_quality_score(history, text, target)
            for text, history, target in zip(completions, histories, targets)
        ]

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
