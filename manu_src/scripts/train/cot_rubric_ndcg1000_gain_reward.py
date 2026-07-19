#!/usr/bin/env python3
"""Rubric-weighted relative-NDCG reward for recommendation GRPO.

This plugin is intentionally separate from ``cot_sim_ndcg_reward.py`` and
registers a distinct ms-swift reward name: ``cot_rubric_ndcg1000_gain``.

For each completion i in a prompt group, it computes

    delta_ndcg_i = NDCG@K(new_cot_i) - NDCG@K(reference_cot)
    joint_i = q_i ** gamma * max(delta_ndcg_i, 0)
              - lambda_negative * max(-delta_ndcg_i, 0)
    reward_i = w_similarity * group_zscore(similarity_i)
               + w_joint * group_zscore(joint_i)

``q_i`` is produced by an API rubric judge that sees history, the generated
completion, and the held-out target item. The reference CoT is fixed prompt
metadata and never enters policy generation.

Preferred inputs are ``reference_ndcg`` values cached offline. When those are
absent, the plugin accepts ``reference_rank`` or ``reference_cot`` and computes
the reference NDCG with the same frozen embedder, item formatter, catalog, and
seen-item mask used for new completions.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from swift.plugin import ORM, orms
except Exception:  # Allows syntax and pure-helper tests without ms-swift.
    class ORM:  # type: ignore[no-redef]
        pass

    orms: dict[str, Any] = {}

from manu_src.scripts.train.cot_sim_ndcg_reward import (
    CotRetrievalRewardState,
    _as_int_list,
    _as_int_set_list,
    _as_text_list,
    _env_bool,
    _expand_values,
    _group_key,
    _group_zscore,
    _ndcg_at_rank,
    _pairwise_conflict_rate,
    _safe_mean,
)
from manu_src.api_info.rubric_target_relevance_api import (
    build_target_relevance_judge_api_client,
)
from rubric_cot_pipeline.rubric import rule_score
from rubric_cot_pipeline.rubric_scorer import OrdinalRubricHead, RubricScorerConfig


ENV_PREFIX = "COT_RUBRIC_NDCG_GAIN"
_STATE: "CotRetrievalRewardState | None" = None
_RUBRIC_SCORER: "FrozenRubricScorer | None" = None
_REFERENCE_NDCG_CACHE: dict[tuple[Any, ...], tuple[float, int]] = {}
_CALL_COUNT = 0


def _env(name: str, default: str = "") -> str:
    return os.getenv(f"{ENV_PREFIX}_{name}", default)


def _as_float_list(value: Any, n: int) -> list[float | None]:
    output: list[float | None] = []
    for item in _expand_values(value, n, None):
        try:
            number = float(item)
        except (TypeError, ValueError):
            output.append(None)
            continue
        output.append(number if math.isfinite(number) else None)
    return output


def _first_text_list(n: int, *values: Any) -> list[str]:
    for value in values:
        if value is None:
            continue
        texts = _as_text_list(value, n)
        if any(texts):
            return texts
    return ["" for _ in range(n)]


def _first_float_list(n: int, *values: Any) -> list[float | None]:
    for value in values:
        if value is None:
            continue
        numbers = _as_float_list(value, n)
        if any(number is not None for number in numbers):
            return numbers
    return [None for _ in range(n)]


def _first_int_list(n: int, *values: Any) -> list[int | None]:
    for value in values:
        if value is None:
            continue
        numbers = _as_int_list(value, n)
        if any(number is not None for number in numbers):
            return numbers
    return [None for _ in range(n)]


def _normalized_identifier(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    path = Path(text).expanduser()
    return str(path.resolve()) if path.exists() else text.rstrip("/")


def _parse_api_keys(value: Any) -> list[str]:
    """Parse a JSON array or a comma/newline separated API-key list."""
    if isinstance(value, (list, tuple)):
        candidates = [str(item or "").strip() for item in value]
    else:
        text = str(value or "").strip()
        if not text:
            return []
        candidates: list[str]
        if text.startswith("["):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{ENV_PREFIX}_API_KEYS must be a JSON array or a delimited list"
                ) from exc
            if not isinstance(parsed, list):
                raise ValueError(f"{ENV_PREFIX}_API_KEYS JSON value must be an array")
            candidates = [str(item or "").strip() for item in parsed]
        else:
            candidates = [part.strip() for part in re.split(r"[,;\s]+", text)]
    return list(dict.fromkeys(key for key in candidates if key))


def _project_api_defaults(provider: str) -> tuple[str, str, list[str]]:
    """Read one project API endpoint/model/key pool without logging secrets."""
    try:
        from manu_src.api_info import API_CONFIG
    except Exception:
        return "", "", []
    normalized_provider = str(provider or "").strip().lower()
    config_name = "ks_tokenverse" if normalized_provider in {
        "ks", "ks_tokenverse", "tokenverse"
    } else "glm_official"
    provider_configs = getattr(API_CONFIG, "API_PROVIDER_CONFIGS", {})
    if isinstance(provider_configs, dict):
        provider_config = provider_configs.get(config_name, {})
        if isinstance(provider_config, dict) and provider_config:
            return (
                str(provider_config.get("base_url") or "").strip(),
                str(provider_config.get("model") or "").strip(),
                _parse_api_keys(provider_config.get("api_key_list", [])),
            )
    base_url = str(getattr(API_CONFIG, "GLM_OFFICIAL_API_BASE_URL", "") or "").strip()
    model = str(getattr(API_CONFIG, "OFFICIAL_DEFAULT_MODEL", "") or "").strip()
    keys = _parse_api_keys(getattr(API_CONFIG, "GLM_OFFICIAL_API_KEY_LIST", []))
    return base_url, model, keys


def _api_score_norm(
    client,
    history: str,
    completion: str,
    target_item: str,
) -> tuple[float | None, str, str]:
    """Call the judge with the held-out target and validate its normalized score."""
    result = client.score(
        history,
        completion,
        target_item=target_item,
        target_usage="relevance",
    )
    if result.score is None:
        return None, str(result.raw or ""), str(result.provider or "")
    try:
        value = float(result.score["score_norm"])
    except (KeyError, TypeError, ValueError):
        return None, str(result.raw or ""), str(result.provider or "")
    if not math.isfinite(value):
        return None, str(result.raw or ""), str(result.provider or "")
    return min(1.0, max(0.0, value)), str(result.raw or ""), str(result.provider or "")


def _joint_gain(
    rubric_score: float,
    delta_ndcg: float,
    *,
    quality_power: float,
    negative_gain_weight: float,
) -> float:
    """Quality-weight positive gain; never attenuate negative rank gain."""
    if quality_power < 0:
        raise ValueError("quality_power must be non-negative")
    if negative_gain_weight < 0:
        raise ValueError("negative_gain_weight must be non-negative")
    quality = min(1.0, max(0.0, float(rubric_score))) ** quality_power
    if delta_ndcg >= 0.0:
        return quality * delta_ndcg
    return negative_gain_weight * delta_ndcg


def _combine_group_rewards(
    similarities: list[float],
    joint_gains: list[float],
    *,
    similarity_weight: float,
    joint_weight: float,
    epsilon: float,
) -> tuple[list[float], list[float], list[float]]:
    if len(similarities) != len(joint_gains):
        raise ValueError("similarity and joint-gain group sizes differ")
    similarity_z = _group_zscore(similarities, epsilon=epsilon)
    joint_z = _group_zscore(joint_gains, epsilon=epsilon)
    rewards = [
        similarity_weight * sim_value + joint_weight * joint_value
        for sim_value, joint_value in zip(similarity_z, joint_z)
    ]
    return rewards, similarity_z, joint_z


def _apply_group_ndcg_only_fallback(
    rubric_scores: list[float | None],
    grouped_indices: dict[tuple[str, str | int], list[int]],
) -> tuple[list[float], list[bool], int]:
    """Use q=1 for a whole GRPO group when any API rubric score is missing."""
    resolved = [0.0 for _ in rubric_scores]
    fallback_mask = [False for _ in rubric_scores]
    fallback_groups = 0
    for indices in grouped_indices.values():
        use_fallback = any(rubric_scores[index] is None for index in indices)
        fallback_groups += int(use_fallback)
        for index in indices:
            score = 1.0 if use_fallback else rubric_scores[index]
            if score is None:
                raise RuntimeError("unresolved rubric score after group fallback")
            resolved[index] = min(1.0, max(0.0, float(score)))
            fallback_mask[index] = use_fallback
    return resolved, fallback_mask, fallback_groups


class FrozenRubricScorer:
    """Frozen no-target rubric scorer; online API judge is the default."""

    def __init__(self, retrieval_state: CotRetrievalRewardState) -> None:
        self.state = retrieval_state
        self.mode = _env("RUBRIC_SCORER", "api").strip().lower()
        self.head = None
        self.config = None
        self.device = retrieval_state.device
        self.api_clients: list[Any] = []
        self.api_provider = ""
        self.api_model = ""
        self.api_concurrency = 1
        self.api_concurrency_per_key = 10
        self.api_key_attempts = 1
        self.api_fallback = "error"
        self._api_next_client = 0
        self._api_lock = threading.Lock()
        self._api_semaphores: list[threading.BoundedSemaphore] = []
        self._api_cache: dict[tuple[str, str, str, str, str], tuple[float, str]] = {}
        self.last_batch_meta: dict[str, Any] = {}

        if self.mode in {"rules", "rule", "local"}:
            return
        if self.mode in {"api", "llm", "judge", "llm_judge"}:
            self._init_api_clients()
            return
        if self.mode not in {"classifier", "clf", "rubric_classifier"}:
            raise ValueError(
                f"Unsupported {ENV_PREFIX}_RUBRIC_SCORER={self.mode}; "
                "use api, classifier, or rules"
            )

        import torch
        from manu_src.scripts.models.train_embedding import QUERY_INSTRUCTION

        checkpoint = Path(_env("RUBRIC_CHECKPOINT").strip())
        config_path = checkpoint / "rubric_scorer_config.json"
        head_path = checkpoint / "rubric_scorer_head.pt"
        if not config_path.is_file() or not head_path.is_file():
            raise RuntimeError(
                f"{ENV_PREFIX}_RUBRIC_CHECKPOINT must contain "
                "rubric_scorer_config.json and rubric_scorer_head.pt"
            )
        config = RubricScorerConfig.from_dict(
            json.loads(config_path.read_text(encoding="utf-8"))
        )
        if _env_bool(f"{ENV_PREFIX}_STRICT_RUBRIC_ENCODER_MATCH", True):
            if _normalized_identifier(config.encoder_model) != _normalized_identifier(
                retrieval_state.model_path
            ):
                raise RuntimeError(
                    "Rubric scorer encoder does not match the retrieval reward embedder: "
                    f"{config.encoder_model!r} != {retrieval_state.model_path!r}"
                )
            if int(config.encoder_max_length) != retrieval_state.max_length:
                raise RuntimeError(
                    "Rubric scorer max_length does not match retrieval reward max_length: "
                    f"{config.encoder_max_length} != {retrieval_state.max_length}"
                )
            if str(config.query_instruction or "") != QUERY_INSTRUCTION:
                raise RuntimeError("Rubric scorer query instruction is inconsistent")

        device_name = _env("RUBRIC_DEVICE", str(retrieval_state.device)).strip()
        self.device = torch.device(device_name)
        self.config = config
        self.head = OrdinalRubricHead(config)
        try:
            weights = torch.load(head_path, map_location="cpu", weights_only=True)
        except TypeError:
            weights = torch.load(head_path, map_location="cpu")
        self.head.load_state_dict(weights)
        self.head.to(self.device)
        self.head.eval()
        for parameter in self.head.parameters():
            parameter.requires_grad_(False)

    def _init_api_clients(self) -> None:
        self.api_provider = (
            _env("API_PROVIDER")
            or os.getenv("RUBRIC_REWARD_API_PROVIDER", "")
            or os.getenv("RUBRIC_JUDGE_API_PROVIDER", "")
            or "zhipu_glm"
        ).strip().lower()
        project_base_url, project_model, project_keys = _project_api_defaults(
            self.api_provider
        )
        base_url = (
            _env("API_BASE_URL")
            or os.getenv("RUBRIC_REWARD_API_BASE_URL", "")
            or os.getenv("RUBRIC_JUDGE_API_BASE_URL", "")
            or project_base_url
        ).strip()
        self.api_model = (
            _env("API_MODEL")
            or os.getenv("RUBRIC_REWARD_API_MODEL", "")
            or os.getenv("RUBRIC_JUDGE_API_MODEL", "")
            or project_model
            or "glm-5.2"
        ).strip()
        timeout = float(
            _env("API_TIMEOUT")
            or os.getenv("RUBRIC_REWARD_API_TIMEOUT", "")
            or os.getenv("RUBRIC_JUDGE_API_TIMEOUT", "60")
        )
        max_retries = int(
            _env("API_MAX_RETRIES")
            or os.getenv("RUBRIC_REWARD_API_MAX_RETRIES", "")
            or os.getenv("RUBRIC_JUDGE_API_MAX_RETRIES", "2")
        )
        self.api_concurrency_per_key = int(_env("API_CONCURRENCY_PER_KEY", "10"))
        self.api_fallback = _env("API_FALLBACK", "error").strip().lower()
        if timeout <= 0 or max_retries < 0 or self.api_concurrency_per_key <= 0:
            raise ValueError("API timeout/concurrency must be positive and retries non-negative")
        if self.api_fallback not in {
            "error",
            "raise",
            "none",
            "rules",
            "rule",
            "local",
            "ndcg_only_group",
            "group_ndcg",
            "ndcg",
        }:
            raise ValueError(
                f"Unsupported {ENV_PREFIX}_API_FALLBACK={self.api_fallback}; "
                "use error, rules, or ndcg_only_group"
            )

        # judge_api.py reads these two request-shape controls from the shared
        # names. setdefault preserves an explicitly configured shared value.
        api_max_tokens = _env("API_MAX_TOKENS").strip()
        api_thinking = _env("API_THINKING").strip()
        api_min_interval = _env("API_MIN_INTERVAL").strip()
        if api_max_tokens:
            os.environ.setdefault("RUBRIC_REWARD_API_MAX_TOKENS", api_max_tokens)
        if api_thinking:
            os.environ.setdefault("RUBRIC_REWARD_API_THINKING", api_thinking)
        if api_min_interval:
            os.environ.setdefault("RUBRIC_JUDGE_API_MIN_INTERVAL", api_min_interval)

        keys = _parse_api_keys(_env("API_KEYS"))
        if not keys:
            keys = _parse_api_keys(
                _env("API_KEY")
                or os.getenv("RUBRIC_REWARD_API_KEY", "")
                or os.getenv("RUBRIC_JUDGE_API_KEY", "")
                or os.getenv("BIGMODEL_API_KEY", "")
            )
        if not keys:
            keys = project_keys
        if not keys:
            # Mock and unauthenticated local OpenAI-compatible endpoints may
            # legitimately use an empty key. Official GLM will reject it here.
            keys = [""]

        self.api_clients = [
            build_target_relevance_judge_api_client(
                provider=self.api_provider,
                base_url=base_url,
                api_key=key,
                model=self.api_model,
                timeout=timeout,
                max_retries=max_retries,
            )
            for key in keys
        ]
        configured_total_concurrency = _env("API_CONCURRENCY").strip()
        self.api_concurrency = (
            int(configured_total_concurrency)
            if configured_total_concurrency
            else self.api_concurrency_per_key * len(self.api_clients)
        )
        if self.api_concurrency <= 0:
            raise ValueError(f"{ENV_PREFIX}_API_CONCURRENCY must be positive")
        self._api_semaphores = [
            threading.BoundedSemaphore(self.api_concurrency_per_key)
            for _ in self.api_clients
        ]
        configured_key_attempts = _env("API_KEY_ATTEMPTS").strip()
        self.api_key_attempts = (
            int(configured_key_attempts)
            if configured_key_attempts
            else len(self.api_clients)
        )
        if self.api_key_attempts <= 0:
            raise ValueError(f"{ENV_PREFIX}_API_KEY_ATTEMPTS must be positive")

    def _reserve_api_start_index(self) -> int:
        with self._api_lock:
            index = self._api_next_client % len(self.api_clients)
            self._api_next_client += 1
        return index

    def _score_api_one(
        self,
        history: str,
        completion: str,
        target_item: str,
    ) -> tuple[float | None, str, float]:
        cache_key = (
            self.api_provider,
            self.api_model,
            history,
            completion,
            target_item,
        )
        with self._api_lock:
            cached = self._api_cache.get(cache_key)
        if cached is not None:
            score, original_source = cached
            return score, f"cache_{original_source}", 0.0

        started = time.perf_counter()
        score: float | None = None
        errors: list[str] = []
        provider = self.api_provider
        start_index = self._reserve_api_start_index()
        for attempt in range(self.api_key_attempts):
            raw = ""
            try:
                client_index = (start_index + attempt) % len(self.api_clients)
                client = self.api_clients[client_index]
                semaphore = self._api_semaphores[client_index]
                with semaphore:
                    score, raw, result_provider = _api_score_norm(
                        client, history, completion, target_item
                    )
                provider = result_provider or provider
            except Exception as exc:
                raw = f"{type(exc).__name__}: {exc}"
            if score is not None:
                break
            errors.append(raw.replace("\n", " ")[:500])
        elapsed = time.perf_counter() - started
        source = "api_success"
        if score is None:
            if self.api_fallback in {"rules", "rule", "local"}:
                score = float(rule_score(history, completion)["score_norm"])
                source = "fallback_rules"
            elif self.api_fallback in {
                "ndcg_only_group",
                "group_ndcg",
                "ndcg",
            }:
                return None, "fallback_ndcg_only_pending", elapsed
            else:
                safe_error = " | ".join(errors)[:1000]
                raise RuntimeError(
                    "Rubric API judge failed with fallback disabled; "
                    f"provider={provider!r}, model={self.api_model!r}, "
                    f"error={safe_error!r}"
                )
        score = min(1.0, max(0.0, float(score)))
        with self._api_lock:
            self._api_cache[cache_key] = (score, source)
        return score, source, elapsed

    def _score_api_batch(
        self,
        histories: list[str],
        completions: list[str],
        target_items: list[str],
    ) -> list[float | None]:
        n = len(completions)
        concurrency = max(1, min(self.api_concurrency, n))
        scores: list[float | None] = [None for _ in range(n)]
        sources = ["" for _ in range(n)]
        item_seconds = [0.0 for _ in range(n)]

        if concurrency == 1:
            for index, (history, completion, target_item) in enumerate(
                zip(histories, completions, target_items)
            ):
                scores[index], sources[index], item_seconds[index] = self._score_api_one(
                    history, completion, target_item
                )
        else:
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = {
                    executor.submit(
                        self._score_api_one,
                        history,
                        completion,
                        target_item,
                    ): index
                    for index, (history, completion, target_item) in enumerate(
                        zip(histories, completions, target_items)
                    )
                }
                for future in as_completed(futures):
                    index = futures[future]
                    scores[index], sources[index], item_seconds[index] = future.result()

        self.last_batch_meta = {
            "provider": self.api_provider,
            "model": self.api_model,
            "key_count": len(self.api_clients),
            "concurrency": concurrency,
            "concurrency_per_key": self.api_concurrency_per_key,
            "key_attempts_per_item": self.api_key_attempts,
            "fallback": self.api_fallback,
            "source_counts": dict(Counter(sources)),
            "failed_items": sum(score is None for score in scores),
            "api_item_sec_mean": _safe_mean(item_seconds),
            "api_item_sec_max": max(item_seconds, default=0.0),
        }
        return scores

    def score(
        self,
        histories: list[str],
        completions: list[str],
        query_embeddings,
        target_items: list[str] | None = None,
    ) -> list[float | None]:
        if len(histories) != len(completions):
            raise ValueError("history and completion batch sizes differ")
        targets = target_items or ["" for _ in completions]
        if len(targets) != len(completions):
            raise ValueError("target and completion batch sizes differ")
        if self.mode in {"api", "llm", "judge", "llm_judge"}:
            if _env_bool(f"{ENV_PREFIX}_STRICT_API_TARGET", True) and any(
                not target for target in targets
            ):
                raise RuntimeError(
                    "API rubric scoring requires a non-empty target_item_text "
                    "or target_item_title for every completion"
                )
            return self._score_api_batch(histories, completions, targets)
        if self.mode in {"rules", "rule", "local"}:
            self.last_batch_meta = {
                "source_counts": {"rules": len(completions)},
                "concurrency": 1,
            }
            return [
                float(rule_score(history, completion)["score_norm"])
                for history, completion in zip(histories, completions)
            ]

        import torch

        if self.head is None or self.config is None:
            raise RuntimeError("Rubric classifier was not initialized")
        embeddings = query_embeddings
        if self.config.encoder_output_dim > 0:
            embeddings = embeddings[:, : self.config.encoder_output_dim]
        if int(embeddings.shape[1]) != int(self.config.input_dim):
            raise RuntimeError(
                "Rubric scorer input dimension does not match query embeddings: "
                f"{self.config.input_dim} != {embeddings.shape[1]}"
            )
        with torch.inference_mode():
            values = self.head.quality(embeddings.to(self.device)).float().cpu().tolist()
        self.last_batch_meta = {
            "source_counts": {"classifier": len(completions)},
            "concurrency": 1,
        }
        return [min(1.0, max(0.0, float(value))) for value in values]


def _get_state() -> CotRetrievalRewardState:
    global _STATE
    if _STATE is None:
        _STATE = CotRetrievalRewardState(env_prefix=ENV_PREFIX)
    return _STATE


def _get_rubric_scorer(state: CotRetrievalRewardState) -> FrozenRubricScorer:
    global _RUBRIC_SCORER
    if _RUBRIC_SCORER is None:
        _RUBRIC_SCORER = FrozenRubricScorer(state)
    return _RUBRIC_SCORER


def _consistent_group_value(
    values: list[Any],
    indices: list[int],
    *,
    name: str,
) -> Any | None:
    present = [values[index] for index in indices if values[index] not in {None, ""}]
    if not present:
        return None
    first = present[0]
    for value in present[1:]:
        if isinstance(first, float) or isinstance(value, float):
            if not math.isclose(float(first), float(value), rel_tol=0.0, abs_tol=1e-12):
                raise RuntimeError(f"{name} differs inside one GRPO prompt group")
        elif value != first:
            raise RuntimeError(f"{name} differs inside one GRPO prompt group")
    if len(present) != len(indices):
        raise RuntimeError(f"{name} is only partially populated inside one group")
    return first


def _resolve_reference_ndcgs(
    *,
    state: CotRetrievalRewardState,
    grouped_indices: dict[tuple[str, str | int], list[int]],
    histories: list[str],
    target_ids: list[int | None],
    history_sets: list[set[int]],
    reference_ndcg_values: list[float | None],
    reference_rank_values: list[int | None],
    reference_cots: list[str],
    ndcg_k: int,
    temperature: float,
) -> tuple[list[float], list[int | None], dict[str, int]]:
    output_ndcgs = [0.0 for _ in histories]
    output_ranks: list[int | None] = [None for _ in histories]
    sources = {"cached_metadata": 0, "cached_rank": 0, "computed_cot": 0, "memory_cache": 0}
    pending: list[tuple[list[int], tuple[Any, ...], str]] = []

    for indices in grouped_indices.values():
        supplied_ndcg = _consistent_group_value(
            reference_ndcg_values, indices, name="reference_ndcg"
        )
        supplied_rank = _consistent_group_value(
            reference_rank_values, indices, name="reference_rank"
        )
        supplied_cot = _consistent_group_value(
            reference_cots, indices, name="reference_cot"
        )
        representative = indices[0]

        if supplied_ndcg is not None:
            value = float(supplied_ndcg)
            if not 0.0 <= value <= 1.0:
                raise RuntimeError(f"reference_ndcg must be in [0, 1], got {value}")
            rank = int(supplied_rank) if supplied_rank is not None else None
            source = "cached_metadata"
            for index in indices:
                output_ndcgs[index] = value
                output_ranks[index] = rank
            sources[source] += 1
            continue

        if supplied_rank is not None:
            rank = int(supplied_rank)
            value = _ndcg_at_rank(rank, ndcg_k)
            for index in indices:
                output_ndcgs[index] = value
                output_ranks[index] = rank
            sources["cached_rank"] += 1
            continue

        if not supplied_cot:
            raise RuntimeError(
                "cot_rubric_ndcg1000_gain requires reference_ndcg, "
                "reference_rank, or a non-empty fixed reference_cot"
            )
        target_id = target_ids[representative]
        cache_key = (
            histories[representative],
            str(supplied_cot),
            target_id,
            tuple(sorted(history_sets[representative])),
            ndcg_k,
            state.model_path,
        )
        cached = _REFERENCE_NDCG_CACHE.get(cache_key)
        if cached is not None:
            value, rank = cached
            for index in indices:
                output_ndcgs[index] = value
                output_ranks[index] = rank
            sources["memory_cache"] += 1
            continue
        pending.append((indices, cache_key, str(supplied_cot)))

    if pending:
        queries = [
            f"{histories[indices[0]]}\n\nRecommendation reasoning:\n{reference_cot}"
            for indices, _, reference_cot in pending
        ]
        pending_targets = [target_ids[indices[0]] for indices, _, _ in pending]
        pending_history_sets = [history_sets[indices[0]] for indices, _, _ in pending]
        embeddings = state.encode_queries(queries)
        _, _, ndcgs, ranks, _ = state.score_queries(
            embeddings,
            pending_targets,
            pending_history_sets,
            temperature=temperature,
            ndcg_k=ndcg_k,
        )
        for (indices, cache_key, _), value, rank in zip(pending, ndcgs, ranks):
            _REFERENCE_NDCG_CACHE[cache_key] = (float(value), int(rank))
            for index in indices:
                output_ndcgs[index] = float(value)
                output_ranks[index] = int(rank)
            sources["computed_cot"] += 1

    return output_ndcgs, output_ranks, sources


def _log_components(summary: dict[str, Any], items: list[dict[str, Any]]) -> None:
    log_path = _env("COMPONENT_LOG").strip()
    if log_path:
        rank = os.getenv("RANK", "0")
        local_rank = os.getenv("LOCAL_RANK", "0")
        path = Path(log_path.format(rank=rank, local_rank=local_rank))
        path.parent.mkdir(parents=True, exist_ok=True)
        record: dict[str, Any] = {
            "kind": "cot_rubric_ndcg1000_gain",
            "call_index": _CALL_COUNT,
            "rank": rank,
            "local_rank": local_rank,
            "summary": summary,
        }
        if _env_bool(f"{ENV_PREFIX}_COMPONENT_LOG_DETAILS", True):
            record["items"] = items
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    log_every = int(_env("LOG_EVERY", "0"))
    if log_every > 0 and _CALL_COUNT % log_every == 0:
        print(
            "[cot_rubric_ndcg1000_gain] "
            f"call={_CALL_COUNT} n={summary['n']} groups={summary['groups']} "
            f"win_rate={summary['reference_win_rate']:.6f} "
            f"delta_ndcg_mean={summary['delta_ndcg_mean']:.6f} "
            f"rubric_mean={summary['rubric_mean']:.6f} "
            f"rubric_fallback_groups={summary['rubric_fallback_groups']} "
            f"zero_joint_groups={summary['zero_joint_std_groups']} "
            f"total_sec={summary['total_sec']:.3f}",
            flush=True,
        )


class CotRubricNdcg1000GainReward(ORM):
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
        reference_cot=None,
        ref_cot=None,
        origin_cot=None,
        selected_cot=None,
        reference_ndcg=None,
        ref_ndcg=None,
        origin_ndcg=None,
        reference_rank=None,
        ref_rank=None,
        origin_rank=None,
        **kwargs,
    ) -> list[float]:
        global _CALL_COUNT
        started = time.perf_counter()
        _CALL_COUNT += 1

        completion_texts = [str(value or "").strip() for value in completions]
        n = len(completion_texts)
        if n == 0:
            return []
        histories = _as_text_list(
            user_history if user_history is not None else source_prompt, n
        )
        if any(not history for history in histories):
            raise RuntimeError(
                "cot_rubric_ndcg1000_gain requires non-empty user_history metadata"
            )
        target_ids = _as_int_list(target_item_id, n)
        target_items = _first_text_list(
            n,
            target_item_text,
            target_item_title,
            kwargs.get("target_item_text"),
            kwargs.get("target_item_title"),
        )
        history_sets = _as_int_set_list(
            history_item_ids if history_item_ids is not None else history_item_id, n
        )
        example_ids = _as_text_list(kwargs.get("example_id"), n)

        ndcg_k = int(_env("K", "1000"))
        temperature = float(_env("TEMPERATURE", "0.05"))
        similarity_weight = float(_env("SIM_WEIGHT", "0.6"))
        joint_weight = float(_env("JOINT_WEIGHT", "0.4"))
        negative_gain_weight = float(_env("NEGATIVE_GAIN_WEIGHT", "1.0"))
        quality_power = float(_env("RUBRIC_POWER", "1.0"))
        epsilon = float(_env("ZSCORE_EPSILON", "1e-6"))
        group_size = int(_env("GROUP_SIZE", "4"))
        strict_group_size = _env_bool(f"{ENV_PREFIX}_STRICT_GROUP_SIZE", True)

        if ndcg_k <= 0 or group_size <= 0 or temperature <= 0 or epsilon <= 0:
            raise ValueError("K, group size, temperature, and epsilon must be positive")
        if similarity_weight < 0 or joint_weight < 0 or similarity_weight + joint_weight <= 0:
            raise ValueError("reward weights must be non-negative with a positive sum")
        if negative_gain_weight < 0 or quality_power < 0:
            raise ValueError("gain weight and rubric power must be non-negative")

        grouped_indices: dict[tuple[str, str | int], list[int]] = {}
        for index in range(n):
            key = _group_key(index, example_ids, histories, group_size)
            grouped_indices.setdefault(key, []).append(index)
        if strict_group_size:
            invalid_groups = {
                str(key): len(indices)
                for key, indices in grouped_indices.items()
                if len(indices) != group_size
            }
            if invalid_groups:
                raise RuntimeError(
                    f"GRPO groups must contain exactly {group_size} completions: "
                    f"{invalid_groups}"
                )

        queries = [
            f"{history}\n\nRecommendation reasoning:\n{completion}"
            if completion
            else history
            for history, completion in zip(histories, completion_texts)
        ]
        state = _get_state()
        query_embeddings = state.encode_queries(queries)
        similarities, cosines, new_ndcgs, new_ranks, masked_counts = state.score_queries(
            query_embeddings,
            target_ids,
            history_sets,
            temperature=temperature,
            ndcg_k=ndcg_k,
        )

        rubric_started = time.perf_counter()
        rubric_scorer = _get_rubric_scorer(state)
        raw_rubric_scores = rubric_scorer.score(
            histories,
            completion_texts,
            query_embeddings,
            target_items,
        )
        rubric_scores, rubric_fallback_mask, rubric_fallback_groups = (
            _apply_group_ndcg_only_fallback(raw_rubric_scores, grouped_indices)
        )
        rubric_sec = time.perf_counter() - rubric_started

        reference_cots = _first_text_list(
            n,
            reference_cot,
            ref_cot,
            origin_cot,
            selected_cot,
            kwargs.get("reference_cot"),
            kwargs.get("ref_cot"),
            kwargs.get("origin_cot"),
            kwargs.get("selected_cot"),
            kwargs.get("cot"),
        )
        reference_ndcg_values = _first_float_list(
            n,
            reference_ndcg,
            ref_ndcg,
            origin_ndcg,
            kwargs.get("reference_ndcg"),
            kwargs.get("ref_ndcg"),
            kwargs.get("origin_ndcg"),
        )
        reference_rank_values = _first_int_list(
            n,
            reference_rank,
            ref_rank,
            origin_rank,
            kwargs.get("reference_rank"),
            kwargs.get("ref_rank"),
            kwargs.get("origin_rank"),
        )
        reference_started = time.perf_counter()
        reference_ndcgs, reference_ranks, reference_sources = _resolve_reference_ndcgs(
            state=state,
            grouped_indices=grouped_indices,
            histories=histories,
            target_ids=target_ids,
            history_sets=history_sets,
            reference_ndcg_values=reference_ndcg_values,
            reference_rank_values=reference_rank_values,
            reference_cots=reference_cots,
            ndcg_k=ndcg_k,
            temperature=temperature,
        )
        reference_sec = time.perf_counter() - reference_started

        delta_ndcgs = [new - old for new, old in zip(new_ndcgs, reference_ndcgs)]
        joint_gains = [
            _joint_gain(
                quality,
                delta,
                quality_power=quality_power,
                negative_gain_weight=negative_gain_weight,
            )
            for quality, delta in zip(rubric_scores, delta_ndcgs)
        ]

        rewards = [0.0 for _ in range(n)]
        similarity_z = [0.0 for _ in range(n)]
        joint_z = [0.0 for _ in range(n)]
        group_conflicts: list[float] = []
        zero_similarity_std_groups = 0
        zero_joint_std_groups = 0
        for indices in grouped_indices.values():
            group_similarities = [similarities[index] for index in indices]
            group_joint = [joint_gains[index] for index in indices]
            group_rewards, group_similarity_z, group_joint_z = _combine_group_rewards(
                group_similarities,
                group_joint,
                similarity_weight=similarity_weight,
                joint_weight=joint_weight,
                epsilon=epsilon,
            )
            zero_similarity_std_groups += int(max(group_similarity_z) == min(group_similarity_z))
            zero_joint_std_groups += int(max(group_joint_z) == min(group_joint_z))
            group_conflicts.append(
                _pairwise_conflict_rate(group_similarities, group_joint)
            )
            for offset, index in enumerate(indices):
                rewards[index] = float(group_rewards[offset])
                similarity_z[index] = float(group_similarity_z[offset])
                joint_z[index] = float(group_joint_z[offset])

        items = [
            {
                "index": index,
                "example_id": example_ids[index],
                "target_item_id": target_ids[index],
                "target_cosine": cosines[index],
                "relative_similarity_logprob": similarities[index],
                "new_rank": new_ranks[index],
                f"new_ndcg@{ndcg_k}": new_ndcgs[index],
                "reference_rank": reference_ranks[index],
                f"reference_ndcg@{ndcg_k}": reference_ndcgs[index],
                "delta_ndcg": delta_ndcgs[index],
                "rubric_score": rubric_scores[index],
                "rubric_fallback_ndcg_only": rubric_fallback_mask[index],
                "joint_gain": joint_gains[index],
                "similarity_z": similarity_z[index],
                "joint_z": joint_z[index],
                "reward": rewards[index],
                "masked_history_items": masked_counts[index],
            }
            for index in range(n)
        ]
        wins = sum(delta > 0.0 for delta in delta_ndcgs)
        losses = sum(delta < 0.0 for delta in delta_ndcgs)
        ties = n - wins - losses
        summary = {
            "n": n,
            "groups": len(grouped_indices),
            "group_size": group_size,
            "embedding_model": state.model_path,
            "rubric_scorer": rubric_scorer.mode,
            "rubric_batch": rubric_scorer.last_batch_meta,
            "candidate_items": len(state.item_ids),
            "temperature": temperature,
            "ndcg_k": ndcg_k,
            "similarity_weight": similarity_weight,
            "joint_weight": joint_weight,
            "negative_gain_weight": negative_gain_weight,
            "rubric_power": quality_power,
            "relative_similarity_mean": _safe_mean(similarities),
            "new_ndcg_mean": _safe_mean(new_ndcgs),
            "reference_ndcg_mean": _safe_mean(reference_ndcgs),
            "delta_ndcg_mean": _safe_mean(delta_ndcgs),
            "reference_win_rate": wins / n,
            "reference_loss_rate": losses / n,
            "reference_tie_rate": ties / n,
            "rubric_mean": _safe_mean(rubric_scores),
            "rubric_fallback_groups": rubric_fallback_groups,
            "rubric_fallback_items": sum(rubric_fallback_mask),
            "joint_gain_mean": _safe_mean(joint_gains),
            "new_rank_mean": _safe_mean([float(rank) for rank in new_ranks]),
            "pairwise_conflict_rate": _safe_mean(group_conflicts),
            "zero_similarity_std_groups": zero_similarity_std_groups,
            "zero_joint_std_groups": zero_joint_std_groups,
            "reference_sources": reference_sources,
            "reward_mean": _safe_mean(rewards),
            "reward_min": min(rewards),
            "reward_max": max(rewards),
            "rubric_sec": rubric_sec,
            "reference_sec": reference_sec,
            "total_sec": time.perf_counter() - started,
        }
        _log_components(summary, items)
        return rewards


orms["cot_rubric_ndcg1000_gain"] = CotRubricNdcg1000GainReward
