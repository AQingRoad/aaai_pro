#!/usr/bin/env python3
"""Similarity plus reference-relative NDCG@K reward for recommendation GRPO.

This plugin is the no-rubric control for ``cot_rubric_ndcg1000_gain_reward``.
For completion i in one prompt group, it computes

    delta_ndcg_i = NDCG@K(new_cot_i) - NDCG@K(reference_cot)
    reward_i = w_similarity * group_zscore(similarity_i)
               + w_gain * group_zscore(delta_ndcg_i)

The reference score must be cached in prompt metadata as ``reference_ndcg`` or
``reference_rank``. The plugin never calls a rubric scorer or an external API.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
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


ENV_PREFIX = "COT_SIM_NDCG1000_GAIN"
_STATE: "CotRetrievalRewardState | None" = None
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


def _resolve_cached_references(
    *,
    grouped_indices: dict[tuple[str, str | int], list[int]],
    reference_ndcg_values: list[float | None],
    reference_rank_values: list[int | None],
    ndcg_k: int,
) -> tuple[list[float], list[int | None], dict[str, int]]:
    """Resolve one fixed reference value per prompt group without online calls."""
    n = len(reference_ndcg_values)
    output_ndcgs = [0.0 for _ in range(n)]
    output_ranks: list[int | None] = [None for _ in range(n)]
    sources = {"cached_metadata": 0, "cached_rank": 0}

    for indices in grouped_indices.values():
        supplied_ndcg = _consistent_group_value(
            reference_ndcg_values, indices, name="reference_ndcg"
        )
        supplied_rank = _consistent_group_value(
            reference_rank_values, indices, name="reference_rank"
        )
        if supplied_ndcg is not None:
            value = float(supplied_ndcg)
            if not 0.0 <= value <= 1.0:
                raise RuntimeError(f"reference_ndcg must be in [0, 1], got {value}")
            rank = int(supplied_rank) if supplied_rank is not None else None
            source = "cached_metadata"
        elif supplied_rank is not None:
            rank = int(supplied_rank)
            if rank <= 0:
                raise RuntimeError(f"reference_rank must be positive, got {rank}")
            value = _ndcg_at_rank(rank, ndcg_k)
            source = "cached_rank"
        else:
            raise RuntimeError(
                "cot_sim_ndcg1000_gain requires cached reference_ndcg or "
                "reference_rank metadata for every prompt group"
            )
        for index in indices:
            output_ndcgs[index] = value
            output_ranks[index] = rank
        sources[source] += 1

    return output_ndcgs, output_ranks, sources


def _combine_group_rewards(
    similarities: list[float],
    ndcg_gains: list[float],
    *,
    similarity_weight: float,
    gain_weight: float,
    epsilon: float,
) -> tuple[list[float], list[float], list[float]]:
    if len(similarities) != len(ndcg_gains):
        raise ValueError("similarity and NDCG-gain group sizes differ")
    similarity_z = _group_zscore(similarities, epsilon=epsilon)
    gain_z = _group_zscore(ndcg_gains, epsilon=epsilon)
    rewards = [
        similarity_weight * similarity_value + gain_weight * gain_value
        for similarity_value, gain_value in zip(similarity_z, gain_z)
    ]
    return rewards, similarity_z, gain_z


def _get_state() -> CotRetrievalRewardState:
    global _STATE
    if _STATE is None:
        _STATE = CotRetrievalRewardState(env_prefix=ENV_PREFIX)
    return _STATE


def _log_components(summary: dict[str, Any], items: list[dict[str, Any]]) -> None:
    log_path = _env("COMPONENT_LOG").strip()
    if log_path:
        rank = os.getenv("RANK", "0")
        local_rank = os.getenv("LOCAL_RANK", "0")
        path = Path(log_path.format(rank=rank, local_rank=local_rank))
        path.parent.mkdir(parents=True, exist_ok=True)
        record: dict[str, Any] = {
            "kind": "cot_sim_ndcg1000_gain",
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
            "[cot_sim_ndcg1000_gain] "
            f"call={_CALL_COUNT} n={summary['n']} groups={summary['groups']} "
            f"win_rate={summary['reference_win_rate']:.6f} "
            f"delta_ndcg_mean={summary['delta_ndcg_mean']:.6f} "
            f"zero_gain_groups={summary['zero_gain_std_groups']} "
            f"total_sec={summary['total_sec']:.3f}",
            flush=True,
        )


class CotSimilarityNdcg1000GainReward(ORM):
    def __call__(
        self,
        completions,
        user_history=None,
        source_prompt=None,
        target_item_id=None,
        history_item_ids=None,
        history_item_id=None,
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
                "cot_sim_ndcg1000_gain requires non-empty user_history metadata"
            )
        target_ids = _as_int_list(target_item_id, n)
        history_sets = _as_int_set_list(
            history_item_ids if history_item_ids is not None else history_item_id, n
        )
        example_ids = _as_text_list(kwargs.get("example_id"), n)

        ndcg_k = int(_env("K", "1000"))
        temperature = float(_env("TEMPERATURE", "0.05"))
        similarity_weight = float(_env("SIM_WEIGHT", "0.6"))
        gain_weight = float(_env("GAIN_WEIGHT", "0.4"))
        epsilon = float(_env("ZSCORE_EPSILON", "1e-6"))
        group_size = int(_env("GROUP_SIZE", "4"))
        strict_group_size = _env_bool(f"{ENV_PREFIX}_STRICT_GROUP_SIZE", True)

        if ndcg_k <= 0 or group_size <= 0 or temperature <= 0 or epsilon <= 0:
            raise ValueError("K, group size, temperature, and epsilon must be positive")
        if similarity_weight < 0 or gain_weight < 0:
            raise ValueError("reward weights must be non-negative")
        if similarity_weight + gain_weight <= 0:
            raise ValueError("at least one reward weight must be positive")

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
        similarities, cosines, new_ndcgs, new_ranks, masked_counts = (
            state.score_queries(
                query_embeddings,
                target_ids,
                history_sets,
                temperature=temperature,
                ndcg_k=ndcg_k,
            )
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
        reference_ndcgs, reference_ranks, reference_sources = (
            _resolve_cached_references(
                grouped_indices=grouped_indices,
                reference_ndcg_values=reference_ndcg_values,
                reference_rank_values=reference_rank_values,
                ndcg_k=ndcg_k,
            )
        )
        delta_ndcgs = [new - old for new, old in zip(new_ndcgs, reference_ndcgs)]

        rewards = [0.0 for _ in range(n)]
        similarity_z = [0.0 for _ in range(n)]
        gain_z = [0.0 for _ in range(n)]
        group_conflicts: list[float] = []
        zero_similarity_std_groups = 0
        zero_gain_std_groups = 0
        for indices in grouped_indices.values():
            group_similarities = [similarities[index] for index in indices]
            group_gains = [delta_ndcgs[index] for index in indices]
            group_rewards, group_similarity_z, group_gain_z = _combine_group_rewards(
                group_similarities,
                group_gains,
                similarity_weight=similarity_weight,
                gain_weight=gain_weight,
                epsilon=epsilon,
            )
            zero_similarity_std_groups += int(
                max(group_similarity_z) == min(group_similarity_z)
            )
            zero_gain_std_groups += int(max(group_gain_z) == min(group_gain_z))
            group_conflicts.append(
                _pairwise_conflict_rate(group_similarities, group_gains)
            )
            for offset, index in enumerate(indices):
                rewards[index] = float(group_rewards[offset])
                similarity_z[index] = float(group_similarity_z[offset])
                gain_z[index] = float(group_gain_z[offset])

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
                "similarity_z": similarity_z[index],
                "ndcg_gain_z": gain_z[index],
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
            "candidate_items": len(state.item_ids),
            "temperature": temperature,
            "ndcg_k": ndcg_k,
            "similarity_weight": similarity_weight,
            "gain_weight": gain_weight,
            "relative_similarity_mean": _safe_mean(similarities),
            "target_cosine_mean": _safe_mean(cosines),
            "new_ndcg_mean": _safe_mean(new_ndcgs),
            "reference_ndcg_mean": _safe_mean(reference_ndcgs),
            "delta_ndcg_mean": _safe_mean(delta_ndcgs),
            "reference_win_rate": wins / n,
            "reference_loss_rate": losses / n,
            "reference_tie_rate": ties / n,
            "new_rank_mean": _safe_mean([float(rank) for rank in new_ranks]),
            "pairwise_conflict_rate": _safe_mean(group_conflicts),
            "zero_similarity_std_groups": zero_similarity_std_groups,
            "zero_gain_std_groups": zero_gain_std_groups,
            "reference_sources": reference_sources,
            "reward_mean": _safe_mean(rewards),
            "reward_min": min(rewards),
            "reward_max": max(rewards),
            "total_sec": time.perf_counter() - started,
        }
        _log_components(summary, items)
        return rewards


orms["cot_sim_ndcg1000_gain"] = CotSimilarityNdcg1000GainReward
