#!/usr/bin/env python3
"""GRPO reward: CoT-trained relative similarity plus NDCG@K.

The reward intentionally has only two components:

1. Full-catalog target log-softmax similarity.
2. Full-catalog NDCG@K for the same target.

Both components are standardized inside each GRPO generation group before the
configured weighted sum. The plugin does not inspect, reward, or penalize the
completion format.
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

from manu_src.scripts.pre_datas.format_positive import format_positive


_STATE: "CotRetrievalRewardState | None" = None
_CALL_COUNT = 0


def _env_bool(name: str, default: bool = True) -> bool:
    value = os.getenv(name, "").strip().lower()
    if not value:
        return default
    return value not in {"0", "false", "off", "no", "none", "null"}


def _safe_mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _expand_values(value: Any, n: int, default: Any = "") -> list[Any]:
    """Broadcast prompt-level metadata to completion-level metadata."""
    if value is None:
        return [default for _ in range(n)]
    if isinstance(value, (str, int, float)):
        return [value for _ in range(n)]
    try:
        values = list(value)
    except TypeError:
        return [value for _ in range(n)]
    if len(values) == n:
        return values
    if len(values) == 1:
        return values * n
    if values and n % len(values) == 0:
        repeats = n // len(values)
        return [item for item in values for _ in range(repeats)]
    raise ValueError(
        f"Cannot align metadata of length {len(values)} with {n} completions"
    )


def _as_text_list(value: Any, n: int) -> list[str]:
    return [str(item or "").strip() for item in _expand_values(value, n, "")]


def _as_int_list(value: Any, n: int) -> list[int | None]:
    output: list[int | None] = []
    for item in _expand_values(value, n, None):
        try:
            output.append(int(item))
        except (TypeError, ValueError):
            output.append(None)
    return output


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
        return {int(part) for part in text.replace(",", " ").split() if part.lstrip("-").isdigit()}
    if isinstance(value, (int, float)):
        return {int(value)}
    try:
        return {int(item) for item in value if item is not None}
    except TypeError:
        return set()


def _as_int_set_list(value: Any, n: int) -> list[set[int]]:
    if value is None:
        return [set() for _ in range(n)]
    if isinstance(value, str):
        parsed = _parse_int_set(value)
        return [set(parsed) for _ in range(n)]
    try:
        values = list(value)
    except TypeError:
        parsed = _parse_int_set(value)
        return [set(parsed) for _ in range(n)]

    # A flat integer list represents one history and must be broadcast.
    if not values or all(isinstance(item, (int, float)) for item in values):
        parsed = _parse_int_set(values)
        return [set(parsed) for _ in range(n)]

    return [_parse_int_set(item) for item in _expand_values(values, n, [])]


def _group_zscore(values: list[float], epsilon: float = 1e-6) -> list[float]:
    """Population z-score; a constant component contributes zero reward."""
    if not values:
        return []
    mean = _safe_mean(values)
    variance = _safe_mean([(value - mean) ** 2 for value in values])
    std = math.sqrt(max(0.0, variance))
    if std < epsilon:
        return [0.0 for _ in values]
    return [(value - mean) / (std + epsilon) for value in values]


def _ndcg_at_rank(rank: int, k: int) -> float:
    if rank <= 0 or rank > k:
        return 0.0
    return 1.0 / math.log2(rank + 1.0)


def _pairwise_conflict_rate(similarities: list[float], ndcgs: list[float]) -> float:
    conflicts = 0
    comparable = 0
    for left in range(len(similarities)):
        for right in range(left + 1, len(similarities)):
            sim_delta = similarities[left] - similarities[right]
            ndcg_delta = ndcgs[left] - ndcgs[right]
            if sim_delta == 0.0 or ndcg_delta == 0.0:
                continue
            comparable += 1
            conflicts += int(sim_delta * ndcg_delta < 0.0)
    return conflicts / comparable if comparable else 0.0


def _combine_group_rewards(
    similarities: list[float],
    ndcgs: list[float],
    similarity_weight: float,
    ndcg_weight: float,
) -> tuple[list[float], list[float], list[float]]:
    if len(similarities) != len(ndcgs):
        raise ValueError("similarity and NDCG group sizes differ")
    similarity_z = _group_zscore(similarities)
    ndcg_z = _group_zscore(ndcgs)
    rewards = [
        similarity_weight * sim_z + ndcg_weight * ndcg_value_z
        for sim_z, ndcg_value_z in zip(similarity_z, ndcg_z)
    ]
    return rewards, similarity_z, ndcg_z


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path} line {line_number} is invalid JSON") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path} line {line_number} is not an object")
            rows.append(row)
    return rows


class CotRetrievalRewardState:
    """Frozen CoT-trained embedder and its full-catalog item table."""

    def __init__(self) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        from manu_src.scripts.models.train_embedding import encode, format_query

        self.torch = torch
        self.encode = encode
        self.format_query = format_query
        self.model_path = os.getenv("COT_SIM_NDCG_EMBEDDING_MODEL", "").strip()
        self.item_info_path = Path(os.getenv("COT_SIM_NDCG_ITEM_INFO", "").strip())
        self.max_length = int(os.getenv("COT_SIM_NDCG_MAX_LENGTH", "4096"))
        self.item_batch_size = int(os.getenv("COT_SIM_NDCG_ITEM_BATCH_SIZE", "128"))
        self.query_batch_size = int(os.getenv("COT_SIM_NDCG_QUERY_BATCH_SIZE", "16"))
        self.device = torch.device(
            os.getenv("COT_SIM_NDCG_DEVICE", f"cuda:{os.getenv('LOCAL_RANK', '0')}")
        )
        self.dtype_name = os.getenv("COT_SIM_NDCG_TORCH_DTYPE", "bfloat16").lower()
        self.attn_implementation = os.getenv(
            "COT_SIM_NDCG_ATTN_IMPLEMENTATION", "flash_attention_2"
        )
        self.expected_items = int(os.getenv("COT_SIM_NDCG_EXPECTED_ITEMS", "12000"))

        if not self.model_path or not Path(self.model_path).is_dir():
            raise RuntimeError(
                "COT_SIM_NDCG_EMBEDDING_MODEL must point to a frozen CoT-trained checkpoint"
            )
        if not self.item_info_path.is_file():
            raise RuntimeError("COT_SIM_NDCG_ITEM_INFO must point to item_info.jsonl")
        if self.max_length <= 0 or self.item_batch_size <= 0 or self.query_batch_size <= 0:
            raise ValueError("embedding lengths and batch sizes must be positive")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the online embedding reward")

        dtype_map = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
        }
        if self.dtype_name not in dtype_map:
            raise ValueError(f"Unsupported COT_SIM_NDCG_TORCH_DTYPE={self.dtype_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            use_fast=True,
            padding_side="left",
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModel.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=dtype_map[self.dtype_name],
            attn_implementation=self.attn_implementation,
        ).to(self.device)
        self.model.eval()

        item_rows = sorted(
            _read_jsonl(self.item_info_path), key=lambda row: int(row["item_id"])
        )
        self.item_ids: list[int] = []
        item_texts: list[str] = []
        for row in item_rows:
            item_id = int(row["item_id"])
            if item_id == 0:
                continue
            fallback_title = str(row.get("title") or "").strip() or f"item_{item_id}"
            if not str(row.get("title") or "").strip():
                row = {**row, "title": fallback_title}
            self.item_ids.append(item_id)
            item_texts.append(format_positive(row, fallback_title))

        if len(self.item_ids) != self.expected_items or len(set(self.item_ids)) != self.expected_items:
            raise RuntimeError(
                f"Expected {self.expected_items} unique candidate items, got {len(set(self.item_ids))}"
            )
        self.item_index = {item_id: index for index, item_id in enumerate(self.item_ids)}
        self.item_embeddings = self._encode_batches(
            item_texts, batch_size=self.item_batch_size, is_query=False
        ).to(self.device)

    def _encode_batches(self, texts: list[str], *, batch_size: int, is_query: bool):
        outputs = []
        with self.torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                batch_texts = texts[start : start + batch_size]
                if is_query:
                    batch_texts = [self.format_query(text) for text in batch_texts]
                batch_embeddings = self.encode(
                    self.model,
                    self.tokenizer,
                    batch_texts,
                    self.device,
                    self.max_length,
                    is_query,
                )
                outputs.append(batch_embeddings.detach().cpu())
        return self.torch.cat(outputs, dim=0)

    def encode_queries(self, queries: list[str]):
        return self._encode_batches(
            queries, batch_size=self.query_batch_size, is_query=True
        ).to(self.device)

    def score_queries(
        self,
        query_embeddings,
        target_ids: list[int | None],
        history_item_ids: list[set[int]],
        *,
        temperature: float,
        ndcg_k: int,
    ) -> tuple[list[float], list[float], list[float], list[int], list[int]]:
        if temperature <= 0:
            raise ValueError("COT_SIM_NDCG_TEMPERATURE must be positive")
        with self.torch.inference_mode():
            scores = query_embeddings @ self.item_embeddings.T
            target_indices: list[int] = []
            masked_counts: list[int] = []
            for row_index, (target_id, history_ids) in enumerate(
                zip(target_ids, history_item_ids)
            ):
                if target_id is None or target_id not in self.item_index:
                    raise RuntimeError(f"Missing target item in full catalog: {target_id}")
                target_index = self.item_index[target_id]
                target_indices.append(target_index)
                seen_ids = set(history_ids)
                seen_ids.discard(target_id)
                masked = [
                    self.item_index[item_id]
                    for item_id in seen_ids
                    if item_id in self.item_index
                ]
                if masked:
                    scores[row_index, masked] = -self.torch.inf
                masked_counts.append(len(masked))

            target_indices_tensor = self.torch.tensor(
                target_indices, dtype=self.torch.long, device=self.device
            )
            rows = self.torch.arange(len(target_indices), device=self.device)
            target_scores = scores[rows, target_indices_tensor]
            ranks = 1 + scores.gt(target_scores[:, None]).sum(dim=1)
            relative_similarity = (
                target_scores / temperature
                - self.torch.logsumexp(scores / temperature, dim=1)
            )

        cosine_values = [float(value) for value in target_scores.cpu().tolist()]
        similarity_values = [float(value) for value in relative_similarity.cpu().tolist()]
        rank_values = [int(value) for value in ranks.cpu().tolist()]
        ndcg_values = [_ndcg_at_rank(rank, ndcg_k) for rank in rank_values]
        return similarity_values, cosine_values, ndcg_values, rank_values, masked_counts


def _get_state() -> CotRetrievalRewardState:
    global _STATE
    if _STATE is None:
        _STATE = CotRetrievalRewardState()
    return _STATE


def _group_key(
    index: int,
    example_ids: list[str],
    histories: list[str],
    fallback_group_size: int,
) -> tuple[str, str | int]:
    if example_ids[index]:
        return ("example_id", example_ids[index])
    if histories[index]:
        return ("history", histories[index])
    return ("chunk", index // max(1, fallback_group_size))


def _log_components(summary: dict[str, Any], items: list[dict[str, Any]]) -> None:
    global _CALL_COUNT
    log_path = os.getenv("COT_SIM_NDCG_COMPONENT_LOG", "").strip()
    if log_path:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "kind": "cot_sim_ndcg",
                        "call_index": _CALL_COUNT,
                        "rank": os.getenv("RANK", ""),
                        "local_rank": os.getenv("LOCAL_RANK", ""),
                        "summary": summary,
                        "items": items,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    log_every = int(os.getenv("COT_SIM_NDCG_LOG_EVERY", "0"))
    if log_every > 0 and _CALL_COUNT % log_every == 0:
        print(
            "[cot_sim_ndcg] "
            f"call={_CALL_COUNT} n={summary['n']} groups={summary['groups']} "
            f"similarity_mean={summary['relative_similarity_mean']:.6f} "
            f"ndcg_mean={summary['ndcg_mean']:.6f} "
            f"rank_mean={summary['rank_mean']:.2f} "
            f"conflict_rate={summary['pairwise_conflict_rate']:.6f} "
            f"reward_mean={summary['reward_mean']:.6f} "
            f"total_sec={summary['total_sec']:.3f}",
            flush=True,
        )


class CotSimilarityNdcgReward(ORM):
    def __call__(
        self,
        completions,
        user_history=None,
        source_prompt=None,
        target_item_id=None,
        history_item_ids=None,
        history_item_id=None,
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
            raise RuntimeError("cot_sim_ndcg requires non-empty user_history metadata")
        target_ids = _as_int_list(target_item_id, n)
        history_sets = _as_int_set_list(
            history_item_ids if history_item_ids is not None else history_item_id, n
        )
        example_ids = _as_text_list(kwargs.get("example_id"), n)

        temperature = float(os.getenv("COT_SIM_NDCG_TEMPERATURE", "0.05"))
        ndcg_k = int(os.getenv("COT_SIM_NDCG_K", "100"))
        similarity_weight = float(os.getenv("COT_SIM_NDCG_SIM_WEIGHT", "0.8"))
        ndcg_weight = float(os.getenv("COT_SIM_NDCG_NDCG_WEIGHT", "0.2"))
        group_size = int(os.getenv("COT_SIM_NDCG_GROUP_SIZE", "4"))
        strict_group_size = _env_bool("COT_SIM_NDCG_STRICT_GROUP_SIZE", True)
        if ndcg_k <= 0 or group_size <= 0:
            raise ValueError("NDCG K and group size must be positive")
        if similarity_weight < 0 or ndcg_weight < 0 or similarity_weight + ndcg_weight <= 0:
            raise ValueError("reward weights must be non-negative with a positive sum")

        queries = [
            f"{history}\n\nRecommendation reasoning:\n{completion}"
            if completion
            else history
            for history, completion in zip(histories, completion_texts)
        ]
        state = _get_state()
        query_embeddings = state.encode_queries(queries)
        similarities, cosines, ndcgs, ranks, masked_counts = state.score_queries(
            query_embeddings,
            target_ids,
            history_sets,
            temperature=temperature,
            ndcg_k=ndcg_k,
        )

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
                    f"GRPO groups must contain exactly {group_size} completions: {invalid_groups}"
                )

        rewards = [0.0 for _ in range(n)]
        similarity_z = [0.0 for _ in range(n)]
        ndcg_z = [0.0 for _ in range(n)]
        group_conflicts: list[float] = []
        for indices in grouped_indices.values():
            group_similarities = [similarities[index] for index in indices]
            group_ndcgs = [ndcgs[index] for index in indices]
            group_rewards, group_similarity_z, group_ndcg_z = _combine_group_rewards(
                group_similarities,
                group_ndcgs,
                similarity_weight,
                ndcg_weight,
            )
            group_conflicts.append(
                _pairwise_conflict_rate(group_similarities, group_ndcgs)
            )
            for offset, index in enumerate(indices):
                rewards[index] = float(group_rewards[offset])
                similarity_z[index] = float(group_similarity_z[offset])
                ndcg_z[index] = float(group_ndcg_z[offset])

        items = [
            {
                "index": index,
                "example_id": example_ids[index],
                "target_item_id": target_ids[index],
                "target_cosine": cosines[index],
                "relative_similarity_logprob": similarities[index],
                "rank": ranks[index],
                f"ndcg@{ndcg_k}": ndcgs[index],
                "similarity_z": similarity_z[index],
                "ndcg_z": ndcg_z[index],
                "reward": rewards[index],
                "masked_history_items": masked_counts[index],
            }
            for index in range(n)
        ]
        summary = {
            "n": n,
            "groups": len(grouped_indices),
            "group_size": group_size,
            "embedding_model": state.model_path,
            "candidate_items": len(state.item_ids),
            "temperature": temperature,
            "ndcg_k": ndcg_k,
            "similarity_weight": similarity_weight,
            "ndcg_weight": ndcg_weight,
            "relative_similarity_mean": _safe_mean(similarities),
            "target_cosine_mean": _safe_mean(cosines),
            "ndcg_mean": _safe_mean(ndcgs),
            "rank_mean": _safe_mean([float(rank) for rank in ranks]),
            "pairwise_conflict_rate": _safe_mean(group_conflicts),
            "reward_mean": _safe_mean(rewards),
            "reward_min": min(rewards),
            "reward_max": max(rewards),
            "zero_ndcg_count": sum(value == 0.0 for value in ndcgs),
            "total_sec": time.perf_counter() - started,
        }
        _log_components(summary, items)
        return rewards


orms["cot_sim_ndcg"] = CotSimilarityNdcgReward
orms["cot_similarity_ndcg"] = CotSimilarityNdcgReward
