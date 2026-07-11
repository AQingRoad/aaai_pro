from __future__ import annotations

import hashlib
import math
from typing import Any

import torch
import torch.nn as nn


class GainRouter(nn.Module):
    def __init__(self, input_dim: int, model_type: str, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        if model_type == "logistic":
            self.network = nn.Linear(input_dim, 1)
        elif model_type == "mlp":
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
        else:
            raise ValueError(f"Unknown model_type={model_type}")

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def user_group_split(user_ids: list[str], *, valid_fraction: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    if not 0 < valid_fraction < 1:
        raise ValueError("valid_fraction must be in (0, 1)")
    valid: list[bool] = []
    boundary = int(valid_fraction * 1_000_000)
    for user_id in user_ids:
        digest = hashlib.sha256(f"{seed}:{user_id}".encode()).digest()
        value = int.from_bytes(digest[:8], "big") % 1_000_000
        valid.append(value < boundary)
    valid_mask = torch.tensor(valid, dtype=torch.bool)
    train_mask = ~valid_mask
    if not train_mask.any() or not valid_mask.any():
        raise ValueError("Hash split produced an empty partition")
    return train_mask, valid_mask


def average_precision(probabilities: torch.Tensor, labels: torch.Tensor) -> float:
    positives = int(labels.sum().item())
    if positives == 0:
        return 0.0
    order = torch.argsort(probabilities, descending=True)
    ordered = labels[order]
    precision = torch.cumsum(ordered, dim=0) / torch.arange(
        1, len(ordered) + 1, dtype=torch.float32
    )
    return float((precision * ordered).sum().item() / positives)


def select_threshold(
    probabilities: torch.Tensor,
    delta_ndcg: torch.Tensor,
    *,
    max_trigger_rate: float,
) -> dict[str, float | int]:
    if not 0 < max_trigger_rate <= 1:
        raise ValueError("max_trigger_rate must be in (0, 1]")
    count = len(probabilities)
    max_selected = min(count, math.floor(count * max_trigger_rate))
    order = torch.argsort(probabilities, descending=True)
    gains = delta_ndcg[order]
    cumulative = torch.cat([torch.zeros(1), torch.cumsum(gains, dim=0)])
    best_count = int(torch.argmax(cumulative[: max_selected + 1]).item())
    if best_count == 0:
        threshold = float(probabilities.max().item()) + 1e-6
    elif best_count == count:
        threshold = float(probabilities.min().item()) - 1e-6
    else:
        upper = float(probabilities[order[best_count - 1]].item())
        lower = float(probabilities[order[best_count]].item())
        threshold = (upper + lower) / 2
    route = probabilities > threshold
    return {
        "threshold": threshold,
        "selected": int(route.sum().item()),
        "trigger_rate": float(route.float().mean().item()),
        "gain_sum": float(delta_ndcg[route].sum().item()),
    }


def route_metrics(
    probabilities: torch.Tensor,
    *,
    threshold: float,
    labels: torch.Tensor,
    baseline_ndcg: torch.Tensor,
    cot_ndcg: torch.Tensor,
    baseline_rank: torch.Tensor,
    cot_rank: torch.Tensor,
    ndcg_k: int = 20,
) -> dict[str, Any]:
    route = probabilities > threshold
    routed_ndcg = torch.where(route, cot_ndcg, baseline_ndcg)
    routed_rank = torch.where(route, cot_rank, baseline_rank)
    oracle_ndcg = torch.maximum(baseline_ndcg, cot_ndcg)
    baseline_mean = float(baseline_ndcg.mean().item())
    oracle_mean = float(oracle_ndcg.mean().item())
    routed_mean = float(routed_ndcg.mean().item())
    oracle_gain = oracle_mean - baseline_mean
    selected_delta = cot_ndcg[route] - baseline_ndcg[route]
    return {
        "samples": len(labels),
        "threshold": threshold,
        "trigger_count": int(route.sum().item()),
        "trigger_rate": float(route.float().mean().item()),
        "selected_positive": int(labels[route].sum().item()),
        "selected_nonpositive": int(route.sum().item() - labels[route].sum().item()),
        "selected_delta_ndcg_sum": float(selected_delta.sum().item()),
        "baseline_ndcg": baseline_mean,
        "cot_ndcg": float(cot_ndcg.mean().item()),
        "routed_ndcg": routed_mean,
        "oracle_ndcg": oracle_mean,
        "absolute_ndcg_gain": routed_mean - baseline_mean,
        "oracle_recovery": (routed_mean - baseline_mean) / oracle_gain if oracle_gain > 0 else None,
        "baseline_hr": float((baseline_rank <= ndcg_k).float().mean().item()),
        "cot_hr": float((cot_rank <= ndcg_k).float().mean().item()),
        "routed_hr": float((routed_rank <= ndcg_k).float().mean().item()),
        "oracle_hr": float((torch.minimum(baseline_rank, cot_rank) <= ndcg_k).float().mean().item()),
        "average_precision": average_precision(probabilities, labels),
    }
