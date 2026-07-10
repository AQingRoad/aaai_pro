from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn


@dataclass
class PreferenceScorerConfig:
    embedding_dim: int
    hidden_dim: int = 512
    dropout: float = 0.1
    encoder_model: str = ""
    encoder_max_length: int = 4096
    query_instruction: str = ""
    text_mode: str = "think_only"
    feature_mode: str = "joint_delta_product"

    @property
    def input_dim(self) -> int:
        if self.feature_mode == "joint_delta_product":
            return self.embedding_dim * 3
        raise ValueError(f"Unsupported feature_mode: {self.feature_mode}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PreferenceScorerConfig":
        return cls(**value)


def build_preference_features(
    history_embeddings: torch.Tensor,
    joint_embeddings: torch.Tensor,
    feature_mode: str = "joint_delta_product",
) -> torch.Tensor:
    if history_embeddings.shape != joint_embeddings.shape:
        raise ValueError(
            f"History and joint embedding shapes differ: {history_embeddings.shape} vs {joint_embeddings.shape}"
        )
    if feature_mode == "joint_delta_product":
        return torch.cat(
            [
                joint_embeddings,
                joint_embeddings - history_embeddings,
                joint_embeddings * history_embeddings,
            ],
            dim=-1,
        )
    raise ValueError(f"Unsupported feature_mode: {feature_mode}")


class PairwisePreferenceHead(nn.Module):
    def __init__(self, config: PreferenceScorerConfig):
        super().__init__()
        self.config = config
        self.net = nn.Sequential(
            nn.LayerNorm(config.input_dim),
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features.float()).squeeze(-1)
