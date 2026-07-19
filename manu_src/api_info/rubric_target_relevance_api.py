#!/usr/bin/env python3
"""OpenAI-compatible API client for the target-relevance Rubric Judge."""

from __future__ import annotations

import http.client
import json
import os
import ssl
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from manu_src.prompts.rubric_target_relevance_judge_v1 import build_judge_messages
from rubric_cot_pipeline.rubric import normalize_judge_score, parse_judge_json


@dataclass
class JudgeAPIResult:
    score: dict[str, Any] | None
    raw: str
    provider: str


def _extract_score(raw_response: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(raw_response)
        message = payload["choices"][0].get("message", {})
    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
        return None
    if str(message.get("reasoning_content") or "").strip():
        return None
    for field in ("content",):
        parsed = parse_judge_json(str(message.get(field) or ""))
        if parsed:
            normalized = normalize_judge_score(parsed)
            if normalized:
                return normalized
    return None


class TargetRelevanceJudgeAPIClient:
    def __init__(
        self,
        *,
        provider: str,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float,
        max_retries: int,
    ) -> None:
        if not base_url:
            raise ValueError("Rubric API base URL is required")
        if not model:
            raise ValueError("Rubric API model is required")
        if timeout <= 0 or max_retries < 0:
            raise ValueError("Rubric API timeout must be positive and retries non-negative")
        self.provider = provider
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries

    @property
    def chat_completions_url(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return f"{self.base_url}/chat/completions"

    def score(
        self,
        user_history: str,
        candidate_reasoning: str,
        target_item: str = "",
        target_usage: str = "relevance",
    ) -> JudgeAPIResult:
        if str(target_usage or "relevance").strip().lower() != "relevance":
            raise ValueError("This client only supports target_usage='relevance'")
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": build_judge_messages(
                user_history,
                candidate_reasoning,
                target_item,
            ),
            "temperature": 0,
            "do_sample": False,
            "stream": False,
            "max_tokens": int(
                os.getenv("COT_RUBRIC_NDCG_GAIN_API_MAX_TOKENS", "128")
            ),
            "response_format": {"type": "json_object"},
        }
        thinking = os.getenv(
            "COT_RUBRIC_NDCG_GAIN_API_THINKING",
            "disabled" if "bigmodel.cn" in self.base_url else "",
        ).strip()
        if thinking.lower() not in {"", "0", "false", "off", "none", "null"}:
            payload["thinking"] = {"type": thinking}
        if self.provider == "ks_tokenverse" and thinking.lower() == "disabled":
            payload["enable_thinking"] = False
            payload["reasoning_effort"] = "none"
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        last_error = ""
        retry_base_seconds = float(
            os.getenv("COT_RUBRIC_NDCG_GAIN_API_RETRY_BASE_SECONDS", "5")
        )
        retry_max_seconds = float(
            os.getenv("COT_RUBRIC_NDCG_GAIN_API_RETRY_MAX_SECONDS", "60")
        )
        if retry_base_seconds <= 0 or retry_max_seconds <= 0:
            raise ValueError("Rubric API retry delays must be positive")
        for attempt in range(self.max_retries + 1):
            retry_after_seconds = 0.0
            retryable = True
            request = urllib.request.Request(
                self.chat_completions_url,
                data=body,
                headers=headers,
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    raw = response.read().decode("utf-8")
                return JudgeAPIResult(
                    score=_extract_score(raw),
                    raw=raw,
                    provider=self.provider,
                )
            except urllib.error.HTTPError as exc:
                try:
                    response_body = exc.read().decode("utf-8", errors="ignore")
                except Exception:
                    response_body = ""
                last_error = f"HTTPError {exc.code}: {response_body[:500]}"
                retryable = exc.code in {408, 409, 425, 429} or exc.code >= 500
                try:
                    retry_after_seconds = float(exc.headers.get("Retry-After") or 0)
                except (TypeError, ValueError):
                    retry_after_seconds = 0.0
            except (
                urllib.error.URLError,
                TimeoutError,
                http.client.RemoteDisconnected,
                ConnectionError,
                ssl.SSLError,
                json.JSONDecodeError,
            ) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            if not retryable:
                break
            if attempt < self.max_retries:
                delay = min(retry_base_seconds * (2**attempt), retry_max_seconds)
                time.sleep(max(delay, retry_after_seconds))
        return JudgeAPIResult(
            score=None,
            raw=json.dumps(
                {"error": last_error, "provider": self.provider},
                ensure_ascii=False,
            ),
            provider=self.provider,
        )


def build_target_relevance_judge_api_client(
    *,
    provider: str,
    base_url: str,
    api_key: str,
    model: str,
    timeout: float,
    max_retries: int,
) -> TargetRelevanceJudgeAPIClient:
    normalized_provider = str(provider or "").strip().lower()
    if normalized_provider in {"zhipu", "zhipu_glm", "glm", "bigmodel", "zai"}:
        if not api_key:
            raise ValueError("Official GLM Rubric API requires an API key")
        return TargetRelevanceJudgeAPIClient(
            provider="zhipu_glm",
            base_url=base_url or "https://open.bigmodel.cn/api/paas/v4",
            api_key=api_key,
            model=model or "glm-5.2",
            timeout=timeout,
            max_retries=max_retries,
        )
    if normalized_provider in {
        "openai",
        "openai_compatible",
        "chat_completions",
        "ks",
        "ks_tokenverse",
        "tokenverse",
    }:
        return TargetRelevanceJudgeAPIClient(
            provider=(
                "ks_tokenverse"
                if normalized_provider in {"ks", "ks_tokenverse", "tokenverse"}
                else "openai_compatible"
            ),
            base_url=base_url,
            api_key=api_key,
            model=model,
            timeout=timeout,
            max_retries=max_retries,
        )
    raise ValueError(f"Unsupported target-relevance API provider: {provider}")


__all__ = [
    "JudgeAPIResult",
    "TargetRelevanceJudgeAPIClient",
    "build_target_relevance_judge_api_client",
]
