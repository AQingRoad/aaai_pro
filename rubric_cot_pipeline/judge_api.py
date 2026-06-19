from __future__ import annotations

import http.client
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from rubric_cot_pipeline.prompts import build_judge_messages
from rubric_cot_pipeline.rubric import normalize_judge_score, parse_judge_json, rule_score


@dataclass
class JudgeAPIResult:
    score: dict[str, Any] | None
    raw: str
    provider: str


class RubricJudgeAPIClient:
    provider = "base"

    def score(self, user_history: str, cot: str, target_item: str = "") -> JudgeAPIResult:
        raise NotImplementedError


class MockRubricJudgeAPIClient(RubricJudgeAPIClient):
    """Deterministic API-shaped judge for offline pipeline development."""

    provider = "mock"

    def score(self, user_history: str, cot: str, target_item: str = "") -> JudgeAPIResult:
        proxy = rule_score(user_history, cot)
        payload = {
            "preference_grounding": proxy["preference_grounding"],
            "taste_specificity": proxy["taste_specificity"],
            "transitional_reasoning": proxy["transitional_reasoning"],
            "discriminative_framing": proxy["discriminative_framing"],
            "conciseness": proxy["conciseness"],
            "comment": "mock API judge; deterministic local proxy with API-compatible response shape",
            "provider": self.provider,
            "target_visible": bool(target_item),
        }
        raw = json.dumps(payload, ensure_ascii=False)
        return JudgeAPIResult(score=normalize_judge_score(payload), raw=raw, provider=self.provider)


class ZhipuGLMMockRubricJudgeAPIClient(MockRubricJudgeAPIClient):
    """Mock response wrapped like Zhipu GLM chat/completions."""

    provider = "zhipu_glm_mock"

    def score(self, user_history: str, cot: str, target_item: str = "") -> JudgeAPIResult:
        result = super().score(user_history, cot, target_item)
        payload = {
            "id": "mock-zhipu-rubric-judge",
            "request_id": "mock-request",
            "created": int(time.time()),
            "model": "glm-mock-rubric",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": result.raw,
                        "reasoning_content": "",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        return JudgeAPIResult(score=result.score, raw=json.dumps(payload, ensure_ascii=False), provider=self.provider)


RUBRIC_DIMENSIONS = [
    "preference_grounding",
    "taste_specificity",
    "transitional_reasoning",
    "discriminative_framing",
    "conciseness",
]


def _extract_score_from_free_text(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    scores: dict[str, int] = {}
    for dim in RUBRIC_DIMENSIONS:
        label = re.escape(dim)
        label_with_spaces = re.escape(dim.replace("_", " "))
        patterns = [
            rf"(?:{label}|{label_with_spaces})[\s\S]{{0,1200}}?score\s*[:=]?\s*([1-5])\b",
            rf"(?:{label}|{label_with_spaces})[\s\S]{{0,1200}}?([1-5])\s*/\s*5\b",
            rf"(?:{label}|{label_with_spaces})\s*[:=]\s*([1-5])\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                scores[dim] = int(match.group(1))
                break
    if len(scores) != len(RUBRIC_DIMENSIONS):
        return None
    comments = []
    for dim in RUBRIC_DIMENSIONS:
        match = re.search(rf"{re.escape(dim)}[^\n\r]{{0,500}}", text, flags=re.IGNORECASE)
        if match:
            comments.append(match.group(0).strip())
    return {**scores, "comment": " | ".join(comments)[:500] or "scores extracted from free-text judge response"}


def _extract_score_from_chat_response(raw_response: str) -> dict[str, Any] | None:
    parsed = json.loads(raw_response)
    message = parsed["choices"][0].get("message", {})
    texts = [
        str(message.get("content") or ""),
        str(message.get("reasoning_content") or ""),
    ]
    for text in texts:
        obj = parse_judge_json(text)
        if obj:
            normalized = normalize_judge_score(obj)
            if normalized:
                return normalized
    for text in texts:
        obj = _extract_score_from_free_text(text)
        if obj:
            return normalize_judge_score(obj)
    return None


class OpenAICompatibleRubricJudgeAPIClient(RubricJudgeAPIClient):
    """Minimal OpenAI-compatible chat/completions client.

    This keeps the real API boundary in one place. The caller receives the same
    normalized rubric dict as the mock provider.
    """

    provider = "openai_compatible"

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 60.0,
        max_retries: int = 2,
    ) -> None:
        if not base_url:
            raise ValueError("api base url is required for openai_compatible provider")
        if not model:
            raise ValueError("api model is required for openai_compatible provider")
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

    def score(self, user_history: str, cot: str, target_item: str = "") -> JudgeAPIResult:
        payload = {
            "model": self.model,
            "messages": build_judge_messages(user_history, cot, target_item),
            "temperature": 0,
            "stream": False,
            "max_tokens": int(os.getenv("RUBRIC_REWARD_API_MAX_TOKENS") or os.getenv("RUBRIC_JUDGE_API_MAX_TOKENS", "128")),
            "response_format": {"type": "json_object"},
        }
        thinking_mode = (
            os.getenv("RUBRIC_REWARD_API_THINKING")
            or os.getenv("RUBRIC_JUDGE_API_THINKING")
            or ""
        ).strip()
        if not thinking_mode and any(host in self.base_url for host in ("bigmodel.cn", "zhipu")):
            thinking_mode = "disabled"
        if thinking_mode.lower() not in {"", "0", "false", "off", "none", "null"}:
            payload["thinking"] = {"type": thinking_mode}
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        last_error = ""
        for attempt in range(self.max_retries + 1):
            request = urllib.request.Request(
                self.chat_completions_url,
                data=body,
                headers=headers,
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    raw_response = response.read().decode("utf-8")
                score = _extract_score_from_chat_response(raw_response)
                return JudgeAPIResult(score=score, raw=raw_response, provider=self.provider)
            except (KeyError, json.JSONDecodeError, urllib.error.URLError, TimeoutError, http.client.RemoteDisconnected) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < self.max_retries:
                    time.sleep(min(2**attempt, 8))

        raw = json.dumps({"error": last_error, "provider": self.provider}, ensure_ascii=False)
        return JudgeAPIResult(score=None, raw=raw, provider=self.provider)


class ZhipuGLMRubricJudgeAPIClient(OpenAICompatibleRubricJudgeAPIClient):
    """Rubric judge through Zhipu GLM chat/completions."""

    provider = "zhipu_glm"

    def __init__(
        self,
        base_url: str = "",
        api_key: str = "",
        model: str = "",
        timeout: float = 60.0,
        max_retries: int = 2,
        require_api_key: bool = True,
    ) -> None:
        if require_api_key and not api_key:
            raise ValueError(
                "Zhipu API key is required. Set RUBRIC_JUDGE_API_KEY, "
                "RUBIREC_ZHIPU_API_KEY, ZHIPUAI_API_KEY, ZHIPU_API_KEY, ZAI_API_KEY, or BIGMODEL_API_KEY."
            )
        super().__init__(
            base_url=base_url or "https://open.bigmodel.cn/api/paas/v4",
            api_key=api_key,
            model=model or "glm-4-flash-250414",
            timeout=timeout,
            max_retries=max_retries,
        )
        self.min_interval = float(os.getenv("RUBRIC_JUDGE_API_MIN_INTERVAL", "0"))
        self._request_lock = threading.Lock()
        self._last_request_ts = 0.0

    def _throttle(self) -> None:
        if self.min_interval <= 0:
            return
        with self._request_lock:
            now = time.time()
            wait = self.min_interval - (now - self._last_request_ts)
            if wait > 0:
                time.sleep(wait)
            self._last_request_ts = time.time()

    def score(self, user_history: str, cot: str, target_item: str = "") -> JudgeAPIResult:
        payload = {
            "model": self.model,
            "messages": build_judge_messages(user_history, cot, target_item),
            "temperature": 0,
            "do_sample": False,
            "stream": False,
            "max_tokens": int(os.getenv("RUBRIC_REWARD_API_MAX_TOKENS") or os.getenv("RUBRIC_JUDGE_API_MAX_TOKENS", "128")),
            "response_format": {"type": "json_object"},
        }
        thinking_mode = (
            os.getenv("RUBRIC_REWARD_API_THINKING")
            or os.getenv("RUBRIC_JUDGE_API_THINKING")
            or "disabled"
        ).strip()
        if thinking_mode.lower() not in {"", "0", "false", "off", "none", "null"}:
            payload["thinking"] = {"type": thinking_mode}
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        last_error = ""
        for attempt in range(self.max_retries + 1):
            self._throttle()
            request = urllib.request.Request(
                self.chat_completions_url,
                data=body,
                headers=headers,
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    raw_response = response.read().decode("utf-8")
                score = _extract_score_from_chat_response(raw_response)
                return JudgeAPIResult(score=score, raw=raw_response, provider=self.provider)
            except urllib.error.HTTPError as exc:
                try:
                    body_text = exc.read().decode("utf-8", errors="ignore")
                except Exception:
                    body_text = ""
                last_error = f"HTTPError {exc.code}: {body_text[:500]}"
                if attempt < self.max_retries:
                    retry_after = exc.headers.get("Retry-After")
                    if retry_after:
                        try:
                            sleep_s = float(retry_after)
                        except ValueError:
                            sleep_s = 20.0
                    elif exc.code == 429:
                        sleep_s = max(20.0, 10.0 * (attempt + 1))
                    else:
                        sleep_s = min(2**attempt, 8)
                    time.sleep(sleep_s)
            except (KeyError, json.JSONDecodeError, urllib.error.URLError, TimeoutError, http.client.RemoteDisconnected) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < self.max_retries:
                    time.sleep(min(2**attempt, 8))

        raw = json.dumps({"error": last_error, "provider": self.provider}, ensure_ascii=False)
        return JudgeAPIResult(score=None, raw=raw, provider=self.provider)


def _first_env(*names: str) -> str:
    for name in names:
        value = os.getenv(name, "")
        if value:
            return value
    return ""


def build_judge_api_client(
    provider: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
) -> RubricJudgeAPIClient:
    provider = (provider or os.getenv("RUBRIC_JUDGE_API_PROVIDER", "mock")).strip().lower()
    if provider in {"mock", "mock_api", "offline"}:
        return MockRubricJudgeAPIClient()
    if provider in {"zhipu_glm_mock", "glm_mock", "zhipu_mock"}:
        return ZhipuGLMMockRubricJudgeAPIClient()
    if provider in {"zhipu", "zhipu_glm", "glm", "bigmodel", "zai"}:
        return ZhipuGLMRubricJudgeAPIClient(
            base_url=base_url or _first_env("RUBRIC_JUDGE_API_BASE_URL", "RUBIREC_ZHIPU_BASE_URL", "ZHIPUAI_BASE_URL"),
            api_key=api_key
            or _first_env(
                "RUBRIC_JUDGE_API_KEY",
                "RUBIREC_ZHIPU_API_KEY",
                "ZHIPUAI_API_KEY",
                "ZHIPU_API_KEY",
                "ZAI_API_KEY",
                "BIGMODEL_API_KEY",
            ),
            model=model or _first_env("RUBRIC_JUDGE_API_MODEL", "RUBIREC_ZHIPU_MODEL") or "glm-4-flash-250414",
            timeout=timeout if timeout is not None else float(os.getenv("RUBRIC_JUDGE_API_TIMEOUT", "60")),
            max_retries=max_retries if max_retries is not None else int(os.getenv("RUBRIC_JUDGE_API_MAX_RETRIES", "2")),
        )
    if provider in {"zhipu_glm_local", "glm_local", "local_glm"}:
        return ZhipuGLMRubricJudgeAPIClient(
            base_url=base_url
            or _first_env("RUBRIC_JUDGE_API_BASE_URL", "RUBIREC_ZHIPU_BASE_URL", "ZHIPUAI_BASE_URL")
            or "http://127.0.0.1:18080/api/paas/v4",
            api_key=api_key
            or _first_env(
                "RUBRIC_JUDGE_API_KEY",
                "RUBIREC_ZHIPU_API_KEY",
                "ZHIPUAI_API_KEY",
                "ZHIPU_API_KEY",
                "ZAI_API_KEY",
                "BIGMODEL_API_KEY",
            ),
            model=model or _first_env("RUBRIC_JUDGE_API_MODEL", "RUBIREC_ZHIPU_MODEL") or "glm-4-flash",
            timeout=timeout if timeout is not None else float(os.getenv("RUBRIC_JUDGE_API_TIMEOUT", "60")),
            max_retries=max_retries if max_retries is not None else int(os.getenv("RUBRIC_JUDGE_API_MAX_RETRIES", "2")),
            require_api_key=False,
        )
    if provider in {"openai", "openai_compatible", "chat_completions"}:
        return OpenAICompatibleRubricJudgeAPIClient(
            base_url=base_url or os.getenv("RUBRIC_JUDGE_API_BASE_URL", ""),
            api_key=api_key or os.getenv("RUBRIC_JUDGE_API_KEY", ""),
            model=model or os.getenv("RUBRIC_JUDGE_API_MODEL", ""),
            timeout=timeout if timeout is not None else float(os.getenv("RUBRIC_JUDGE_API_TIMEOUT", "60")),
            max_retries=max_retries if max_retries is not None else int(os.getenv("RUBRIC_JUDGE_API_MAX_RETRIES", "2")),
        )
    raise ValueError(f"Unsupported judge API provider: {provider}")
