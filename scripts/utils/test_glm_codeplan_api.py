#!/usr/bin/env python3
from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Any


DEFAULT_BASE_URL = "https://open.bigmodel.cn/api/coding/paas/v4"
DEFAULT_MODEL = "glm-5.2"
ENV_KEY_NAMES = (
    "BIGMODEL_API_KEY",
    "ZAI_API_KEY",
    "ZHIPUAI_API_KEY",
    "ZHIPU_API_KEY",
    "GLM_API_KEY",
)


def first_env(names: tuple[str, ...]) -> str:
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return ""


def chat_completions_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"


def read_api_key(args: argparse.Namespace) -> str:
    if args.api_key_stdin:
        if sys.stdin.isatty():
            return getpass.getpass("").strip()
        return sys.stdin.readline().strip()
    return first_env(ENV_KEY_NAMES)


def short_text(value: Any, limit: int = 1200) -> str:
    text = "" if value is None else str(value)
    return text if len(text) <= limit else text[:limit] + "\n...[truncated]"


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test GLM Coding Plan OpenAI-compatible chat/completions API.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-key-stdin", action="store_true", help="Read API key from the first stdin line.")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--thinking", choices=["enabled", "disabled", ""], default="disabled")
    parser.add_argument(
        "--prompt",
        default="用 Python 写一个函数 add(a, b)，只返回代码和一个最短示例。",
    )
    args = parser.parse_args()

    api_key = read_api_key(args)
    if not api_key:
        names = ", ".join(ENV_KEY_NAMES)
        print(f"missing API key: set one of {names}, or pass --api-key-stdin", file=sys.stderr)
        return 2

    payload: dict[str, Any] = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": "你是编程助手，回答要简洁，优先给可运行代码。"},
            {"role": "user", "content": args.prompt},
        ],
        "temperature": args.temperature,
        "stream": False,
        "max_tokens": args.max_tokens,
    }
    if args.thinking:
        payload["thinking"] = {"type": args.thinking}

    url = chat_completions_url(args.base_url)
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    print(f"request_url={url}")
    print(f"model={args.model} thinking={args.thinking or 'omitted'} max_tokens={args.max_tokens}")
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            raw = response.read().decode("utf-8")
            status = response.status
    except urllib.error.HTTPError as exc:
        elapsed = time.perf_counter() - start
        error_body = exc.read().decode("utf-8", errors="replace")
        print(f"http_status={exc.code} elapsed_seconds={elapsed:.3f}", file=sys.stderr)
        print(short_text(error_body), file=sys.stderr)
        return 1
    except (urllib.error.URLError, TimeoutError) as exc:
        elapsed = time.perf_counter() - start
        print(f"request_failed={type(exc).__name__}: {exc} elapsed_seconds={elapsed:.3f}", file=sys.stderr)
        return 1

    elapsed = time.perf_counter() - start
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        print(f"http_status={status} elapsed_seconds={elapsed:.3f}")
        print(short_text(raw))
        return 1

    choice = (obj.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    print(f"http_status={status} elapsed_seconds={elapsed:.3f}")
    print(f"response_id={obj.get('id', '')}")
    print(f"finish_reason={choice.get('finish_reason', '')}")
    print(f"usage={json.dumps(obj.get('usage', {}), ensure_ascii=False)}")
    reasoning = message.get("reasoning_content")
    if reasoning:
        print(f"reasoning_content_chars={len(str(reasoning))}")
    print("content:")
    print(short_text(message.get("content", "")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
