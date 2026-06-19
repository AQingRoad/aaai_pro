#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urljoin


def load_env_file(path: str) -> None:
    if not path:
        return
    env_path = Path(path).expanduser()
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        try:
            parsed = shlex.split(value, comments=False, posix=True)
            os.environ[key] = parsed[0] if parsed else ""
        except ValueError:
            os.environ[key] = value.strip().strip("\"'")


def join_url(base_url: str, path: str) -> str:
    base = base_url.rstrip("/") + "/"
    clean_path = path.lstrip("/")
    if clean_path.startswith("v1/") and base.endswith("/v1/"):
        clean_path = clean_path[3:]
    return urljoin(base, clean_path)


class ProxyHandler(BaseHTTPRequestHandler):
    server_version = "LocalOpenAIProxy/0.1"

    def do_GET(self) -> None:
        self._proxy()

    def do_POST(self) -> None:
        self._proxy()

    def log_message(self, fmt: str, *args: object) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _proxy(self) -> None:
        length = int(self.headers.get("Content-Length", "0") or "0")
        body = self.rfile.read(length) if length else None
        target_url = join_url(self.server.upstream_base_url, self.path)  # type: ignore[attr-defined]
        headers = {"Content-Type": self.headers.get("Content-Type", "application/json")}
        if self.server.api_key:  # type: ignore[attr-defined]
            headers["Authorization"] = f"Bearer {self.server.api_key}"  # type: ignore[attr-defined]

        request = urllib.request.Request(target_url, data=body, headers=headers, method=self.command)
        try:
            with urllib.request.urlopen(request, timeout=self.server.timeout) as response:  # type: ignore[attr-defined]
                response_body = response.read()
                status = response.status
                content_type = response.headers.get("Content-Type", "application/json")
        except urllib.error.HTTPError as exc:
            response_body = exc.read()
            status = exc.code
            content_type = exc.headers.get("Content-Type", "application/json")
        except urllib.error.URLError as exc:
            response_body = json.dumps({"error": str(exc.reason)}).encode("utf-8")
            status = 502
            content_type = "application/json"

        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(response_body)))
        self.end_headers()
        self.wfile.write(response_body)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18081)
    parser.add_argument("--env-file", default="")
    parser.add_argument("--api-base", default="")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    load_env_file(args.env_file)
    api_base = args.api_base or os.getenv("RUBIREC_API_BASE") or os.getenv("TOKENVERSE_API_BASE")
    api_key = args.api_key or os.getenv("RUBIREC_API_KEY") or os.getenv("TOKENVERSE_API_KEY")
    if not api_base:
        raise SystemExit("Missing upstream API base. Set RUBIREC_API_BASE or pass --api-base.")
    if not api_key:
        raise SystemExit("Missing upstream API key. Set RUBIREC_API_KEY or pass --api-key.")

    server = ThreadingHTTPServer((args.host, args.port), ProxyHandler)
    server.upstream_base_url = api_base
    server.api_key = api_key
    server.timeout = args.timeout
    print(f"Local OpenAI-compatible proxy listening on http://{args.host}:{args.port}", flush=True)
    print(f"Forwarding to {api_base.rstrip('/')}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
