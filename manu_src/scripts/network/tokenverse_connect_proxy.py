#!/usr/bin/env python3
"""Minimal allowlisted HTTP CONNECT proxy for a local Tokenverse SSH tunnel."""

from __future__ import annotations

import argparse
import select
import socket
import socketserver
import sys
from typing import ClassVar


MAX_HEADER_BYTES = 64 * 1024


class ConnectHandler(socketserver.BaseRequestHandler):
    allowed_host: ClassVar[str]
    allowed_port: ClassVar[int]
    connect_timeout: ClassVar[float]

    def _read_headers(self) -> bytes:
        payload = bytearray()
        while b"\r\n\r\n" not in payload:
            chunk = self.request.recv(4096)
            if not chunk:
                break
            payload.extend(chunk)
            if len(payload) > MAX_HEADER_BYTES:
                raise ValueError("CONNECT headers exceed the size limit")
        return bytes(payload)

    def _reject(self, status: str) -> None:
        self.request.sendall(
            f"HTTP/1.1 {status}\r\nConnection: close\r\n\r\n".encode("ascii")
        )

    def handle(self) -> None:
        upstream: socket.socket | None = None
        try:
            headers = self._read_headers()
            request_line = headers.split(b"\r\n", 1)[0].decode("ascii", errors="replace")
            parts = request_line.split()
            if len(parts) != 3 or parts[0].upper() != "CONNECT":
                self._reject("405 Method Not Allowed")
                return
            authority = parts[1]
            host, separator, port_text = authority.rpartition(":")
            if not separator or host.lower() != self.allowed_host or int(port_text) != self.allowed_port:
                self._reject("403 Forbidden")
                return
            upstream = socket.create_connection(
                (self.allowed_host, self.allowed_port), timeout=self.connect_timeout
            )
            upstream.setblocking(False)
            self.request.setblocking(False)
            self.request.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            sockets = [self.request, upstream]
            while True:
                readable, _, exceptional = select.select(sockets, [], sockets, 60.0)
                if exceptional or not readable:
                    break
                for source in readable:
                    target = upstream if source is self.request else self.request
                    data = source.recv(64 * 1024)
                    if not data:
                        return
                    target.sendall(data)
        except (OSError, ValueError):
            try:
                self._reject("502 Bad Gateway")
            except OSError:
                pass
        finally:
            if upstream is not None:
                upstream.close()


class ThreadingConnectServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, default=18765)
    parser.add_argument("--allowed-host", default="tokenverse.corp.kuaishou.com")
    parser.add_argument("--allowed-port", type=int, default=443)
    parser.add_argument("--connect-timeout", type=float, default=10.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.listen_port <= 65535 or not 1 <= args.allowed_port <= 65535:
        raise ValueError("ports must be in [1, 65535]")
    if args.connect_timeout <= 0:
        raise ValueError("connect timeout must be positive")
    ConnectHandler.allowed_host = args.allowed_host.strip().lower()
    ConnectHandler.allowed_port = args.allowed_port
    ConnectHandler.connect_timeout = args.connect_timeout
    with ThreadingConnectServer(
        (args.listen_host, args.listen_port), ConnectHandler
    ) as server:
        print(
            f"CONNECT proxy listening on {args.listen_host}:{args.listen_port}; "
            f"allowlist={ConnectHandler.allowed_host}:{ConnectHandler.allowed_port}",
            file=sys.stderr,
            flush=True,
        )
        server.serve_forever(poll_interval=0.5)


if __name__ == "__main__":
    main()
