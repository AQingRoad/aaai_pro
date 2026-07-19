#!/usr/bin/env bash
set -euo pipefail

SSH_TARGET=${SSH_TARGET:-A100-1}
LOCAL_PROXY_HOST=${LOCAL_PROXY_HOST:-127.0.0.1}
LOCAL_PROXY_PORT=${LOCAL_PROXY_PORT:-18765}
REMOTE_PROXY_HOST=${REMOTE_PROXY_HOST:-127.0.0.1}
REMOTE_PROXY_PORT=${REMOTE_PROXY_PORT:-18766}
RECONNECT_DELAY_SECONDS=${RECONNECT_DELAY_SECONDS:-10}
SERVER_ALIVE_INTERVAL=${SERVER_ALIVE_INTERVAL:-30}
SERVER_ALIVE_COUNT_MAX=${SERVER_ALIVE_COUNT_MAX:-10}

child_pid=""
cleanup() {
  if [[ -n "$child_pid" ]] && kill -0 "$child_pid" 2>/dev/null; then
    kill "$child_pid" 2>/dev/null || true
    wait "$child_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

while true; do
  printf '%s starting reverse tunnel %s:%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$REMOTE_PROXY_HOST" "$REMOTE_PROXY_PORT" >&2
  ssh -N \
    -o BatchMode=yes \
    -o ExitOnForwardFailure=yes \
    -o ConnectTimeout=20 \
    -o TCPKeepAlive=yes \
    -o ServerAliveInterval="$SERVER_ALIVE_INTERVAL" \
    -o ServerAliveCountMax="$SERVER_ALIVE_COUNT_MAX" \
    -R "$REMOTE_PROXY_HOST:$REMOTE_PROXY_PORT:$LOCAL_PROXY_HOST:$LOCAL_PROXY_PORT" \
    "$SSH_TARGET" &
  child_pid=$!
  set +e
  wait "$child_pid"
  status=$?
  set -e
  child_pid=""
  printf '%s reverse tunnel exited with status %s; retrying in %ss\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$status" "$RECONNECT_DELAY_SECONDS" >&2
  sleep "$RECONNECT_DELAY_SECONDS"
done
