#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/Users/tanqing/Desktop/aaai_pro}
WORK_DIR=${WORK_DIR:-$ROOT/manu_src/work/tokenverse_proxy}
PYTHON=${PYTHON:-/usr/bin/python3}
PROXY_SCRIPT=${PROXY_SCRIPT:-$ROOT/manu_src/scripts/network/tokenverse_connect_proxy.py}
TUNNEL_SCRIPT=${TUNNEL_SCRIPT:-$ROOT/manu_src/scripts/network/run_tokenverse_reverse_tunnel.sh}
RESTART_DELAY_SECONDS=${RESTART_DELAY_SECONDS:-10}

mkdir -p "$WORK_DIR"

proxy_supervisor_pid=""
tunnel_supervisor_pid=""

cleanup() {
  for pid in "$tunnel_supervisor_pid" "$proxy_supervisor_pid"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT INT TERM

supervise_proxy() {
  while true; do
    printf '%s starting local Tokenverse CONNECT proxy\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$WORK_DIR/connect_proxy.supervisor.log"
    set +e
    "$PYTHON" "$PROXY_SCRIPT" \
      --listen-host 127.0.0.1 \
      --listen-port 18765 \
      --allowed-host tokenverse.corp.kuaishou.com \
      --allowed-port 443 \
      --connect-timeout 30 \
      --idle-timeout 600 \
      >> "$WORK_DIR/connect_proxy.stdout.log" \
      2>> "$WORK_DIR/connect_proxy.stderr.log"
    status=$?
    set -e
    printf '%s local proxy exited with status %s; retrying in %ss\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$status" "$RESTART_DELAY_SECONDS" \
      >> "$WORK_DIR/connect_proxy.supervisor.log"
    sleep "$RESTART_DELAY_SECONDS"
  done
}

supervise_proxy &
proxy_supervisor_pid=$!

bash "$TUNNEL_SCRIPT" \
  >> "$WORK_DIR/reverse_tunnel.stdout.log" \
  2>> "$WORK_DIR/reverse_tunnel.stderr.log" &
tunnel_supervisor_pid=$!

printf 'Tokenverse 本地桥接已启动。关闭此窗口会停止代理和反向隧道。\n'
printf '本地代理: 127.0.0.1:18765；A100 代理入口: 127.0.0.1:18766。\n'
printf '网络中断后两个进程会自动重试；日志目录: %s\n' "$WORK_DIR"

set +e
wait "$tunnel_supervisor_pid"
status=$?
set -e
exit "$status"
