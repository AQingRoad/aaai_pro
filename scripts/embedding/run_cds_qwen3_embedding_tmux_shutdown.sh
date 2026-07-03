#!/usr/bin/env bash
set -euo pipefail

# Start the CDs_and_Vinyl Qwen3-Embedding-0.6B training/eval workflow in tmux.
# Worker mode runs inside tmux, logs to RUN_LOG, and powers off the server when
# the workflow exits.

ROOT=${ROOT:-/root/autodl-tmp/rec/aaai_pro}
TMUX_SESSION=${TMUX_SESSION:-cds_qwen3_embed_4096}
TMUX_WINDOW=${TMUX_WINDOW:-train}
LOG_DIR=${LOG_DIR:-$ROOT/logs/embedding}
RUN_LOG=${RUN_LOG:-$LOG_DIR/${TMUX_SESSION}_$(date +%Y%m%d_%H%M%S).log}
SHUTDOWN_ON_EXIT=${SHUTDOWN_ON_EXIT:-1}
SHUTDOWN_DELAY_MIN=${SHUTDOWN_DELAY_MIN:-1}

run_worker() {
  mkdir -p "$(dirname "$RUN_LOG")"
  exec > >(stdbuf -oL tee -a "$RUN_LOG") 2>&1

  finish() {
    local status=$?
    echo "WORKFLOW_EXIT_STATUS=$status"
    echo "WORKFLOW_END_TIME=$(date -Is)"
    sync || true
    if [[ "$SHUTDOWN_ON_EXIT" == "1" || "$SHUTDOWN_ON_EXIT" == "true" ]]; then
      echo "Scheduling server shutdown in ${SHUTDOWN_DELAY_MIN} minute(s)."
      if shutdown_bin=$(command -v shutdown 2>/dev/null); then
        "$shutdown_bin" -h "+${SHUTDOWN_DELAY_MIN}" "Qwen3 embedding training/eval workflow finished with status ${status}." || true
      elif poweroff_bin=$(command -v poweroff 2>/dev/null); then
        nohup sh -c "sleep '${SHUTDOWN_DELAY_MIN}m'; '$poweroff_bin'" >/tmp/qwen3_embedding_poweroff.log 2>&1 &
      elif halt_bin=$(command -v halt 2>/dev/null); then
        nohup sh -c "sleep '${SHUTDOWN_DELAY_MIN}m'; '$halt_bin'" >/tmp/qwen3_embedding_halt.log 2>&1 &
      else
        echo "No shutdown, poweroff, or halt command found; cannot power off automatically."
      fi
    else
      echo "SHUTDOWN_ON_EXIT=$SHUTDOWN_ON_EXIT, skip shutdown."
    fi
    exit "$status"
  }
  trap finish EXIT

  cd "$ROOT"
  echo "WORKFLOW_START_TIME=$(date -Is)"
  echo "ROOT=$ROOT"
  echo "RUN_LOG=$RUN_LOG"
  echo "TMUX_SESSION=$TMUX_SESSION"
  echo "SHUTDOWN_ON_EXIT=$SHUTDOWN_ON_EXIT"
  echo "SHUTDOWN_DELAY_MIN=$SHUTDOWN_DELAY_MIN"

  export ROOT=/root/autodl-tmp/rec/aaai_pro
  export VENV=/root/miniconda3/envs/swift
  export PYTHON_BIN=/root/miniconda3/envs/swift/bin/python
  export COT_CANDIDATE_LISTS=/root/autodl-tmp/rec/aaai_pro/outputs/rrec_amazon/CDs_and_Vinyl/cot_candidate_lists_glm47_meta_compact_one_train_raw.jsonl
  export COT_TEXT_MODE=tagged
  export INCLUDE_HISTORY=0
  export INCLUDE_COT=1
  export COT_EMBEDDER_DATASET=/root/autodl-tmp/rec/aaai_pro/outputs/rrec_amazon/CDs_and_Vinyl/phase0_embedder_cds_glm47_meta_compact_one_tagged_cot_only.jsonl
  export BASE_EMBEDDING_MODEL=/root/autodl-tmp/modelscope_cache/models/Qwen/Qwen3-Embedding-0.6B
  export EMBEDDER_OUT=/root/autodl-tmp/rec/aaai_pro/checkpoints/rrec_amazon_CDs_and_Vinyl/qwen3_embedding_0p6b_cds_glm47_meta_compact_one_tagged_cot_only_len4096_batch32_accum8_epoch5
  export EMBEDDER_CUDA_VISIBLE_DEVICES=0
  export EMBEDDER_NPROC_PER_NODE=1
  export EMBEDDER_BATCH_SIZE=32
  export EMBEDDER_GRAD_ACCUM=8
  export EMBEDDER_MAX_LENGTH=4096
  export MAX_COT_CHARS=0
  export MAX_ITEM_CHARS=0
  export EMBEDDER_EPOCHS=5
  export EMBEDDER_MAX_STEPS=-1
  export EMBEDDER_LR=3e-6
  export EMBEDDER_SAVE_STEPS=auto
  export EMBEDDER_TORCH_DTYPE=bfloat16
  export EMBEDDER_GRADIENT_CHECKPOINTING=auto
  export EMBEDDER_CROSS_GPU_NEGATIVES=0
  export FORCE_REBUILD_DATASET=0

  train_status=0
  bash scripts/embedding/run_train_cds_cot_embedding_tidal.sh || train_status=$?
  echo "TRAIN_EXIT_STATUS=$train_status"

  eval_status=0
  if [[ "$train_status" -eq 0 ]]; then
    export CHECKPOINT_ROOT="$EMBEDDER_OUT"
    export EVAL_DIR=/root/autodl-tmp/rec/aaai_pro/outputs/rrec_amazon/eval/qwen3_embedding_0p6b_cds_glm47_meta_compact_one_tagged_cot_only_len4096_batch32_accum8_epoch5_test
    export SPLIT=test
    export MAX_EXAMPLES=0
    export CUDA_VISIBLE_DEVICES=0
    export EMBEDDING_DEVICE=cuda:0
    export EMBEDDING_BATCH_SIZE=${EMBEDDING_BATCH_SIZE:-256}
    export EMBEDDING_MAX_LENGTH=4096
    export EMBEDDING_TORCH_DTYPE=bfloat16
    export FORCE_EVAL=1
    bash scripts/embedding/run_eval_cds_embedding_checkpoints_tidal.sh || eval_status=$?
  else
    echo "Skip evaluation because training failed."
    eval_status=99
  fi
  echo "EVAL_EXIT_STATUS=$eval_status"

  if [[ "$train_status" -ne 0 ]]; then
    exit "$train_status"
  fi
  exit "$eval_status"
}

if [[ "${TMUX_WORKER:-0}" == "1" ]]; then
  run_worker
fi

if ! command -v tmux >/dev/null 2>&1; then
  echo "Missing tmux. Install tmux before running this script." >&2
  exit 1
fi

mkdir -p "$LOG_DIR"
SCRIPT_PATH=$(readlink -f "$0")

if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
  echo "tmux session already exists: $TMUX_SESSION" >&2
  echo "Attach: tmux attach -t $TMUX_SESSION" >&2
  exit 1
fi

tmux new-session -d -s "$TMUX_SESSION" -n "$TMUX_WINDOW" \
  "cd '$ROOT' && TMUX_WORKER=1 ROOT='$ROOT' RUN_LOG='$RUN_LOG' TMUX_SESSION='$TMUX_SESSION' SHUTDOWN_ON_EXIT='$SHUTDOWN_ON_EXIT' SHUTDOWN_DELAY_MIN='$SHUTDOWN_DELAY_MIN' bash '$SCRIPT_PATH'"

echo "Started tmux session: $TMUX_SESSION"
echo "Attach: tmux attach -t $TMUX_SESSION"
echo "Tail log: tail -f $RUN_LOG"
echo "Shutdown on exit: $SHUTDOWN_ON_EXIT, delay minutes: $SHUTDOWN_DELAY_MIN"
