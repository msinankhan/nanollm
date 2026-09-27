#!/usr/bin/env bash
set -Eeuo pipefail

# Single-GPU Colab G4 reference pipeline. Override any value by exporting it
# before invoking this script, e.g. MODEL_TAG=my-test ./base_train.sh preflight.
# Full local-machine Colab CLI setup and recovery commands are documented in
# COLAB_TRAINING.md. Provision/mount with the CLI, enter with `colab ssh`, then
# run this file's staged actions inside /content/nanollm.
ACTION="${1:-status}"
MODEL_TAG="${MODEL_TAG:-d24-r12-bf16-reference}"
WANDB_RUN="${WANDB_RUN:-$MODEL_TAG}"
BASE_DIR="${NANOLLM_BASE_DIR:-/content/drive/MyDrive/nanollm-runs/reference-d24-r12}"
DATA_DIR="${NANOLLM_DATA_DIR:-/content/nanollm-data}"
PERSISTENT_DATA_DIR="${NANOLLM_PERSISTENT_DATA_DIR:-$BASE_DIR/static_data/fineweb-edu-170}"
STAGING_DIR="${NANOLLM_CHECKPOINT_STAGING_DIR:-/content/nanollm-checkpoint-staging}"
DEPTH="${DEPTH:-24}"
DATA_RATIO="${DATA_RATIO:-12}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-16}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:--1}"
MAX_RUNTIME_MINUTES="${MAX_RUNTIME_MINUTES:-1320}"
EXIT_GUARD_MINUTES="${EXIT_GUARD_MINUTES:-30}"
SAVE_EVERY_MINUTES="${SAVE_EVERY_MINUTES:-45}"
SFT_SAVE_EVERY="${SFT_SAVE_EVERY:-200}"
RL_SAVE_EVERY="${RL_SAVE_EVERY:-20}"
KEEP_CHECKPOINTS="${KEEP_CHECKPOINTS:-3}"
TOKENIZER_VOCAB_SIZE="${TOKENIZER_VOCAB_SIZE:-32768}"
DATA_SHARDS="${DATA_SHARDS:-170}"

export NANOLLM_BASE_DIR="$BASE_DIR"
export NANOLLM_DATA_DIR="$DATA_DIR"
export NANOLLM_ATTN_BACKEND="${NANOLLM_ATTN_BACKEND:-fa2_hub}"
export NANOCHAT_DTYPE="${NANOCHAT_DTYPE:-bfloat16}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

BASE_CHECKPOINT_DIR="$BASE_DIR/base_checkpoints/$MODEL_TAG"
BASE_COMPLETE_MARKER="$BASE_CHECKPOINT_DIR/training.complete.json"
SFT_COMPLETE_MARKER="$BASE_DIR/chatsft_checkpoints/$MODEL_TAG/training.complete.json"
RL_COMPLETE_MARKER="$BASE_DIR/chatrl_checkpoints/$MODEL_TAG/training.complete.json"
BASE_EVAL_COMPLETE_MARKER="$BASE_CHECKPOINT_DIR/base_eval.complete"
SFT_EVAL_COMPLETE_MARKER="$BASE_DIR/chatsft_checkpoints/$MODEL_TAG/chat_eval.complete"

die() {
    echo "ERROR: $*" >&2
    exit 1
}

ensure_environment() {
    environment_extra="${1:-gpu}"
    command -v uv >/dev/null 2>&1 || die "uv is not installed"
    uv sync --extra "$environment_extra"
    # shellcheck disable=SC1091
    source .venv/bin/activate
}

require_single_cuda_gpu() {
    python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")
if torch.cuda.device_count() != 1:
    raise SystemExit(f"Expected exactly one visible CUDA GPU, found {torch.cuda.device_count()}")
print(f"GPU ready: {torch.cuda.get_device_name(0)}")
PY
}

verify_assets() {
    python - "$DATA_SHARDS" "$TOKENIZER_VOCAB_SIZE" <<'PY'
import glob
import os
import sys
from nanollm.tokenizer import get_tokenizer

expected_shards = int(sys.argv[1])
expected_vocab = int(sys.argv[2])
data_dir = os.environ["NANOLLM_DATA_DIR"]
shards = glob.glob(os.path.join(data_dir, "shard_*.parquet"))
if len(shards) < expected_shards:
    raise SystemExit(f"Need at least {expected_shards} dataset shards in {data_dir}; found {len(shards)}")
tokenizer = get_tokenizer()
actual_vocab = tokenizer.get_vocab_size()
if actual_vocab != expected_vocab:
    raise SystemExit(f"Expected tokenizer vocab {expected_vocab}; found {actual_vocab}")
print(f"Assets ready: {len(shards)} shards, vocab size {actual_vocab}")
PY
}

latest_checkpoint_step() {
    python - "$1" <<'PY'
import sys
from nanollm.checkpoint_manager import find_last_step
try:
    print(find_last_step(sys.argv[1]))
except FileNotFoundError:
    print("")
PY
}

base_train_args=(
    --depth="$DEPTH"
    --window-pattern=SSSL
    --target-params-data-ratio="$DATA_RATIO"
    --device-batch-size="$DEVICE_BATCH_SIZE"
    --total-batch-size="$TOTAL_BATCH_SIZE"
    --model-tag="$MODEL_TAG"
    --run="$WANDB_RUN"
    --checkpoint-staging-dir="$STAGING_DIR"
    --save-every-minutes="$SAVE_EVERY_MINUTES"
    --first-save-minutes=15
    --keep-checkpoints="$KEEP_CHECKPOINTS"
    --max-runtime-minutes="$MAX_RUNTIME_MINUTES"
    --exit-guard-minutes="$EXIT_GUARD_MINUTES"
)

case "$ACTION" in
    prepare)
        ensure_environment cpu
        mkdir -p "$BASE_DIR" "$PERSISTENT_DATA_DIR"
        NANOLLM_DATA_DIR="$PERSISTENT_DATA_DIR" python -m nanollm.dataset -n "$DATA_SHARDS"
        if [[ ! -f "$BASE_DIR/tokenizer_dir/tokenizer.pkl" ]]; then
            NANOLLM_DATA_DIR="$PERSISTENT_DATA_DIR" python -m scripts.tok_train --vocab_size "$TOKENIZER_VOCAB_SIZE"
        fi
        NANOLLM_DATA_DIR="$PERSISTENT_DATA_DIR" python -m scripts.tok_eval
        python -m scripts.prepare_posttrain_data
        NANOLLM_DATA_DIR="$PERSISTENT_DATA_DIR" verify_assets
        ;;
    hydrate)
        ensure_environment
        [[ -d "$PERSISTENT_DATA_DIR" ]] || die "Persistent dataset not found: $PERSISTENT_DATA_DIR"
        mkdir -p "$DATA_DIR"
        cp -an "$PERSISTENT_DATA_DIR/." "$DATA_DIR/"
        verify_assets
        ;;
    preflight)
        ensure_environment
        require_single_cuda_gpu
        verify_assets
        python -m scripts.base_train \
            --num-iterations="${PREFLIGHT_STEPS:-20}" \
            --depth="$DEPTH" \
            --window-pattern=SSSL \
            --device-batch-size="$DEVICE_BATCH_SIZE" \
            --total-batch-size="${PREFLIGHT_TOTAL_BATCH_SIZE:-65536}" \
            --model-tag="${MODEL_TAG}-preflight" \
            --run=dummy \
            --eval-every=-1 \
            --core-metric-every=-1 \
            --sample-every=-1 \
            --save-every-minutes=-1
        ;;
    pretrain)
        ensure_environment
        require_single_cuda_gpu
        verify_assets
        mkdir -p "$STAGING_DIR"
        if [[ -f "$BASE_COMPLETE_MARKER" ]]; then
            echo "Base training is already complete: $BASE_COMPLETE_MARKER"
            exit 0
        fi
        resume_step="$(latest_checkpoint_step "$BASE_CHECKPOINT_DIR")"
        if [[ -n "$resume_step" ]]; then
            echo "Resuming $MODEL_TAG from validated checkpoint step $resume_step"
            base_train_args+=(--resume-from-step=latest)
        else
            echo "Starting a new base run: $MODEL_TAG"
        fi
        python -m scripts.base_train "${base_train_args[@]}"
        if [[ ! -f "$BASE_COMPLETE_MARKER" ]]; then
            echo "Session ended safely with a resumable checkpoint; base training is not finished yet."
            exit 75
        fi
        ;;
    posttrain)
        ensure_environment
        require_single_cuda_gpu
        verify_assets
        [[ -f "$BASE_COMPLETE_MARKER" ]] || die "Base training is not marked complete; refusing to start SFT"
        if [[ ! -f "$BASE_EVAL_COMPLETE_MARKER" ]]; then
            python -m scripts.base_eval --model-tag "$MODEL_TAG" --device-batch-size "$DEVICE_BATCH_SIZE"
            touch "$BASE_EVAL_COMPLETE_MARKER"
        else
            echo "Base evaluation is already complete; skipping"
        fi
        if [[ ! -f "$SFT_COMPLETE_MARKER" ]]; then
            sft_args=(
                --model-tag "$MODEL_TAG"
                --run "${WANDB_RUN}-sft"
                --save-every "$SFT_SAVE_EVERY"
                --checkpoint-staging-dir "$STAGING_DIR/sft"
                --max-runtime-minutes "$MAX_RUNTIME_MINUTES"
                --exit-guard-minutes "$EXIT_GUARD_MINUTES"
            )
            sft_checkpoint_dir="$BASE_DIR/chatsft_checkpoints/$MODEL_TAG"
            if [[ -n "$(latest_checkpoint_step "$sft_checkpoint_dir")" ]]; then
                sft_args+=(--resume-from-step latest)
            fi
            python -m scripts.chat_sft "${sft_args[@]}"
            if [[ ! -f "$SFT_COMPLETE_MARKER" ]]; then
                echo "SFT session ended safely with a resumable checkpoint; SFT is not finished yet."
                exit 75
            fi
        else
            echo "SFT is already complete; skipping training"
        fi
        if [[ ! -f "$SFT_EVAL_COMPLETE_MARKER" ]]; then
            python -m scripts.chat_eval --source sft --model-tag "$MODEL_TAG"
            touch "$SFT_EVAL_COMPLETE_MARKER"
        else
            echo "SFT evaluation is already complete; skipping"
        fi
        if [[ ! -f "$RL_COMPLETE_MARKER" ]]; then
            rl_args=(
                --model-tag "$MODEL_TAG"
                --run "${WANDB_RUN}-rl"
                --save-every "$RL_SAVE_EVERY"
                --checkpoint-staging-dir "$STAGING_DIR/rl"
                --max-runtime-minutes "$MAX_RUNTIME_MINUTES"
                --exit-guard-minutes "$EXIT_GUARD_MINUTES"
            )
            rl_checkpoint_dir="$BASE_DIR/chatrl_checkpoints/$MODEL_TAG"
            if [[ -n "$(latest_checkpoint_step "$rl_checkpoint_dir")" ]]; then
                rl_args+=(--resume-from-step latest)
            fi
            python -m scripts.chat_rl "${rl_args[@]}"
            if [[ ! -f "$RL_COMPLETE_MARKER" ]]; then
                echo "ChatRL session ended safely with a resumable checkpoint; ChatRL is not finished yet."
                exit 75
            fi
        else
            echo "ChatRL is already complete; skipping training"
        fi
        python -m scripts.chat_eval --source rl --model-tag "$MODEL_TAG"
        ;;
    status)
        echo "Model tag:       $MODEL_TAG"
        echo "Base directory:  $BASE_DIR"
        echo "Data directory:  $DATA_DIR"
        echo "Data source:     $PERSISTENT_DATA_DIR"
        echo "Staging:         $STAGING_DIR"
        echo "Recipe:          d${DEPTH}, ratio ${DATA_RATIO}, BF16, SSSL, device batch ${DEVICE_BATCH_SIZE}"
        if [[ -f "$BASE_COMPLETE_MARKER" ]]; then
            echo "Base status:     complete"
        elif [[ -d "$BASE_CHECKPOINT_DIR" ]]; then
            echo "Base status:     incomplete (run pretrain to resume)"
        else
            echo "Base status:     not started"
        fi
        [[ -f "$SFT_COMPLETE_MARKER" ]] && echo "SFT status:      complete" || echo "SFT status:      not complete"
        [[ -f "$RL_COMPLETE_MARKER" ]] && echo "ChatRL status:   complete" || echo "ChatRL status:   not complete"
        ;;
    *)
        die "Unknown action '$ACTION'. Use: prepare | hydrate | preflight | pretrain | posttrain | status"
        ;;
esac
