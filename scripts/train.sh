#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export TOKENIZERS_PARALLELISM=false
export WANDB_MODE="${WANDB_MODE:-disabled}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
TORCHRUN="${TORCHRUN:-${HOME}/miniconda3/envs/icae_v2/bin/torchrun}"
NUM_GPUS="${NUM_GPUS:-2}"
SEED="${SEED:-42}"
BF16=True
ENCODER_LAST_HIDDEN_ONLY=True
REPORT_TO=none

MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.2-1B-Instruct}"
MODEL_SHORT="${MODEL_NAME##*/}"
COMPRESS_RATIO="${COMPRESS_RATIO:-32}"
COMPRESSOR_VERSION="${COMPRESSOR_VERSION:-context_adaptive_v1}"
BUDGET_MODE=strict_context
LEGACY_BUDGET_MODE=strict_context

QUERY_ATTENTION_MODE=dot
QUERY_MODE=multi_slot
DIVERSITY_MODE=attention
QUERY_SLOTS=8
RELEVANCE_DIM=256
QUERY_PHRASE_WIDTHS=(2 4)

CONTEXT_WINDOW_WIDTHS=(8 32)
BLOCK_WIDTH="${BLOCK_WIDTH:-128}"
BOUNDARY_MODE="${BOUNDARY_MODE:-semantic}"
CONTEXT_BUDGET_MODE="${CONTEXT_BUDGET_MODE:-adaptive}"
MERGE_MODE="${MERGE_MODE:-hybrid}"
NOVELTY_WEIGHT="${NOVELTY_WEIGHT:-0.5}"
BOUNDARY_RADIUS="${BOUNDARY_RADIUS:-0.25}"
ANCHOR_WEIGHT="${ANCHOR_WEIGHT:-0.35}"

TAG="${TAG:-${COMPRESSOR_VERSION}}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/output/${TAG}/${MODEL_SHORT}/ratio_${COMPRESS_RATIO}}"

TRAIN_SCRIPT="${ROOT}/instruction_finetune.py"
TRAIN_FILE="${TRAIN_FILE:-/home/huangzj/proj/clustering/data/mrqa_train_24000_converted.jsonl}"
DEV_FILE="${DEV_FILE:-/home/huangzj/proj/clustering/data/mrqa_train_sample_debug_3000_converted.jsonl}"
MASTER_PORT="${MASTER_PORT:-29524}"
PER_DEVICE_TRAIN_BATCH_SIZE="${BATCH_SIZE:-2}"
MAX_STEPS="${MAX_STEPS:-20000}"
SAVE_STEPS="${SAVE_STEPS:-10000}"
TRAIN=True
DDP_FIND_UNUSED_PARAMETERS=True
CONFIRM_TRAIN="${CONFIRM_TRAIN:-1}"

cd "$ROOT"

if [[ "$CONFIRM_TRAIN" != 1 ]]; then
    echo "Training disabled. Explicitly set CONFIRM_TRAIN=1 when training is approved." >&2
    exit 2
fi

if [[ ! -f "$TRAIN_FILE" || ! -f "$DEV_FILE" ]]; then
    echo "Prepare train/dev data first" >&2
    exit 2
fi

if [[ -e "$OUTPUT_DIR" ]]; then
    echo "Use a new OUTPUT_DIR; refusing overwrite" >&2
    exit 2
fi

"$TORCHRUN" \
    --nproc_per_node="$NUM_GPUS" \
    --master_port="$MASTER_PORT" \
    "$TRAIN_SCRIPT" \
    --model_name_or_path "$MODEL_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --train_file "$TRAIN_FILE" \
    --test_file "$DEV_FILE" \
    --train "$TRAIN" \
    --bf16 "$BF16" \
    --encoder_last_hidden_only "$ENCODER_LAST_HIDDEN_ONLY" \
    --report_to "$REPORT_TO" \
    --max_steps "$MAX_STEPS" \
    --save_steps "$SAVE_STEPS" \
    --seed "$SEED" \
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
    --ddp_find_unused_parameters "$DDP_FIND_UNUSED_PARAMETERS" \
    --compressor_version "$COMPRESSOR_VERSION" \
    --compress_ratio "$COMPRESS_RATIO" \
    --budget_mode "$BUDGET_MODE" \
    --legacy_budget_mode "$LEGACY_BUDGET_MODE" \
    --query_attention_mode "$QUERY_ATTENTION_MODE" \
    --query_mode "$QUERY_MODE" \
    --diversity_mode "$DIVERSITY_MODE" \
    --query_slots "$QUERY_SLOTS" \
    --relevance_dim "$RELEVANCE_DIM" \
    --query_phrase_widths "${QUERY_PHRASE_WIDTHS[@]}" \
    --context_window_widths "${CONTEXT_WINDOW_WIDTHS[@]}" \
    --allocation_block_width "$BLOCK_WIDTH" \
    --context_boundary_mode "$BOUNDARY_MODE" \
    --context_budget_mode "$CONTEXT_BUDGET_MODE" \
    --context_merge_mode "$MERGE_MODE" \
    --context_novelty_weight "$NOVELTY_WEIGHT" \
    --context_boundary_radius "$BOUNDARY_RADIUS" \
    --context_anchor_weight "$ANCHOR_WEIGHT" \
    "$@"
