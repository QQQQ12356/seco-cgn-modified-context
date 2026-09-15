#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

TRAIN_SCRIPT="${ROOT}/instruction_finetune.py"
TRAIN_FILE="${TRAIN_FILE:-"/home/huangzj/proj/clustering/data/mrqa_train_24000_converted.jsonl"}"
DEV_FILE="${DEV_FILE:-"/home/huangzj/proj/clustering/data/mrqa_dev_24000_converted.jsonl"}"
MASTER_PORT="${MASTER_PORT:-29524}"
PER_DEVICE_TRAIN_BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_STEPS="${MAX_STEPS:-20000}"
SAVE_STEPS="${SAVE_STEPS:-5000}"
TRAIN=True
DDP_FIND_UNUSED_PARAMETERS=True
CONFIRM_TRAIN="${CONFIRM_TRAIN:-0}"

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
