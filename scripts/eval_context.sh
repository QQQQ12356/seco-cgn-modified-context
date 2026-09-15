#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

EVAL_SCRIPT="${ROOT}/ft_inference_all.py"
RESTORE_FROM="${RESTORE_FROM:-}"
TEST_FILE="${TEST_FILE:-}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${ROOT}/eval_result/${TAG}}"
MASTER_PORT="${MASTER_PORT:-29525}"
PER_DEVICE_EVAL_BATCH_SIZE=1
EVAL_SAMPLES="${EVAL_SAMPLES:-2000}"
SHUFFLE_EVAL=True

cd "$ROOT"

: "${RESTORE_FROM:?Set RESTORE_FROM to a checkpoint model.safetensors}"
: "${TEST_FILE:?Set TEST_FILE to a held-out converted JSONL file}"

if [[ ! -f "$RESTORE_FROM" || ! -f "$TEST_FILE" ]]; then
    echo "Missing checkpoint or test file" >&2
    exit 2
fi

"$TORCHRUN" \
    --nproc_per_node="$NUM_GPUS" \
    --master_port="$MASTER_PORT" \
    "$EVAL_SCRIPT" \
    --model_name_or_path "$MODEL_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --test_file "$TEST_FILE" \
    --restore_from "$RESTORE_FROM" \
    --eval_output_dir "$EVAL_OUTPUT_DIR" \
    --bf16 "$BF16" \
    --encoder_last_hidden_only "$ENCODER_LAST_HIDDEN_ONLY" \
    --report_to "$REPORT_TO" \
    --seed "$SEED" \
    --shuffle_eval "$SHUFFLE_EVAL" \
    --eval_samples "$EVAL_SAMPLES" \
    --per_device_eval_batch_size "$PER_DEVICE_EVAL_BATCH_SIZE" \
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
