#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export TOKENIZERS_PARALLELISM=false
export WANDB_MODE="${WANDB_MODE:-disabled}"

TORCHRUN="${TORCHRUN:-${HOME}/miniconda3/envs/icae_v2/bin/torchrun}"
NUM_GPUS="${NUM_GPUS:-1}"
SEED="${SEED:-42}"
BF16=True
ENCODER_LAST_HIDDEN_ONLY=True
REPORT_TO=none

MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.2-1B-Instruct}"
MODEL_SHORT="${MODEL_NAME##*/}"
COMPRESS_RATIO="${COMPRESS_RATIO:-16}"
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

EVAL_SCRIPT="${ROOT}/ft_inference_all.py"
RESTORE_FROM="${RESTORE_FROM:-${ROOT}/output/context_adaptive_v1/Llama-3.2-1B-Instruct/ratio_16/checkpoint-20000/model.safetensors}"
TEST_FILE="${TEST_FILE:-/home/huangzj/proj/clustering/data/mrqa_test_58221_9633_converted.jsonl}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${ROOT}/eval_result/${TAG}}"
MASTER_PORT="${MASTER_PORT:-29525}"
PER_DEVICE_EVAL_BATCH_SIZE=1
EVAL_SAMPLES="${EVAL_SAMPLES:-67854}"
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
