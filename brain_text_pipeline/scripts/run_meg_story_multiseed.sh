#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

log() {
  printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*"
}

run_if_missing() {
  local target="$1"
  shift
  if [[ -e "${target}" ]]; then
    log "skip: ${target} already exists"
  else
    log "run: $*"
    "$@"
  fi
}

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DEVICE="${DEVICE:-cuda}"
MODEL_NAME="${MODEL_NAME:-t5-small}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-brain_text_pipeline/data/meg_aligned_postword_story_train/manifest.json}"
TEST_MANIFEST="${TEST_MANIFEST:-brain_text_pipeline/data/meg_aligned_postword_story_test/manifest.json}"
RUN_ROOT="${RUN_ROOT:-brain_text_pipeline/runs}"
PRIMARY_PREFIX="${PRIMARY_PREFIX:-t5_meg_postword_story_hybrid_last1}"
PRIMARY_NORM="${PRIMARY_NORM:-per_example}"
SEEDS="${SEEDS:-42 7 123}"
RUN_ALT_NORM="${RUN_ALT_NORM:-1}"
ALT_NORM="${ALT_NORM:-none}"
ALT_NORM_SEEDS="${ALT_NORM_SEEDS:-42}"
SUMMARY_JSON="${SUMMARY_JSON:-${RUN_ROOT}/${PRIMARY_PREFIX}_multiseed_summary.json}"

train_and_eval() {
  local seed="$1"
  local brain_norm="$2"
  local run_dir="$3"
  local eval_json="${run_dir}/eval_story_test_50k.json"

  run_if_missing "${run_dir}/brain_encoder.pt" \
    "${PYTHON_BIN}" brain_text_pipeline/scripts/train_t5_brain_crossattn.py \
    --mode meg_supervised \
    --model_name_or_path "${MODEL_NAME}" \
    --meg_dataset_path "${TRAIN_MANIFEST}" \
    --output_dir "${run_dir}" \
    --batch_size "${TRAIN_BATCH_SIZE:-32}" \
    --lr "${TRAIN_LR:-5e-5}" \
    --epochs "${TRAIN_EPOCHS:-6}" \
    --bf16 \
    --freeze_t5 \
    --unfreeze_cross_attn \
    --cross_attn_last_n "${CROSS_ATTN_LAST_N:-6}" \
    --unfreeze_last_n "${UNFREEZE_LAST_N:-1}" \
    --decoder_context_mode "${DECODER_CONTEXT_MODE:-target_only}" \
    --brain_norm "${brain_norm}" \
    --max_text_len "${MAX_TEXT_LEN:-8}" \
    --max_brain_len "${MAX_BRAIN_LEN:-120}" \
    --log_interval "${LOG_INTERVAL:-100}" \
    --cpu_threads "${CPU_THREADS:-8}" \
    --num_workers "${NUM_WORKERS:-0}" \
    --device "${DEVICE}" \
    --seed "${seed}"

  run_if_missing "${eval_json}" \
    "${PYTHON_BIN}" brain_text_pipeline/scripts/eval_brain_controls.py \
    --model_name_or_path "${run_dir}" \
    --brain_encoder_ckpt "${run_dir}/brain_encoder.pt" \
    --meg_dataset_path "${TEST_MANIFEST}" \
    --samples "${EVAL_SAMPLES:-50000}" \
    --batch_size "${EVAL_BATCH_SIZE:-32}" \
    --device "${DEVICE}" \
    --decoder_context_mode "${DECODER_CONTEXT_MODE:-target_only}" \
    --brain_norm "${brain_norm}" \
    --max_text_len "${MAX_TEXT_LEN:-8}" \
    --max_brain_len "${MAX_BRAIN_LEN:-120}" \
    --seed "${seed}" \
    --out_json "${eval_json}"
}

declare -a eval_jsons=()

for seed in ${SEEDS}; do
  if [[ "${seed}" == "42" && "${PRIMARY_NORM}" == "per_example" ]]; then
    run_dir="${RUN_ROOT}/${PRIMARY_PREFIX}"
  else
    run_dir="${RUN_ROOT}/${PRIMARY_PREFIX}_seed${seed}"
  fi
  train_and_eval "${seed}" "${PRIMARY_NORM}" "${run_dir}"
  eval_jsons+=("${run_dir}/eval_story_test_50k.json")
done

if [[ "${RUN_ALT_NORM}" == "1" ]]; then
  for seed in ${ALT_NORM_SEEDS}; do
    run_dir="${RUN_ROOT}/${PRIMARY_PREFIX}_${ALT_NORM}_seed${seed}"
    train_and_eval "${seed}" "${ALT_NORM}" "${run_dir}"
    eval_jsons+=("${run_dir}/eval_story_test_50k.json")
  done
fi

log "summarize: ${SUMMARY_JSON}"
"${PYTHON_BIN}" brain_text_pipeline/scripts/summarize_meg_story_multiseed.py \
  --out_json "${SUMMARY_JSON}" \
  "${eval_jsons[@]}"

log "done: ${SUMMARY_JSON}"
