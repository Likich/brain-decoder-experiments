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

# Thread caps for long GPU runs.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# Core config.
PYTHON_BIN="${PYTHON_BIN:-python3}"
DEVICE="${DEVICE:-cuda}"
SOURCE_TOKENIZER="${SOURCE_TOKENIZER:-t5-small}"
TEXT_MODEL="${TEXT_MODEL:-bert-base-uncased}"
MODEL_NAME="${MODEL_NAME:-t5-small}"
SEED="${SEED:-42}"

TRAIN_MANIFEST="${TRAIN_MANIFEST:-brain_text_pipeline/data/meg_aligned_postword_story_train/manifest.json}"
TEST_MANIFEST="${TEST_MANIFEST:-brain_text_pipeline/data/meg_aligned_postword_story_test/manifest.json}"

AUX_TRAIN_DIR="${AUX_TRAIN_DIR:-brain_text_pipeline/data/aux_bert_postword_story_train}"
AUX_TEST_DIR="${AUX_TEST_DIR:-brain_text_pipeline/data/aux_bert_postword_story_test}"
COMBO_TRAIN_DIR="${COMBO_TRAIN_DIR:-brain_text_pipeline/data/megplusbert_postword_story_train}"
COMBO_TEST_DIR="${COMBO_TEST_DIR:-brain_text_pipeline/data/megplusbert_postword_story_test}"
RUN_DIR="${RUN_DIR:-brain_text_pipeline/runs/t5_megplusbert_postword_story_hybrid_last1}"
EVAL_JSON="${EVAL_JSON:-${RUN_DIR}/eval_story_test_50k_megonly.json}"

# Build text-aux data.
run_if_missing "${AUX_TRAIN_DIR}/manifest.json" \
  "${PYTHON_BIN}" brain_text_pipeline/scripts/build_text_aux_dataset.py \
  --source_manifest "${TRAIN_MANIFEST}" \
  --out_dir "${AUX_TRAIN_DIR}" \
  --source_tokenizer_name_or_path "${SOURCE_TOKENIZER}" \
  --text_model_name_or_path "${TEXT_MODEL}" \
  --batch_size "${AUX_BUILD_BATCH_SIZE:-32}" \
  --max_text_length "${AUX_MAX_TEXT_LENGTH:-128}" \
  --device "${DEVICE}"

run_if_missing "${AUX_TEST_DIR}/manifest.json" \
  "${PYTHON_BIN}" brain_text_pipeline/scripts/build_text_aux_dataset.py \
  --source_manifest "${TEST_MANIFEST}" \
  --out_dir "${AUX_TEST_DIR}" \
  --source_tokenizer_name_or_path "${SOURCE_TOKENIZER}" \
  --text_model_name_or_path "${TEXT_MODEL}" \
  --batch_size "${AUX_BUILD_BATCH_SIZE:-32}" \
  --max_text_length "${AUX_MAX_TEXT_LENGTH:-128}" \
  --device "${DEVICE}"

# Combine MEG + BERT.
run_if_missing "${COMBO_TRAIN_DIR}/manifest.json" \
  "${PYTHON_BIN}" brain_text_pipeline/scripts/combine_meg_text_aux_dataset.py \
  --meg_manifest "${TRAIN_MANIFEST}" \
  --text_aux_manifest "${AUX_TRAIN_DIR}/manifest.json" \
  --out_dir "${COMBO_TRAIN_DIR}"

run_if_missing "${COMBO_TEST_DIR}/manifest.json" \
  "${PYTHON_BIN}" brain_text_pipeline/scripts/combine_meg_text_aux_dataset.py \
  --meg_manifest "${TEST_MANIFEST}" \
  --text_aux_manifest "${AUX_TEST_DIR}/manifest.json" \
  --out_dir "${COMBO_TEST_DIR}"

# Train combined model.
run_if_missing "${RUN_DIR}/brain_encoder.pt" \
  "${PYTHON_BIN}" brain_text_pipeline/scripts/train_t5_brain_crossattn.py \
  --mode meg_supervised \
  --model_name_or_path "${MODEL_NAME}" \
  --meg_dataset_path "${COMBO_TRAIN_DIR}/manifest.json" \
  --output_dir "${RUN_DIR}" \
  --batch_size "${TRAIN_BATCH_SIZE:-32}" \
  --lr "${TRAIN_LR:-5e-5}" \
  --epochs "${TRAIN_EPOCHS:-6}" \
  --bf16 \
  --freeze_t5 \
  --unfreeze_cross_attn \
  --cross_attn_last_n "${CROSS_ATTN_LAST_N:-6}" \
  --unfreeze_last_n "${UNFREEZE_LAST_N:-1}" \
  --decoder_context_mode "${DECODER_CONTEXT_MODE:-target_only}" \
  --brain_norm "${BRAIN_NORM:-per_example}" \
  --max_text_len "${MAX_TEXT_LEN:-8}" \
  --max_brain_len "${MAX_BRAIN_LEN:-120}" \
  --log_interval "${LOG_INTERVAL:-100}" \
  --cpu_threads "${CPU_THREADS:-8}" \
  --num_workers "${NUM_WORKERS:-0}" \
  --device "${DEVICE}" \
  --seed "${SEED}"

# Evaluate while perturbing only the MEG feature slice.
run_if_missing "${EVAL_JSON}" \
  "${PYTHON_BIN}" brain_text_pipeline/scripts/eval_brain_controls.py \
  --model_name_or_path "${RUN_DIR}" \
  --brain_encoder_ckpt "${RUN_DIR}/brain_encoder.pt" \
  --meg_dataset_path "${COMBO_TEST_DIR}/manifest.json" \
  --samples "${EVAL_SAMPLES:-50000}" \
  --batch_size "${EVAL_BATCH_SIZE:-32}" \
  --device "${DEVICE}" \
  --decoder_context_mode "${DECODER_CONTEXT_MODE:-target_only}" \
  --brain_norm "${BRAIN_NORM:-per_example}" \
  --max_text_len "${MAX_TEXT_LEN:-8}" \
  --max_brain_len "${MAX_BRAIN_LEN:-120}" \
  --seed "${SEED}" \
  --control_feature_group meg_only \
  --out_json "${EVAL_JSON}"

log "done: ${EVAL_JSON}"
