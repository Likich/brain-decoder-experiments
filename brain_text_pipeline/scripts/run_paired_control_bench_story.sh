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
RUN_DIR="${RUN_DIR:-brain_text_pipeline/runs/t5_meg_postword_story_hybrid_last1}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-brain_text_pipeline/data/meg_aligned_postword_story_train/manifest.json}"
TEST_MANIFEST="${TEST_MANIFEST:-brain_text_pipeline/data/meg_aligned_postword_story_test/manifest.json}"
MODEL_PATH="${MODEL_PATH:-${RUN_DIR}}"
BRAIN_ENCODER_CKPT="${BRAIN_ENCODER_CKPT:-${RUN_DIR}/brain_encoder.pt}"
SAMPLES="${SAMPLES:-50000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
SEED="${SEED:-42}"
DECODER_CONTEXT_MODE="${DECODER_CONTEXT_MODE:-target_only}"
BRAIN_NORM="${BRAIN_NORM:-per_example}"
MAX_TEXT_LEN="${MAX_TEXT_LEN:-8}"
MAX_BRAIN_LEN="${MAX_BRAIN_LEN:-120}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-10000}"
RUN_SENSOR_ABLATION="${RUN_SENSOR_ABLATION:-0}"
MEG_ROOT="${MEG_ROOT:-}"

MAIN_JSON="${MAIN_JSON:-${RUN_DIR}/eval_story_test_50k.json}"
EXAMPLES_JSONL="${EXAMPLES_JSONL:-${RUN_DIR}/eval_story_test_50k_examples.jsonl}"
CLUSTERED_JSON="${CLUSTERED_JSON:-${RUN_DIR}/eval_story_test_50k_clustered.json}"
CHAR_JSON="${CHAR_JSON:-${RUN_DIR}/eval_story_test_50k_characterization.json}"
SENSOR_JSON="${SENSOR_JSON:-${RUN_DIR}/eval_story_test_50k_sensor_ablation.json}"

run_eval() {
  local out_json="$1"
  shift
  run_if_missing "${out_json}" \
    "${PYTHON_BIN}" brain_text_pipeline/scripts/eval_brain_controls.py \
    --model_name_or_path "${MODEL_PATH}" \
    --brain_encoder_ckpt "${BRAIN_ENCODER_CKPT}" \
    --meg_dataset_path "${TEST_MANIFEST}" \
    --samples "${SAMPLES}" \
    --batch_size "${BATCH_SIZE}" \
    --device "${DEVICE}" \
    --decoder_context_mode "${DECODER_CONTEXT_MODE}" \
    --brain_norm "${BRAIN_NORM}" \
    --max_text_len "${MAX_TEXT_LEN}" \
    --max_brain_len "${MAX_BRAIN_LEN}" \
    --seed "${SEED}" \
    "$@" \
    --out_json "${out_json}"
}

run_if_missing "${MAIN_JSON}" \
  "${PYTHON_BIN}" brain_text_pipeline/scripts/eval_brain_controls.py \
  --model_name_or_path "${MODEL_PATH}" \
  --brain_encoder_ckpt "${BRAIN_ENCODER_CKPT}" \
  --meg_dataset_path "${TEST_MANIFEST}" \
  --samples "${SAMPLES}" \
  --batch_size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  --decoder_context_mode "${DECODER_CONTEXT_MODE}" \
  --brain_norm "${BRAIN_NORM}" \
  --max_text_len "${MAX_TEXT_LEN}" \
  --max_brain_len "${MAX_BRAIN_LEN}" \
  --seed "${SEED}" \
  --out_json "${MAIN_JSON}" \
  --out_examples_jsonl "${EXAMPLES_JSONL}"

run_if_missing "${CLUSTERED_JSON}" \
  "${PYTHON_BIN}" brain_text_pipeline/scripts/clustered_paired_controls.py \
  --examples_jsonl "${EXAMPLES_JSONL}" \
  --cluster_spec subject \
  --cluster_spec sound \
  --cluster_spec sequence_id \
  --bootstrap_samples "${BOOTSTRAP_SAMPLES}" \
  --seed "${SEED}" \
  --out_json "${CLUSTERED_JSON}"

run_if_missing "${CHAR_JSON}" \
  "${PYTHON_BIN}" brain_text_pipeline/scripts/analyze_meg_effect_characterization.py \
  --examples_jsonl "${EXAMPLES_JSONL}" \
  --train_manifest "${TRAIN_MANIFEST}" \
  --out_json "${CHAR_JSON}"

run_eval "${RUN_DIR}/eval_story_test_50k_globalshuf.json" \
  --shuf_mode global_sample

run_eval "${RUN_DIR}/eval_story_test_50k_shuf_subject.json" \
  --shuf_mode within_group \
  --shuf_group_keys subject

run_eval "${RUN_DIR}/eval_story_test_50k_shuf_sound.json" \
  --shuf_mode within_group \
  --shuf_group_keys sound

run_eval "${RUN_DIR}/eval_story_test_50k_shuf_circshift.json" \
  --shuf_mode circular_time_shift

run_eval "${RUN_DIR}/eval_story_test_50k_shuf_block10.json" \
  --shuf_mode block_permute \
  --shuf_block_size 10

run_eval "${RUN_DIR}/eval_story_test_50k_shuf_phase.json" \
  --shuf_mode phase_randomized

if [[ "${RUN_SENSOR_ABLATION}" == "1" ]]; then
  if [[ -z "${MEG_ROOT}" ]]; then
    log "skip: sensor ablation requested but MEG_ROOT is empty"
  else
    run_if_missing "${SENSOR_JSON}" \
      "${PYTHON_BIN}" brain_text_pipeline/scripts/eval_meg_sensor_ablation.py \
      --model_name_or_path "${MODEL_PATH}" \
      --brain_encoder_ckpt "${BRAIN_ENCODER_CKPT}" \
      --meg_dataset_path "${TEST_MANIFEST}" \
      --meg_root "${MEG_ROOT}" \
      --samples "${SAMPLES}" \
      --batch_size "${BATCH_SIZE}" \
      --device "${DEVICE}" \
      --decoder_context_mode "${DECODER_CONTEXT_MODE}" \
      --brain_norm "${BRAIN_NORM}" \
      --max_text_len "${MAX_TEXT_LEN}" \
      --max_brain_len "${MAX_BRAIN_LEN}" \
      --groups left_temporal,right_temporal,frontal,occipital,left,right \
      --random_match_group left_temporal \
      --random_match_repeats 3 \
      --seed "${SEED}" \
      --out_json "${SENSOR_JSON}"
  fi
fi

log "paired-control benchmark outputs are ready under ${RUN_DIR}"
