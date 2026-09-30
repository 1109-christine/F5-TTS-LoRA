#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 3 ]]; then
  echo 'Usage: bash lora_lab/run.sh GPU_INDEX_OR_UUID full|r8|r16|r32 RUN_NAME' >&2
  exit 2
fi
gpu=$1
recipe=$2
run_name=$3
if [[ ! "$gpu" =~ ^([0-9]+|GPU-[a-fA-F0-9-]+)$ || ! "$run_name" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo 'Invalid GPU identifier or run name' >&2
  exit 2
fi
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
: "${BASE_CKPT:?Set BASE_CKPT to the official v1 Base safetensors file}"
: "${DATA_ROOT:?Set DATA_ROOT to the prepared LJSpeech directory}"
: "${SPLIT_DIR:?Set SPLIT_DIR to the prepared train/val manifest directory}"
VOCAB=${VOCAB:-"$PWD/src/f5_tts/infer/examples/vocab.txt"}
for file in "$BASE_CKPT" "$VOCAB" "$SPLIT_DIR/train.jsonl" "$SPLIT_DIR/val.jsonl" lora_lab/UPSTREAM_COMMIT; do
  [[ -f "$file" ]] || { echo "Missing file: $file" >&2; exit 1; }
done
case "$recipe" in
  full) mode=full; rank=8; alpha=16; lr=1e-5 ;;
  r8) mode=lora; rank=8; alpha=16; lr=1e-4 ;;
  r16) mode=lora; rank=16; alpha=32; lr=1e-4 ;;
  r32) mode=lora; rank=32; alpha=64; lr=1e-4 ;;
  *) echo 'Unknown recipe' >&2; exit 2 ;;
esac
extra=()
if [[ "${RESUME:-0}" == 1 ]]; then extra+=(--resume); fi
if [[ -n "${WANDB_PROJECT:-}" ]]; then extra+=(--wandb-project "$WANDB_PROJECT"); fi
export CUDA_VISIBLE_DEVICES="$gpu"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
mkdir -p logs
python -m lora_lab.train \
  --base "$BASE_CKPT" --vocab "$VOCAB" --data-root "$DATA_ROOT" \
  --train "$SPLIT_DIR/train.jsonl" --val "$SPLIT_DIR/val.jsonl" \
  --out "runs/$run_name" --mode "$mode" --rank "$rank" --alpha "$alpha" \
  --lr "${LR:-$lr}" --updates "${UPDATES:-500}" --warmup "${WARMUP:-50}" \
  --accum "${ACCUM:-8}" --save-every "${SAVE_EVERY:-50}" \
  --seed "${TRAIN_SEED:-666}" --precision "${PRECISION:-bf16}" \
  --checkpoint-activations "${extra[@]}" 2>&1 | tee -a "logs/$run_name.log"
