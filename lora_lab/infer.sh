#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 2 ]]; then
  echo 'Usage: bash lora_lab/infer.sh GPU_INDEX_OR_UUID base|full|r8|r16|r32' >&2
  exit 2
fi
gpu=$1
method=$2
[[ "$gpu" =~ ^([0-9]+|GPU-[a-fA-F0-9-]+)$ ]] || { echo 'Invalid GPU identifier' >&2; exit 2; }
case "$method" in base|full|r8|r16|r32) ;; *) echo 'Unknown method' >&2; exit 2 ;; esac
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
: "${BASE_CKPT:?Set BASE_CKPT}"
: "${VOCOS_DIR:?Set VOCOS_DIR}"
: "${DATA_ROOT:?Set DATA_ROOT}"
: "${SPLIT_DIR:?Set SPLIT_DIR}"
VOCAB=${VOCAB:-"$PWD/src/f5_tts/infer/examples/vocab.txt"}
TRAIN_SEED=${TRAIN_SEED:-666}
INFER_SEED=${INFER_SEED:-666}
extra=()
if [[ "$method" != base ]]; then
  checkpoint=${CHECKPOINT:-"runs/lj2h_${method}_500_seed${TRAIN_SEED}/best.pt"}
  [[ -f "$checkpoint" ]] || { echo "Missing checkpoint: $checkpoint" >&2; exit 1; }
  extra+=(--checkpoint "$checkpoint")
fi
manifest=${MANIFEST:-"$SPLIT_DIR/test.jsonl"}
output=${INFER_OUT:-"runs/infer_lj2h_test100_${method}_seed${INFER_SEED}"}
[[ ! -e "$output" ]] || { echo "Output exists: $output" >&2; exit 1; }
for file in "$BASE_CKPT" "$VOCAB" "$manifest"; do
  [[ -f "$file" ]] || { echo "Missing file: $file" >&2; exit 1; }
done
export CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
python -u -m lora_lab.generate \
  --base "$BASE_CKPT" --vocab "$VOCAB" --vocos "$VOCOS_DIR" \
  --data-root "$DATA_ROOT" --manifest "$manifest" --out "$output" \
  --seed "$INFER_SEED" "${extra[@]}"
