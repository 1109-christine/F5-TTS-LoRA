# Copy to lora_lab/env.local.sh (gitignored), edit paths, then source it.
# Run from the repository root. This file contains no personal credentials.
export BASE_CKPT=/path/to/F5TTS_v1_Base/model_1250000.safetensors
export VOCAB="$PWD/src/f5_tts/infer/examples/vocab.txt"
export DATA_ROOT="$PWD/data/ljspeech_2h_seed666"
export SPLIT_DIR="$DATA_ROOT"
export VOCOS_DIR=/path/to/vocos-mel-24khz
export ASR_DIR=/path/to/faster-whisper-large-v3
export SIM_CKPT=/path/to/wavlm/wavlm_large_finetune.pth
export UTMOS_HUB="$HOME/.cache/torch/hub/tarepan_SpeechMOS_v1.2.0"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
# Optional; leaving unset keeps training independent of W&B network service.
# export WANDB_PROJECT=F5TTS-LoRA-LJSpeech
# export WANDB_MODE=offline
