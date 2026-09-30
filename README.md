# F5-TTS LoRA: single-speaker adaptation on LJSpeech

A small, reproducible experiment layer on top of [SWivid/F5-TTS](https://github.com/SWivid/F5-TTS).
Compare the official F5-TTS v1 Base with full fine-tuning and attention LoRA at ranks 8, 16 and 32.
The original F5-TTS implementation, attribution and license are retained.

**[中文实验指南](lora_lab/README.zh.md) · [Results and limitations](lora_lab/benchmarks/RESULTS.md) · [Upstream README](README.upstream.md)**

## Completed experiment

2 hours of LJSpeech training audio, 500 updates, one training seed (666), and the same
100 held-out test utterances for all five methods.

| Method | WER (%) ↓ | Speaker similarity ↑ | UTMOS ↑ |
|---|---:|---:|---:|
| Official Base | 4.6837 | 0.6470 | 4.2590 |
| Full fine-tuning | 4.7445 | **0.7597** | 4.3046 |
| LoRA r8 | 4.5620 | 0.7266 | 4.3318 |
| LoRA r16 | 4.5620 | 0.7289 | **4.3589** |
| LoRA r32 | **4.5012** | 0.7398 | 4.3475 |

These are single-seed observations, not claims of statistical significance or a universally best rank.
UTMOS is a learned predictor, not human MOS. The complete protocol and provenance limitations
are documented in the results page. Reported results came from the original server run;
new release utilities have not rerun the GPU experiment.

## What's added

- `lora_lab/common.py`: LoRA on Q/K/V/output projections across 22 attention blocks;
  frozen base, zero-initialized B, strict adapter loading and merging.
- `train.py`, `run.sh`: independent single-GPU full/LoRA training, validation selection,
  checkpointing, exact-configuration resume, optional W&B.
- `generate.py`, `infer.sh`: matched five-arm inference using local Vocos.
- `evaluate.py`: WER, SIM and UTMOS using the same evaluation operations as the completed run.
- `publish_snapshot.py`: export actual split IDs and scores without local paths or model weights.
- `prepare_ljspeech.py`: restore frozen utterance IDs/order from public LJSpeech metadata.
- `verify.py`, `test_lora.py`: real-model GPU verification and core CPU tests.
- `export.py`: merge LoRA for deployment; no merged-model latency benchmark is claimed.

## Setup and execution

Use Python 3.11 and a compatible PyTorch/torchaudio CUDA installation. Install the
upstream project according to [its installation instructions](README.upstream.md),
then run commands from the repository root. `lora_lab` is a repository-local module;
it is not added to the upstream wheel's package layout.

The observed evaluation environment included torch 2.4.1, faster-whisper 0.10.1,
CTranslate2 4.5.0, cuDNN 9.1.0.70 and cuBLAS 12.9.2.10. This is a partial version record,
not a complete environment lockfile or an instruction to blindly upgrade an existing environment.
Upstream `pyproject.toml` includes the main and optional evaluation dependencies.
SpeechMOS and WavLM additionally require their model/code dependencies and local caches.

```bash
cp lora_lab/env.example.sh lora_lab/env.local.sh
# Edit env.local.sh to set your own checkpoint, vocoder, evaluator and data paths.
source lora_lab/env.local.sh
python -m unittest lora_lab.test_lora lora_lab.test_publication -v
CUDA_VISIBLE_DEVICES=0 python -m lora_lab.verify --base "$BASE_CKPT" --vocab "$VOCAB"
```

Obtain the official v1 Base checkpoint and its matching vocabulary; do not substitute
another dataset's character vocabulary. Obtain LJSpeech-1.1 separately. No audio,
base model, trained adapter, or third-party evaluator weights are included here.

Before public release, the experiment owner must export the original split and scores
with `publish_snapshot.py` (see Chinese guide). The review archive did not contain
those files. Once exported, a new checkout can restore the frozen split:

```bash
python -m lora_lab.prepare_ljspeech \
  --source-root /path/to/LJSpeech-1.1 \
  --split-ids lora_lab/benchmarks/lj2h_seed666/split_ids.json \
  --out "$DATA_ROOT"
```

Existing output directories are rejected. Existing prepared experimental data should be
kept as-is; recreating the WAVs with the newly documented resampler does not establish
byte-identical reconstruction of historical preprocessing.

```bash
# One command per allocated GPU/terminal; use GPU IDs assigned to you.
bash lora_lab/run.sh 0 full lj2h_full_500_seed666
bash lora_lab/run.sh 1 r8   lj2h_r8_500_seed666
bash lora_lab/run.sh 2 r16  lj2h_r16_500_seed666
bash lora_lab/run.sh 3 r32  lj2h_r32_500_seed666

# Then generate all five groups (one command per terminal for concurrency).
bash lora_lab/infer.sh 0 base
bash lora_lab/infer.sh 1 full
bash lora_lab/infer.sh 2 r8
bash lora_lab/infer.sh 3 r16
bash lora_lab/infer.sh 4 r32

# Evaluate on one allocated GPU, with cuDNN search paths set before Python starts.
source lora_lab/cuda_eval_env.sh
CUDA_VISIBLE_DEVICES=1 python -m lora_lab.evaluate \
  --data-root "$DATA_ROOT" --manifest "$SPLIT_DIR/test.jsonl" \
  --asr "$ASR_DIR" --sim "$SIM_CKPT" --utmos-hub "$UTMOS_HUB"
```

Each training/inference command above runs in the foreground. Run separate terminal/tmux
windows to execute concurrently. Training outputs refuse accidental overwrites.
Evaluation requires `--reuse-existing` to reuse already-completed results from identical inputs.

## Attribution and publication

This repository extends F5-TTS; the underlying architecture, upstream training/inference
implementation and pretrained base are credited to their original authors.
Pinned upstream SHA: `283252563dbf91be625e0c27926acfaac449186c`.
See [LICENSE](LICENSE), [upstream README](README.upstream.md) and upstream references.
Third-party datasets/models retain their own terms; none are redistributed in this code release.

This update bundle is an **overlay for an existing F5-TTS checkout**, not a standalone
replacement for `src/`. Keep the upstream source and license in the published repository.
