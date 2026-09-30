# LJSpeech 2h / seed 666: reported experiment

These are completed remote-server results supplied by the experiment owner,
not new measurements produced while preparing this release.

| Method | Corpus WER (%) ↓ | SIM ↑ | UTMOS ↑ |
|---|---:|---:|---:|
| Base | 4.6837 | 0.6470 | 4.2590 |
| Full | 4.7445 | **0.7597** | 4.3046 |
| LoRA r8 | 4.5620 | 0.7266 | 4.3318 |
| LoRA r16 | 4.5620 | 0.7289 | **4.3589** |
| LoRA r32 | **4.5012** | 0.7398 | 4.3475 |

- Train: 1,066 utterances / 119.98788 minutes. Validation: 100 / 11.56596 minutes.
- Test: 100 / 10.61974 minutes. Reference: 2 / 0.24714 minutes.
- Utterance-level split with normalized-text deduplication; not chapter-disjoint.
- Audio: mono 24 kHz, 2–12 seconds. Same test order and reserved references for all methods.
- 500 optimizer updates, microbatch 1, accumulation 8 (approximately 3.75 training passes).
- Full LR 1e-5; LoRA LR 1e-4; ranks 8/16/32, alpha 16/32/64, dropout 0.
- Warmup 50, save/validate every 50 updates, bf16, activation checkpointing, no EMA.
- Validation-selected best checkpoints were all at update 500 in this run.
- Training seed 666; inference seed 666 with per-item seed = 666 + manifest row index.
- Inference: FP32, Euler, NFE 32, CFG 2.0, sway -1, speed 1.0, local Vocos.
- LoRA inference uses unmerged adapters. Exported merged models were not used for this table.
- WER: local faster-whisper large-v3, beam 5, English, upstream punctuation removal/lowercasing.
- SIM: upstream WavLM speaker-verification pipeline, 16 kHz resampling, separate held-out SIM reference.
- UTMOS: local SpeechMOS v1.2.0, utmos22_strong, mean over 100 utterances; not human MOS.

All methods have 1,644 reference words. Total word errors are Base 77, Full 78,
r8 75, r16 75, r32 74. These small differences do not establish a significant WER advantage.
The measurements suggest improved speaker similarity and predicted quality within
this experiment. They do not establish multi-seed robustness, a universally best rank,
pretraining-data disjointness, or generalization to other speakers/languages.
Different learning rates mean this is a comparison of specified recipes, not an isolated
causal estimate of the LoRA mechanism. No hyperparameter sweep was performed for this table.

## Source and reproducibility status

`reported_results.json` preserves the supplied numeric summaries.
Before publishing, run `python -m lora_lab.publish_snapshot ...` on the original
server to create `lj2h_seed666/`: exact split IDs/order, text hashes, original
configuration hashes and all per-utterance scores. These files were absent from
the review archive and are intentionally not fabricated here.

The new preparation script restores those IDs from LJSpeech-1.1 and documents its
resampling recipe. The historical ad-hoc resampling script was not supplied, so
byte-identical historical WAV reconstruction is not yet established. Preserve the
existing private prepared audio for exact reruns. No audio or checkpoint is distributed.

GPU/server logs confirmed real-model verification, 500-step training and evaluation.
New publication/preparation wrappers are separate from that historical execution;
local validation details are in RELEASE_NOTES.md. Timing and memory from ten-step smoke
tests are not promoted to full-run benchmark measurements.
