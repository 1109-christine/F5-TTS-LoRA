# Publication update — 2026-09-30

## Preserved

`common.py`, `train.py`, `generate.py`, `export.py`, `verify.py`, `test_lora.py`,
`prepare_arrow.py`, `__init__.py`, `UPSTREAM_COMMIT` are byte-identical to the supplied review archive.
Upstream source, LICENSE, pyproject.toml and .gitmodules are not modified by the overlay.
The old root README is preserved as README.upstream.md.

## Added or updated

- README pages now describe the completed LJSpeech experiment and actual metrics.
- `run.sh` uses dataset-neutral DATA_ROOT and defaults to the actual save interval of 50.
- New inference wrapper, local-path environment example and cuDNN search-path helper.
- New evaluation CLI packages the successful WER/SIM/UTMOS procedure.
- New snapshot tool exports frozen IDs, per-utterance scores and selected original provenance.
- New data preparation CLI restores exported IDs; historical resampler identity remains unverified.
- Ignore rules additionally cover private local configuration, model weights and crash dumps.

## Validation performed for this update

- Python syntax compilation for all modules.
- Shell syntax for all four shell helpers.
- CLI help entry points for prepare_ljspeech, evaluate and publish_snapshot.
- Three dependency-free tests: snapshot roundtrip/path scrubbing/source preservation,
  mismatch refusal, and reference/text leakage rejection.
- The package installer was checked on a temporary copy for successful application,
  backups/idempotence, and refusal to overwrite unexpected local changes.

The local review environment has no PyTorch, audio libraries, GPU weights or real
LJSpeech audio. The existing seven torch tests and real GPU/ASR/SIM/UTMOS computations
were NOT rerun here. Historical GPU verification/training/evaluation success comes
from the supplied remote-server logs. New audio preprocessing requires validation
on the actual server before claiming full end-to-end reproduction.

## Required before publishing the completed experiment

Run publish_snapshot on the original server and commit its output. Actual split IDs,
per-utterance evaluation records and private run configs were not in the supplied archive.
The release contains supplied summary numbers, not fabricated detailed records.

Adding Python helpers changes the training implementation hash. Existing deploy/best
weights remain loadable, but the strict old-run resume guard will reject the changed
source directory. Keep the installer's backup if recovering an unfinished old run.
