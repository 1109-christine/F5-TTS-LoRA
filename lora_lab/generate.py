import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

from .common import build_base, load_deployment, read_jsonl, seed_all, sha256


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', required=True)
    p.add_argument('--vocab', required=True)
    p.add_argument('--vocos', required=True)
    p.add_argument('--data-root', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--checkpoint', default=None)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Expose exactly one GPU')
    from f5_tts.infer.utils_infer import infer_batch_process, load_vocoder
    out, root = Path(a.out), Path(a.data_root)
    out.mkdir(parents=True, exist_ok=False)
    model, _ = build_base(a.base, a.vocab)
    if a.checkpoint:
        payload = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
        load_deployment(model, payload, sha256(a.base), sha256(a.vocab))
        del payload
    # Fixed FP32 inference for all arms; slower but avoids cross-method dtype differences.
    model = model.cuda().eval()
    vocoder = load_vocoder(vocoder_name='vocos', is_local=True,
                           local_path=a.vocos, device='cuda')
    config = dict(vars(a), nfe_step=32, cfg_strength=2.0, sway_sampling_coef=-1,
                  seed_policy='seed + manifest row index', solver='euler',
                  inference_dtype='float32', duration_policy='prompt-text-ratio',
                  manifest_sha256=sha256(a.manifest), base_sha256=sha256(a.base),
                  vocab_sha256=sha256(a.vocab),
                  checkpoint_sha256=sha256(a.checkpoint) if a.checkpoint else None)
    (out / 'generation_config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    for i, row in enumerate(read_jsonl(a.manifest)):
        seed_all(a.seed + i)
        prompt, sr = torchaudio.load(str(root / row['prompt_audio']))
        prompt = prompt.mean(0, keepdim=True)
        if not row['prompt_text'].strip() or prompt.square().mean() < 1e-10:
            raise ValueError('Empty reference prompt')
        # One text per call: no simultaneous chunk generation sharing a global RNG.
        torch.cuda.synchronize()
        tick = time.perf_counter()
        wav, sr_out, _ = next(infer_batch_process(
            (prompt, sr), row['prompt_text'], [row['text']], model, vocoder,
            progress=None, nfe_step=32, cfg_strength=2.0, sway_sampling_coef=-1,
            speed=1.0, fix_duration=None, device='cuda'))
        torch.cuda.synchronize()
        seconds = time.perf_counter() - tick
        if wav is None or not len(wav) or not np.isfinite(wav).all():
            raise ValueError(f"Invalid generation: {row['utt']}; retain the failure in the report")
        sf.write(out / f"{row['utt']}.wav", wav, sr_out, subtype='FLOAT')
        record = dict(utt=row['utt'], generation_seconds=seconds,
                      audio_seconds=len(wav) / sr_out, rtf=seconds / (len(wav) / sr_out))
        with (out / 'generation_metrics.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(record) + '\n')
        print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
