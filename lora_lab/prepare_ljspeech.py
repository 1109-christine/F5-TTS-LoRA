"""Rebuild a frozen LJSpeech split from published IDs; never redraw a reported test set.

The IDs must be exported from the actual experiment using publish_snapshot.py.
Waveforms are regenerated with librosa's soxr_hq resampler; this is not a claim
that the original, unavailable ad-hoc preprocessing script was identical.
"""
import argparse
import json
from pathlib import Path

from .artifacts import SPLITS, check_id, check_splits, digest, text_digest, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', required=True, help='LJSpeech-1.1 containing metadata.csv and wavs/')
    p.add_argument('--split-ids', required=True, help='Published split_ids.json from the actual experiment')
    p.add_argument('--out', required=True, help='New output directory; existing directories are rejected')
    a = p.parse_args()
    source, out = Path(a.source_root), Path(a.out)
    if out.exists():
        raise FileExistsError(out)
    frozen = json.loads(Path(a.split_ids).read_text(encoding='utf-8'))
    if frozen.get('format_version') != 1:
        raise ValueError('Unsupported split IDs format')
    texts = {}
    for line in (source / 'metadata.csv').read_text(encoding='utf-8-sig').splitlines():
        if not line.strip():
            continue
        fields = line.split('|')  # Quotes in LJSpeech text are literal characters.
        if len(fields) != 3:
            raise ValueError('Expected id|original text|normalized text')
        utt, _, text = fields
        utt, text = check_id(utt.strip()), text.strip()
        if utt in texts or not text:
            raise ValueError(f'Duplicate ID or empty text: {utt}')
        texts[utt] = text
    for name in SPLITS:
        for row in frozen['splits'][name]:
            utt = check_id(row['utt'])
            if utt not in texts or text_digest(texts[utt]) != row['text_sha256']:
                raise ValueError(f'Text differs from the published experiment: {utt}')
            if not (source / 'wavs' / f'{utt}.wav').is_file():
                raise FileNotFoundError(utt)

    import librosa
    import numpy as np
    import soundfile as sf

    # Validate the complete reconstructed set before writing output waveforms.
    splits, audio = {}, {}
    for name in SPLITS:
        splits[name] = []
        for item in frozen['splits'][name]:
            utt = item['utt']
            wav, sr = sf.read(source / 'wavs' / f'{utt}.wav', dtype='float32', always_2d=True)
            wav = wav.mean(axis=1)
            if sr != 24000:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=24000, res_type='soxr_hq')
            duration = len(wav) / 24000
            if not np.isfinite(wav).all() or not len(wav) or np.mean(wav ** 2) < 1e-10:
                raise ValueError(f'Invalid/silent audio: {utt}')
            if abs(duration - item['duration']) > 0.002:
                raise ValueError(f'Duration differs from frozen split: {utt}')
            row = dict(utt=utt, audio=f'wavs/{utt}.wav', text=texts[utt], duration=duration, speaker='LJ')
            if name in ('val', 'test'):
                prompt, sim = check_id(item['prompt_utt']), check_id(item['sim_ref_utt'])
                row.update(prompt_audio=f'wavs/{prompt}.wav', prompt_text=texts[prompt],
                           sim_ref_audio=f'wavs/{sim}.wav')
            splits[name].append(row)
            audio[utt] = wav
    check_splits(splits)
    (out / 'wavs').mkdir(parents=True)
    for utt, wav in audio.items():
        sf.write(out / 'wavs' / f'{utt}.wav', wav, 24000, subtype='FLOAT')
    for name, rows in splits.items():
        with (out / f'{name}.jsonl').open('w', encoding='utf-8') as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
    report = dict(dataset='LJSpeech-1.1', sample_rate=24000,
                  split_method='restored frozen utterance IDs and order; no resampling of the split',
                  chapter_disjoint=False, split_ids_sha256=digest(a.split_ids),
                  metadata_sha256=digest(source / 'metadata.csv'),
                  waveform_recipe='mono mean; librosa soxr_hq to 24000 Hz; FLOAT WAV',
                  historical_waveform_identity_verified=False,
                  librosa_version=librosa.__version__, soundfile_version=sf.__version__)
    for name, rows in splits.items():
        report[name] = dict(samples=len(rows), minutes=sum(r['duration'] for r in rows) / 60)
    write_json(out / 'split_report.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
