"""Build a small speaker-adaptation split from an existing LibriTTS Arrow file.

Reuses normalized text from prior preprocessing; no .normalized.txt assumption.
Whole chapters are reserved for reference, validation, and test respectively.
"""
import argparse
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path


def choose_split(rows, minutes=10, eval_n=20, seed=666):
    seen, groups = set(), defaultdict(list)
    for row in sorted(rows, key=lambda x: x['utt']):
        key = ' '.join(re.findall(r'[a-z0-9]+', row['text'].lower()))
        if not key or key in seen:
            continue
        seen.add(key)
        groups[row['chapter']].append(dict(row))
    chapters = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(chapters)
    if len(chapters) < 4:
        return None
    refs = [r for r in groups[chapters[0]] if 3 <= r['duration'] <= 10]
    val, test = groups[chapters[1]], groups[chapters[2]]
    for pool in (refs, val, test):
        rng.shuffle(pool)
    if len(refs) < 2 or min(len(val), len(test)) < eval_n:
        return None
    pool = [r for c in chapters[3:] for r in groups[c]]
    rng.shuffle(pool)
    train, seconds = [], 0.0
    for row in pool:
        if seconds + row['duration'] <= minutes * 60:
            train.append(row)
            seconds += row['duration']
    if seconds < 0.95 * minutes * 60:
        return None
    val, test = val[:eval_n], test[:eval_n]
    for row in val + test:
        row.update(prompt_audio=refs[0]['audio'], prompt_text=refs[0]['text'],
                   sim_ref_audio=refs[1]['audio'])
    return dict(train=train, val=val, test=test, reference=refs[:2])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arrow', required=True)
    p.add_argument('--root', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--speaker', default=None)
    p.add_argument('--minutes', type=float, default=10)
    p.add_argument('--eval-n', type=int, default=20)
    p.add_argument('--seed', type=int, default=666)
    a = p.parse_args()
    if not math.isfinite(a.minutes) or a.minutes <= 0 or a.eval_n < 1:
        raise ValueError('minutes and eval-n must be positive')
    from datasets import Dataset
    from .common import sha256
    root, out = Path(a.root).resolve(), Path(a.out)
    if out.exists():
        raise FileExistsError('Do not overwrite an existing split; use a new --out')
    data = Dataset.from_file(a.arrow)
    if not {'audio_path', 'text', 'duration'} <= set(data.column_names):
        raise ValueError('Expected audio_path, text, duration columns')
    speakers = defaultdict(list)
    for row in data:
        path = Path(row['audio_path'])
        parts = path.parts
        if 'train-clean-100' not in parts or not 2 <= row['duration'] <= 12:
            continue
        start = parts.index('train-clean-100')
        tail = parts[start:]
        if len(tail) != 4:
            raise ValueError(f'Unexpected LibriTTS path: {path}')
        _, speaker, chapter, _ = tail
        if a.speaker and speaker != a.speaker:
            continue
        text = row['text']
        if isinstance(text, list):
            text = ''.join(text)
        if not isinstance(text, str):
            raise ValueError(f'Unsupported text format: {path}')
        speakers[speaker].append(dict(utt=path.stem, audio=str(Path(*tail)),
                                     text=text, duration=float(row['duration']),
                                     speaker=speaker, chapter=chapter))
    result = None
    for speaker in sorted(speakers):
        result = choose_split(speakers[speaker], a.minutes, a.eval_n, a.seed)
        if result:
            break
    if result is None:
        raise RuntimeError('No eligible speaker with >=4 chapters and sufficient audio. '
                           'No files were written; inspect data or adjust the split recipe.')
    for rows in result.values():
        for row in rows:
            if not (root / row['audio']).is_file():
                raise FileNotFoundError(root / row['audio'])
    report = dict(speaker=speaker, seed=a.seed, requested_train_minutes=a.minutes,
                  source_arrow_sha256=sha256(a.arrow), split='train-clean-100',
                  selection='lexicographically first eligible speaker unless explicitly specified')
    out.mkdir(parents=True)
    for name, rows in result.items():
        with (out / f'{name}.jsonl').open('w', encoding='utf-8') as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
        report[name] = dict(samples=len(rows), minutes=sum(r['duration'] for r in rows)/60,
                            chapters=sorted({r['chapter'] for r in rows}))
    (out / 'split_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
