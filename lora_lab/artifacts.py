"""Small dependency-free helpers shared by preparation and publication tools."""
import hashlib
import json
import math
import re
from pathlib import Path

METHODS = ('base', 'full', 'r8', 'r16', 'r32')
SPLITS = ('train', 'val', 'test', 'reference')


def read_jsonl(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def text_digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def text_key(text):
    return ' '.join(re.findall(r'[a-z0-9]+', text.lower()))


def check_id(utt):
    if not isinstance(utt, str) or not re.fullmatch(r'LJ\d{3}-\d{4}', utt):
        raise ValueError(f'Unexpected LJSpeech identifier: {utt!r}')
    return utt


def check_splits(splits):
    """Reject duplicate utterances/texts and references leaking into other splits."""
    seen_ids, seen_texts = set(), set()
    for name in SPLITS:
        if not splits[name]:
            raise ValueError(f'Empty split: {name}')
        for row in splits[name]:
            utt = check_id(row['utt'])
            key = text_key(row['text'])
            if not key or utt in seen_ids or key in seen_texts:
                raise ValueError(f'Duplicate ID/normalized text or empty text: {utt}')
            if not math.isfinite(row['duration']) or not 2 <= row['duration'] <= 12:
                raise ValueError(f'Invalid duration: {utt}')
            seen_ids.add(utt)
            seen_texts.add(key)
    refs = {r['audio']: r for r in splits['reference']}
    for name in ('val', 'test'):
        for row in splits[name]:
            prompt = row['prompt_audio']
            sim = row['sim_ref_audio']
            if prompt == sim or prompt not in refs or sim not in refs:
                raise ValueError(f'Prompt/SIM references must be distinct reserved clips: {row["utt"]}')
            if row['prompt_text'] != refs[prompt]['text']:
                raise ValueError(f'Prompt text mismatch: {row["utt"]}')
