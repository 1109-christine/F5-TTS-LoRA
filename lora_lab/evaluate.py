"""Evaluate generated LJSpeech WAVs with the project's WER/SIM and local UTMOS.

Existing complete JSONL results are reused only with --reuse-existing. This
checks IDs and values, not provenance: use that flag only for the same WAVs,
manifest, reference policy and evaluator checkpoints.
"""
import argparse
import json
import math
import os
import string
from pathlib import Path

from .artifacts import METHODS, check_id, read_jsonl, write_json


def validate(records, ids, metric, path):
    key = 'utt' if metric == 'utmos' else 'wav'
    if [r[key] for r in records] != ids or not all(math.isfinite(r[metric]) for r in records):
        raise ValueError(f'Incomplete, reordered or non-finite results: {path}')


def summarize(records, metric):
    if metric != 'wer':
        return dict(count=len(records), **{f'{metric}_mean': sum(r[metric] for r in records) / len(records)})
    from jiwer import process_words
    from zhon.hanzi import punctuation

    def normalize(text):
        for ch in punctuation + string.punctuation:
            text = text.replace(ch, '')
        return text.replace('  ', ' ').lower()

    measures = process_words([normalize(r['truth']) for r in records],
                             [normalize(r['hypo']) for r in records])
    return dict(count=len(records), corpus_wer_percent=100 * measures.wer,
                mean_utterance_wer_percent=100 * sum(r['wer'] for r in records) / len(records),
                substitutions=measures.substitutions, deletions=measures.deletions,
                insertions=measures.insertions,
                reference_words=measures.hits + measures.substitutions + measures.deletions)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--infer-root', default='runs')
    p.add_argument('--folder-template', default='infer_lj2h_test100_{method}_seed666')
    p.add_argument('--out', default='runs/eval_lj2h_test100_seed666')
    p.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    p.add_argument('--metrics', nargs='+', choices=['wer', 'sim', 'utmos'], default=['wer', 'sim', 'utmos'])
    p.add_argument('--asr', help='Local faster-whisper-large-v3 directory')
    p.add_argument('--sim', help='Local wavlm_large_finetune.pth')
    p.add_argument('--utmos-hub', help='Trusted local SpeechMOS v1.2.0 checkout/cache containing hubconf.py')
    p.add_argument('--reuse-existing', action='store_true')
    a = p.parse_args()
    if len(set(a.methods)) != len(a.methods) or len(set(a.metrics)) != len(a.metrics):
        raise ValueError('Duplicate methods/metrics')
    root, out = Path(a.data_root), Path(a.out)
    rows = read_jsonl(a.manifest)
    ids = [check_id(r['utt']) for r in rows]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('Empty manifest or duplicate utterances')
    n = len(rows)
    items, folders = {}, {}
    for method in a.methods:
        if '{method}' not in a.folder_template:
            raise ValueError('folder-template must contain {method}')
        folder = Path(a.infer_root) / a.folder_template.format(method=method)
        folders[method] = folder
        items[method] = []
        for row in rows:
            wav, ref = folder / f'{row["utt"]}.wav', root / row['sim_ref_audio']
            if not wav.is_file() or not ref.is_file():
                raise FileNotFoundError(f'{wav} or {ref}')
            items[method].append((str(wav.resolve()), str(ref.resolve()), row['text']))
    out.mkdir(parents=True, exist_ok=True)
    for metric in a.metrics:
        pending = []
        for method in a.methods:
            path = out / f'{metric}_{method}.jsonl'
            if path.exists():
                if not a.reuse_existing:
                    raise FileExistsError(f'{path}; use --reuse-existing only for unchanged inputs')
                validate(read_jsonl(path), ids, metric, path)
                print(f'Reusing {path}', flush=True)
            else:
                pending.append(method)
        if pending:
            import torch
            torch.set_num_threads(4)
            visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
            if not visible or ',' in visible or torch.cuda.device_count() != 1:
                raise RuntimeError('Set CUDA_VISIBLE_DEVICES to exactly one allocated GPU before Python starts')
            print(f'{metric.upper()}: {len(pending)} groups, {n * len(pending)} WAVs', flush=True)
            if metric in ('wer', 'sim'):
                from f5_tts.eval.utils_eval import run_asr_wer, run_sim
                combined = [item for m in pending for item in items[m]]
                if metric == 'wer':
                    if not a.asr or not Path(a.asr).is_dir():
                        raise FileNotFoundError('Provide --asr pointing to a local Whisper directory')
                    # Upstream sets CUDA_VISIBLE_DEVICES=str(rank); preserve its existing value.
                    results = run_asr_wer((visible, 'en', combined, a.asr))
                else:
                    if not a.sim or not Path(a.sim).is_file():
                        raise FileNotFoundError('Provide --sim pointing to the speaker-verification checkpoint')
                    results = run_sim((0, combined, a.sim))
                if len(results) != len(combined):
                    raise ValueError('Evaluator returned an unexpected number of records')
                parts = {m: results[i*n:(i+1)*n] for i, m in enumerate(pending)}
            else:
                import librosa
                if not a.utmos_hub or not (Path(a.utmos_hub) / 'hubconf.py').is_file():
                    raise FileNotFoundError('Provide --utmos-hub with trusted local SpeechMOS code')
                predictor = torch.hub.load(a.utmos_hub, 'utmos22_strong', source='local').cuda().eval()
                parts = {}
                for method in pending:
                    records = []
                    for i, row in enumerate(rows, 1):
                        wav_path = folders[method] / f'{row["utt"]}.wav'
                        wav, sr = librosa.load(str(wav_path), sr=None, mono=True)
                        tensor = torch.from_numpy(wav).unsqueeze(0).cuda()
                        if not tensor.numel() or not torch.isfinite(tensor).all():
                            raise ValueError(f'Invalid WAV: {wav_path}')
                        with torch.inference_mode():
                            score = float(predictor(tensor, sr).item())
                        records.append(dict(utt=row['utt'], wav=str(wav_path), utmos=score))
                        if i % 20 == 0 or i == n:
                            print(f'{method}: {i}/{n}', flush=True)
                    parts[method] = records
                del predictor
            for method in pending:
                path = out / f'{metric}_{method}.jsonl'
                validate(parts[method], ids, metric, path)
                temp = path.with_suffix('.jsonl.tmp')
                with temp.open('w', encoding='utf-8') as f:
                    for record in parts[method]:
                        f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
                temp.replace(path)
            torch.cuda.empty_cache()
        summary = {m: summarize(read_jsonl(out / f'{metric}_{m}.jsonl'), metric) for m in a.methods}
        write_json(out / f'{metric}_summary.json', summary)
        print(json.dumps({metric: summary}, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
