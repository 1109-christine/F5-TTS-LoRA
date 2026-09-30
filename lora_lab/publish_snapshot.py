"""Export the actual split and per-utterance scores, without WAVs, weights or local paths.

Read-only with respect to the experiment. Creates a NEW public output directory.
This does not claim new training or new metric computation.
"""
import argparse
import csv
import json
import math
from pathlib import Path

from .artifacts import (METHODS, SPLITS, check_splits, digest, read_jsonl,
                        text_digest, write_json)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', required=True)
    p.add_argument('--split', required=True)
    p.add_argument('--eval-dir', required=True)
    p.add_argument('--runs', default='runs')
    p.add_argument('--out', default='lora_lab/benchmarks/lj2h_seed666')
    a = p.parse_args()
    root, split, eval_dir, out = map(Path, (a.data_root, a.split, a.eval_dir, a.out))
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite public snapshot: {out}')
    splits = {name: read_jsonl(split / f'{name}.jsonl') for name in SPLITS}
    # Canonical relative paths permit equivalent absolute paths in private manifests.
    for rows in splits.values():
        for row in rows:
            for key in ('audio', 'prompt_audio', 'sim_ref_audio'):
                if key in row:
                    path = (root / row[key]).resolve()
                    if not path.is_file():
                        raise FileNotFoundError(path)
                    row[key] = path.relative_to(root.resolve()).as_posix()
    check_splits(splits)
    refs = {r['audio']: r['utt'] for r in splits['reference']}
    frozen = dict(format_version=1, dataset='LJSpeech-1.1', split_seed=666,
                  selection='exported from actual manifests, preserving order', splits={})
    for name, rows in splits.items():
        frozen['splits'][name] = []
        for row in rows:
            record = dict(utt=row['utt'], text_sha256=text_digest(row['text']), duration=row['duration'])
            if name in ('val', 'test'):
                record.update(prompt_utt=refs[row['prompt_audio']], sim_ref_utt=refs[row['sim_ref_audio']])
            frozen['splits'][name].append(record)
    ids = [r['utt'] for r in splits['test']]
    scores, summaries = {}, {}
    for metric in ('wer', 'sim', 'utmos'):
        summaries[metric] = json.loads((eval_dir / f'{metric}_summary.json').read_text(encoding='utf-8'))
        for method in METHODS:
            records = read_jsonl(eval_dir / f'{metric}_{method}.jsonl')
            key = 'utt' if metric == 'utmos' else 'wav'
            if [r[key] for r in records] != ids or not all(math.isfinite(r[metric]) for r in records):
                raise ValueError(f'Invalid {metric}/{method} records')
            # Whitelist fields; WER hypotheses remain useful for error analysis.
            scores[metric, method] = [dict(utt=r[key], **{metric: r[metric]},
                                          **({k: r[k] for k in ('truth', 'hypo')} if metric == 'wer' else {}))
                                      for r in records]
            summary = summaries[metric][method]
            if summary['count'] != len(ids):
                raise ValueError(f'Summary count mismatch: {metric}/{method}')
            if metric != 'wer':
                mean = sum(r[metric] for r in records) / len(records)
                if not math.isclose(mean, summary[f'{metric}_mean'], rel_tol=1e-10, abs_tol=1e-10):
                    raise ValueError('Summary does not match per-utterance scores')
    # Whitelist provenance, excluding private filesystem paths and W&B identity.
    provenance = {'original_manifest_sha256': {name: digest(split / f'{name}.jsonl') for name in SPLITS},
                  'training': {}, 'generation': {}, 'split_sizes': {}}
    train_keys = ('mode', 'rank', 'alpha', 'dropout', 'lr', 'updates', 'warmup', 'accum',
                  'save_every', 'precision', 'checkpoint_activations', 'seed')
    meta_keys = ('base_sha256', 'vocab_sha256', 'train_sha256', 'val_sha256', 'upstream_sha',
                 'code_sha', 'implementation_sha256', 'trainable_params', 'total_params', 'ema')
    gen_keys = ('seed', 'nfe_step', 'cfg_strength', 'sway_sampling_coef', 'seed_policy', 'solver',
                'inference_dtype', 'duration_policy', 'manifest_sha256', 'base_sha256',
                'vocab_sha256', 'checkpoint_sha256')
    for name, rows in splits.items():
        provenance['split_sizes'][name] = dict(samples=len(rows), minutes=sum(r['duration'] for r in rows) / 60)
    for method in METHODS:
        gen = Path(a.runs) / f'infer_lj2h_test100_{method}_seed666' / 'generation_config.json'
        config = json.loads(gen.read_text(encoding='utf-8'))
        if config['manifest_sha256'] != digest(split / 'test.jsonl'):
            raise ValueError(f'Inference used a different test manifest: {method}')
        provenance['generation'][method] = {k: config[k] for k in gen_keys if k in config}
        if method != 'base':
            config_path = Path(a.runs) / f'lj2h_{method}_500_seed666' / 'config.json'
            config = json.loads(config_path.read_text(encoding='utf-8'))
            for name in ('train', 'val'):
                if config['meta'][f'{name}_sha256'] != digest(split / f'{name}.jsonl'):
                    raise ValueError(f'Training manifest mismatch: {method}/{name}')
            provenance['training'][method] = {
                'args': {k: config['args'][k] for k in train_keys if k in config['args']},
                'meta': {k: config['meta'][k] for k in meta_keys if k in config['meta']}}
    out.mkdir(parents=True)
    write_json(out / 'split_ids.json', frozen)
    write_json(out / 'provenance.json', provenance)
    for metric in ('wer', 'sim', 'utmos'):
        # Only publish the documented numeric summary fields.
        keys = (('count', 'corpus_wer_percent', 'mean_utterance_wer_percent', 'substitutions',
                 'deletions', 'insertions', 'reference_words') if metric == 'wer'
                else ('count', f'{metric}_mean'))
        public_summary = {m: {k: summaries[metric][m][k] for k in keys} for m in METHODS}
        write_json(out / f'{metric}_summary.json', public_summary)
        for method in METHODS:
            with (out / f'{metric}_{method}.jsonl').open('w', encoding='utf-8') as f:
                for row in scores[metric, method]:
                    f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    with (out / 'results.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['method', 'count', 'corpus_wer_percent', 'sim_mean', 'utmos_mean'])
        for method in METHODS:
            writer.writerow([method, len(ids), summaries['wer'][method]['corpus_wer_percent'],
                             summaries['sim'][method]['sim_mean'], summaries['utmos'][method]['utmos_mean']])
    print(f'Public snapshot written: {out.resolve()}')
    print('No audio, weights, private paths or login credentials were copied.')


if __name__ == '__main__':
    main()
