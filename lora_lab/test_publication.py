"""CPU-only publication checks; no torch or evaluator weights required."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from .artifacts import METHODS, SPLITS, check_splits, digest, text_digest


class PublicationTests(unittest.TestCase):
    def fixture(self, root):
        data, runs, scores = root / 'data', root / 'runs', root / 'scores'
        data.mkdir()
        scores.mkdir()
        splits = {}
        for i, name in enumerate(SPLITS):
            rows = []
            for j in range(2):
                utt = f'LJ001-{i*2+j:04d}'
                audio = data / 'wavs' / f'{utt}.wav'
                audio.parent.mkdir(exist_ok=True)
                # Snapshot checks file identity/existence, not audio decoding.
                audio.write_bytes(b'fixture-audio')
                rows.append(dict(utt=utt, audio=str(audio), text=f'Unique sentence {i} {j}', duration=3.0))
            splits[name] = rows
        ref1, ref2 = splits['reference']
        for name in ('val', 'test'):
            for row in splits[name]:
                row.update(prompt_audio=ref1['audio'], sim_ref_audio=ref2['audio'], prompt_text=ref1['text'])
        for name, rows in splits.items():
            (data / f'{name}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        for method in METHODS:
            folder = runs / f'infer_lj2h_test100_{method}_seed666'
            folder.mkdir(parents=True)
            (folder / 'generation_config.json').write_text(json.dumps({
                'manifest_sha256': digest(data / 'test.jsonl'), 'seed': 666,
                'base': '/private/person/base.safetensors'}))
            if method != 'base':
                folder = runs / f'lj2h_{method}_500_seed666'
                folder.mkdir()
                (folder / 'config.json').write_text(json.dumps({'args': {'seed': 666, 'base': '/private/model'},
                     'meta': {'train_sha256': digest(data / 'train.jsonl'),
                              'val_sha256': digest(data / 'val.jsonl')}}))
        for metric in ('wer', 'sim', 'utmos'):
            summary = {}
            for method in METHODS:
                records = []
                for row in splits['test']:
                    if metric == 'utmos':
                        record = dict(utt=row['utt'], wav='/private/person/audio.wav', utmos=4.0)
                    else:
                        record = dict(wav=row['utt'], **{metric: 0.0 if metric == 'wer' else 0.7})
                        if metric == 'wer':
                            record.update(truth=row['text'], hypo=row['text'])
                    records.append(record)
                (scores / f'{metric}_{method}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
                if metric == 'wer':
                    summary[method] = dict(count=2, corpus_wer_percent=0.0, mean_utterance_wer_percent=0.0,
                                           substitutions=0, deletions=0, insertions=0, reference_words=8)
                else:
                    summary[method] = dict(count=2, **{f'{metric}_mean': 4.0 if metric == 'utmos' else 0.7})
            (scores / f'{metric}_summary.json').write_text(json.dumps(summary))
        return data, runs, scores, splits

    def run_export(self, data, runs, scores, out):
        return subprocess.run([sys.executable, '-m', 'lora_lab.publish_snapshot',
                               '--data-root', str(data), '--split', str(data), '--runs', str(runs),
                               '--eval-dir', str(scores), '--out', str(out)], capture_output=True, text=True)

    def test_snapshot_preserves_ids_and_removes_private_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, runs, scores, splits = self.fixture(root)
            before = {p: digest(p) for p in root.rglob('*') if p.is_file()}
            out = root / 'public'
            result = self.run_export(data, runs, scores, out)
            self.assertEqual(result.returncode, 0, result.stderr)
            for path, sha in before.items():
                self.assertEqual(digest(path), sha)
            frozen = json.loads((out / 'split_ids.json').read_text())
            for name in SPLITS:
                self.assertEqual([r['utt'] for r in frozen['splits'][name]], [r['utt'] for r in splits[name]])
                self.assertEqual(frozen['splits'][name][0]['text_sha256'], text_digest(splits[name][0]['text']))
            for path in out.rglob('*'):
                if path.is_file():
                    self.assertNotIn('/private/', path.read_text())
                    self.assertNotIn(str(root), path.read_text())
            again = self.run_export(data, runs, scores, out)
            self.assertNotEqual(again.returncode, 0)

    def test_tampered_manifest_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, runs, scores, _ = self.fixture(root)
            with (data / 'test.jsonl').open('a') as f:
                f.write('\n')  # Same rows, different original manifest bytes.
            result = self.run_export(data, runs, scores, root / 'public')
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('different test manifest', result.stderr)
            self.assertFalse((root / 'public').exists())

    def test_reference_and_text_leakage_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, _, splits = self.fixture(Path(directory))
            check_splits(splits)
            splits['test'][0]['sim_ref_audio'] = splits['test'][0]['prompt_audio']
            with self.assertRaises(ValueError):
                check_splits(splits)
            splits['test'][0]['sim_ref_audio'] = splits['reference'][1]['audio']
            splits['test'][0]['text'] = splits['train'][0]['text'].upper() + '!'
            with self.assertRaises(ValueError):
                check_splits(splits)


if __name__ == '__main__':
    unittest.main()
