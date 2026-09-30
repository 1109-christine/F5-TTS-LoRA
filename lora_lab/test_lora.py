import copy
import tempfile
import unittest
import torch
from torch import nn
from lora_lab.common import LoRALinear, adapter_state, load_adapter, merge_lora, inject_lora
from lora_lab.prepare_arrow import choose_split


class LoRATest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_zero_init_and_freezing(self):
        base = nn.Linear(7, 5)
        x = torch.randn(3, 7)
        expected = base(x).detach()
        m = LoRALinear(base, rank=2, alpha=4)
        torch.testing.assert_close(m(x), expected, rtol=0, atol=0)
        weight = base.weight.detach().clone()
        opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=0.01)
        m(x).square().mean().backward()
        self.assertIsNone(base.weight.grad)
        self.assertGreater(m.lora_B.weight.grad.abs().sum().item(), 0)
        # A can have exactly zero gradient on the first step because B is initialized to zero.
        opt.step()
        torch.testing.assert_close(base.weight, weight, rtol=0, atol=0)
        self.assertGreater(m.lora_B.weight.abs().sum().item(), 0)

    def test_adapter_round_trip_and_merge(self):
        base = nn.Linear(7, 5)
        one = nn.Sequential(LoRALinear(copy.deepcopy(base), 2, 4))
        two = nn.Sequential(LoRALinear(copy.deepcopy(base), 2, 4))
        with torch.no_grad():
            one[0].lora_B.weight.normal_(0, 0.02)
        with tempfile.TemporaryFile() as f:
            torch.save(adapter_state(one), f)
            f.seek(0)
            state = torch.load(f, map_location='cpu', weights_only=True)
        load_adapter(two, state)
        x = torch.randn(3, 7)
        one.eval()
        two.eval()
        torch.testing.assert_close(one(x), two(x))
        expected = one(x).detach()
        # Named child required by merge_lora's parent-path traversal.
        outer = nn.ModuleDict({'body': one})
        outer.eval()
        merge_lora(outer)
        torch.testing.assert_close(outer['body'](x), expected, atol=1e-6, rtol=1e-5)
        with self.assertRaises(ValueError):
            merge_lora(outer)
        with self.assertRaises(ValueError):
            load_adapter(two, {})

    def test_root_child_merge_and_dropout_guard(self):
        model = nn.Sequential(LoRALinear(nn.Linear(7, 5), 2, 4, dropout=0.1))
        with self.assertRaises(ValueError):
            merge_lora(model)
        model.eval()
        with torch.no_grad():
            model[0].lora_B.weight.normal_()
        x = torch.randn(3, 7)
        expected = model(x).detach()
        merge_lora(model)
        torch.testing.assert_close(model(x), expected, rtol=1e-5, atol=1e-6)

    def test_invalid_state_does_not_partially_load(self):
        model = nn.Sequential(LoRALinear(nn.Linear(7, 5), 2, 4))
        original = adapter_state(model)
        bad = {k: v + 1 for k, v in original.items()}
        bad['0.lora_B.weight'] = torch.zeros(100, 100)
        with self.assertRaises(ValueError):
            load_adapter(model, bad)
        for key, value in adapter_state(model).items():
            torch.testing.assert_close(value, original[key], rtol=0, atol=0)

    def test_invalid_hyperparameters(self):
        for rank, alpha, dropout in [(0, 4, 0), (2, float('nan'), 0), (2, 4, 1)]:
            with self.assertRaises(ValueError):
                LoRALinear(nn.Linear(7, 5), rank, alpha, dropout)

    def test_inject_all_22_blocks(self):
        model = nn.Module()
        model.transformer = nn.Module()
        blocks = nn.ModuleList()
        for _ in range(22):
            block = nn.Module()
            block.attn = nn.Module()
            for name in ('to_q', 'to_k', 'to_v'):
                setattr(block.attn, name, nn.Linear(8, 8))
            block.attn.to_out = nn.ModuleList([nn.Linear(8, 8), nn.Dropout(0)])
            blocks.append(block)
        model.transformer.transformer_blocks = blocks
        self.assertEqual(len(inject_lora(model, 2, 4)), 88)
        self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad), 2816)
        self.assertTrue(all('lora_' in n for n, p in model.named_parameters() if p.requires_grad))
        with self.assertRaises(RuntimeError):
            inject_lora(model, 2, 4)

    def test_chapter_split_disjoint_and_reproducible(self):
        rows = [dict(utt=f'{c}_{i}', chapter=str(c), audio=f'{c}/{i}.wav',
                     text=f'Unique sentence chapter {c} item {i}', duration=5)
                for c in range(5) for i in range(30)]
        splits = choose_split(rows, minutes=2, eval_n=5, seed=666)
        self.assertIsNotNone(splits)
        self.assertEqual(splits, choose_split(rows, minutes=2, eval_n=5, seed=666))
        seen = set()
        for values in splits.values():
            chapters = {r['chapter'] for r in values}
            self.assertFalse(seen & chapters)
            seen |= chapters


if __name__ == '__main__':
    unittest.main()
