"""Run on the user's GPU with the real official Base checkpoint, before training."""
import argparse
import hashlib
import tempfile

import torch

from .common import (adapter_state, build_base, inject_lora, load_adapter,
                     merge_lora, seed_all)


def frozen_hash(model):
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        if not p.requires_grad:
            h.update(name.encode())
            h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', required=True)
    parser.add_argument('--vocab', required=True)
    a = parser.parse_args()
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Expose exactly one GPU')
    seed_all(42)
    model, _ = build_base(a.base, a.vocab, checkpoint_activations=True)
    model = model.cuda().eval()
    x = torch.randn(1, 64, 100, device='cuda')
    args = dict(x=x, cond=torch.zeros_like(x), text=torch.tensor([[1, 2, 3]], device='cuda'),
                time=torch.tensor([0.5], device='cuda'), cache=False)
    with torch.no_grad():
        before = model.transformer(**args)
    targets = inject_lora(model, 8, 16)
    assert len(targets) == 88
    params = [p for p in model.parameters() if p.requires_grad]
    assert len(params) == 176
    assert sum(p.numel() for p in params) == 1441792
    with torch.no_grad():
        after = model.transformer(**args)
    torch.testing.assert_close(before, after, rtol=1e-5, atol=1e-6)
    print('Real model zero-init equivalence: PASS', flush=True)
    fingerprint = frozen_hash(model)
    model.train()
    optimizer = torch.optim.AdamW(params, lr=1e-4)
    loss, _, _ = model(inp=x, text=['test'])
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    b_grad = sum(p.grad.abs().sum().item() for n, p in model.named_parameters()
                 if '.lora_B.' in n and p.grad is not None)
    assert b_grad > 0
    optimizer.step()
    assert frozen_hash(model) == fingerprint
    print('Real model gradients and frozen base: PASS', flush=True)
    model.eval()
    with torch.no_grad():
        expected = model.transformer(**args)
    with tempfile.TemporaryFile() as f:
        torch.save(adapter_state(model), f)
        f.seek(0)
        state = torch.load(f, map_location='cpu', weights_only=True)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if '.lora_B.' in n:
                p.zero_()
    load_adapter(model, state)
    with torch.no_grad():
        restored = model.transformer(**args)
    torch.testing.assert_close(expected, restored, rtol=1e-5, atol=1e-6)
    merge_lora(model)
    with torch.no_grad():
        merged = model.transformer(**args)
    torch.testing.assert_close(expected, merged, rtol=1e-3, atol=1e-3)
    print('Real model adapter serialization and merge: PASS', flush=True)
    print('Merge max absolute error:', (merged - expected).abs().max().item())


if __name__ == '__main__':
    main()
