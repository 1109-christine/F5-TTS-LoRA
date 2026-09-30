import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torchaudio

from .common import (adapter_state, atomic_save, build_base, inject_lora,
                     load_adapter, read_jsonl, restore_rng, rng_state, seed_all, sha256)


def cache_mels(model, rows, root):
    from f5_tts.model.utils import convert_char_to_pinyin
    cache = []
    # Precompute deterministic features on CPU, keeping GPU timings independent of disk reads.
    for row in rows:
        wav, sr = torchaudio.load(str(root / row['audio']))
        wav = wav.mean(0, keepdim=True)
        if sr != 24000 or not 2 <= wav.shape[-1] / sr <= 12:
            raise ValueError(f"Unexpected sampling rate/duration: {row['utt']}")
        if not torch.isfinite(wav).all() or wav.square().mean() < 1e-10:
            raise ValueError(f"Invalid/silent audio: {row['utt']}")
        text = convert_char_to_pinyin([row['text']])
        missing = sorted({ch for ch in text[0] if ch not in model.vocab_char_map})
        if missing:
            raise ValueError(f"OOV tokens in {row['utt']}: {missing}")
        with torch.no_grad():
            mel = model.mel_spec(wav).permute(0, 2, 1).contiguous()
        cache.append((mel, text))
    if not cache:
        raise ValueError('Empty dataset')
    return cache


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', required=True)
    p.add_argument('--vocab', required=True)
    p.add_argument('--data-root', required=True)
    p.add_argument('--train', required=True)
    p.add_argument('--val', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--mode', choices=['full', 'lora'], required=True)
    p.add_argument('--rank', type=int, default=8)
    p.add_argument('--alpha', type=float, default=16)
    p.add_argument('--dropout', type=float, default=0.0)
    p.add_argument('--lr', type=float, required=True)
    p.add_argument('--updates', type=int, default=500)
    p.add_argument('--warmup', type=int, default=50)
    p.add_argument('--accum', type=int, default=8)
    p.add_argument('--save-every', type=int, default=100)
    p.add_argument('--precision', choices=['bf16', 'fp16'], default='bf16')
    p.add_argument('--checkpoint-activations', action='store_true')
    p.add_argument('--seed', type=int, default=666)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--wandb-project', default=None)
    a = p.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Expose exactly one GPU with CUDA_VISIBLE_DEVICES')
    if a.accum < 1 or a.updates < 1 or not 0 <= a.warmup < a.updates or a.save_every < 1:
        raise ValueError('Invalid update/accumulation schedule')
    if a.precision == 'bf16' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('bf16 unsupported; run all compared methods with --precision fp16')
    torch.set_num_threads(4)
    seed_all(a.seed)
    out = Path(a.out)
    if a.resume:
        if not (out / 'last.pt').exists():
            raise FileNotFoundError(out / 'last.pt')
    elif out.exists():
        raise FileExistsError('Use a new experiment directory, or explicitly pass --resume')
    model, cfg = build_base(a.base, a.vocab, a.checkpoint_activations)
    train_rows, val_rows = read_jsonl(a.train), read_jsonl(a.val)
    if {x['utt'] for x in train_rows} & {x['utt'] for x in val_rows}:
        raise ValueError('Train/validation overlap')
    if not np.isfinite(a.lr) or a.lr <= 0:
        raise ValueError('learning rate must be finite and positive')
    root = Path(a.data_root)
    train_data = cache_mels(model, train_rows, root)
    val_data = cache_mels(model, val_rows, root)
    targets = []
    if a.mode == 'lora':
        targets = inject_lora(model, a.rank, a.alpha, a.dropout)
    else:
        model.requires_grad_(True)
    model = model.cuda()  # Keep master weights FP32; autocast handles compute dtype.
    params = [x for x in model.parameters() if x.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.999), weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=a.precision == 'fp16')
    dtype = torch.bfloat16 if a.precision == 'bf16' else torch.float16
    trainable = sum(x.numel() for x in params)
    total = sum(x.numel() for x in model.parameters())
    config = {k: v for k, v in vars(a).items() if k not in {'resume', 'wandb_project'}}
    meta = dict(mode=a.mode, rank=a.rank, alpha=a.alpha, dropout=a.dropout,
                targets=targets, base_sha256=sha256(a.base), vocab_sha256=sha256(a.vocab),
                train_sha256=sha256(a.train), val_sha256=sha256(a.val),
                upstream_sha=Path(__file__).with_name('UPSTREAM_COMMIT').read_text().strip(),
                code_sha=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                implementation_sha256=hashlib.sha256(b''.join(
                    p.name.encode() + p.read_bytes()
                    for p in sorted(Path(__file__).parent.glob('*.py')))).hexdigest(),
                model_config=cfg, trainable_params=trainable, total_params=total,
                ema=False, format_version=1)
    start, best_val, resume_rng = 0, float('inf'), None
    # Local last.pt contains RNG objects. Only load your own trusted training artifacts.
    if a.resume:
        saved = torch.load(out / 'last.pt', map_location='cpu', weights_only=False)
        if saved['config'] != config or saved['meta'] != meta:
            raise ValueError('Resume config/code/data/base mismatch')
        if a.mode == 'lora':
            load_adapter(model, saved['state'])
        else:
            model.load_state_dict(saved['state'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        scaler.load_state_dict(saved['scaler'])
        start = saved['update']
        best_val = saved['best_val']
        resume_rng = saved['rng']
        del saved
    out.mkdir(parents=True, exist_ok=True)
    (out / 'config.json').write_text(json.dumps({'args': config, 'meta': meta}, indent=2), encoding='utf-8')
    wandb_run = None
    if a.wandb_project:
        import wandb
        id_path = out / 'wandb_id.txt'
        run_id = id_path.read_text().strip() if a.resume and id_path.exists() else wandb.util.generate_id()
        wandb_run = wandb.init(project=a.wandb_project, name=out.name, id=run_id,
                              resume='allow', config={'args': config, 'meta': meta})
        id_path.write_text(run_id, encoding='utf-8')
    print(json.dumps({'trainable': trainable, 'total': total, 'fraction': trainable / total,
                      'train_samples': len(train_data), 'micro_batch_samples': 1,
                      'effective_samples_per_update': a.accum}), flush=True)
    perm_epoch, order = -1, None
    # Adapter initialization and logging setup must not shift training noise across ranks/modes.
    if resume_rng is None:
        seed_all(a.seed)
    else:
        restore_rng(resume_rng)
    torch.cuda.reset_peak_memory_stats()
    for update in range(start, a.updates):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        # Recomputable schedule: no scheduler object to forget in checkpoint restore.
        mult = min((update + 1) / max(a.warmup, 1), 1.0)
        if update >= a.warmup:
            mult = (a.updates - update) / (a.updates - a.warmup)
        for group in optimizer.param_groups:
            group['lr'] = a.lr * mult
        torch.cuda.synchronize()
        tick, loss_value = time.perf_counter(), 0.0
        for micro in range(a.accum):
            position = update * a.accum + micro
            epoch, offset = divmod(position, len(train_data))
            if epoch != perm_epoch:
                order = np.random.default_rng(a.seed + epoch).permutation(len(train_data))
                perm_epoch = epoch
            mel, text = train_data[int(order[offset])]
            mel = mel.cuda()
            with torch.autocast('cuda', dtype=dtype):
                loss, _, _ = model(inp=mel, text=text)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite loss at update {update + 1}')
            loss_value += loss.detach().item() / a.accum
            scaler.scale(loss / a.accum).backward()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        if not torch.isfinite(norm):
            raise FloatingPointError('Non-finite gradients; stop rather than count a skipped update')
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        seconds = time.perf_counter() - tick
        log = dict(update=update + 1, loss=loss_value, lr=optimizer.param_groups[0]['lr'],
                   seconds_per_update=seconds,
                   peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                   peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)
        if (update + 1) % a.save_every == 0 or update + 1 == a.updates:
            saved_rng = rng_state()
            seed_all(12345)
            model.eval()
            losses = []
            with torch.no_grad(), torch.autocast('cuda', dtype=dtype):
                for mel, text in val_data:
                    loss, _, _ = model(inp=mel.cuda(), text=text)
                    losses.append(loss.item())
            restore_rng(saved_rng)
            log['val_loss'] = float(np.mean(losses))
            if not np.isfinite(log['val_loss']):
                raise FloatingPointError('Non-finite validation loss')
            state = adapter_state(model) if a.mode == 'lora' else {
                n: v.detach().cpu().clone() for n, v in model.state_dict().items()}
            deploy = dict(meta=meta, state=state, update=update + 1)
            atomic_save(deploy, out / f'deploy_{update + 1:06d}.pt')
            if log['val_loss'] < best_val:
                best_val = log['val_loss']
                atomic_save(deploy, out / 'best.pt')
            atomic_save(dict(**deploy, config=config, optimizer=optimizer.state_dict(),
                             scaler=scaler.state_dict(), rng=rng_state(), best_val=best_val), out / 'last.pt')
        with (out / 'metrics.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(log) + '\n')
        print(json.dumps(log), flush=True)
        if wandb_run:
            wandb_run.log(log, step=update + 1)
    if wandb_run:
        wandb_run.finish()


if __name__ == '__main__':
    main()
