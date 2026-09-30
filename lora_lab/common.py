import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(x) for x in f if x.strip()]


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())


def restore_rng(s):
    random.setstate(s['python'])
    np.random.set_state(s['numpy'])
    torch.set_rng_state(s['torch'])
    torch.cuda.set_rng_state_all(s['cuda'])


class LoRALinear(nn.Module):
    def __init__(self, base, rank=8, alpha=16, dropout=0.0):
        super().__init__()
        if not isinstance(base, nn.Linear) or not isinstance(rank, int) or rank <= 0:
            raise ValueError('LoRA requires an nn.Linear and positive rank')
        if not math.isfinite(alpha) or alpha <= 0 or not 0 <= dropout < 1:
            raise ValueError('Require finite alpha > 0 and 0 <= dropout < 1')
        self.base = base
        self.base.requires_grad_(False)
        self.scale = float(alpha) / rank
        self.dropout = nn.Dropout(dropout)
        kw = dict(device=base.weight.device, dtype=base.weight.dtype)
        self.lora_A = nn.Linear(base.in_features, rank, bias=False, **kw)
        self.lora_B = nn.Linear(rank, base.out_features, bias=False, **kw)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.base(x) + self.scale * self.lora_B(self.lora_A(self.dropout(x)))


def inject_lora(model, rank, alpha, dropout=0.0):
    if any(isinstance(m, LoRALinear) for m in model.modules()):
        raise RuntimeError('LoRA already injected')
    blocks = model.transformer.transformer_blocks
    if len(blocks) != 22:
        raise ValueError('This reference implementation targets v1 Base, depth=22')
    # Validate all targets before mutating the model.
    if not isinstance(rank, int) or rank <= 0 or not math.isfinite(alpha) or alpha <= 0 or not 0 <= dropout < 1:
        raise ValueError('Invalid LoRA configuration')
    for block in blocks:
        if not all(isinstance(m, nn.Linear) for m in
                   [block.attn.to_q, block.attn.to_k, block.attn.to_v, block.attn.to_out[0]]):
            raise TypeError('Attention projections differ from the supported upstream')
    model.requires_grad_(False)
    targets = []
    for i, block in enumerate(blocks):
        for name in ('to_q', 'to_k', 'to_v'):
            old = getattr(block.attn, name)
            setattr(block.attn, name, LoRALinear(old, rank, alpha, dropout))
            targets.append(f'transformer.transformer_blocks.{i}.attn.{name}')
        block.attn.to_out[0] = LoRALinear(block.attn.to_out[0], rank, alpha, dropout)
        targets.append(f'transformer.transformer_blocks.{i}.attn.to_out.0')
    return targets


def adapter_state(model):
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters()
            if '.lora_A.' in n or '.lora_B.' in n}


def load_adapter(model, state):
    expected = set(adapter_state(model))
    if not expected or expected != set(state):
        raise ValueError('Adapter tensor names mismatch')
    params = dict(model.named_parameters())
    for name, value in state.items():
        if params[name].shape != value.shape or not torch.isfinite(value).all():
            raise ValueError(f'Adapter shape/non-finite mismatch: {name}')
    with torch.no_grad():
        for name, value in state.items():
            params[name].copy_(value)


def merge_lora(model):
    # In-place deployment export. Never continue training this merged instance.
    names = [n for n, m in model.named_modules() if isinstance(m, LoRALinear)]
    if not names:
        raise ValueError('No unmerged LoRA modules')
    if model.training:
        raise ValueError('Call eval() before merging adapters for deployment')
    for name in names:
        parent_name, _, attr = name.rpartition('.')
        parent = model.get_submodule(parent_name) if parent_name else model
        layer = model.get_submodule(name)
        with torch.no_grad():
            delta = layer.lora_B.weight.float() @ layer.lora_A.weight.float()
            layer.base.weight.add_((delta * layer.scale).to(layer.base.weight.dtype))
        setattr(parent, attr, layer.base)
    return model


def build_base(base_path, vocab_path, checkpoint_activations=False):
    import torch.utils.checkpoint
    from importlib.resources import files
    from omegaconf import OmegaConf
    from safetensors.torch import load_file
    from f5_tts.model import CFM, DiT
    from f5_tts.model.utils import get_tokenizer

    cfg_path = files('f5_tts').joinpath('configs/F5TTS_v1_Base.yaml')
    cfg = OmegaConf.to_container(OmegaConf.load(str(cfg_path))['model'], resolve=True)
    arch = dict(cfg['arch'])
    if (arch['dim'], arch['depth'], arch['heads']) != (1024, 22, 16):
        raise ValueError('Unexpected Base architecture; use the pinned clean checkout')
    arch['checkpoint_activations'] = checkpoint_activations
    vocab, size = get_tokenizer(str(vocab_path), 'custom')
    if vocab.get(' ') != 0:
        raise ValueError('Official vocabulary requires space at index 0')
    model = CFM(transformer=DiT(**arch, text_num_embeds=size, mel_dim=100),
                mel_spec_kwargs=cfg['mel_spec'], vocab_char_map=vocab,
                odeint_kwargs={'method': 'euler'}).float()
    if Path(base_path).suffix != '.safetensors':
        raise ValueError('Use official v1 Base model_1250000.safetensors for this recipe')
    raw = load_file(str(base_path), device='cpu')
    state = {}
    obsolete = {'mel_spec.mel_stft.mel_scale.fb', 'mel_spec.mel_stft.spectrogram.window'}
    for key, value in raw.items():
        if key in {'initted', 'step', 'update'}:
            continue
        name = key.removeprefix('ema_model.')
        if name not in obsolete:
            state[name] = value
    model.load_state_dict(state, strict=True)
    return model, cfg


def load_deployment(model, payload, base_hash, vocab_hash):
    meta = payload['meta']
    if meta['base_sha256'] != base_hash or meta['vocab_sha256'] != vocab_hash:
        raise ValueError('Wrong base checkpoint or vocabulary')
    if meta['mode'] == 'lora':
        targets = inject_lora(model, meta['rank'], meta['alpha'], meta['dropout'])
        if targets != meta['targets']:
            raise ValueError('LoRA target list mismatch')
        load_adapter(model, payload['state'])
    elif meta['mode'] == 'full':
        model.load_state_dict(payload['state'], strict=True)
    else:
        raise ValueError('Unknown deployment mode')
    return model


def atomic_save(payload, path):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, temp)
    temp.replace(path)
