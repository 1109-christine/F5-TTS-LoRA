"""Export LoRA/full deployment payload as plain, merged model weights.

The exported safetensors file has ordinary CFM state keys and no EMA prefix.
Load it with the official load_checkpoint(..., use_ema=False).
"""
import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from .common import build_base, load_deployment, merge_lora, sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', required=True)
    p.add_argument('--vocab', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    out = Path(a.out)
    if out.suffix != '.safetensors' or out.exists():
        raise ValueError('Use a new .safetensors output path')
    payload = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    model, _ = build_base(a.base, a.vocab)
    load_deployment(model, payload, sha256(a.base), sha256(a.vocab))
    model.eval()
    if payload['meta']['mode'] == 'lora':
        merge_lora(model)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(out))
    meta = dict(source_meta=payload['meta'], update=payload['update'], ema=False,
                source_checkpoint_sha256=sha256(a.checkpoint), exported_sha256=sha256(out))
    out.with_suffix('.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(f'Exported {out}; load as ordinary weights with use_ema=False')


if __name__ == '__main__':
    main()
