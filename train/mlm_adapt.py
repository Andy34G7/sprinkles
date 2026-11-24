"""Bidirectional Masked Language Modeling (MLM) adaptation for Crouton backbones.

Converts a causally pretrained Crouton checkpoint into a bidirectional encoder by
continuing training on uint16 memmap shards (`--data-dir`) with `is_causal=False`
and ModernBERT-style 30% token masking (80% mask token, 10% random, 10% unchanged).

The saved checkpoint uses the standard Crouton format and can be passed directly to
`train/train_splade.py --arch crouton --base-checkpoint <out>/final.pt --mntp-steps 0`.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import math
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

HOME = Path.home()
CROUTON_SRC = HOME / "projects/crouton/src"
LERNEA_PY = HOME / "projects/tokenisers/python"


def cosine_lr_mult(step: int, total_steps: int, warmup_steps: int, min_ratio: float = 0.05) -> float:
    if step < warmup_steps:
        return max(1e-3, float(step + 1) / float(max(1, warmup_steps)))
    prog = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + math.cos(math.pi * min(1.0, prog)))


def apply_mlm_mask(
    input_ids: torch.Tensor,
    mask_token_id: int,
    vocab_size: int,
    mask_prob: float = 0.30,
) -> tuple[torch.Tensor, torch.Tensor]:
    """ModernBERT-style MLM corruption (80% [MASK], 10% random token, 10% unchanged)."""
    labels = input_ids.clone()
    probs = torch.full_like(input_ids, mask_prob, dtype=torch.float32)
    mask = torch.bernoulli(probs).bool()
    # Guarantee at least one masked token per batch to avoid NaN loss
    if not mask.any():
        mask[0, 0] = True

    corrupted = input_ids.clone()
    r = torch.rand_like(probs)
    replace_mask = mask & (r < 0.8)
    random_mask = mask & (r >= 0.8) & (r < 0.9)

    corrupted[replace_mask] = mask_token_id
    if random_mask.any():
        corrupted[random_mask] = torch.randint(0, vocab_size, (int(random_mask.sum().item()),), device=input_ids.device)

    labels[~mask] = -100
    return corrupted, labels


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-checkpoint", type=str, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True, help="Directory of uint16 memmap train_*.bin shards")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=2000, help="2000 steps x mb=16 x ga=4 x 1024 = ~131M tokens")
    ap.add_argument("--micro-batch", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--mask-prob", type=float, default=0.30)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-steps", type=int, default=100)
    ap.add_argument("--save-interval", type=int, default=1000)
    ap.add_argument("--log-interval", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-bf16", action="store_true")
    a = ap.parse_args()

    sys.path[:0] = [str(CROUTON_SRC), str(LERNEA_PY)]
    from tokenisers import WpBpePpigTokenizer
    from crouton.data import ShardedMemmapDataset
    from crouton.data.pretrain_data import resolve_pad_token_id
    from crouton.train.common import (
        instantiate_checkpoint_model,
        load_checkpoint_payload,
        save_checkpoint,
        validate_tokenizer_vocab,
    )

    torch.manual_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = (not a.no_bf16) and device.type == "cuda" and torch.cuda.is_bf16_supported()
    amp_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else nullcontext()

    ckpt, model_cfg, _ = load_checkpoint_payload(a.base_checkpoint)
    tok = WpBpePpigTokenizer.from_pretrained(str(a.tokenizer))
    validate_tokenizer_vocab(tok, model_cfg)
    mask_token_id = resolve_pad_token_id(tok)
    vocab_size = model_cfg.vocab_size

    model = instantiate_checkpoint_model(ckpt, model_cfg, device)
    model.train()

    ds = ShardedMemmapDataset(str(a.data_dir), seq_len=a.seq_len, seed=a.seed)
    loader = DataLoader(ds, batch_size=a.micro_batch)
    it = iter(loader)

    decay = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    nodecay = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
    opt = AdamW(
        [{"params": decay, "weight_decay": a.weight_decay}, {"params": nodecay, "weight_decay": 0.0}],
        lr=a.lr,
        betas=(0.9, 0.95),
    )

    toks_per_step = a.micro_batch * a.grad_accum * a.seq_len
    total_toks = toks_per_step * a.steps
    print(
        f"Bidirectional MLM adapt: {a.steps:,} steps x {toks_per_step:,} tok/step = {total_toks / 1e6:.1f}M tokens "
        f"(mask_prob={a.mask_prob}, device={device}, bf16={use_bf16})",
        flush=True,
    )

    a.output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    for step in range(a.steps):
        mult = cosine_lr_mult(step, a.steps, a.warmup_steps)
        for pg in opt.param_groups:
            pg["lr"] = a.lr * mult

        opt.zero_grad(set_to_none=True)
        tot_loss = 0.0
        for _ in range(a.grad_accum):
            try:
                x, _ = next(it)
            except StopIteration:
                it = iter(loader)
                x, _ = next(it)
            x = x.to(device)
            corrupted, labels = apply_mlm_mask(x, mask_token_id, vocab_size, mask_prob=a.mask_prob)
            with amp_ctx:
                out = model(corrupted, is_causal=False)
                loss = F.cross_entropy(out.logits.float().view(-1, vocab_size), labels.view(-1), ignore_index=-100)
                loss = loss / a.grad_accum
            loss.backward()
            tot_loss += float(loss.item())

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % a.log_interval == 0 or step == a.steps - 1:
            print(
                f"[mlm step {step:5d}/{a.steps}] loss={tot_loss:.4f} lr_mult={mult:.3f} "
                f"toks={(step + 1) * toks_per_step / 1e6:.1f}M",
                flush=True,
            )
        if a.save_interval > 0 and step > 0 and step % a.save_interval == 0:
            save_checkpoint(a.output_dir / f"step_{step}.pt", model, model_cfg, step)

    final_path = a.output_dir / "final.pt"
    save_checkpoint(final_path, model, model_cfg, a.steps)
    print(f"Saved bidirectional MLM checkpoint -> {final_path} ({(time.time() - t0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
