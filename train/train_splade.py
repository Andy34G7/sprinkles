"""Unified SPLADE trainer for Crouton and Shortcake (v2 InfoNCE and v3 Margin-MSE distillation).

Uses the exact same mixed-domain JSONL, FLOPS schedule, and checkpoint formats loaded by
`harness/eval_suite.py`:
  - v2 (no distillation): pass `--margin-mse-weight 0.0` (InfoNCE + quadratic FLOPS warmup)
  - v3 (distillation):    pass scored JSONL with `pos_score`/`neg_score` and `--margin-mse-weight 1.0`
  - Shortcake `--echo`:   feeds `[tokens, tokens]` and max-pools only over the second copy for
                          full-sequence visibility through the causal Mamba SSM.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys
import time
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

HOME = Path.home()
CROUTON_SRC = HOME / "projects/crouton/src"
DEFAULT_CROUTON_TOK = HOME / "projects/crouton/artifacts/tokenizer/wp_bpe_ppig_32k.json"
LERNEA_PY = HOME / "projects/tokenisers/python"
SHORTCAKE_DIR = HOME / "projects/shortcake"
DEFAULT_SHORTCAKE_TOK = SHORTCAKE_DIR / "tokenizer.json"


# ---------------------------------------------------------------- losses & schedules

def quad_reg_weight(step: int, warmup_steps: int, max_weight: float) -> float:
    if max_weight <= 0.0:
        return 0.0
    if warmup_steps <= 0 or step >= warmup_steps:
        return max_weight
    frac = max(0.0, float(step) / float(warmup_steps))
    return max_weight * frac * frac


def lr_mult_cosine(step: int, total_steps: int, warmup_steps: int, min_ratio: float = 0.1) -> float:
    if step < warmup_steps:
        return max(1e-3, float(step + 1) / float(max(1, warmup_steps)))
    prog = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    import math
    return min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + math.cos(math.pi * min(1.0, prog)))


def infonce_loss(q_vec: torch.Tensor, p_vec: torch.Tensor, n_vec: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    q_f, p_f, n_f = q_vec.float(), p_vec.float(), n_vec.float()
    scores = torch.cat([q_f @ p_f.T, q_f @ n_f.T], dim=1)
    if temperature != 1.0:
        scores = scores / temperature
    labels = torch.arange(q_f.size(0), device=q_f.device, dtype=torch.long)
    return F.cross_entropy(scores, labels)


def margin_mse_loss(
    q_vec: torch.Tensor,
    p_vec: torch.Tensor,
    n_vec: torch.Tensor,
    pos_score: torch.Tensor,
    neg_score: torch.Tensor,
) -> torch.Tensor:
    s_pos = (q_vec.float() * p_vec.float()).sum(dim=-1)
    s_neg = (q_vec.float() * n_vec.float()).sum(dim=-1)
    return F.mse_loss(s_pos - s_neg, pos_score.float() - neg_score.float())


# ---------------------------------------------------------------- dataset

class TripletJSONLDataset(Dataset):
    def __init__(self, path: Path, max_examples: int = 0) -> None:
        self.rows: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("query") and r.get("pos") and r.get("neg"):
                    self.rows.append(r)
                    if max_examples and len(self.rows) >= max_examples:
                        break
        if not self.rows:
            raise ValueError(f"No valid triplets found in {path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.rows[idx]


def _pad(seqs: list[list[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    ml = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), ml), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), ml), dtype=torch.bool)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        mask[i, : len(s)] = True
    return ids, mask


# ---------------------------------------------------------------- adapters

class CroutonAdapter:
    def __init__(self, base_ckpt: str, tokenizer_path: Path, device: torch.device, bidirectional: bool = True) -> None:
        sys.path[:0] = [str(CROUTON_SRC), str(LERNEA_PY)]
        from tokenisers import WpBpePpigTokenizer
        from crouton.data.pretrain_data import resolve_pad_token_id
        from crouton.model.splade import CroutonSPLADE
        from crouton.train.common import instantiate_checkpoint_model, load_checkpoint_payload, validate_tokenizer_vocab
        from crouton.train.splade import run_mntp_adaptation, save_splade_checkpoint

        ckpt, cfg, _ = load_checkpoint_payload(base_ckpt)
        self.tok = WpBpePpigTokenizer.from_pretrained(str(tokenizer_path))
        validate_tokenizer_vocab(self.tok, cfg)
        self.pad_id = resolve_pad_token_id(self.tok)
        backbone = instantiate_checkpoint_model(ckpt, cfg, device)
        self.model = CroutonSPLADE(backbone, tie_lm_head=True, pad_token_id=self.pad_id, bidirectional=bidirectional).to(device)
        self._fmt = CroutonSPLADE.format_bpe_text
        self._mntp = run_mntp_adaptation
        self._save = save_splade_checkpoint
        self.bidirectional = bidirectional

    def encode_batch(self, texts: list[str], max_len: int, is_query: bool) -> tuple[torch.Tensor, torch.Tensor]:
        seqs = []
        for t in texts:
            ids = self.tok.encode(self._fmt(t, prefix=" "), split_special_tokens=True)[:max_len]
            seqs.append(ids if ids else [self.pad_id])
        return _pad(seqs, self.pad_id)

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.model(ids, attention_mask=mask)
        return out.sparse_vector, out.flops_loss, out.l1_loss

    def warmup(self, loader: DataLoader, device: torch.device, amp_ctx, steps: int, lr: float) -> None:
        if steps > 0 and self.bidirectional:
            self._mntp(self.model, loader, device, amp_ctx, steps=steps, lr=lr, mask_token_id=self.pad_id)

    def save(self, path: Path, step: int) -> None:
        self._save(path, self.model, step)


class ShortcakeAdapter:
    def __init__(self, base_ckpt: str, tokenizer_path: Path, device: torch.device, echo: bool = False, use_suffix: bool = True) -> None:
        sys.path.insert(0, str(SHORTCAKE_DIR))
        from tokenizers import Tokenizer
        from model import Shortcake as Backbone
        from splade_head import ShortcakeSPLADE
        from train_splade import DOC_SUFFIX, QUERY_SUFFIX

        ckpt = torch.load(base_ckpt, map_location=device, weights_only=False)
        self.config = ckpt["config"]
        self.tok = Tokenizer.from_file(str(tokenizer_path))
        pad = self.tok.token_to_id("<pad>")
        self.pad_id = 1 if pad is None else pad
        backbone = Backbone(self.config)
        backbone.load_state_dict(ckpt["model_state_dict"])
        self.model = ShortcakeSPLADE(backbone, tie_lm_head=True, pad_token_id=self.pad_id).to(device)
        self.echo = echo
        self.use_suffix = use_suffix
        self.sfx = {True: QUERY_SUFFIX if use_suffix else "", False: DOC_SUFFIX if use_suffix else ""}

    def encode_batch(self, texts: list[str], max_len: int, is_query: bool) -> tuple[torch.Tensor, torch.Tensor]:
        sfx = self.sfx[is_query]
        encs = self.tok.encode_batch([t + sfx for t in texts])
        seqs = [e.ids[:max_len] if e.ids else [self.pad_id] for e in encs]
        if not self.echo:
            return _pad(seqs, self.pad_id)
        # Echo embeddings: [s + s], mask True ONLY on second copy so every pooled token saw the full sequence
        ml = max(len(s) * 2 for s in seqs)
        ids = torch.full((len(seqs), ml), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(seqs), ml), dtype=torch.bool)
        for i, s in enumerate(seqs):
            n = len(s)
            ids[i, :n] = torch.tensor(s, dtype=torch.long)
            ids[i, n : 2 * n] = torch.tensor(s, dtype=torch.long)
            mask[i, n : 2 * n] = True
        return ids, mask

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.model(ids, attention_mask=mask)
        sv = out["sparse_vector"]
        l1 = sv.float().sum(dim=-1).mean()
        return sv, out["flops_loss"], l1

    def warmup(self, loader: DataLoader, device: torch.device, amp_ctx, steps: int, lr: float) -> None:
        return

    def save(self, path: Path, step: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "step": step,
                "model_state_dict": self.model.state_dict(),
                "config": self.config,
                "use_suffix": self.use_suffix,
                "echo": self.echo,
            },
            path,
        )


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", choices=["crouton", "shortcake"], required=True)
    ap.add_argument("--base-checkpoint", type=str, required=True)
    ap.add_argument("--tokenizer", type=Path, default=None)
    ap.add_argument("--triplets", type=Path, required=True, help="JSONL from build_mixed_triplets.py or score_teacher_margins.py")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--max-examples", type=int, default=0)
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--mntp-steps", type=int, default=1000, help="Crouton bidirectional MNTP warmup steps")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--max-query-len", type=int, default=96)
    ap.add_argument("--max-doc-len", type=int, default=256)
    ap.add_argument("--lr", type=float, default=4e-5)
    ap.add_argument("--query-reg-type", choices=["l1", "flops"], default="l1")
    ap.add_argument("--lambda-q", type=float, default=0.015)
    ap.add_argument("--lambda-d", type=float, default=0.04)
    ap.add_argument("--reg-warmup-steps", type=int, default=2000)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--margin-mse-weight", type=float, default=0.0, help="0.0 = v2 (InfoNCE only); >0 = v3 (Margin-MSE distillation)")
    ap.add_argument("--infonce-weight", type=float, default=1.0)
    ap.add_argument("--echo", action="store_true", help="Shortcake only: Echo embeddings (2x input, pool second copy)")
    ap.add_argument("--save-interval", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    amp_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else nullcontext()

    tok_path = a.tokenizer or (DEFAULT_CROUTON_TOK if a.arch == "crouton" else DEFAULT_SHORTCAKE_TOK)
    adapter = (
        CroutonAdapter(a.base_checkpoint, tok_path, device)
        if a.arch == "crouton"
        else ShortcakeAdapter(a.base_checkpoint, tok_path, device, echo=a.echo)
    )

    ds = TripletJSONLDataset(a.triplets, max_examples=a.max_examples)

    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        q_ids, q_mask = adapter.encode_batch([b["query"] for b in batch], a.max_query_len, is_query=True)
        p_ids, p_mask = adapter.encode_batch([b["pos"] for b in batch], a.max_doc_len, is_query=False)
        n_ids, n_mask = adapter.encode_batch([b["neg"] for b in batch], a.max_doc_len, is_query=False)
        out: dict[str, Any] = {
            "q_ids": q_ids, "q_mask": q_mask,
            "p_ids": p_ids, "p_mask": p_mask,
            "n_ids": n_ids, "n_mask": n_mask,
        }
        if all(b.get("pos_score") is not None and b.get("neg_score") is not None for b in batch):
            out["pos_score"] = torch.tensor([float(b["pos_score"]) for b in batch], dtype=torch.float32)
            out["neg_score"] = torch.tensor([float(b["neg_score"]) for b in batch], dtype=torch.float32)
        return out

    loader = DataLoader(ds, batch_size=a.batch_size, shuffle=True, drop_last=len(ds) >= a.batch_size, collate_fn=collate)
    print(f"Training {a.arch} SPLADE on {len(ds):,} triplets ({device}, bf16={use_bf16}, echo={a.echo})", flush=True)

    adapter.warmup(loader, device, amp_ctx, steps=a.mntp_steps, lr=a.lr)

    decay_params = [p for p in adapter.model.parameters() if p.requires_grad and p.ndim >= 2]
    nodecay_params = [p for p in adapter.model.parameters() if p.requires_grad and p.ndim < 2]
    opt = AdamW(
        [{"params": decay_params, "weight_decay": 0.01}, {"params": nodecay_params, "weight_decay": 0.0}],
        lr=a.lr,
        betas=(0.9, 0.95),
    )

    warmup_steps = min(500, max(1, a.steps // 10))
    it = iter(loader)
    adapter.model.train()
    t0 = time.time()

    for step in range(a.steps):
        mult = lr_mult_cosine(step, a.steps, warmup_steps)
        for pg in opt.param_groups:
            pg["lr"] = a.lr * mult
        lq = quad_reg_weight(step, a.reg_warmup_steps, a.lambda_q)
        ld = quad_reg_weight(step, a.reg_warmup_steps, a.lambda_d)

        opt.zero_grad(set_to_none=True)
        tot_loss = tot_rank = tot_l0_q = tot_l0_d = 0.0

        for _ in range(a.grad_accum):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader)
                batch = next(it)

            q_ids, q_mask = batch["q_ids"].to(device), batch["q_mask"].to(device)
            p_ids, p_mask = batch["p_ids"].to(device), batch["p_mask"].to(device)
            n_ids, n_mask = batch["n_ids"].to(device), batch["n_mask"].to(device)

            with amp_ctx:
                q_vec, q_flops, q_l1 = adapter.forward(q_ids, q_mask)
                p_vec, p_flops, _ = adapter.forward(p_ids, p_mask)
                n_vec, n_flops, _ = adapter.forward(n_ids, n_mask)

                rank = torch.tensor(0.0, device=device)
                if a.infonce_weight > 0.0:
                    rank = rank + a.infonce_weight * infonce_loss(q_vec, p_vec, n_vec, temperature=a.temperature)
                if a.margin_mse_weight > 0.0 and "pos_score" in batch:
                    mmse = margin_mse_loss(
                        q_vec, p_vec, n_vec, batch["pos_score"].to(device), batch["neg_score"].to(device)
                    )
                    rank = rank + a.margin_mse_weight * mmse

                q_reg = q_l1 if a.query_reg_type == "l1" else q_flops
                reg = lq * q_reg + ld * 0.5 * (p_flops + n_flops)
                loss = (rank + reg) / a.grad_accum

            loss.backward()
            tot_loss += float(loss.item())
            tot_rank += float(rank.item()) / a.grad_accum
            with torch.no_grad():
                tot_l0_q += float((q_vec > 0).float().sum(dim=-1).mean().item()) / a.grad_accum
                tot_l0_d += float((p_vec > 0).float().sum(dim=-1).mean().item()) / a.grad_accum

        torch.nn.utils.clip_grad_norm_(adapter.model.parameters(), 1.0)
        opt.step()

        if step % 50 == 0 or step == a.steps - 1:
            print(
                f"[{a.arch} step {step:5d}/{a.steps}] loss={tot_loss:.4f} rank={tot_rank:.4f} "
                f"l0_q={tot_l0_q:.1f} l0_d={tot_l0_d:.1f} lq={lq:.4f} ld={ld:.4f} lr_mult={mult:.3f}",
                flush=True,
            )
        if a.save_interval > 0 and step > 0 and step % a.save_interval == 0:
            adapter.save(a.output_dir / f"step_{step}.pt", step)

    final_name = "splade_final.pt" if a.arch == "crouton" else "best_splade_model.pt"
    final_path = a.output_dir / final_name
    adapter.save(final_path, a.steps)
    print(f"Saved final checkpoint -> {final_path} ({(time.time() - t0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
