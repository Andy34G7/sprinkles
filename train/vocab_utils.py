"""Vocabulary utilities for SPLADE v7:
  1. Case & leading-space canonical variant merging (SPARSEUP 2611.09680 / Geng & Yang SIGIR '26)
  2. Document-frequency weighted FLOPS (DF-FLOPS, Porco et al. SIGIR '25)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import torch


def build_canonical_vocab_map(tokenizer_json: Path, vocab_size: int | None = None) -> tuple[torch.Tensor, int]:
    """Map raw BPE token IDs -> canonical lowercase/space-stripped IDs.

    Works on standard HuggingFace tokenizer.json files (both Crouton wp-bpe-ppig and Shortcake BPE).
    Special tokens (e.g. <|pad|>, <|endoftext|>) keep their own unique canonical IDs.
    """
    data = json.loads(Path(tokenizer_json).read_text(encoding="utf-8"))
    vocab: dict[str, int] = {}
    if isinstance(data.get("model"), dict) and isinstance(data["model"].get("vocab"), dict):
        vocab.update(data["model"]["vocab"])
    for tok in data.get("added_tokens", []):
        if isinstance(tok, dict) and "content" in tok and "id" in tok:
            vocab[tok["content"]] = int(tok["id"])

    v_size = vocab_size or (max(vocab.values()) + 1 if vocab else 0)
    id_to_tok = [""] * v_size
    for tok_str, idx in vocab.items():
        if 0 <= idx < v_size:
            id_to_tok[idx] = tok_str

    canon_to_id: dict[str, int] = {}
    mapping = [0] * v_size
    for idx, tok_str in enumerate(id_to_tok):
        if not tok_str or (tok_str.startswith("<") and tok_str.endswith(">")):
            key = f"__SPECIAL_{idx}__"
        else:
            stripped = tok_str.lstrip(" \u0120\u2581").lower()
            key = stripped if stripped else f"__WS_{idx}__"
        if key not in canon_to_id:
            canon_to_id[key] = len(canon_to_id)
        mapping[idx] = canon_to_id[key]

    return torch.tensor(mapping, dtype=torch.long), len(canon_to_id)


def merge_sparse_variants(sv: torch.Tensor, canon_ids: torch.Tensor, num_canon: int) -> torch.Tensor:
    """Differentiable max-pool of case/leading-space equivalent BPE columns."""
    idx = canon_ids.to(device=sv.device).unsqueeze(0).expand_as(sv)
    out = torch.zeros((sv.size(0), num_canon), device=sv.device, dtype=sv.dtype)
    return out.scatter_reduce(1, idx, sv, reduce="amax", include_self=False)


def compute_df_flops_weights(
    docs: list[str],
    tokenize_ids: Callable[[str], list[int]],
    vocab_size: int,
    canon_ids: torch.Tensor | None = None,
    num_canon: int | None = None,
    alpha: float = 0.5,
    max_docs: int = 10_000,
) -> torch.Tensor:
    """Compute per-term DF-FLOPS weights `(df / mean_df)^alpha` clamped to `[0.25, 4.0]`."""
    eff_v = num_canon if (canon_ids is not None and num_canon is not None) else vocab_size
    cmap = canon_ids.tolist() if canon_ids is not None else None
    counts = torch.zeros(eff_v, dtype=torch.float32)

    sample = docs[:max_docs]
    for d in sample:
        ids = tokenize_ids(d)
        if cmap is not None:
            uniq = {cmap[i] for i in ids if 0 <= i < len(cmap)}
        else:
            uniq = {i for i in ids if 0 <= i < eff_v}
        for u in uniq:
            counts[u] += 1.0

    # Smooth with Laplace +1 so unseen terms get mild weight (0.25) rather than 0
    df = (counts + 1.0) / float(len(sample) + 1)
    mean_df = df.mean().clamp(min=1e-8)
    weights = (df / mean_df).pow(alpha).clamp(0.25, 4.0)
    # Normalize mean weight to 1.0 so lambda_d keeps the exact same overall scale
    weights = weights / weights.mean()
    return weights
