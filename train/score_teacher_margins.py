"""Precompute dual-teacher Margin-MSE scores (for v3) on mixed General + Code triplets.

Scores:
  - `msmarco` rows with `naver/splade-v3` (general SPLADE teacher)
  - `stackoverflow-qa` and `codesearchnet` rows with `naver/splade-code-06B` (code SPLADE teacher)

All scoring runs at `max_len=256` (matching eval) and writes `pos_score` / `neg_score` into a new JSONL.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


@torch.no_grad()
def score_group(
    rows: list[dict],
    teacher_repo: str,
    max_len: int,
    batch_size: int,
    device: str,
    margin_min: float,
) -> list[dict]:
    if not rows:
        return []
    from sentence_transformers import SparseEncoder

    print(f"Scoring {len(rows):,} rows with {teacher_repo} (max_len={max_len}, bs={batch_size})...", flush=True)
    mkw = {"dtype": torch.bfloat16} if (device == "cuda" and torch.cuda.is_bf16_supported()) else {}
    enc = SparseEncoder(teacher_repo, device=device, trust_remote_code=True, model_kwargs=mkw)
    enc.max_seq_length = max_len

    kept: list[dict] = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        qs = [r["query"] for r in chunk]
        ps = [r["pos"] for r in chunk]
        ns = [r["neg"] for r in chunk]

        q_v = enc.encode_query(qs, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=False)
        p_v = enc.encode_document(ps, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=False)
        n_v = enc.encode_document(ns, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=False)

        pos_scores = (q_v.float() * p_v.float()).sum(dim=-1).cpu().tolist()
        neg_scores = (q_v.float() * n_v.float()).sum(dim=-1).cpu().tolist()

        for r, ps_i, ns_i in zip(chunk, pos_scores, neg_scores):
            if abs(ps_i - ns_i) >= margin_min:
                out = dict(r)
                out["pos_score"] = round(float(ps_i), 4)
                out["neg_score"] = round(float(ns_i), 4)
                out["teacher"] = teacher_repo
                kept.append(out)

        if (start // batch_size) % 100 == 0:
            print(f"  [{teacher_repo}] {min(start + batch_size, len(rows)):,}/{len(rows):,} (kept {len(kept):,})", flush=True)

    del enc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return kept


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--general-teacher", default="naver/splade-v3")
    ap.add_argument("--code-teacher", default="naver/splade-code-06B")
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--general-bs", type=int, default=64)
    ap.add_argument("--code-bs", type=int, default=16)
    ap.add_argument("--margin-min", type=float, default=0.5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    rows = [json.loads(line) for line in a.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    gen_rows = [r for r in rows if r.get("source") == "msmarco"]
    code_rows = [r for r in rows if r.get("source") != "msmarco"]

    scored = []
    scored.extend(score_group(gen_rows, a.general_teacher, a.max_len, a.general_bs, a.device, a.margin_min))
    scored.extend(score_group(code_rows, a.code_teacher, a.max_len, a.code_bs, a.device, a.margin_min))

    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("w", encoding="utf-8") as f:
        for r in scored:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(scored):,} teacher-scored triplets -> {a.output}", flush=True)


if __name__ == "__main__":
    main()
