# Retrieval Ablation, Tuning & Testing Plan (`v5 → v7`)

Updated: 2026-11-24. Grounded in `v0 → v4-2` results ([`docs/EXPERIMENTS.md`](EXPERIMENTS.md)) and the tokenizer study ([`docs/TOKENISATION.md`](TOKENISATION.md)).

---

## 1. Evidence & Diagnosed Bottlenecks

| Status | Finding | Evidence |
| --- | --- | --- |
| **Solved** | Sparsity scaling across vocab sizes | `temp=1.0`, L1 on queries (`0.005`), vocab-normalized FLOPS (`V / 30522`, `0.008`) align 16k Shortcake and 32k Crouton to `~90–180 Q / ~170–295 D`. |
| **Solved** | CSN docstring leakage | Stripping docstrings (`v4-2`) doubled Shortcake `cosqa` (`0.0908 → 0.1863`, R@100 `0.7140`) and gave our highest mixed Code avg (`0.3871`). |
| **Solved** | Dual-teacher Margin-MSE scale mismatch | Normalizing each teacher's `(pos - neg)` RMS to `8.0` (`splade-v3 × 1.001`, `splade-code-06B × 0.019`) fixed the `2,745×` MSE imbalance. |
| **Bottleneck 1** | Easy random negatives on Code | `splade-code-06B` had `0.0%` negative margins on CoIR (`so-qa`, `csn`) vs. `21.1%` on MS MARCO. |
| **Bottleneck 2** | Overfitting to 90k triplets | By step `7,500/8,000` (`5.7` epochs), `margin_mse` fell to `0.20–0.40` while out-of-domain `nfcorpus` regressed on both models. |
| **Bottleneck 3** | Backbone & bidirectionality gap | Crouton v1 used a 0%-code tokenizer + only `500` steps (`~4M` tokens) of MNTP; Shortcake is strictly causal (`--echo` not yet ablated). |

---

## 2. Phase-by-Phase Roadmap

### Phase 1 — Finish `crouton` v2.0.0 + Bidirectional MLM Adaptation
1. **Finish `crouton-base-50m` v2.0.0:** Auto-resume from Drive (`step_4000/6000 → 15,000`) + `midtrain-4k → sft → dpo` (`tiny-14m` and `small-27m` are already complete in `/MyDrive/crouton_v2/checkpoints/`).
2. **Bidirectional MLM Adaptation (`train/mlm_adapt.py`):** Run `100M–150M` tokens of 30%-masking bidirectional MLM (`is_causal=False`) directly on `/MyDrive/crouton_v2/data_2b/` starting from `crouton-base-50m-base.pt` (`~30 min` on L4) → `crouton-base-50m-mlm.pt`.

### Phase 2 — High-Leverage SPLADE Ablations (`v5`, 1 Variable at a Time)
All runs evaluate intermediate checkpoints (`step 2k / 4k / 6k`) on `core` to catch overfitting early.

- **`v5a` — BM25 Hard-Negative Mining + Multi-Negative KL-Div Distillation:**
  - Mine `k=4–8` BM25 hard negatives per query (`bm25s` top-50 excluding positive) across MS MARCO, StackOverflow-QA, and docstring-stripped CodeSearchNet (`~150k–200k` queries).
  - Score `(q, pos, neg_1..neg_k)` with `naver/splade-v3` (General) and `naver/splade-code-06B` (Code).
  - Train with **Listwise KL-divergence** (`KL(softmax(t/T) || softmax(s/T))`), which is naturally teacher-scale-invariant and directly teaches ranking over hard confusers.
- **`v5b` — Crouton Tokenizer & Bidirectional MLM Isolation:**
  - Same `v5a` dataset across `crouton-base-v1` (0% code tok, 500 MNTP) vs. `crouton-base-v2` (70/30 tok, 500 MNTP) vs. `crouton-base-v2-mlm` (70/30 tok, 150M-tok MLM).
- **`v5c` — Shortcake `--echo` Ablation:**
  - Same `v5a` dataset on Shortcake causal vs. Shortcake `--echo` (`[s, s]` max-pooled over the second copy) to measure how much of Shortcake's General BEIR gap (`0.205` vs. `0.273–0.348`) is causal blindness vs. 16k vocab / pretraining mix.
  - Also test `use_suffix=True` vs. `False` under `--echo`.
  - *Go/no-go gate:* Only invest in native bidirectional Shortcake (bidirectional `AttentionBlock` + BiMamba) if `--echo` produces a clear General BEIR jump and 1× inference latency is needed.
- **`v5d` — Sparsity Pareto Sweep:**
  - Sweep `(lambda_q, lambda_d)` across `{0.002/0.004, 0.005/0.008, 0.010/0.015}` on the winning recipe to map **nDCG@10 vs. Avg Doc Terms** against `splade-v3` (`138/247`) and `ettin-17m` (`1534/1434`).

### Phase 3 — Size Scaling & Efficiency Frontier (`v6`)
- Apply the winning `v5` recipe to `crouton-tiny-14m` and `crouton-small-27m` (plus a smaller Shortcake tier if available).
- Compare `14M / 27M / 50M` against `ettin-17m` (`17M`, `1534/1434` terms) and `splade-v3` (`110M`) across **nDCG@10 vs. Params, Index NNZ, and CPU/GPU Latency**.

### Phase 4 — Broader Validation & Diagnostics (`v7`)
- **Full suite (`--suite full`):** Evaluate winning models + baselines on `beir:fiqa` and multi-language `codesearchnet` (`go`, `java`, `javascript`).
- **Hybrid BM25 + SPLADE (RRF):** Measure whether our sparse neural models complement BM25 in a hybrid retrieval stack.
- **Token-expansion error analysis:** Inspect top-20 expanded query/doc terms on sample `scifact`, `nfcorpus`, and `cosqa` queries vs. `splade-v3` and `ettin-17m`.
