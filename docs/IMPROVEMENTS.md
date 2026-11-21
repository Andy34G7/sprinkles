# SPLADE improvements: shortcake and crouton

Written 2026-11-21. Diagnoses come from these runs' logs and code. Expected gains come from the SPLADE
literature (SPLADE++, SPLADE-v3); they have not been measured on these models yet.

## Baseline results (2026-11-21)

### Shortcake (50M Mamba SSM), 1,000 held-out CodeSearchNet-python pairs, docstrings stripped

| Model | MRR@10 | R@1 | R@10 | Terms Q/D |
|---|---|---|---|---|
| Shortcake SPLADE (fine-tuned in-domain, 2000 steps, 50k pairs) | 0.7375 | 0.623 | 0.943 | 64 / 80 |
| SPLADE++ `naver/splade-cocondenser-ensembledistil` (zero-shot) | 0.5710 | 0.448 | 0.811 | 88 / 101 |
| SPLADE-v3 `naver/splade-v3` (zero-shot) | 0.5632 | 0.435 | 0.816 | 156 / 226 |

Not apples to apples: shortcake is trained in-domain, the baselines are zero-shot. The 1k pool is much
easier than standard benchmarks.

### Crouton-base (50M), BEIR SciFact (300 queries, 5,000 docs, identical pool)

| Model | nDCG@10 | MRR@10 | R@100 | Terms Q/D |
|---|---|---|---|---|
| SPLADE++ | 0.7041 | 0.6717 | 0.9353 | 80 / 189 |
| crouton-base SPLADE (300 MNTP + 5000 InfoNCE steps, msmarco-bm25) | 0.5046 | 0.4672 | 0.8383 | 60 / 248 |
| SPLADE-v3 `naver/splade-v3` | 0.7109 | 0.6858 | 0.9343 | 129 / 260 |
| BM25 (published BEIR) | ~0.665 | | | |

Artifacts:
- `~/projects/crouton/checkpoints/splade/splade_final.pt`, `splade.log`
- `~/projects/shortcake/checkpoints_splade/best_splade_model.pt`
- Baseline scripts: `crouton.eval.eval_splade --hf-baseline <hf id>`, `shortcake/eval_splade_baseline.py`

## Improvements, ranked by expected gain per effort

### Both models

1. **Cross-encoder distillation.** This is the main ingredient in SPLADE-v3 and the biggest gap.
   - Score each (query, pos, neg) with `cross-encoder/ms-marco-MiniLM-L6-v2` (minutes on an L4).
   - Train with Margin-MSE, or KL over several negatives.
   - Crouton's trainer already supports it: give `--local-jsonl` records `pos_score` / `neg_score`.
   - msmarco-bm25 has no scores, so the first run was InfoNCE only.
   - The literature reports roughly +3 to 5 nDCG on BEIR.
2. **Self-mined hard negatives.** Crouton used BM25 negatives; shortcake used in-batch negatives only.
   - Round 2: index the corpus with the round-1 model, take its rank 10 to 50 hits as negatives, filter
     false negatives with the cross-encoder, and retrain.
3. **Much more training.**
   - Crouton saw 5000 x 64 = ~320k triplets, under one pass over MS MARCO.
   - Shortcake used 2000 steps on 50k pairs, and MRR was still rising at the end (0.711, 0.737, 0.7375).
   - Bigger *real* batches add in-batch negatives. Gradient accumulation does not: crouton's 2 x 32
     micro-batches give 32 negatives per query, not 64.
4. **Fold case variants together.** Crouton's top terms split budget across `Ġtree` / `ĠTree` /
   `Ġtrees` and `ĠSearch` / `Ġsearch`. For shortcake, lowercasing the input is fine. For crouton, do
   NOT lowercase: with its PPIG vocab, `ĠPython` is one token but `Ġpython` fragments (see
   `tokenisation.md`). Instead, max-pool case variants into one canonical id at index time.
5. **Loosen the query penalty.** Crouton queries collapsed to ~13 active terms during training
   (`lambda_q=0.05`, L1). Recall@100 of 0.84 vs 0.94 points to under-expansion. Try `lambda_q` around
   0.01, or FLOPS regularisation on queries.

### Crouton-specific

6. **Longer bidirectional adaptation.** Only 300 MNTP steps converted the causal model; LLM2Vec-style
   conversion uses 1k+. Better still, run a few thousand steps of MLM-style training on the MS MARCO
   passages first (the CoCondenser idea from SPLADE++), so the vocabulary head learns to predict related
   terms. That is where expansion comes from.
7. **Longer docs.** SciFact abstracts often exceed 256 BPE tokens, and crouton handles 4k context.
   Train and eval docs at 512.
8. **Capacity.** Try `crouton-medium` (85M); BERT-base is 110M.
9. **Tokenisation.** The lernea PPIG vocab fragments word-initial technical terms (`Ġp ython`,
   `Ġj son`), 20/102 vs 7/102 for GPT-2. Not an encoder bug. See `tokenisation.md`.

### Shortcake-specific

10. **Bidirectionality for a causal SSM.**
    - Echo embeddings: feed the text twice and max-pool over the second copy only. No model changes;
      costs 2x compute.
    - Bidirectional SSM: sum a forward and a reversed scan (Caduceus / Vim style). A bigger change.
11. **More data and harder negatives.** Use the full CodeSearchNet-python train set (~400k pairs),
    other languages, and mined look-alike negatives.
12. **Standard benchmarks.** Evaluate on the full CodeSearchNet test set or CoIR, plus one BEIR task, so
    the numbers compare with each other and with published work.

## Suggested order

1. Cross-encoder scores plus Margin-MSE (#1).
2. Lowercasing (#4) and `lambda_q` (#5): one config tweak each.
3. Longer training (#3).
4. Self-mined negatives (#2).
5. Architecture work (#6, #10) only after these.
