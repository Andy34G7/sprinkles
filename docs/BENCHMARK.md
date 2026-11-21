# Comparison target: pes-innovation-lab/semantic-search-code-docs ("InferSPLADE")

Repo: https://github.com/pes-innovation-lab/semantic-search-code-docs (commit `2a99980`).

## What the project is

- **Goal:** semantic search for static documentation sites (Sphinx, Doxygen, MkDocs, Docusaurus), fully in
  the browser, with no vector DB and no API. SPLADE sparse vectors drop into the inverted index the docs
  site already ships.
- **Two research tracks** (`MIM-Slides/Semantic-Search-Project-Brief.pdf`):
  1. Tiny code-SPLADE models that run client-side, built on Ettin backbones of 17M / 32M / 68M params,
     vs 600M+ teachers.
  2. Inference-free or lightweight query encoders: the heavy model runs only at build time. The GK paper,
     `GK-Paper-Replication-Trying/reneuir2026_paper_13.pdf`, "Exploring Lightweight Query Encoders for
     Fixed Sparse Document Indexes", freezes the document index and distils only the query encoder,
     comparing layer-pruned and static token-lookup students.
- **Evaluation:** CoIR as a proxy for doc search, comparing retrieval quality and latency (dense vs
  static, GPU vs CPU, at both build and query time).

## Benchmark protocol

- **Primary task:** CoIR StackOverflow-QA.
  - Corpus: `CoIR-Retrieval/stackoverflow-qa-queries-corpus`, 19,931 docs.
  - Test qrels: `CoIR-Retrieval/stackoverflow-qa-qrels`, 1,994 queries. The train split has 13,951 qrels
    (training data).
- **Secondary task:** `CoIR-Retrieval/codesearchnet` (NL to code).
- **Doc text:** `(title + " " + text).strip()`.
- **Retrieval:** exact sparse dot product over the full corpus, top-100 per query.
- **Metrics:** via `ranx`: recall@5/10/100, mrr@10, **ndcg@10** (headline), precision@10.
- **Max length:** 256 for evaluation (always state `max_len` alongside numbers).
- **Lexical baselines:** our harness reports both sublinear TF-IDF with l2 norm (`tfidf`) and Okapi BM25
  (`bm25`, `k1=1.5, b=0.75`).
- **Evaluation notes:**
  - Report zero-shot and fine-tuned models separately (in StackOverflow-QA, each corpus doc answers one
    query across train/val/test).
  - Always report sparsity (`avg_query_terms`, `avg_doc_terms`, `index_nnz`) alongside nDCG@10 so
    browser-shipped index sizes are compared fairly.

## Reference numbers

| System | Params | StackOverflow-QA nDCG@10 | Avg Q/D terms | Source |
|---|---|---|---|---|
| `Cdn13/ettin-splade-stackoverflowqa` (HF Hub checkpoint) | 17M | **0.7417** (MRR@10 0.7083, R@100 0.9468) | 1348 / 1397 | `MIM-4-Training/run_bm25_eval.py` |
| `Cdn13/ettin-splade-stackoverflowqa` (earlier run) | 17M | 0.656 | — | `MIM-4-Training/commands.md` |
| Same recipe, TF-IDF hard negatives | 17M | 0.333 | — | `MIM-4-Training/commands.md` |
| `naver/splade-code-06B` (teacher) | ~600M | **0.9208** (MRR@10 0.9059, R@100 0.9885) | 1538 / 1668 | `MIM-4-Training/run_bm25_eval.py` |
| BM25 / TF-IDF | 0 | — | — | measured via `harness/eval_coir.py` |

We re-measure every reference system with `harness/eval_coir.py` so all rows share one protocol.

### Reference training recipe (`MIM-4-Training/train_mvp.py`)

- MarginMSE distillation from `naver/splade-code-06B`; skip triplets with `|teacher margin| < 0.5`.
- Random negatives (3 per query) beat TF-IDF hard negatives: the teacher's margins on those were too
  small for a 17M student.
- FLOPS reg with `lambda_q=0.10`, `lambda_d=0.005`, ramped up quadratically over the first 33% of steps.
- lr 1e-4, batch 16, grad accumulation 2, max_len 128, 50k triplets, 3 to 5 epochs.
- About 93% sparse index; ~21 min per epoch on an A10G (Modal).

## What we compare (shortcake, crouton)

Report for every system, on the same protocol:
- Quality: nDCG@10 (headline), MRR@10, R@100.
- Index size: avg doc terms and total nnz.
- Query cost: avg query terms, params, batch-1 query-encode latency (CPU first; browser via
  Transformers.js / ONNX later).
- Build cost: index build time.

Rows:

| Group | Systems |
|---|---|
| Baselines | `bm25`, `tfidf` (their approximation) |
| Reference | `hf:Cdn13/ettin-splade-stackoverflowqa` (target: 0.656), `hf:naver/splade-code-06B` (teacher ceiling), `hf:naver/splade-v3` (general SPLADE) |
| Ours, current | `shortcake:~/projects/shortcake/checkpoints_splade/best_splade_model.pt` (trained on CodeSearchNet docstring to code), `crouton:~/projects/crouton/checkpoints/splade/splade_final.pt` (trained on MS MARCO) |
| Ours, next | trained on the StackOverflow-QA *train* qrels with splade-code-06B MarginMSE (their recipe), then with the improvements in `README.md` |

### Planned experiments, in order

1. Baselines plus reference rows on the full test set. Confirm we land near their 0.656 for the Ettin
   model, which validates the harness.
2. Zero-shot rows for current shortcake and crouton.
3. Distil shortcake and crouton with their recipe on StackOverflow-QA train (same teacher, same
   negatives). This gives a like-for-like comparison against Ettin-17M; shortcake and crouton tiny/small
   sizes compare directly with 17M.
4. Apply `README.md` improvements: self-mined negatives, query-reg tuning, crouton MNTP/MLM adaptation,
   shortcake echo embeddings.
5. Inference-free / frozen-index track: keep the splade-code-06B document index fixed, distil a tiny
   shortcake or crouton query encoder (GK-paper setup). This is where small causal models could matter.
6. Latency: batch-1 CPU now; browser runtime later via their `Experiments/coir-js-latency-eval`
   (Transformers.js, needs ONNX export).

### Open issues to handle before step 3

- Crouton's tokenizer fragments code terms (`tokenisation.md`). That matters more on StackOverflow than
  on SciFact.
- splade-code-06B is decoder-based (Qwen). The harness loads it through `SparseEncoder` with
  `trust_remote_code`, as the reference does. It needs a GPU; on CPU, encoding 19,931 docs at 600M
  params is impractical.

## Harness

```
cd ~/Documents/research/splade-improvements/harness
uv venv --python 3.13 .venv && uv pip install --python .venv/bin/python -r requirements.txt
# smoke test
.venv/bin/python eval_coir.py --model bm25 --model tfidf --limit-queries 50 --limit-corpus 2000
# full
.venv/bin/python eval_coir.py --model bm25 --model tfidf --model hf:Cdn13/ettin-splade-stackoverflowqa --latency
```

- Results are written as JSON to `results/<task>__<model>.json`.
- Python 3.13 is required: lernea's `_core` is a cp313 extension.
- On this 15 GB machine, run inside `systemd-run --user --scope -p MemoryMax=4G`.
- Local venv uses the CPU torch wheel: `uv pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu`
  first, then the rest. On a GPU box, use the default wheel.
- Status 2026-11-21: smoke-tested on 50 queries x 2,000 docs (CPU), all 5 model kinds working. Results
  are in `results/*__smoke.json`; the pool is tiny, so absolute numbers are inflated.

  | Model | smoke nDCG@10 | Q/D terms | CPU bs=1 query ms |
  |---|---|---|---|
  | bm25 | 0.826 | n/a | n/a |
  | tfidf | 0.853 | 68/78 | 0.4 |
  | ettin-splade-stackoverflowqa | 0.898 | 1340/1395 | 89 |
  | shortcake (CodeSearchNet-trained) | 0.727 | 121/124 | 318 (CPU fallback scan, no Triton) |
  | crouton (MS MARCO-trained) | 0.619 | 148/206 | 158 |

- Full runs: do splade-code-06B on a GPU (Colab L4/A100). Full-corpus CPU runs of the small models take
  roughly 10 x the smoke time.
