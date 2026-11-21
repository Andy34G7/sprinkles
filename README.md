# exp-retrieval-splade

Document retrieval and sparse neural retrieval (SPLADE) benchmark and distillation experiments, comparing custom small language model backbones ([`crouton`](../crouton), [`shortcake`](../shortcake)) against published SPLADE models (`naver/splade-v3`, `naver/splade-code-06B`, `Cdn13/ettin-splade-stackoverflowqa`) and lexical baselines (BM25, sublinear TF-IDF).

## Structure

- [`harness/eval_suite.py`](harness/eval_suite.py) — unified General (BEIR) + Code (CoIR) sparse-retrieval evaluation suite (`bm25`, `tfidf`, Hugging Face `SparseEncoder`, `crouton`, and `shortcake` checkpoints), reporting nDCG@10, MRR@10, Recall@5/10/100, active query/doc terms, index `nnz`, and batch-1 query latency.
- [`harness/summarize.py`](harness/summarize.py) — builds a Markdown scoreboard from `results/*.json` with General / Code / Overall macro-averages and sparsity.
- [`harness/eval_coir.py`](harness/eval_coir.py) — original single-task CoIR runner (superseded by `eval_suite.py`).
- [`docs/BENCHMARK.md`](docs/BENCHMARK.md) — comparison protocol (CoIR StackOverflow-QA, matched to `pes-innovation-lab/semantic-search-code-docs`), reference numbers, and experiment roadmap.
- [`docs/IMPROVEMENTS.md`](docs/IMPROVEMENTS.md) — initial BEIR SciFact and CodeSearchNet baseline measurements and ranked SPLADE improvement plan for `crouton` and `shortcake`.
- [`docs/TOKENISATION.md`](docs/TOKENISATION.md) — investigation of `wp-bpe-ppig` vs. frequency BPE on prose and 70/30 prose+code corpora.
- [`tokeniser-exp/`](tokeniser-exp/) — reproducible scripts and results for the tokenizer ablation experiments (`tokexp.py`, `exp_e.py`, `results.json`).
- [`results/`](results/) — JSON evaluation outputs.

## Benchmark Suites

| Suite | Tasks | Coverage |
|---|---|---|
| `core` | `beir:scifact`, `beir:nfcorpus`, `beir:arguana`, `coir:stackoverflow-qa`, `coir:cosqa` | 3 General + 2 Code (fast default) |
| `full` | `core` + `beir:fiqa`, `coir:codesearchnet-python` | 4 General + 3 Code |
| `general` | `beir:scifact`, `beir:nfcorpus`, `beir:arguana`, `beir:fiqa` | General-domain BEIR |
| `code` | `coir:stackoverflow-qa`, `coir:cosqa`, `coir:codesearchnet-python` | Code & technical CoIR |

## Quickstart

Requires Python 3.13 (for `tokenisers`'s `cp313` extension) and sibling checkouts of `../crouton`, `../shortcake`, and `../tokenisers` when evaluating local checkpoints.

```bash
cd harness
uv venv --python 3.13 .venv
# Local CPU machine (install CPU torch first to avoid large CUDA wheels):
uv pip install --python .venv/bin/python torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python -r requirements.txt

# Smoke test across the full core suite (3 general + 2 code)
.venv/bin/python eval_suite.py --suite core --model bm25 --model tfidf --limit-queries 50 --limit-corpus 2000 --tag smoke

# Full core suite lexical baselines
.venv/bin/python eval_suite.py --suite core --model bm25 --model tfidf --latency

# Print Markdown leaderboard from results/
.venv/bin/python summarize.py
```
