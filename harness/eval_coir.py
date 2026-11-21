"""CoIR sparse-retrieval eval, protocol-matched to pes-innovation-lab/semantic-search-code-docs.

Matches the reference repo:
  - data: CoIR-Retrieval/<task>-queries-corpus + <task>-qrels, test split, full corpus
  - doc text: (title + " " + text).strip()   (MIM-4-Training/run_bm25_eval.py)
  - retrieval: exact sparse dot product over the whole corpus, top-100 per query
  - metrics: ranx recall@5/10/100, mrr@10, ndcg@10, precision@10   (scripts/evaluation/metrics.py)

Model specs (repeatable --model):
  hf:<repo or path>       SentenceTransformers SparseEncoder (reference repo's loader)
  crouton:<ckpt.pt>       CroutonSPLADE checkpoint (~/projects/crouton)
  shortcake:<ckpt.pt>     ShortcakeSPLADE checkpoint (~/projects/shortcake)
  tfidf                   sublinear TF-IDF with l2 norm
  bm25                    Okapi BM25 (bm25s, k1=1.5, b=0.75)

Example:
  .venv/bin/python eval_coir.py --model bm25 --model hf:Cdn13/ettin-splade-stackoverflowqa \
      --model crouton:~/projects/crouton/checkpoints/splade/splade_final.pt
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

HOME = Path.home()
CROUTON_SRC = HOME / "projects/crouton/src"
CROUTON_TOKENIZER = HOME / "projects/crouton/artifacts/tokenizer/wp_bpe_ppig_32k.json"
LERNEA_PY = HOME / "projects/tokenisers/python"
SHORTCAKE_DIR = HOME / "projects/shortcake"
RESULTS = Path(__file__).resolve().parent.parent / "results"


# ---------------------------------------------------------------- data

def load_coir(task: str, split: str, limit_queries: int, limit_corpus: int, seed: int = 0):
    from datasets import load_dataset

    cq = load_dataset(f"CoIR-Retrieval/{task}-queries-corpus")
    qrels_ds = load_dataset(f"CoIR-Retrieval/{task}-qrels", split=split)
    corpus = {r["_id"]: ((r.get("title") or "") + " " + (r.get("text") or "")).strip() for r in cq["corpus"]}
    queries_all = {r["_id"]: r["text"] for r in cq["queries"]}

    qrels: dict[str, dict[str, int]] = {}
    for r in qrels_ds:
        qid, cid, s = str(r["query_id"]), str(r["corpus_id"]), int(r.get("score", 1))
        if s >= 1 and qid in queries_all and cid in corpus:
            qrels.setdefault(qid, {})[cid] = s
    qids = sorted(qrels)
    if limit_queries:
        qids = sorted(random.Random(seed).sample(qids, min(limit_queries, len(qids))))
        qrels = {q: qrels[q] for q in qids}
    if limit_corpus:  # smoke tests only: keep every relevant doc, then fill
        keep = {d for q in qids for d in qrels[q]}
        rest = [d for d in corpus if d not in keep]
        keep |= set(random.Random(seed).sample(rest, max(0, min(limit_corpus - len(keep), len(rest)))))
        corpus = {d: corpus[d] for d in corpus if d in keep}
    return corpus, {q: queries_all[q] for q in qids}, qrels


# ---------------------------------------------------------------- encoders
# Each encoder: .encode(texts, is_query) -> scipy CSR (n, vocab); .params; .name

def dense_to_csr(x: torch.Tensor) -> sp.csr_matrix:
    x = x.float().cpu()
    nz = x.nonzero(as_tuple=True)
    return sp.csr_matrix((x[nz].numpy(), (nz[0].numpy(), nz[1].numpy())), shape=tuple(x.shape))


def batched(texts, bs):
    for i in range(0, len(texts), bs):
        yield texts[i : i + bs]


class HFSparse:
    def __init__(self, repo, device, max_len, bs):
        from sentence_transformers import SparseEncoder

        self.m = SparseEncoder(os.path.expanduser(repo), device=device, trust_remote_code=True)
        self.m.max_seq_length = max_len
        self.bs, self.name = bs, f"hf:{repo}"
        self.params = sum(p.numel() for p in self.m.parameters())

    def encode(self, texts, is_query):
        fn = self.m.encode_query if is_query else self.m.encode_document  # model-specific prompts if any
        out = fn(texts, batch_size=self.bs, convert_to_sparse_tensor=True, show_progress_bar=False)
        out = out.coalesce().cpu()
        i, v = out.indices().numpy(), out.values().float().numpy()
        return sp.csr_matrix((v, (i[0], i[1])), shape=tuple(out.shape))


class Crouton:
    def __init__(self, ckpt, device, max_len, bs):
        sys.path[:0] = [str(CROUTON_SRC), str(LERNEA_PY)]
        from tokenisers import WpBpePpigTokenizer
        from crouton.eval.eval_splade import encode_texts_splade, load_splade_model

        self.dev = torch.device(device)
        self.m, _ = load_splade_model(os.path.expanduser(ckpt), device=self.dev)
        self.tok = WpBpePpigTokenizer.from_pretrained(str(CROUTON_TOKENIZER))
        self.enc, self.max_len, self.bs = encode_texts_splade, max_len, bs
        self.name, self.params = f"crouton:{ckpt}", sum(p.numel() for p in self.m.parameters())

    def encode(self, texts, is_query):
        return sp.vstack([dense_to_csr(self.enc(self.m, self.tok, b, device=self.dev, max_len=self.max_len, batch_size=self.bs))
                          for b in batched(texts, self.bs * 8)]).tocsr()


class Shortcake:
    def __init__(self, ckpt, device, max_len, bs):
        sys.path.insert(0, str(SHORTCAKE_DIR))
        from tokenizers import Tokenizer
        from model import Shortcake as Backbone
        from splade_head import ShortcakeSPLADE
        from train_splade import DOC_SUFFIX, QUERY_SUFFIX, encode_texts

        c = torch.load(os.path.expanduser(ckpt), map_location="cpu", weights_only=False)
        self.tok = Tokenizer.from_file(str(SHORTCAKE_DIR / "tokenizer.json"))
        pad = self.tok.token_to_id("<pad>")
        self.pad = 1 if pad is None else pad
        self.m = ShortcakeSPLADE(Backbone(c["config"]), tie_lm_head=True, pad_token_id=self.pad)
        self.m.load_state_dict(c["model_state_dict"])
        self.m.to(device).eval()
        use_suffix = c.get("use_suffix", True)
        self.sfx = {True: QUERY_SUFFIX if use_suffix else "", False: DOC_SUFFIX if use_suffix else ""}
        self.enc, self.dev, self.max_len, self.bs = encode_texts, device, max_len, bs
        self.name, self.params = f"shortcake:{ckpt}", sum(p.numel() for p in self.m.parameters())

    @torch.no_grad()
    def encode(self, texts, is_query):
        rows = []
        for b in batched(texts, self.bs):
            ids, mask = self.enc(self.tok, b, self.max_len, self.pad, suffix=self.sfx[is_query])
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.dev == "cuda"):
                rows.append(dense_to_csr(self.m(ids.to(self.dev), attention_mask=mask.to(self.dev))["sparse_vector"]))
        return sp.vstack(rows).tocsr()


class TfidfApprox:
    """Sublinear TF-IDF with l2 norm (matching MIM-4-Training/run_bm25_eval.py)."""
    name, params = "tfidf", 0

    def fit(self, docs):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.v = TfidfVectorizer(sublinear_tf=True, max_features=200_000, norm="l2")
        return self.v.fit_transform(docs).tocsr()

    def encode(self, texts, is_query):
        return self.v.transform(texts).tocsr()


def bm25_run(corpus_texts, query_texts, k):
    import bm25s

    r = bm25s.BM25(k1=1.5, b=0.75)
    r.index(bm25s.tokenize(corpus_texts, stopwords="en", show_progress=False), show_progress=False)
    idx, scores = r.retrieve(bm25s.tokenize(query_texts, stopwords="en", show_progress=False), k=k, show_progress=False)
    return idx, scores


# ---------------------------------------------------------------- eval

def retrieve(D: sp.csr_matrix, Q: sp.csr_matrix, k: int):
    S = (Q @ D.T).toarray() if Q.shape[0] * D.shape[0] <= 4e8 else None
    out_idx, out_s = [], []
    for i in range(Q.shape[0]):
        s = S[i] if S is not None else (D @ Q[i].T).toarray().ravel()
        top = np.argpartition(-s, min(k, len(s)) - 1)[:k]
        top = top[np.argsort(-s[top])]
        out_idx.append(top)
        out_s.append(s[top])
    return out_idx, out_s


def query_latency_ms(enc, queries, n=100):
    """Single-query (batch=1) encode latency on whatever device the encoder sits on."""
    qs = queries[:n]
    enc.encode(qs[:3], True)  # warm-up
    t = time.perf_counter()
    for q in qs:
        enc.encode([q], True)
    return 1000 * (time.perf_counter() - t) / len(qs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--task", default="stackoverflow-qa", help="CoIR task prefix, e.g. stackoverflow-qa, codesearchnet")
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-len", type=int, default=256, help="reference scripts/evaluation uses 256")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--limit-queries", type=int, default=0)
    ap.add_argument("--limit-corpus", type=int, default=0)
    ap.add_argument("--latency", action="store_true", help="also time batch=1 query encoding (100 queries)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tag", default="", help="suffix for the results file name")
    a = ap.parse_args()

    from ranx import Qrels, Run, evaluate

    corpus, queries, qrels = load_coir(a.task, a.split, a.limit_queries, a.limit_corpus)
    dids, qids = list(corpus), list(queries)
    dtexts, qtexts = [corpus[d] for d in dids], [queries[q] for q in qids]
    print(f"[{a.task}/{a.split}] {len(qids)} queries x {len(dids)} docs on {a.device}", flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    metrics = ["recall@5", "recall@10", "recall@100", "mrr@10", "ndcg@10", "precision@10"]

    for spec in a.model:
        kind, _, arg = spec.partition(":")
        t0 = time.time()
        rec = {"model": spec, "task": a.task, "split": a.split, "n_queries": len(qids), "n_docs": len(dids),
               "max_len": a.max_len, "device": a.device, "limited": bool(a.limit_queries or a.limit_corpus)}
        if kind == "bm25":
            idx, sc = bm25_run(dtexts, qtexts, a.top_k)
            enc = None
        else:
            if kind == "tfidf":
                enc = TfidfApprox()
                D = enc.fit(dtexts)
            else:
                enc = {"hf": HFSparse, "crouton": Crouton, "shortcake": Shortcake}[kind](arg, a.device, a.max_len, a.batch_size)
                D = enc.encode(dtexts, False)
            rec["index_build_s"] = round(time.time() - t0, 1)
            Q = enc.encode(qtexts, True)
            idx, sc = retrieve(D, Q, a.top_k)
            rec |= {"params": enc.params, "avg_doc_terms": D.nnz / D.shape[0], "avg_query_terms": Q.nnz / Q.shape[0],
                    "index_nnz": int(D.nnz)}
        run = {q: {dids[j]: float(s) for j, s in zip(idx[i], sc[i])} for i, q in enumerate(qids)}
        rec["metrics"] = {k: float(v) for k, v in evaluate(Qrels(qrels), Run(run), metrics, make_comparable=True).items()}
        if a.latency and enc is not None:
            rec["query_latency_ms_bs1"] = query_latency_ms(enc, qtexts)
        rec["wall_s"] = round(time.time() - t0, 1)

        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", spec.replace(str(HOME), "~"))[-80:]
        out = RESULTS / f"{a.task}__{slug}{('__' + a.tag) if a.tag else ''}.json"
        out.write_text(json.dumps(rec, indent=2))
        m = rec["metrics"]
        print(f"{spec:60s} nDCG@10={m['ndcg@10']:.4f} MRR@10={m['mrr@10']:.4f} R@100={m['recall@100']:.4f} "
              f"terms Q/D={rec.get('avg_query_terms', 0):.0f}/{rec.get('avg_doc_terms', 0):.0f} -> {out.name}", flush=True)
        del enc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
