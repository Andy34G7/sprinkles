"""Unified General (BEIR) + Code (CoIR) sparse-retrieval evaluation suite.

Supports:
  General (BEIR):
    beir:scifact, beir:nfcorpus, beir:arguana, beir:fiqa (or any beir:<dataset>)
  Code & Technical (CoIR):
    coir:stackoverflow-qa, coir:cosqa, coir:codesearchnet-<lang> (python/go/java/javascript/php/ruby)

Suites (--suite):
  core     beir:scifact, beir:nfcorpus, beir:arguana, coir:stackoverflow-qa, coir:cosqa
  full     core + beir:fiqa, coir:codesearchnet-python
  general  beir:scifact, beir:nfcorpus, beir:arguana, beir:fiqa
  code     coir:stackoverflow-qa, coir:cosqa, coir:codesearchnet-python

Model specs (repeatable --model):
  bm25                    Okapi BM25 (bm25s, k1=1.5, b=0.75)
  tfidf                   sublinear TF-IDF with l2 norm
  hf:<repo or path>       SentenceTransformers SparseEncoder (SPLADE-v3, splade-code-06B, Ettin, etc.)
  crouton:<ckpt.pt>       CroutonSPLADE checkpoint (~/projects/crouton)
  shortcake:<ckpt.pt>     ShortcakeSPLADE checkpoint (~/projects/shortcake)

Examples:
  .venv/bin/python eval_suite.py --suite core --model bm25 --model tfidf --latency
  .venv/bin/python eval_suite.py --task beir:scifact --task coir:stackoverflow-qa --model hf:naver/splade-v3
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
from typing import Any

import numpy as np
import scipy.sparse as sp
import torch

HOME = Path.home()
CROUTON_SRC = HOME / "projects/crouton/src"
DEFAULT_CROUTON_TOK = HOME / "projects/crouton/artifacts/tokenizer/wp_bpe_ppig_32k.json"
LERNEA_PY = HOME / "projects/tokenisers/python"
SHORTCAKE_DIR = HOME / "projects/shortcake"
DEFAULT_SHORTCAKE_TOK = SHORTCAKE_DIR / "tokenizer.json"
RESULTS = Path(__file__).resolve().parent.parent / "results"

SUITES: dict[str, list[str]] = {
    "core": [
        "beir:scifact",
        "beir:nfcorpus",
        "beir:arguana",
        "coir:stackoverflow-qa",
        "coir:cosqa",
    ],
    "full": [
        "beir:scifact",
        "beir:nfcorpus",
        "beir:arguana",
        "beir:fiqa",
        "coir:stackoverflow-qa",
        "coir:cosqa",
        "coir:codesearchnet-python",
    ],
    "general": ["beir:scifact", "beir:nfcorpus", "beir:arguana", "beir:fiqa"],
    "code": ["coir:stackoverflow-qa", "coir:cosqa", "coir:codesearchnet-python"],
}

METRICS = ["recall@5", "recall@10", "recall@100", "mrr@10", "ndcg@10", "precision@10"]


# ---------------------------------------------------------------- data loaders

def _doc_text(row: dict[str, Any]) -> str:
    return ((row.get("title") or "") + " " + (row.get("text") or "")).strip()


def _parse_qrels(qrels_ds, queries_all: dict[str, str], corpus: dict[str, str]) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    for r in qrels_ds:
        qid = str(r.get("query_id") if r.get("query_id") is not None else r.get("query-id"))
        cid = str(r.get("corpus_id") if r.get("corpus_id") is not None else r.get("corpus-id"))
        s = int(r.get("score", 1))
        if s >= 1 and qid in queries_all and cid in corpus:
            qrels.setdefault(qid, {})[cid] = s
    return qrels


def normalize_task(task: str) -> str:
    if ":" in task:
        return task
    return f"coir:{task}"


def task_domain(task: str) -> str:
    prefix, _, _ = normalize_task(task).partition(":")
    return "general" if prefix == "beir" else "code"


def load_task(task: str, split: str, limit_queries: int, limit_corpus: int, seed: int = 0):
    from datasets import load_dataset

    task = normalize_task(task)
    family, _, name = task.partition(":")

    if family == "beir":
        corpus_ds = load_dataset(f"BeIR/{name}", "corpus", split="corpus")
        queries_ds = load_dataset(f"BeIR/{name}", "queries", split="queries")
        qrels_ds = load_dataset(f"BeIR/{name}-qrels", split=split)
        exclude_self = name == "arguana"  # BEIR arguana includes query texts in corpus
    elif family == "coir":
        exclude_self = False
        if name.startswith("codesearchnet-"):
            lang = name.split("-", 1)[1]
            corpus_ds = load_dataset("CoIR-Retrieval/codesearchnet", f"{lang}-corpus", split="corpus")
            queries_ds = load_dataset("CoIR-Retrieval/codesearchnet", f"{lang}-queries", split="queries")
            qrels_ds = load_dataset("CoIR-Retrieval/codesearchnet", f"{lang}-qrels", split=split)
        elif name == "cosqa":
            corpus_ds = load_dataset("CoIR-Retrieval/cosqa", "corpus", split="corpus")
            queries_ds = load_dataset("CoIR-Retrieval/cosqa", "queries", split="queries")
            qrels_ds = load_dataset("CoIR-Retrieval/cosqa-qrels", split=split)
        else:
            cq = load_dataset(f"CoIR-Retrieval/{name}-queries-corpus")
            corpus_ds = cq["corpus"]
            queries_ds = cq["queries"]
            qrels_ds = load_dataset(f"CoIR-Retrieval/{name}-qrels", split=split)
    else:
        raise ValueError(f"Unknown task family {family!r} in {task!r}; expected 'beir:<name>' or 'coir:<name>'")

    corpus = {str(r["_id"]): _doc_text(r) for r in corpus_ds}
    queries_all = {str(r["_id"]): str(r["text"]) for r in queries_ds}
    qrels = _parse_qrels(qrels_ds, queries_all, corpus)

    qids = sorted(qrels)
    if limit_queries:
        qids = sorted(random.Random(seed).sample(qids, min(limit_queries, len(qids))))
        qrels = {q: qrels[q] for q in qids}
    if limit_corpus:
        keep = {d for q in qids for d in qrels[q]}
        rest = [d for d in corpus if d not in keep]
        keep |= set(random.Random(seed).sample(rest, max(0, min(limit_corpus - len(keep), len(rest)))))
        corpus = {d: corpus[d] for d in corpus if d in keep}

    return corpus, {q: queries_all[q] for q in qids}, qrels, exclude_self


# ---------------------------------------------------------------- encoders

def dense_to_csr(x: torch.Tensor) -> sp.csr_matrix:
    x = x.float().cpu()
    nz = x.nonzero(as_tuple=True)
    return sp.csr_matrix((x[nz].numpy(), (nz[0].numpy(), nz[1].numpy())), shape=tuple(x.shape))


def batched(texts: list[str], bs: int):
    for i in range(0, len(texts), bs):
        yield texts[i : i + bs]


class HFSparse:
    def __init__(self, repo: str, device: str, max_len: int, bs: int):
        from sentence_transformers import SparseEncoder

        self.m = SparseEncoder(os.path.expanduser(repo), device=device, trust_remote_code=True)
        self.m.max_seq_length = max_len
        self.bs, self.name = bs, f"hf:{repo}"
        self.params = sum(p.numel() for p in self.m.parameters())

    def encode(self, texts: list[str], is_query: bool) -> sp.csr_matrix:
        fn = self.m.encode_query if is_query else self.m.encode_document
        out = fn(texts, batch_size=self.bs, convert_to_sparse_tensor=True, show_progress_bar=False)
        out = out.coalesce().cpu()
        i, v = out.indices().numpy(), out.values().float().numpy()
        return sp.csr_matrix((v, (i[0], i[1])), shape=tuple(out.shape))


class Crouton:
    def __init__(self, ckpt: str, device: str, max_len: int, bs: int, tokenizer_path: Path = DEFAULT_CROUTON_TOK):
        sys.path[:0] = [str(CROUTON_SRC), str(LERNEA_PY)]
        from tokenisers import WpBpePpigTokenizer
        from crouton.eval.eval_splade import encode_texts_splade, load_splade_model

        self.dev = torch.device(device)
        self.m, _ = load_splade_model(os.path.expanduser(ckpt), device=self.dev)
        self.tok = WpBpePpigTokenizer.from_pretrained(str(tokenizer_path))
        self.enc, self.max_len, self.bs = encode_texts_splade, max_len, bs
        self.name, self.params = f"crouton:{ckpt}", sum(p.numel() for p in self.m.parameters())

    def encode(self, texts: list[str], is_query: bool) -> sp.csr_matrix:
        return sp.vstack(
            [
                dense_to_csr(self.enc(self.m, self.tok, b, device=self.dev, max_len=self.max_len, batch_size=self.bs))
                for b in batched(texts, self.bs * 8)
            ]
        ).tocsr()


class Shortcake:
    def __init__(self, ckpt: str, device: str, max_len: int, bs: int, tokenizer_path: Path = DEFAULT_SHORTCAKE_TOK):
        sys.path.insert(0, str(SHORTCAKE_DIR))
        from tokenizers import Tokenizer
        from model import Shortcake as Backbone
        from splade_head import ShortcakeSPLADE
        from train_splade import DOC_SUFFIX, QUERY_SUFFIX, encode_texts

        c = torch.load(os.path.expanduser(ckpt), map_location="cpu", weights_only=False)
        self.tok = Tokenizer.from_file(str(tokenizer_path))
        pad = self.tok.token_to_id("<pad>")
        self.pad = 1 if pad is None else pad
        self.m = ShortcakeSPLADE(Backbone(c["config"]), tie_lm_head=True, pad_token_id=self.pad)
        self.m.load_state_dict(c["model_state_dict"])
        self.m.to(device).eval()
        use_suffix = c.get("use_suffix", True)
        self.echo = bool(c.get("echo", False))
        self.sfx = {True: QUERY_SUFFIX if use_suffix else "", False: DOC_SUFFIX if use_suffix else ""}
        self.enc, self.dev, self.max_len, self.bs = encode_texts, device, max_len, bs
        self.name, self.params = f"shortcake:{ckpt}", sum(p.numel() for p in self.m.parameters())

    @torch.no_grad()
    def encode(self, texts: list[str], is_query: bool) -> sp.csr_matrix:
        rows = []
        for b in batched(texts, self.bs):
            ids, mask = self.enc(self.tok, b, self.max_len, self.pad, suffix=self.sfx[is_query])
            if self.echo:
                # Duplicate each valid sequence and pool only over the second copy
                lens = mask.sum(dim=1).tolist()
                ml = max(int(n) * 2 for n in lens)
                e_ids = torch.full((len(lens), ml), self.pad, dtype=torch.long)
                e_mask = torch.zeros((len(lens), ml), dtype=torch.bool)
                for i, n in enumerate(lens):
                    n = int(n)
                    e_ids[i, :n] = ids[i, :n]
                    e_ids[i, n : 2 * n] = ids[i, :n]
                    e_mask[i, n : 2 * n] = True
                ids, mask = e_ids, e_mask
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.dev == "cuda"):
                rows.append(dense_to_csr(self.m(ids.to(self.dev), attention_mask=mask.to(self.dev))["sparse_vector"]))
        return sp.vstack(rows).tocsr()


class TfidfApprox:
    """Sublinear TF-IDF with l2 norm."""

    name, params = "tfidf", 0

    def fit(self, docs: list[str]) -> sp.csr_matrix:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.v = TfidfVectorizer(sublinear_tf=True, max_features=200_000, norm="l2")
        return self.v.fit_transform(docs).tocsr()

    def encode(self, texts: list[str], is_query: bool) -> sp.csr_matrix:
        return self.v.transform(texts).tocsr()


def bm25_run(corpus_texts: list[str], query_texts: list[str], k: int, dids: list[str], qids: list[str], exclude_self: bool):
    import bm25s

    r = bm25s.BM25(k1=1.5, b=0.75)
    r.index(bm25s.tokenize(corpus_texts, stopwords="en", show_progress=False), show_progress=False)
    fetch_k = min(len(corpus_texts), k + (1 if exclude_self else 0))
    idx, scores = r.retrieve(bm25s.tokenize(query_texts, stopwords="en", show_progress=False), k=fetch_k, show_progress=False)
    if not exclude_self:
        return idx, scores
    out_idx, out_s = [], []
    for i, qid in enumerate(qids):
        row_i, row_s = [], []
        for j, s in zip(idx[i], scores[i]):
            if dids[int(j)] == qid:
                continue
            row_i.append(int(j))
            row_s.append(float(s))
            if len(row_i) == k:
                break
        out_idx.append(row_i)
        out_s.append(row_s)
    return out_idx, out_s


# ---------------------------------------------------------------- retrieval & timing

def retrieve(D: sp.csr_matrix, Q: sp.csr_matrix, k: int, dids: list[str], qids: list[str], exclude_self: bool):
    S = (Q @ D.T).toarray() if Q.shape[0] * D.shape[0] <= 4e8 else None
    did_to_idx = {d: i for i, d in enumerate(dids)} if exclude_self else {}
    out_idx, out_s = [], []
    for i, qid in enumerate(qids):
        s = S[i].copy() if S is not None else (D @ Q[i].T).toarray().ravel()
        if exclude_self and qid in did_to_idx:
            s[did_to_idx[qid]] = -np.inf
        kk = min(k, len(s))
        top = np.argpartition(-s, kk - 1)[:kk]
        top = top[np.argsort(-s[top])]
        out_idx.append(top)
        out_s.append(s[top])
    return out_idx, out_s


def query_latency_ms(enc, queries: list[str], n: int = 100) -> float:
    qs = queries[:n]
    enc.encode(qs[:3], True)
    t = time.perf_counter()
    for q in qs:
        enc.encode([q], True)
    return 1000.0 * (time.perf_counter() - t) / len(qs)


def build_encoder(spec: str, device: str, max_len: int, bs: int, crouton_tok: Path, shortcake_tok: Path):
    kind, _, arg = spec.partition(":")
    if kind in ("bm25", "tfidf"):
        return None
    if kind == "hf":
        return HFSparse(arg, device, max_len, bs)
    if kind == "crouton":
        return Crouton(arg, device, max_len, bs, tokenizer_path=crouton_tok)
    if kind == "shortcake":
        return Shortcake(arg, device, max_len, bs, tokenizer_path=shortcake_tok)
    raise ValueError(f"Unknown model spec {spec!r}")


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True, help="Model spec (repeatable)")
    ap.add_argument("--suite", choices=sorted(SUITES), default=None, help="Named benchmark suite")
    ap.add_argument("--task", action="append", default=[], help="Individual task, e.g. beir:scifact, coir:stackoverflow-qa")
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--limit-queries", type=int, default=0)
    ap.add_argument("--limit-corpus", type=int, default=0)
    ap.add_argument("--latency", action="store_true", help="Time batch=1 query encoding (100 queries)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tag", default="", help="Optional suffix for result JSON filenames")
    ap.add_argument("--crouton-tokenizer", type=Path, default=DEFAULT_CROUTON_TOK)
    ap.add_argument("--shortcake-tokenizer", type=Path, default=DEFAULT_SHORTCAKE_TOK)
    a = ap.parse_args()

    from ranx import Qrels, Run, evaluate

    tasks: list[str] = []
    if a.suite:
        tasks.extend(SUITES[a.suite])
    for t in a.task:
        nt = normalize_task(t)
        if nt not in tasks:
            tasks.append(nt)
    if not tasks:
        tasks = ["coir:stackoverflow-qa"]

    # Preload datasets once so multiple models share them cleanly
    loaded_tasks = {}
    for task in tasks:
        corpus, queries, qrels, exclude_self = load_task(task, a.split, a.limit_queries, a.limit_corpus)
        dids, qids = list(corpus), list(queries)
        dtexts, qtexts = [corpus[d] for d in dids], [queries[q] for q in qids]
        loaded_tasks[task] = (dids, qids, dtexts, qtexts, qrels, exclude_self)
        print(f"[{task}/{a.split}] {len(qids)} queries x {len(dids)} docs (exclude_self={exclude_self})", flush=True)

    RESULTS.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []

    for spec in a.model:
        kind, _, _ = spec.partition(":")
        shared_enc = build_encoder(spec, a.device, a.max_len, a.batch_size, a.crouton_tokenizer, a.shortcake_tokenizer)
        model_scores: dict[str, float] = {}
        doc_terms_list: list[float] = []

        for task in tasks:
            dids, qids, dtexts, qtexts, qrels, exclude_self = loaded_tasks[task]
            t0 = time.time()
            rec: dict[str, Any] = {
                "model": spec,
                "task": task,
                "domain": task_domain(task),
                "split": a.split,
                "n_queries": len(qids),
                "n_docs": len(dids),
                "max_len": a.max_len,
                "device": a.device,
                "limited": bool(a.limit_queries or a.limit_corpus),
            }
            if kind == "bm25":
                idx, sc = bm25_run(dtexts, qtexts, a.top_k, dids, qids, exclude_self)
                enc = None
            else:
                if kind == "tfidf":
                    enc = TfidfApprox()
                    D = enc.fit(dtexts)
                else:
                    enc = shared_enc
                    D = enc.encode(dtexts, False)
                rec["index_build_s"] = round(time.time() - t0, 1)
                Q = enc.encode(qtexts, True)
                idx, sc = retrieve(D, Q, a.top_k, dids, qids, exclude_self)
                rec |= {
                    "params": enc.params,
                    "vocab_size": int(D.shape[1]),
                    "avg_doc_terms": float(D.nnz / D.shape[0]),
                    "avg_query_terms": float(Q.nnz / Q.shape[0]),
                    "index_nnz": int(D.nnz),
                }
                doc_terms_list.append(rec["avg_doc_terms"])

            run = {q: {dids[int(j)]: float(s) for j, s in zip(idx[i], sc[i])} for i, q in enumerate(qids)}
            rec["metrics"] = {k: float(v) for k, v in evaluate(Qrels(qrels), Run(run), METRICS, make_comparable=True).items()}
            if a.latency and enc is not None:
                rec["query_latency_ms_bs1"] = round(query_latency_ms(enc, qtexts), 2)
            rec["wall_s"] = round(time.time() - t0, 1)

            task_slug = task.replace(":", "_")
            model_slug = re.sub(r"[^A-Za-z0-9._-]+", "_", spec.replace(str(HOME), "~"))[-80:]
            out = RESULTS / f"{task_slug}__{model_slug}{('__' + a.tag) if a.tag else ''}.json"
            out.write_text(json.dumps(rec, indent=2))

            m = rec["metrics"]
            model_scores[task] = m["ndcg@10"]
            print(
                f"{task:26s} | {spec:42s} | nDCG@10={m['ndcg@10']:.4f} MRR@10={m['mrr@10']:.4f} "
                f"R@100={m['recall@100']:.4f} Q/D={rec.get('avg_query_terms', 0):.0f}/{rec.get('avg_doc_terms', 0):.0f} "
                f"-> {out.name}",
                flush=True,
            )

        gen_vals = [v for t, v in model_scores.items() if task_domain(t) == "general"]
        code_vals = [v for t, v in model_scores.items() if task_domain(t) == "code"]
        summary_rows.append(
            {
                "model": spec,
                "scores": model_scores,
                "general_avg": float(np.mean(gen_vals)) if gen_vals else None,
                "code_avg": float(np.mean(code_vals)) if code_vals else None,
                "overall_avg": float(np.mean(list(model_scores.values()))),
                "avg_doc_terms": float(np.mean(doc_terms_list)) if doc_terms_list else None,
            }
        )
        del shared_enc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if len(tasks) > 1 or len(a.model) > 1:
        print("\n=== Suite Summary (nDCG@10) ===")
        header = ["Model"] + tasks + ["General", "Code", "Overall", "Avg Doc Terms"]
        print("| " + " | ".join(header) + " |")
        print("| " + " | ".join(["---"] * len(header)) + " |")
        for row in summary_rows:
            cols = [row["model"]]
            for t in tasks:
                cols.append(f"{row['scores'][t]:.4f}")
            cols.append(f"{row['general_avg']:.4f}" if row["general_avg"] is not None else "—")
            cols.append(f"{row['code_avg']:.4f}" if row["code_avg"] is not None else "—")
            cols.append(f"{row['overall_avg']:.4f}")
            cols.append(f"{row['avg_doc_terms']:.0f}" if row["avg_doc_terms"] is not None else "—")
            print("| " + " | ".join(cols) + " |")


if __name__ == "__main__":
    main()
