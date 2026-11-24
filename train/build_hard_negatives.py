"""Build a multi-negative BM25 hard-negative JSONL for SPLADE v5 KL-div / Margin-MSE distillation.

Sources (strictly train-split only, zero test-split leakage):
  - General: sentence-transformers/msmarco-bm25 (triplet, train — grouped by query+positive)
  - Code QA: CoIR-Retrieval/stackoverflow-qa (train qrels + BM25 top-K mined hard negatives)
  - Code NL->Code: CoIR-Retrieval/codesearchnet (python train qrels, docstrings stripped + BM25 hard negatives)

Output JSONL schema:
  {"query": str, "pos": str, "negs": [str, ...], "source": "msmarco" | "stackoverflow-qa" | "codesearchnet"}
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path


def _strip_python_docstrings(code: str, query: str = "") -> str:
    cleaned = code.replace(query, "") if query else code
    cleaned = re.sub(r'"""[\s\S]*?"""', "", cleaned)
    cleaned = re.sub(r"'''[\s\S]*?'''", "", cleaned)
    return cleaned.strip()


def _doc_text(row: dict) -> str:
    return ((row.get("title") or "") + " " + (row.get("text") or "")).strip()


def load_msmarco_hard(n_queries: int, num_negs: int, rng: random.Random) -> list[dict]:
    if n_queries <= 0:
        return []
    from datasets import load_dataset

    print(f"Loading MS MARCO BM25 hard negatives (target {n_queries:,} queries x {num_negs} negs)...", flush=True)
    ds = load_dataset("sentence-transformers/msmarco-bm25", name="triplet", split="train", streaming=True)
    grouped: dict[tuple[str, str], list[str]] = {}
    neg_pool: list[str] = []
    scan_limit = n_queries * max(num_negs + 2, 6)
    seen = 0
    for r in ds:
        q = (r.get("query") or "").strip()
        p = (r.get("positive") or "").strip()
        neg = (r.get("negative") or "").strip()
        if not (q and p and neg):
            continue
        seen += 1
        key = (q, p)
        if key in grouped:
            if len(grouped[key]) < num_negs and neg not in grouped[key]:
                grouped[key].append(neg)
        elif len(grouped) < n_queries:
            grouped[key] = [neg]
        if len(neg_pool) < 50_000:
            neg_pool.append(neg)
        if len(grouped) >= n_queries and seen >= scan_limit:
            break

    out: list[dict] = []
    for (q, p), negs in grouped.items():
        # Top up with random in-domain MS MARCO passages if a query appeared < num_negs times
        while len(negs) < num_negs and neg_pool:
            cand = rng.choice(neg_pool)
            if cand != p and cand not in negs:
                negs.append(cand)
        if len(negs) >= num_negs:
            out.append({"query": q, "pos": p, "negs": negs[:num_negs], "source": "msmarco"})
    print(f"  MS MARCO ready: {len(out):,} queries (avg mined negs before top-up: {seen / max(1, len(grouped)):.1f})", flush=True)
    return out


def _mine_coir_bm25_hard(
    corpus: dict[str, str],
    queries: dict[str, str],
    qrels_ds,
    source_name: str,
    n_queries: int,
    num_negs: int,
    bm25_top_k: int,
    rng: random.Random,
) -> list[dict]:
    import bm25s

    pos_map: dict[str, list[str]] = {}
    for r in qrels_ds:
        qid = str(r.get("query_id") if r.get("query_id") is not None else r.get("query-id"))
        cid = str(r.get("corpus_id") if r.get("corpus_id") is not None else r.get("corpus-id"))
        if int(r.get("score", 1)) >= 1 and qid in queries and cid in corpus:
            pos_map.setdefault(qid, []).append(cid)

    qids = [qid for qid in pos_map if queries[qid].strip()]
    rng.shuffle(qids)
    qids = qids[:n_queries]
    if not qids:
        return []

    print(f"Indexing {len(corpus):,} {source_name} docs with BM25S for hard-negative mining...", flush=True)
    doc_ids = list(corpus.keys())
    doc_texts = [corpus[d] for d in doc_ids]
    corpus_tokens = bm25s.tokenize(doc_texts, show_progress=False)
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens, show_progress=False)

    q_texts = [queries[qid].strip() for qid in qids]
    q_tokens = bm25s.tokenize(q_texts, show_progress=False)
    k = min(bm25_top_k, len(doc_ids))
    results, _ = retriever.retrieve(q_tokens, k=k, show_progress=False)

    out: list[dict] = []
    for idx, qid in enumerate(qids):
        pos_ids = set(pos_map[qid])
        pid = pos_map[qid][0]
        ptext = corpus[pid].strip()
        if not ptext:
            continue
        # Collect hard negatives from BM25 top-k excluding any positive
        hard_ids = [doc_ids[int(j)] for j in results[idx] if doc_ids[int(j)] not in pos_ids]
        # Sample from top-k hard negatives (preferring top half for genuine confusers)
        pool = hard_ids[: max(num_negs * 3, 20)]
        chosen_ids = rng.sample(pool, min(num_negs, len(pool))) if len(pool) >= num_negs else list(pool)
        negs = [corpus[nid].strip() for nid in chosen_ids if corpus[nid].strip()]
        while len(negs) < num_negs:
            nid = rng.choice(doc_ids)
            if nid not in pos_ids and corpus[nid].strip() and corpus[nid].strip() not in negs:
                negs.append(corpus[nid].strip())
        out.append({"query": q_texts[idx], "pos": ptext, "negs": negs[:num_negs], "source": source_name})

    print(f"  {source_name} ready: {len(out):,} queries x {num_negs} BM25 hard negs", flush=True)
    return out


def load_soqa_hard(n_queries: int, num_negs: int, bm25_top_k: int, rng: random.Random) -> list[dict]:
    if n_queries <= 0:
        return []
    from datasets import load_dataset

    cq = load_dataset("CoIR-Retrieval/stackoverflow-qa-queries-corpus")
    qrels = load_dataset("CoIR-Retrieval/stackoverflow-qa-qrels", split="train")
    corpus = {str(r["_id"]): _doc_text(r) for r in cq["corpus"]}
    queries = {str(r["_id"]): str(r["text"]) for r in cq["queries"]}
    return _mine_coir_bm25_hard(corpus, queries, qrels, "stackoverflow-qa", n_queries, num_negs, bm25_top_k, rng)


def load_csn_hard(
    n_queries: int, num_negs: int, bm25_top_k: int, rng: random.Random, strip_docstrings: bool = True
) -> list[dict]:
    if n_queries <= 0:
        return []
    from datasets import load_dataset

    corpus_ds = load_dataset("CoIR-Retrieval/codesearchnet", "python-corpus", split="corpus")
    queries_ds = load_dataset("CoIR-Retrieval/codesearchnet", "python-queries", split="queries")
    qrels = load_dataset("CoIR-Retrieval/codesearchnet", "python-qrels", split="train")
    if strip_docstrings:
        corpus = {str(r["_id"]): _strip_python_docstrings(_doc_text(r)) for r in corpus_ds}
        corpus = {k: v for k, v in corpus.items() if len(v) >= 30}
    else:
        corpus = {str(r["_id"]): _doc_text(r) for r in corpus_ds}
    queries = {str(r["_id"]): str(r["text"]) for r in queries_ds}
    return _mine_coir_bm25_hard(corpus, queries, qrels, "codesearchnet", n_queries, num_negs, bm25_top_k, rng)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path, default=Path("data/v5_hard_negs_raw.jsonl"))
    ap.add_argument("--n-msmarco", type=int, default=60_000)
    ap.add_argument("--n-soqa", type=int, default=15_000)
    ap.add_argument("--n-csn", type=int, default=25_000)
    ap.add_argument("--num-negs", type=int, default=4, help="Hard negatives per query")
    ap.add_argument("--bm25-top-k", type=int, default=50, help="BM25 retrieval depth for hard-negative pool")
    ap.add_argument("--keep-csn-docstrings", action="store_true", help="Do NOT strip docstrings from CSN code bodies")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    ms = load_msmarco_hard(a.n_msmarco, a.num_negs, rng)
    so = load_soqa_hard(a.n_soqa, a.num_negs, a.bm25_top_k, rng)
    cs = load_csn_hard(a.n_csn, a.num_negs, a.bm25_top_k, rng, strip_docstrings=not a.keep_csn_docstrings)

    rows = ms + so + cs
    rng.shuffle(rows)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print(
        f"Wrote {len(rows):,} hard-negative rows ({a.num_negs} negs/query) -> {a.output} "
        f"(msmarco={len(ms):,}, stackoverflow-qa={len(so):,}, codesearchnet={len(cs):,})",
        flush=True,
    )


if __name__ == "__main__":
    main()
