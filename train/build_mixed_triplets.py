"""Build a reproducible mixed General + Code retrieval triplet JSONL for SPLADE v2/v3.

Sources (all strictly train-split only, zero test-split leakage):
  - General: sentence-transformers/msmarco-bm25 (triplet, train)
  - Code QA: CoIR-Retrieval/stackoverflow-qa (train qrels + random corpus negatives)
  - Code NL->Code: CoIR-Retrieval/codesearchnet (python train qrels + random corpus negatives)

Output JSONL schema:
  {"query": str, "pos": str, "neg": str, "source": "msmarco" | "stackoverflow-qa" | "codesearchnet"}
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


def load_msmarco(n: int) -> list[dict]:
    if n <= 0:
        return []
    from datasets import load_dataset

    ds = load_dataset("sentence-transformers/msmarco-bm25", name="triplet", split="train", streaming=True)
    out: list[dict] = []
    for r in ds:
        q = (r.get("query") or "").strip()
        p = (r.get("positive") or "").strip()
        neg = (r.get("negative") or "").strip()
        if q and p and neg:
            out.append({"query": q, "pos": p, "neg": neg, "source": "msmarco"})
            if len(out) >= n:
                break
    return out


def _build_coir_triplets(
    corpus: dict[str, str],
    queries: dict[str, str],
    qrels_ds,
    source_name: str,
    max_triplets: int,
    negs_per_pos: int,
    rng: random.Random,
) -> list[dict]:
    pos_map: dict[str, set[str]] = {}
    for r in qrels_ds:
        qid = str(r.get("query_id") if r.get("query_id") is not None else r.get("query-id"))
        cid = str(r.get("corpus_id") if r.get("corpus_id") is not None else r.get("corpus-id"))
        if int(r.get("score", 1)) >= 1 and qid in queries and cid in corpus:
            pos_map.setdefault(qid, set()).add(cid)

    corpus_ids = list(corpus.keys())
    out: list[dict] = []
    qids = list(pos_map.keys())
    rng.shuffle(qids)
    for qid in qids:
        pos_ids = pos_map[qid]
        qtext = queries[qid].strip()
        if not qtext:
            continue
        for pid in pos_ids:
            ptext = corpus[pid].strip()
            if not ptext:
                continue
            for _ in range(negs_per_pos):
                nid = rng.choice(corpus_ids)
                while nid in pos_ids:
                    nid = rng.choice(corpus_ids)
                ntext = corpus[nid].strip()
                if ntext:
                    out.append({"query": qtext, "pos": ptext, "neg": ntext, "source": source_name})
                    if len(out) >= max_triplets:
                        return out
    return out


def load_stackoverflow_qa(n: int, rng: random.Random) -> list[dict]:
    if n <= 0:
        return []
    from datasets import load_dataset

    cq = load_dataset("CoIR-Retrieval/stackoverflow-qa-queries-corpus")
    qrels = load_dataset("CoIR-Retrieval/stackoverflow-qa-qrels", split="train")
    corpus = {str(r["_id"]): _doc_text(r) for r in cq["corpus"]}
    queries = {str(r["_id"]): str(r["text"]) for r in cq["queries"]}
    return _build_coir_triplets(corpus, queries, qrels, "stackoverflow-qa", n, negs_per_pos=3, rng=rng)


def load_codesearchnet_python(n: int, rng: random.Random) -> list[dict]:
    if n <= 0:
        return []
    from datasets import load_dataset

    corpus_ds = load_dataset("CoIR-Retrieval/codesearchnet", "python-corpus", split="corpus")
    queries_ds = load_dataset("CoIR-Retrieval/codesearchnet", "python-queries", split="queries")
    qrels = load_dataset("CoIR-Retrieval/codesearchnet", "python-qrels", split="train")
    # Strip Python docstrings from code bodies so models cannot cheat via verbatim docstring overlap
    corpus = {str(r["_id"]): _strip_python_docstrings(_doc_text(r)) for r in corpus_ds}
    corpus = {k: v for k, v in corpus.items() if len(v) >= 30}
    queries = {str(r["_id"]): str(r["text"]) for r in queries_ds}
    return _build_coir_triplets(corpus, queries, qrels, "codesearchnet", n, negs_per_pos=1, rng=rng)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path, default=Path("data/mixed_triplets_200k.jsonl"))
    ap.add_argument("--n-msmarco", type=int, default=120_000)
    ap.add_argument("--n-soqa", type=int, default=40_000)
    ap.add_argument("--n-csn", type=int, default=40_000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    ms = load_msmarco(a.n_msmarco)
    so = load_stackoverflow_qa(a.n_soqa, rng)
    cs = load_codesearchnet_python(a.n_csn, rng)

    rows = ms + so + cs
    rng.shuffle(rows)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print(
        f"Wrote {len(rows):,} mixed triplets to {a.output} "
        f"(msmarco={len(ms):,}, stackoverflow-qa={len(so):,}, codesearchnet={len(cs):,})",
        flush=True,
    )


if __name__ == "__main__":
    main()
