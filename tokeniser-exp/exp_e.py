"""Experiment E: 70/30 prose+code corpus (by bytes, matching train_50k's ~220 MB budget).

Compares PPIG base, PPIG + pmi_floor=1.0 (C1), and plain frequency BPE on the exact same
held-out prose/code/SciFact/CODE_TERMS probes as tokexp.py.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from tokexp import (
    DATA,
    LLAMA3_SPLIT_PATTERN,
    OUT,
    SPECIALS,
    corpus,
    evaluate,
    heldout_code,
    pretok,
    prose_words,
)

E_VARIANTS = {
    "E_base": ("ppig", None, LLAMA3_SPLIT_PATTERN),
    "E_C1": ("ppig", 1.0, LLAMA3_SPLIT_PATTERN),
    "E_freq-bpe": ("frequency", None, LLAMA3_SPLIT_PATTERN),
}


def build_70_30_corpus(held_code: list[str]) -> list[str]:
    out_p = DATA / "train_70_30_code.jsonl"
    if out_p.exists():
        return [json.loads(line) for line in out_p.read_text().splitlines()]

    from datasets import load_dataset

    prose_all, _ = corpus()
    prose = prose_all[:35_000]
    prose_bytes = sum(len(t.encode("utf-8")) for t in prose)
    target_code_bytes = int(prose_bytes * (3.0 / 7.0))

    held_set = set(held_code)
    ds = load_dataset("Nan-Do/code-search-net-python", split="train", streaming=True)
    code_docs: list[str] = []
    code_bytes = 0
    for idx, row in enumerate(ds):
        if idx < 2000:
            continue  # strictly disjoint from heldout_code.jsonl
        c = row.get("code") or ""
        if not isinstance(c, str) or not c.strip() or c in held_set:
            continue
        code_docs.append(c)
        code_bytes += len(c.encode("utf-8"))
        if code_bytes >= target_code_bytes:
            break

    mixed = prose + code_docs
    out_p.write_text("\n".join(json.dumps(t) for t in mixed))
    print(
        f"Built 70/30 corpus: {len(prose):,} prose ({prose_bytes / 1e6:.1f} MB) + "
        f"{len(code_docs):,} code ({code_bytes / 1e6:.1f} MB) = {len(mixed):,} docs",
        flush=True,
    )
    return mixed


def main() -> None:
    from tokenisers import BpeTokenizer, WpBpePpigTokenizer

    _, held_prose = corpus()
    held_code = heldout_code()
    words = prose_words()
    train = build_70_30_corpus(held_code)
    print(f"Loaded E corpus: {len(train):,} docs", flush=True)

    OUT.mkdir(exist_ok=True)
    results_path = HERE / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}

    for name, (scorer, floor, pattern) in E_VARIANTS.items():
        t0 = time.time()
        if scorer == "ppig":
            tok = WpBpePpigTokenizer.train(
                train,
                vocab_size=32768,
                min_frequency=2,
                length_penalty=0.35,
                special_tokens=SPECIALS,
                pretokenizer=pretok(pattern),
                pmi_floor=floor,
            )
        else:
            tok = BpeTokenizer.train(
                train,
                vocab_size=32768,
                scorer="frequency",
                special_tokens=SPECIALS,
                pretokenizer=pretok(pattern),
            )
        path = OUT / f"{name}.json"
        tok.export_hf_json(str(path))
        r = evaluate(path, held_prose, held_code, words)
        r["train_s"] = round(time.time() - t0, 1)
        results[name] = r
        results_path.write_text(json.dumps(results, indent=2))
        print(
            f"{name:12s} code-frag {r['code_terms_fragmented']:3d}/{r['code_terms_total']}  "
            f"prose whole {r['prose_whole_word']:.3f} frag {r['prose_fragmented']:.3f}  "
            f"B/tok prose {r['bytes_per_token_prose']:.3f} code {r['bytes_per_token_code']:.3f}  "
            f"{r['train_s']}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
