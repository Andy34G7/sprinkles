"""lernea PPIG tokenizer experiment: C (PMI floor), D (GPT-2-style letter pre-token), C+D.

Corpus and settings reproduce crouton's wp_bpe_ppig_32k.json exactly (crouton
src/crouton/tokenizer/train_tokenizer.py + pretokenizer.py, copied below), so `base` must match the
shipped artifact and every variant differs from it in one knob only.

Run with the experiment worktree's venv (lernea built with pmi_floor):
  ~/projects/tokenisers/.claude/worktrees/ppig-pmi-floor/.venv/bin/python tokexp.py [--only base,C1]
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA, OUT = HERE / "data", HERE / "tokenizers"
SHIPPED = Path.home() / "projects/crouton/artifacts/tokenizer/wp_bpe_ppig_32k.json"

# ---- copied verbatim from crouton/tokenizer/pretokenizer.py
LLAMA3_SPLIT_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}|"
    r" ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)
# D: letters may only carry an optional leading space (GPT-2 style), not any punctuation char
GPT2_LETTER_PATTERN = LLAMA3_SPLIT_PATTERN.replace(r"[^\r\n\p{L}\p{N}]?\p{L}+", r" ?\p{L}+")
SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|pad|>"]
SOURCES = [("HuggingFaceFW/fineweb-edu", "sample-10BT", "train", "text"),
           ("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", "train", "text")]


def pretok(pattern: str) -> dict:
    return {"hf": {"type": "Sequence", "pretokenizers": [
        {"type": "Split", "pattern": {"Regex": pattern}, "behavior": "Isolated", "invert": False},
        {"type": "Digits", "individual_digits": True}]}}


VARIANTS = {  # name: (scorer, pmi_floor, pattern)
    "base": ("ppig", None, LLAMA3_SPLIT_PATTERN),
    "C0.5": ("ppig", 0.5, LLAMA3_SPLIT_PATTERN),
    "C1": ("ppig", 1.0, LLAMA3_SPLIT_PATTERN),
    "C2": ("ppig", 2.0, LLAMA3_SPLIT_PATTERN),
    "D": ("ppig", None, GPT2_LETTER_PATTERN),
    "C1+D": ("ppig", 1.0, GPT2_LETTER_PATTERN),
    "freq-bpe": ("frequency", None, LLAMA3_SPLIT_PATTERN),
}

CODE_TERMS = ("python javascript java json numpy string array list dict function return integer boolean http "
              "request file path directory parse regex socket thread async django flask sql query database github "
              "linux docker tensor pytorch keras matplotlib pandas variable class method module package import "
              "exception iterator generator lambda decorator recursion pointer struct compile").split()


# ---- data (crouton's round-robin loader, then the next docs of each stream as held-out prose)

def corpus(max_docs=50_000, heldout_per_source=1000):
    train_p, held_p = DATA / "train_50k.jsonl", DATA / "heldout_prose.jsonl"
    if not train_p.exists():
        from datasets import load_dataset

        DATA.mkdir(parents=True, exist_ok=True)
        iters = [iter(load_dataset(n, s, split=sp, streaming=True)) for n, s, sp, _ in SOURCES]
        texts, idx, exhausted = [], 0, set()
        while len(texts) < max_docs and len(exhausted) < len(iters):
            i = idx % len(iters); idx += 1
            if i in exhausted:
                continue
            try:
                row = next(iters[i])
            except StopIteration:
                exhausted.add(i); continue
            v = row.get(SOURCES[i][3], "")
            if isinstance(v, str) and v.strip():
                texts.append(v)
        held = [r[SOURCES[i][3]] for i, it in enumerate(iters) for _, r in zip(range(heldout_per_source), it)]
        train_p.write_text("\n".join(json.dumps(t) for t in texts))
        held_p.write_text("\n".join(json.dumps(t) for t in held))
    load = lambda p: [json.loads(l) for l in p.read_text().splitlines()]
    return load(train_p), load(held_p)


def heldout_code(n=2000):
    p = DATA / "heldout_code.jsonl"
    if not p.exists():
        from datasets import load_dataset

        ds = load_dataset("Nan-Do/code-search-net-python", split="train", streaming=True)
        p.write_text("\n".join(json.dumps(r["code"]) for _, r in zip(range(n), ds)))
    return [json.loads(l) for l in p.read_text().splitlines()]


def prose_words():
    p = DATA / "scifact_words.json"
    if not p.exists():
        from datasets import load_dataset

        qs = [r["text"] for r in load_dataset("BeIR/scifact", "queries", split="queries")]
        qs += [r["text"] for r in load_dataset("BeIR/scifact", "corpus", split="corpus").select(range(300))]
        p.write_text(json.dumps(Counter(w.lower() for q in qs for w in re.findall(r"[A-Za-z]{4,}", q))))
    return json.loads(p.read_text())


# ---- metrics (HF `tokenizers` on the exported JSON = what crouton actually loads)

def evaluate(path: Path, held_prose, held_code, words) -> dict:
    from tokenizers import Tokenizer

    t = Tokenizer.from_file(str(path))
    toks = lambda s: t.encode(s, add_special_tokens=False).tokens
    frag = lambda ts: len(ts) > 1 and len(ts[0].lstrip("Ġ")) <= 1  # word starts with a 1-letter piece

    code_bad = [(f.strip(), toks(f)) for w in CODE_TERMS for f in (" " + w, " " + w.capitalize()) if frag(toks(f))]
    whole = fr = tot = 0
    for w, c in words.items():
        ts = toks(" " + w); tot += c; whole += c * (len(ts) == 1); fr += c * frag(ts)

    def bpt(texts):
        enc = t.encode_batch(texts, add_special_tokens=False)
        return sum(len(s.encode()) for s in texts) / sum(len(e.ids) for e in enc)

    used = set()
    for e in t.encode_batch(held_prose + held_code, add_special_tokens=False):
        used.update(e.ids)
    return {"code_terms_fragmented": len(code_bad), "code_terms_total": 2 * len(CODE_TERMS),
            "code_examples": code_bad[:12], "prose_whole_word": whole / tot, "prose_fragmented": fr / tot,
            "bytes_per_token_prose": bpt(held_prose), "bytes_per_token_code": bpt(held_code),
            "vocab_used_heldout": len(used), "vocab_size": t.get_vocab_size()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma-separated variant names")
    ap.add_argument("--vocab-size", type=int, default=32768)
    a = ap.parse_args()
    from tokenisers import BpeTokenizer, WpBpePpigTokenizer

    train, held_prose = corpus()
    held_code, words = heldout_code(), prose_words()
    print(f"corpus {len(train)} docs, held-out {len(held_prose)} prose / {len(held_code)} code", flush=True)
    OUT.mkdir(exist_ok=True)
    results = json.loads((HERE / "results.json").read_text()) if (HERE / "results.json").exists() else {}
    if "shipped" not in results:
        results["shipped"] = evaluate(SHIPPED, held_prose, held_code, words)

    for name in [n for n in VARIANTS if not a.only or n in a.only.split(",")]:
        scorer, floor, pattern = VARIANTS[name]
        t0 = time.time()
        if scorer == "ppig":
            tok = WpBpePpigTokenizer.train(train, vocab_size=a.vocab_size, min_frequency=2, length_penalty=0.35,
                                           special_tokens=SPECIALS, pretokenizer=pretok(pattern), pmi_floor=floor)
        else:
            tok = BpeTokenizer.train(train, vocab_size=a.vocab_size, scorer="frequency",
                                     special_tokens=SPECIALS, pretokenizer=pretok(pattern))
        path = OUT / f"{name}.json"
        tok.export_hf_json(str(path))
        r = evaluate(path, held_prose, held_code, words)
        r["train_s"] = round(time.time() - t0, 1)
        if name == "base":
            r["matches_shipped_vocab"] = (json.loads(path.read_text())["model"]["vocab"]
                                          == json.loads(SHIPPED.read_text())["model"]["vocab"])
        results[name] = r
        (HERE / "results.json").write_text(json.dumps(results, indent=2))
        print(f"{name:8s} code-frag {r['code_terms_fragmented']:3d}/{r['code_terms_total']}  "
              f"prose whole {r['prose_whole_word']:.3f} frag {r['prose_fragmented']:.3f}  "
              f"B/tok prose {r['bytes_per_token_prose']:.3f} code {r['bytes_per_token_code']:.3f}  "
              f"{r['train_s']}s", flush=True)


if __name__ == "__main__":
    main()
