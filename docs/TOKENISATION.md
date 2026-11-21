# Crouton tokenisation: word-initial fragmentation (lernea `wp_bpe_ppig`)

Investigated 2026-11-21.

## Symptom

A crouton SPLADE query for "python function to sort a list" expanded to the terms `ython`, `Ġp`, `ĠP`.
`Ġp` is a near-meaningless, high-document-frequency term, so it wastes expansion budget and adds noisy
matches.

## Findings

1. **Not an encoder bug.** lernea's `WpBpePpigTokenizer.encode` matches HF `tokenizers` token-for-token
   on the same `wp_bpe_ppig_32k.json`, across every probe tried.
2. **Cause: the learned merges.** `ython` is merge #2030, `Ġpy` is #23027, and there is no
   `Ġp`+`ython` merge. BPE applies merges by rank, so `ython` forms first and `Ġpython` can never form.
   Meanwhile `python` (no space) and `ĠPython` are single tokens.
3. **Mechanism: the PPIG score.** The score is `freq(uv) * max(0, PMI(u,v)) / len(uv)^0.35`
   (`crates/tokenisers_core/src/scoring/wp_bpe.rs`, implemented as documented). PMI divides by
   `freq(u) * freq(v)`. Word-initial pieces like `Ġp` or `Ġj` are extremely frequent, so their PMI with
   anything is near 0, and merges onto them get starved. Word-internal pieces such as `ython` and `son`
   have high PMI, so they merge early and lock out the word-initial path.
   - Likely contributor, untested: the cl100k-style pre-tokenizer (`[^\r\n\p{L}\p{N}]?\p{L}+`) attaches
     punctuation, so code like `.json`, `(json`, `_json` and ` json` split the counts between pre-token
     variants. That matters at 32k vocab.
4. **Scope.**
   - Prose (SciFact queries plus 300 abstracts): no worse than GPT-2. 80.7% whole-word vs 80.8%; 3.2% vs
     2.5% start with a one-letter piece.
   - Programming terms (51 terms, lower and Capitalised forms): 20/102 start with a one-letter piece vs
     7/102 for GPT-2's 50k vocab. Examples: `Ġp ython`, `Ġj ava`, `Ġj son`, `Ġd j ang o`, `Ġg ith ub`,
     `ĠD ock er`, `ĠL ambda`.
   - Shortcake is not affected. It uses a standard HF ByteLevel BPE (16k), where `Ġpython`, `Ġjson` and
     `Ġdjango` are single tokens.

## Options

### No retraining (crouton as-is)

- Do not lowercase inputs: that turns `ĠPython` (one token) into `Ġp ython`.
- At index and query time, fold variant ids (`ĠPython` / `python` / `Python`) into one canonical id
  with a precomputed id map.
- Drop or down-weight one-letter word-start pieces (`Ġa`..`Ġz`, `ĠA`..`ĠZ`) from SPLADE vectors as
  stopword-like terms, or add an IDF-style penalty to them in the FLOPS reg.

### Tokenizer fix in lernea (needs a new tokenizer, then re-pretraining crouton)

- Floor the PMI term, e.g. `freq(uv) * max(PMI, c)`, or blend in a plain-frequency score, so very
  frequent word-initial merges are not starved.
- Protect top-N frequent whole-word pre-tokens (`Ġword`) as required vocab, similar to SentencePiece
  `user_defined_symbols`.
- A/B the pre-tokenizer: attach an optional space only (GPT-2 style) vs any punctuation (current).
- Measure each change with the probes above: the word-initial fragmentation rate on a code-term list and
  on prose, plus bytes/token, so prose compression does not regress.

## Experiment: C (PMI floor), D (GPT-2 letter pre-token), C+D (2026-11-21)

Setup: crouton's tokenizer corpus (50k round-robin docs of fineweb-edu sample-10BT + cosmopedia-v2),
32,768 vocab, length_penalty 0.35, min_freq 2. Held-out sets: 2k prose docs (the next docs of the same
streams) and 2k CodeSearchNet-python functions. Code: `tokeniser-exp/tokexp.py`. Raw numbers:
`tokeniser-exp/results.json`. Tokenizers: `tokeniser-exp/tokenizers/`. lernea change (`pmi_floor`):
branch `exp/ppig-pmi-floor`, worktree `~/projects/tokenisers/.claude/worktrees/ppig-pmi-floor`
(uncommitted).

| Variant | Code terms split badly (of 102) | Prose whole-word | Bytes/token prose | Bytes/token code |
|---|---|---|---|---|
| shipped `wp_bpe_ppig_32k.json` | 20 | 0.807 | 4.761 | 3.445 |
| base (PPIG retrained now) | 18 | 0.810 | 4.768 | 3.448 |
| C, floor 0.5 | 18 | 0.810 | 4.774 | 3.472 |
| C, floor 1 | 18 | 0.810 | 4.775 | 3.476 |
| C, floor 2 | 20 | 0.806 | 4.775 | 3.495 |
| D | 17 | 0.811 | 4.738 | 3.386 |
| C1+D | 17 | 0.811 | 4.747 | 3.405 |
| plain frequency BPE | 17 | 0.811 | 4.794 | 3.539 |

Findings:
- **C and D do not fix it.**
  - The floor changes fragmentation by 0 to +2 terms.
  - D removes one bad split but costs 0.6% (prose) and 1.8% (code) in bytes/token.
  - Plain frequency BPE on the same corpus has the same problem (17/102). So PPIG's PMI term is not
    the main cause, which corrects the mechanism I gave above.
- **The real cause is the corpus.** `json`, `django`, `github` and `javascript` fragment under *every*
  scorer, because the tokenizer corpus has no code. Crouton then pretrains on 30% Python with a
  tokenizer that never saw code.
- **`Ġp ython` in the shipped file was partly luck.** Retrained today on the "same" corpus, every
  variant gets `Ġpython` as one token. The shipped vocab does not match a retrain (`base` differs). The
  likely reasons are a different streaming order or snapshot, or an older lernea build (the main
  checkout's `.so` dates from 2026-11-01).
- **Small positive for C:** a floor of 0.5 to 1 improves bytes/token on code by ~0.8% with no prose
  cost. Plain BPE still compresses best on both (4.794 / 3.539), so PPIG's case over plain BPE is not
  supported by these intrinsic numbers.

## Experiment E: 70/30 prose + Python code corpus (2026-11-21)

Setup: 70/30 mix by bytes matching `train_50k.jsonl`'s ~212 MB budget — 35,000 prose docs (148.3 MB from
`train_50k.jsonl[:35000]`) + 52,064 Python functions (63.6 MB from `Nan-Do/code-search-net-python`
starting after row 2,000, strictly disjoint from `heldout_code.jsonl`). Same 32,768 vocab,
`length_penalty=0.35`, `min_freq=2`, `LLAMA3_SPLIT_PATTERN`, and probes. Code: `tokeniser-exp/exp_e.py`.

| Variant | Code terms split badly (of 102) | Prose whole-word | Bytes/token prose | Bytes/token code |
|---|---|---|---|---|
| `base` (prose-only PPIG) | 18 | 0.810 | 4.768 | 3.448 |
| `E_base` (70/30 PPIG) | **10** | 0.784 | 4.695 | 4.159 |
| `E_C1` (70/30 PPIG, `pmi_floor=1.0`) | **10** | 0.783 | 4.701 | 4.165 |
| `E_freq-bpe` (70/30 plain BPE) | 11 | 0.781 | **4.719** | **4.195** |

Findings:
- **Corpus fix works decisively.** Adding 30% Python code cuts code-term fragmentation nearly in half
  (**18–20/102 → 10/102**) and boosts code compression by **+20.7%** (`3.448 → 4.159` B/tok) for only a
  1.5% prose compression cost (`4.768 → 4.695` B/tok).
- **Remaining 10 splits are almost all non-Python / unusual-case terms:** `javascript`, `Javascript`,
  `java`, `django`, `Django`, `Socket`, plus title-cased acronyms (`Json`, `Http`, `Sql`, `Github`).
  `python`, `json`, `Numpy`, `Dict`, `github`, `docker`, etc. are all fixed.
- **`pmi_floor` is not needed in lernea.** `E_C1` ties `E_base` on fragmentation (10/102) and adds only
  +0.14% compression. The `exp/ppig-pmi-floor` worktree can be dropped cleanly.
- **PPIG vs. plain BPE on 70/30:** PPIG (`E_base`) slightly beats plain BPE on code-term fragmentation
  (10 vs. 11) and prose whole-word rate (0.784 vs. 0.781), while plain BPE is ~0.5–0.9% higher on raw
  bytes/token. No need to abandon PPIG in lernea.
- **Action for future crouton v2:** update `crouton/src/crouton/tokenizer/train_tokenizer.py`'s
  `DEFAULT_TOKENIZER_SOURCES` to include a code/tech mix (matching pretraining) before any future
  pretrain from scratch. Do not re-pretrain crouton now; proceed with SPLADE distillation on the current
  checkpoint first.
