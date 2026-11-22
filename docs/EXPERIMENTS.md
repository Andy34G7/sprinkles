# Experiment Log & Findings

Updated: 2026-11-21. All retrieval evaluations use [`harness/eval_suite.py`](../harness/eval_suite.py) on the `core` suite (`beir:arguana`, `beir:nfcorpus`, `beir:scifact`, `coir:cosqa`, `coir:stackoverflow-qa`) at `max_len=256`.

---

## 1. Baseline Leaderboard (`v0`, `core` suite, nDCG@10)

| Model | `beir:arguana` | `beir:nfcorpus` | `beir:scifact` | `coir:cosqa` | `coir:stackoverflow-qa` | **General** | **Code** | **Overall** | Avg Q/D terms | Lat (ms)* |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `hf:naver/splade-code-06B` (600M) | **0.5963** | 0.2539 | 0.6645 | **0.3605** | **0.9210** | 0.5049 | **0.6408** | **0.5592** | 1666 / 1614 | 47.8 |
| `hf:naver/splade-v3` (110M) | 0.5128 | **0.3576** | **0.7096** | 0.2257 | 0.6570 | **0.5267** | 0.4414 | 0.4925 | 138 / 247 | 13.6 |
| `tfidf` (sublinear, l2) | 0.5239 | 0.3023 | 0.6286 | 0.1299 | 0.7579 | 0.4849 | 0.4439 | 0.4685 | 43 / 90 | 0.3 |
| `bm25` (`bm25s`, k1=1.5, b=0.75) | 0.4860 | 0.3064 | 0.6617 | 0.1426 | 0.6954 | 0.4847 | 0.4190 | 0.4584 | — | — |
| `hf:Cdn13/ettin-splade-stackoverflowqa` (17M) | 0.4860 | 0.1881 | 0.5548 | 0.1966 | 0.7417 | 0.4096 | 0.4691 | 0.4334 | 1534 / 1434 | 9.9 |
| `crouton-base` v0 (50M, MS MARCO InfoNCE) | 0.2880 | 0.2483 | 0.5079 | 0.0980 | 0.4495 | 0.3481 | 0.2738 | 0.3184 | 124 / 248 | 20.1 |
| `shortcake-base` v0 (53M, CSN-py InfoNCE) | 0.1620 | 0.1158 | 0.3181 | 0.2145 | 0.5163 | 0.1986 | 0.3654 | 0.2653 | 120 / 208 | 18.5 |

*\*Batch-1 query latency on L4 GPU for neural models, CPU for `tfidf`.*

### Key baseline observations
- **Two complementary ceilings:** `splade-code-06B` dominates Code (`0.6408`) and `arguana` (`0.5963`) but is dense (`1666/1614` terms); `splade-v3` dominates General (`0.5267`) at **6.5× sparser vectors** (`138/247`).
- **Sparsity vs. generalization niche:** `ettin-17m` hits `0.7417` on `stackoverflow-qa` (its training domain) but drops to `0.4096` General (`0.1881` on `nfcorpus`) at `1534/1434` active terms.
- **Why both `stackoverflow-qa` and `cosqa` matter:** `stackoverflow-qa` has strong lexical overlap (`tfidf` gets `0.7579`), while `cosqa` (web queries → Python code) breaks lexical baselines (`0.13–0.14`) and directly rewards neural lexical expansion.

---

## 2. v1 Ablation: Bug Fixes Only (Same Single-Domain Data & Step Budgets)

| Model | `arguana` | `nfcorpus` | `scifact` | `cosqa` | `stackoverflow-qa` | **General** | **Code** | **Overall** | Avg Q/D |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Shortcake v0 (`temp=0.05`) | 0.1620 | 0.1158 | 0.3181 | 0.2145 | 0.5163 | 0.1986 | 0.3654 | 0.2653 | 120 / 208 |
| **Shortcake v1 (`temp=1.0`)** | 0.1256 | **0.1425** | **0.3902** | **0.2628** | **0.5312** | **0.2194** | **0.3970** | **0.2905** | 470 / 738 |
| **Crouton v0 (`l1`, 300 MNTP)** | **0.2880** | **0.2483** | **0.5079** | **0.0980** | **0.4495** | **0.3481** | **0.2738** | **0.3184** | 124 / 248 |
| Crouton v1 (`flops`, 1000 MNTP) | 0.2704 | 0.2312 | 0.4645 | 0.0346 | 0.3826 | 0.3220 | 0.2086 | 0.2767 | 166 / 282 |

### Findings from v1
1. **Shortcake `temperature=0.05 → 1.0` was a major win (+0.0252 Overall, +0.0483 on `cosqa`):** Dividing unbounded SPLADE dot products by `0.05` had multiplied logits by 20× and caused premature InfoNCE saturation. At `temp=1.0`, Shortcake v1 jumped from `0.7375 → 0.7767` on its held-out CSN-py 1k pool and **beat both `naver/splade-v3` (`0.2257`) and `ettin-17m` (`0.1966`) on `coir:cosqa` (`0.2628`)**. Because its `lambda_q=3e-4, lambda_d=1e-4` were tuned for `temp=0.05`, active terms expanded to `470/738` — fixed in v2 by raising reg to L1 `0.015` / FLOPS `0.04`.
2. **Crouton `l1` → `flops` at `lambda_q=0.05` regressed (-0.0417 Overall):** FLOPS squares the batch mean of each vocab term, making it ~8–16× looser than L1 at the same weight (`l0_q` rose from ~13 to ~100 during training). On 100k MS MARCO triplets (3.2 epochs), this caused memorization (`rank` loss dropped to `0.21–0.35`) and noisy over-expansion out of domain.
3. **Design rule for v2/v3:** Use **L1 regularization on queries (`lambda_q ≈ 0.015`)** + **FLOPS regularization on documents (`lambda_d ≈ 0.04`)**, matching the original SPLADE formulation.

---

## 3. v2 Ablation: Scaled Mixed-Domain InfoNCE (200k Triplets, 10k Steps, No Distillation)

Both models trained identically via [`train/train_splade.py`](../train/train_splade.py) on `200,000` mixed triplets (`120k` MS MARCO + `40k` StackOverflow-QA + `40k` CodeSearchNet-python), `10,000` steps (`bs=32 × ga=2`), `temperature=1.0`, `query_reg_type=l1 (0.015)`, `lambda_d=0.04` (FLOPS), `margin_mse_weight=0.0`.

| Model | `arguana` | `nfcorpus` | `scifact` | `cosqa` | `stackoverflow-qa` | **General** | **Code** | **Overall** | Avg Q/D |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Crouton v0 (100k MS MARCO) | **0.2880** | **0.2483** | **0.5079** | 0.0980 | 0.4495 | **0.3481** | 0.2738 | **0.3184** | 124 / 248 |
| **Crouton v2 (200k mixed)** | 0.1876 | 0.2329 | 0.4720 | **0.1074** | **0.5695** | 0.2975 | **0.3385** | 0.3139 | 101 / 160 |
| Shortcake v1 (50k CSN, `temp=1.0`) | 0.1256 | 0.1425 | **0.3902** | **0.2628** | **0.5312** | **0.2194** | **0.3970** | **0.2905** | 470 / 738 |
| **Shortcake v2 (200k mixed)** | **0.1761** | **0.1648** | 0.2881 | 0.0933 | 0.4650 | 0.2097 | 0.2792 | 0.2375 | 34 / 53 |

### Findings from v2
1. **Mixed data lifted Crouton's Code retrieval strongly (+0.0647 Code avg):** `coir:stackoverflow-qa` jumped **+0.1200** (`0.4495 → 0.5695`, R@100 `0.753 → 0.880`) at nearly **3× sparser** vectors (`37/74` vs. `133/207`), and `coir:cosqa` rose `0.0980 → 0.1074`.
2. **Shortcake v2 improved on `nfcorpus` (`0.1425 → 0.1648`) and `arguana` (`0.1256 → 0.1761`),** showing cross-domain transfer from MS MARCO.
3. **Critical sparsity finding (why `cosqa` & `scifact` under-expanded in v2):**
   - `lambda_d=0.04` (FLOPS) + `lambda_q=0.015` (L1) **over-pruned short code documents** on `coir:cosqa`: Crouton dropped to **`13` active doc terms** and Shortcake collapsed to **`8` active doc terms** (and `34/53` overall!).
   - Because FLOPS ($\sum_j \bar{v}_j^2$) scales inversely with vocab size, Shortcake's **16k vocab** takes a **2× harsher** FLOPS penalty than Crouton's **32k vocab** at the same `lambda_d`.
   - **Rule for v3:** Lower `lambda_d` to **`0.015`** (Crouton 32k) and **`0.005–0.008`** (Shortcake 16k), and `lambda_q` (L1) to **`0.008`** (Crouton) / **`0.003`** (Shortcake), targeting ~100–250 active doc terms across both prose and short code.
4. **Why pure InfoNCE hits a ceiling around `~0.31` Overall without distillation:** Both v0 and v2 Crouton plateau around `0.31–0.32` Overall with binary positive/negative InfoNCE (using random CoIR negatives and BM25 MS MARCO negatives). **v3 (dual-teacher Margin-MSE distillation)** is the key remaining lever to break into the `0.40+` tier.

---

## 4. v3 Ablation: Vocab-Normalized FLOPS (`V / 30522`) + Gentler Lambdas (InfoNCE Only)

Same 200k mixed triplets and 10k steps as v2, with FLOPS normalized by `V / 30522` (`lambda_q=0.005` L1, `lambda_d=0.008` vocab-normalized FLOPS, `margin_mse_weight=0.0`).

| Model | `arguana` | `nfcorpus` | `scifact` | `cosqa` | `stackoverflow-qa` | **General** | **Code** | **Overall** | Avg Q/D |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Crouton v2 (unnorm `0.015/0.04`) | **0.1876** | **0.2329** | **0.4720** | **0.1074** | 0.5695 | **0.2975** | 0.3385 | **0.3139** | 101 / 160 |
| **Crouton v3 (`V/30522`, `0.005/0.008`)** | 0.1460 | 0.2206 | 0.4612 | 0.0957 | **0.6128** | 0.2759 | **0.3543** | 0.3072 | 174 / 284 |
| Shortcake v2 (unnorm `0.015/0.04`) | **0.1761** | 0.1648 | 0.2881 | **0.0933** | 0.4650 | 0.2097 | 0.2792 | 0.2375 | 34 / 53 |
| **Shortcake v3 (`V/30522`, `0.005/0.008`)** | 0.1210 | **0.1719** | **0.3344** | 0.0908 | **0.5928** | **0.2091** | **0.3418** | **0.2622** | 93 / 170 |

### Findings from v3
1. **Vocab-normalized FLOPS worked as designed:** Shortcake (16k) and Crouton (32k) converged to nearly identical sparsity on `stackoverflow-qa` (`45/135` vs. `64/130`) instead of Shortcake collapsing to `22/46`.
2. **Strong `stackoverflow-qa` gains without distillation:**
   - **Crouton v3** hit **`0.6128`** nDCG@10 (**R@100 `0.9072`**, +0.1633 over v0 `0.4495`, +0.0433 over v2) at just `64/130` active terms (**11× sparser** than Ettin-17M's `1348/1397`!).
   - **Shortcake v3** jumped **+0.1278** over v2 to **`0.5928`** (`45/135` terms, +0.0616 over v1), plus its best `nfcorpus` (`0.1719`) and a +0.0463 recovery on `scifact` (`0.3344`).
3. **Root cause of low `coir:cosqa` (`~0.09`) in v2 & v3:** `CoIR-Retrieval/codesearchnet` leaves raw docstrings inside Python function bodies, so binary InfoNCE matched verbatim docstring words in 14–20 tokens instead of learning code-token expansion (unlike Shortcake v0/v1, which stripped docstrings and reached `0.2628` on `cosqa`). Fix for v4: strip docstrings from CodeSearchNet in `build_mixed_triplets.py`.
4. **Confirmed ceiling of pure InfoNCE:** Across v0, v1, v2, and v3, pure InfoNCE without a teacher plateaus at `~0.31` Overall for 50M models (trading General for Code when mixed). **v4 (dual-teacher Margin-MSE distillation)** is the next major step.

---

## 5. Tokenizer Investigation (`docs/TOKENISATION.md`)

- Word-initial code fragmentation in `crouton` v1.0.0 (`18–20/102` code terms) was caused by training the tokenizer on prose only (`fineweb-edu` + `cosmopedia`), **not** by `wp-bpe-ppig` scoring.
- Adding 30% Python code to the tokenizer corpus (Experiment E) cut code fragmentation nearly in half (`18 → 10/102`) and boosted code compression by **+20.7%** (`3.448 → 4.159` B/tok) for only a 1.5% prose cost.
- Unmodified **`wp-bpe-ppig`** beat `pmi_floor` (negligible difference) and outperformed plain frequency BPE on downstream LM BPB (`2.522` vs. `2.549`), Morph F1, Identifier F1, and Exp E code fragmentation (`10` vs. `11`).
- Decision: stick with default `wp-bpe-ppig`, retrain the `crouton` v2.0.0 model family (`tiny`, `small`, `base`) on the 70/30 tokenizer, and evaluate those new backbones as **v4** after the `v0 → v1 → v2 → v3` SPLADE ablation ladder.

---

## 5. Roadmap (`v3 → v4 → v5`)

- **v3 (Vocab-Normalized FLOPS + Gentler Sparsity, InfoNCE only):** Exact same 200k mixed triplets & 10k steps as v2, with FLOPS scaled by `V / 30522` (so 16k Shortcake and 32k Crouton share one effective sparsity scale) and gentler lambdas (`lambda_q=0.005` L1, `lambda_d=0.008` normalized FLOPS) to prevent short-code doc collapse on `cosqa`.
- **v4 (Dual-Teacher Margin-MSE Distillation):** Score the exact same 200k triplets via [`train/score_teacher_margins.py`](../train/score_teacher_margins.py) (`naver/splade-v3` for General, `naver/splade-code-06B` for Code, at `max_len=256`) and retrain with Margin-MSE + optional Shortcake `--echo` embeddings and smaller tiers (`crouton-tiny` 14M, `shortcake-small` 23M vs. `ettin-17m`).
- **v5 (Crouton v2.0.0 Backbones):** Apply the winning v4 recipe to the newly pretrained 70/30-tokenizer `crouton` v2.0.0 family.
