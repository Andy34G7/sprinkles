# Experiment Log & Findings

Updated: 2026-11-28. All retrieval evaluations use [`harness/eval_suite.py`](../harness/eval_suite.py) on the `core` suite (`beir:arguana`, `beir:nfcorpus`, `beir:scifact`, `coir:cosqa`, `coir:stackoverflow-qa`) at `max_len=256`.

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

## 5. v4 & v4-2 Ablation: Dual-Teacher Margin-MSE Distillation + Docstring-Stripped CSN

Both models trained via [`train/train_splade.py`](../train/train_splade.py) on `90,010` teacher-scored mixed triplets (`50,012` MS MARCO scored by `naver/splade-v3` + `39,998` StackOverflow-QA & CodeSearchNet scored by `naver/splade-code-06B`, `margin_min=0.5`, `max_len=256`), `8,000` steps (`bs=32 × ga=2`), `margin_mse_weight=1.0`, `infonce_weight=0.1`, `lambda_q=0.005` (L1), `lambda_d=0.008` (vocab-normalized FLOPS, `V / 30522`), and per-teacher RMS margin normalization (`--teacher-margin-rms 8.0`).
- **v4:** Unstripped CSN (matching v3 data) for a clean InfoNCE → Margin-MSE comparison.
- **v4-2:** Same 70,010 MS MARCO + SO-QA rows + 20,000 freshly scored `--strip-csn-docstrings` CSN rows.

| Model | `arguana` | `nfcorpus` | `scifact` | `cosqa` | `stackoverflow-qa` | **General** | **Code** | **Overall** | Avg Q/D |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Crouton v3 (InfoNCE) | 0.1460 | **0.2206** | **0.4612** | 0.0957 | **0.6128** | **0.2759** | **0.3543** | **0.3072** | 174 / 284 |
| **Crouton v4 (Margin-MSE)** | **0.2189** | 0.1440 | 0.4571 | **0.1356** | 0.5668 | 0.2733 | 0.3512 | 0.3045 | 213 / 286 |
| Shortcake v3 (InfoNCE) | 0.1210 | **0.1719** | 0.3344 | 0.0908 | **0.5928** | **0.2091** | 0.3418 | 0.2622 | 93 / 170 |
| **Shortcake v4 (Margin-MSE)** | **0.1537** | 0.1180 | **0.3439** | 0.1584 | 0.5754 | 0.2052 | 0.3669 | **0.2699** | 175 / 289 |
| **Shortcake v4-2 (Margin-MSE, stripped CSN)** | 0.1464 | 0.1049 | 0.3076 | **0.1863** | 0.5878 | 0.1863 | **0.3871** | 0.2666 | 181 / 295 |

### Findings from v4 & v4-2
1. **Critical multi-teacher distillation lesson (`--teacher-margin-rms`):** `naver/splade-v3` (`~138/247` active terms) has raw margin `RMS = 7.99`, whereas `naver/splade-code-06B` (`~1666/1614` active terms) has raw margin `RMS = 418.62` (**52.4× larger**). Because Margin-MSE is quadratic, unnormalized dual-teacher distillation produces a **~2,745× MSE imbalance** that drowns out MS MARCO and forces `l0_d > 4,000`. Normalizing each teacher's `(pos_score, neg_score)` to `RMS = 8.0` (`×1.00077` for `splade-v3`, `×0.01911` for `splade-code-06B`) immediately restored balanced training (`loss 55 → 0.5`, `l0_q ~ 90–120`, `l0_d ~ 220–240`).
2. **Margin-MSE lifted `cosqa` and `arguana` on both architectures:**
   - **Crouton v4:** `arguana` jumped **+0.0729** (`0.1460 → 0.2189`) and `cosqa` jumped **+0.0399** (`0.0957 → 0.1356`, R@100 `0.6060`).
   - **Shortcake v4:** `cosqa` jumped **+0.0676** (`0.0908 → 0.1584`, R@100 `0.6640`), `arguana` +0.0327 (`0.1210 → 0.1537`), and `scifact` +0.0095 (`0.3344 → 0.3439`).
3. **Stripping CSN docstrings (`v4-2`) delivered the strongest mixed-domain Code score (`0.3871`):**
   - `coir:cosqa` rose another **+0.0279** to **`0.1863`** (**R@100 `0.7140`**, more than **2×** v3's `0.0908`, beating BM25 `0.1426` and TF-IDF `0.1299`, and approaching `ettin-17m` `0.1966`), while `stackoverflow-qa` also improved to **`0.5878`** (`R@100 = 0.8756`).
4. **Why `nfcorpus` dipped and why 50M models still trail `ettin-17m` (`0.4334`) and `splade-v3` (`0.4925`):**
   - Two structural bottlenecks remain: (a) **pretraining scale & objective** — our 50M backbones are causal LMs pretrained on ~2B tokens vs. `ettin-17m`'s 1.7T-token native bidirectional MLM; and (b) **single random/BM25 negatives** — `splade-code-06B` had `0.0%` negative margins on our CoIR negatives (too easy), whereas production SPLADE distillation uses multiple cross-encoder-mined hard negatives per query.

---

## 6. v5 Ablation: BM25 Hard Negatives + Listwise KL-Div + Crouton v2.0.0 + Shortcake `--echo` + Sparsity Sweep

Trained on `78,951` queries × `4` BM25-mined hard negatives (`45,000` MS MARCO scored by `naver/splade-v3` + `13,951` StackOverflow-QA and `20,000` docstring-stripped CodeSearchNet scored by `naver/splade-code-06B`, `max_len=256`, per-teacher RMS normalized to `8.0`), `4,000` steps (`eff_bs=64`), Listwise KL-divergence (`kl_weight=1.0`, `kl_temp=1.0`, `infonce_weight=0.1`). Evaluated at both `2k` and `4k` steps; `4k` checkpoints shown below.

| Model | Recipe (`lq/ld`) | `arguana` | `nfcorpus` | `scifact` | `cosqa` | `stackoverflow-qa` | **General** | **Code** | **Overall** | Avg Doc Terms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`shortcake --echo` (v5c-2)** | KL, `0.005/0.008` | **0.3567** | **0.2437** | 0.5568 | 0.1600 | 0.6503 | **0.3857** | 0.4051 | **0.3935** | **105** |
| **`crouton-v2` (v5b-1)** | KL, `0.005/0.008` | 0.3390 | 0.2153 | **0.5954** | 0.1373 | 0.6656 | 0.3832 | 0.4014 | 0.3905 | 136 |
| **`crouton-v2-mlm` (v5b-2)** | KL, `0.005/0.008` | 0.3528 | 0.2429 | 0.5592 | 0.1140 | 0.6650 | 0.3850 | 0.3895 | 0.3868 | 139 |
| `crouton-v2` (v5d-1) | KL, `0.001/0.002` | 0.3286 | 0.2058 | 0.5782 | 0.1279 | **0.6838** | 0.3709 | 0.4058 | 0.3849 | 217 |
| `crouton-v2-mlm` (v5d-2) | KL, `0.001/0.002` | 0.3299 | 0.2244 | 0.5401 | 0.1260 | 0.6805 | 0.3648 | 0.4033 | 0.3802 | 198 |
| **`shortcake` causal (v5d-3)** | KL, `0.001/0.002` | 0.2777 | 0.2039 | 0.5028 | **0.1633** | 0.6744 | 0.3282 | **0.4189** | 0.3644 | 200 |
| `shortcake` causal (v5c-1) | KL, `0.005/0.008` | 0.2725 | 0.1937 | 0.4773 | 0.1147 | 0.6308 | 0.3145 | 0.3727 | 0.3378 | 113 |

### Findings from v5
1. **BM25 hard negatives + Listwise KL-div produced the largest jump of the project:** Crouton jumped **+0.0860 Overall** (`0.3045 -> 0.3905`, beating `splade-v3` on `so-qa` and beating `ettin-17m` on `scifact` and `nfcorpus` at 10.5x sparser vectors), and causal Shortcake jumped **+0.0712** (`0.2666 -> 0.3378`).
2. **Shortcake `--echo` completely eliminated Shortcake's General BEIR deficit:** Adding `--echo` on identical data lifted all 5 datasets simultaneously (`+0.0712` General, `+0.0324` Code, **`+0.0557` Overall** to **`0.3935`** at **`105` Avg Doc Terms**, 13.7x sparser than `ettin-17m`).
3. **131M-token Bidirectional MLM adaptation (`crouton-v2-mlm`) lifted OOD General BEIR and converged faster:** Improved `nfcorpus` (`0.2153 -> 0.2429`) and `arguana` (`0.3390 -> 0.3528`, R@100 `0.9243`) and reached `0.3869` by `step_2000`.
4. **Sparsity trade-off (`v5d`: `0.001/0.002` vs. `0.005/0.008`):**
   - Looser sparsity boosted Code across every model (`crouton-v2` `so-qa` hit **`0.6838`**; `shortcake` causal jumped across all 5 tasks to **`0.4189` Code** / `0.3644` Overall, tying BM25's `0.4190` Code avg).
   - For Crouton, tighter `0.005/0.008` remained better on General BEIR and Overall (`0.3905` vs. `0.3849`) at 1.6x sparser vectors; for 16k Shortcake, `0.001/0.002` improved both General and Code, making `shortcake --echo` at `0.001/0.002` a clear next candidate.

---

## 7. v6 Ablation: Native ZiG-BiMamba (`--bidirectional`), Echo Light Sparsity, and Crouton Size Scaling (`14M / 27M / 50M`)

Same `78,951` BM25-hard-negative KL dataset and `4,000`-step recipe as v5 (`eff_bs=64`, `kl_weight=1.0`, `infonce_weight=0.1`). Evaluated at `2k` and `4k` steps (`4k` checkpoints below; all 50 JSONs archived in `results/`). Shortcake `--bidirectional` wraps each Mamba mixer with a zero-initialized gated reverse scan (`fwd + gate * rev(mamba(rev(x)))`, +4,608 parameters total, 1x sequence length) alongside unmasked bidirectional attention.

| Model | Params | Recipe (`lq/ld`) | `arguana` | `nfcorpus` | `scifact` | `cosqa` | `stackoverflow-qa` | **General** | **Code** | **Overall** | Avg Q/D | Lat (ms)* |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`shortcake --bidirectional` (v6-3)** | 53.0M | KL, `0.001/0.002` | **0.3720** | **0.2371** | **0.5779** | 0.2103 | **0.7019** | **0.3956** | 0.4561 | **0.4198** | 85 / 203 | 27.8 |
| **`shortcake --echo` (v6-1)** | 53.0M | KL, `0.001/0.002` | 0.3445 | 0.2249 | 0.5527 | **0.2166** | 0.6973 | 0.3740 | **0.4570** | 0.4072 | 83 / 198 | 18.4 |
| **`shortcake --bidirectional` (v6-2)** | 53.0M | KL, `0.005/0.008` | 0.3576 | 0.2352 | 0.5374 | 0.1836 | 0.6600 | 0.3768 | 0.4218 | 0.3948 | **42 / 105** | 28.4 |
| `shortcake --echo` (v5c-2, ref) | 53.0M | KL, `0.005/0.008` | 0.3567 | 0.2437 | 0.5568 | 0.1600 | 0.6503 | 0.3857 | 0.4051 | 0.3935 | 43 / 105 | — |
| `crouton-base-50m` v2 (v5b-1, ref) | 50.0M | KL, `0.005/0.008` | 0.3390 | 0.2153 | 0.5954 | 0.1373 | 0.6656 | 0.3832 | 0.4014 | 0.3905 | 66 / 136 | — |
| **`crouton-small-27m` v2 (v6-5)** | 27.5M | KL, `0.005/0.008` | 0.3244 | 0.0954 | 0.5083 | 0.0588 | 0.5234 | 0.3094 | 0.2911 | 0.3021 | 90 / 163 | 16.8 |
| **`crouton-tiny-14m` v2 (v6-4)** | 14.3M | KL, `0.005/0.008` | 0.2803 | 0.0901 | 0.4096 | 0.0473 | 0.4600 | 0.2600 | 0.2537 | 0.2575 | 136 / 228 | 13.9 |

*\*Mean batch-1 query latency across the 5 `core` datasets on Colab L4.*

### Findings from v6
1. **New project record (`0.4198` Overall) with native ZiG-BiMamba (`shortcake --bidirectional`, v6-3):** Beats Echo at identical light sparsity (`0.4072`) by **+0.0126 Overall** and **+0.0216 General** (`arguana 0.3720`, `nfcorpus 0.2371`, `scifact 0.5779`), hits **`0.7019`** on `stackoverflow-qa` (`R@100 0.9122`), and reaches **`0.4561` Code** (beating BM25 `0.4190`, TF-IDF `0.4439`, and `naver/splade-v3` `0.4414`). Compared to `ettin-17m` (`0.4334` Overall at `1534/1434` active terms), `v6-3` wins on 3 of 5 tasks (`nfcorpus`, `scifact`, `cosqa`) while using **7.1x fewer document terms** (`203` vs. `1434`).
2. **ZiG-BiMamba matches or beats Echo at both sparsity points:** At tight sparsity (`0.005/0.008`), `v6-2` (`0.3948` Overall, `42/105` Q/D, `arguana R@100 0.9286`) edges out `v5c-2` Echo (`0.3935`, `105` D) with a +0.0167 Code lift (`cosqa 0.1836` vs. `0.1600`). At short `max_len=256`, batch-1 latency is faster for single-pass Echo (`18.4 ms`) than two-pass ZiG-BiMamba (`27.8 ms`) because kernel-launch overhead dominates short sequences; on longer documents, ZiG-BiMamba avoids Echo's 2x sequence duplication.
3. **Echo at light sparsity (`v6-1`, `0.001/0.002`) sets our highest Code average (`0.4570`):** Jumps +0.0519 Code (`cosqa 0.1600 -> 0.2166`, `so-qa 0.6503 -> 0.6973`) and +0.0137 Overall (`0.3935 -> 0.4072`) over tight Echo (`v5c-2`), confirming that Shortcake's 16k vocab benefits from `0.001/0.002` across all adaptation modes.
4. **Sharp capacity scaling across `crouton` v2.0.0 (`14M -> 27M -> 50M`):** Under identical `0.005/0.008` KL distillation, Overall nDCG@10 climbs from `0.2575` (`14.3M`) to `0.3021` (`27.5M`, +0.0446) to `0.3905` (`50.0M`, +0.0884). Both smaller tiers track lexical tasks reasonably (`scifact 0.4096 -> 0.5083`, `so-qa 0.4600 -> 0.5234`), while out-of-domain biomedical (`nfcorpus ~0.09` vs. `0.2153`) and semantic code (`cosqa ~0.05-0.06` vs. `0.1373`) show a steep threshold between 27M and 50M pretrained capacity.

---

## 8. Tokenizer Investigation (`docs/TOKENISATION.md`)

- Word-initial code fragmentation in `crouton` v1.0.0 (`18–20/102` code terms) was caused by training the tokenizer on prose only (`fineweb-edu` + `cosmopedia`), **not** by `wp-bpe-ppig` scoring.
- Adding 30% Python code to the tokenizer corpus (Experiment E) cut code fragmentation nearly in half (`18 → 10/102`) and boosted code compression by **+20.7%** (`3.448 → 4.159` B/tok) for only a 1.5% prose cost.
- Unmodified **`wp-bpe-ppig`** beat `pmi_floor` (negligible difference) and outperformed plain frequency BPE on downstream LM BPB (`2.522` vs. `2.549`), Morph F1, Identifier F1, and Exp E code fragmentation (`10` vs. `11`).
- Decision: stick with default `wp-bpe-ppig`, retrain the `crouton` v2.0.0 model family (`tiny`, `small`, `base`) on the 70/30 tokenizer, and evaluate those new backbones as **v5/v6**.

---

## 9. Next Steps (`v7`)

- Code is landed on `main` (`8597f07`) for `v7` ablations: `--init-norm-scale`, `--merge-vocab-variants` (case/space canonicalization: 25.1% merged on Crouton, 31.5% on Shortcake), and `--df-flops-alpha` (document-frequency weighted FLOPS).
- Broader validation (`--suite full` on `beir:fiqa` and multi-language `codesearchnet`) and BM25 + SPLADE hybrid (RRF) evaluation on the `v6-3` / `v7` winners (see [`docs/PLAN.md`](PLAN.md)).
