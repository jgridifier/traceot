# CrossCat modernization for TRACE — optimized Rust rebuild plan

**Date:** 2026-10-26 (America/New_York)  
**Pinned question:** Can we **rebuild** CrossCat’s generative model + inference in a TRACE-owned Rust core so that it **matches** a frozen `probcomp/crosscat` oracle within documented tolerances, then **beats** its wall-clock/memory by a large margin — scoped as TRACE-adjacent structure diagnostics (views / \(\pi\) / co-assignment), **not** a BayesDB product?  
**Constraints:** Rust core, thin Python bindings; no neural nets; no banks/industry mentions; glass-box; novel **or** match rivals while clearly faster; Faraday honesty on crates; do **not** start V5 \(Q_{\mathrm{mix}}\) from this note.  
**Teaching page:** [crosscat_modernization.html](../learn/crosscat_modernization.html)  
**Prior review:** [BAYESDB_CROSSCAT.md](BAYESDB_CROSSCAT.md) · merge `0258408`

Related: [FARADAY_MIXTURE_CRATES.md](FARADAY_MIXTURE_CRATES.md), [BOARD_V5.md](BOARD_V5.md), [BOARD_V4.md](BOARD_V4.md), [LANDSCAPE.md](LANDSCAPE.md).

---

## 0. Rebuild stance (not “inspired by”)

Jared clarified 2026-10-26 ET: this track is an **optimized rebuild**.

| Rule | Meaning |
|------|---------|
| Oracle | Frozen `probcomp/crosscat` (and thin bayeslite predictive hooks if needed) |
| Parity first | Match predictive / co-assignment / cluster structure / likelihood-style checks within **documented** tolerances |
| Then speed | Beat oracle wall-clock **and** memory by a large margin on the same synthetic suite |
| Scope | TRACE-adjacent APIs only; no full BQL DBMS |
| Trees | Do **not** vendor CrossCat/bayeslite/Lace into TRACE core; oracle harness may clone tips in a **separate** path/container |
| Kill | If parity fails after budgeted P1 work, stop — do not quietly loosen gates |

---

## 1. Oracle freeze

Verified via `gh api` on 2026-10-26 (America/New_York):

| Repo | Tip SHA (full) | Tip date (UTC) | License | Stars |
|------|----------------|----------------|---------|-------|
| [probcomp/crosscat](https://github.com/probcomp/crosscat) `master` | `1f2ac5a43a50ebd7aaa89f0c5ac3815a170848c5` | 2019-11-19 | Apache-2.0 | 328 |
| [probcomp/bayeslite](https://github.com/probcomp/bayeslite) tip | `211e5eb3821a464a2fffeb9d35e3097e1b7a99ba` | 2020-04-16 | Apache-2.0 | 942 |

**Bitrot workarounds (required for P0):**

1. Build a Docker/Podman image that pins OS packages + Python + numpy/Cython/setuptools to versions that successfully `python setup.py build` / `./check.sh` at the tip SHA.
2. Record **image digest** + pip freeze in `docs/research/oracle/` (when that harness lands — not in this plan-only PR).
3. Prefer CPU-only; isolate filesystem; heed bayeslite README: *“THIS SOFTWARE SHOULD NOT BE EXPECTED TO TREAT YOUR DATA SECURELY.”*
4. If tip will not build after a documented attempt, **kill/pause** — do not reimplement “what we think CrossCat did” without an oracle.

bayeslite is **optional**: use only when a predictive query shape is awkward to extract from CrossCat’s Python client alone. CrossCat remains the structure/likelihood oracle of record.

---

## 2. Generative model (spec freeze target — JMLR §2)

Outer CRP over columns \(z_d\), inner CRP over rows \(y^v_r\) per view \(v\), type-specific likelihoods \(L_d\):

\[
\alpha_D \sim \mathrm{Gamma}(1,1),\quad
z_d \sim \mathrm{CRP}(\alpha_D),\quad
\alpha_v \sim \mathrm{Gamma}(1,1),\quad
y^v_r \sim \mathrm{CRP}(\alpha_v),
\]
\[
x_{(r,d)} \sim L_d\big(\theta^{d}_{y^{z_d}_r}\big).
\]

Conjugates in the paper: Beta–Bernoulli (binary), Normal–Gamma (continuous), Dirichlet–Discrete (categorical). Collapsed / auxiliary Gibbs (Neal Alg. 8 style) over views, categories, hypers (JMLR §2.4).

**Complexity (paper):** category sweeps \(O(VRCD)\); inter-view sweeps \(O(DVCR)\); memory \(O(D+VC+VR)\). Empirical scale claim: tables up to ~10M cells with ~10–100 parallel chains × ~100–1000 transitions (JMLR §2.4–§2.6).

**Invariants for P0 (must be testable):**

- Exchangeability of CRP seating given \(\alpha\).
- Views partition columns (hard, non-overlapping).
- Given views, row partitions independent across views.
- Missing cells: contribute no likelihood term (as in reference implementation — verify against oracle tests, do not invent).
- Predictive path: assign new-row categories given observed cells, then simulate/score targets; average over posterior samples (JMLR §2.5).

---

## 3. Parity protocol

### 3.1 Shared inputs

- Synthetic generators with **known** view structure (e.g. two independent blocks + distractors), fixed seeds, fixed shapes \((R,D)\), missingness masks.
- Optional: public academic extracts already used in CrossCat examples **only if** redistributable and free of industry/bank framing in TRACE docs (prefer synthetics).

### 3.2 Shared inference knobs

Document in each parity run: `#chains`, `#iters`, init (prior vs other), hyper grids, RNG seeds for both engines. Prefer identical seeds where both stacks expose RNG control; otherwise freeze oracle seeds and map Rust RNG explicitly.

### 3.3 Quantities + suggested tolerances (freeze numbers in P0 review)

| Quantity | Compare how | Straw tolerance (finalize in P0) |
|----------|-------------|-----------------------------------|
| Column co-assignment \(Z_{ij}=\widehat{Pr}[z_i=z_j]\) | Entrywise max + Frobenius / \(\sqrt{D(D-1)}\) | e.g. max \(\le 0.05\) after equal burn-in budget **or** rank-correlation of dependence maps \(\ge 0.95\) |
| Row co-assignment within true views | Pairwise agreement / ARI after label alignment | ARI \(\ge 0.9\) on easy synthetics |
| \(\pi\) (category weights) | Sort masses; \(W_1\) on discrete weights | \(W_1 \le 0.05\) on easy synthetics |
| Predictive | Held-out conditional log score; imputation RMSE / 0-1 | Within 1–2% relative of oracle mean over seeds |
| Joint density of latent+data | JMLR §2.2 style score on aligned state | Relative error within documented float tolerance |

**Label alignment:** Hungarian / majority-map on sampled partitions before comparing labels; prefer co-assignment metrics that are label-invariant.

### 3.4 Kill criteria (parity)

- After P1 budget (oracle green + two Rust sampler iterations), any **core** quantity above fails on the easy synthetic suite → **stop rebuild**.
- Escalation paths (need explicit ask): diagnostic-only approx without full likelihood parity; or Lace-backed engine behind TRACE API.

---

## 4. Algorithms to keep / upgrade

| Piece | Keep from CrossCat | Upgrade candidates |
|-------|--------------------|--------------------|
| Model | Factorial views × per-view DP mixtures | Optional Pitman–Yor **only if** oracle comparison stays fair (default = DP) |
| Sampler | Collapsed Gibbs | + SAMS split-merge (Jain & Neal 2004); parallel chains; later slice/stick-breaking with transcoding (Vicentini 2023) if needed |
| Diagnostics | Multi-chain dependence traces (paper §2.6) | ESS/IAT, Geweke-style on \(\alpha\), view-count traces; reject “single long chain only” |
| Summaries | \(z\)-matrix, context similarity | First-class \(\pi\) export for TRACE composition term |
| VB foil | — | Collapsed VB-DP (Kurihara, Welling, Teh 2007) as **speed foil**, not default TRACE method (glass-box samples preferred) |

Survey context for partition models / compute: arXiv:2303.17182 (2023).

---

## 5. Target Rust architecture (modules)

```
crosscat_trace/           # name TBD; out-of-tree until explicit merge ask
  model/        # generative + hyper priors + invariants
  suffstats/    # typed conjugate tables; SoA layout
  partition/    # CRP state; alignment helpers
  kernel/       # collapsed Gibbs; optional sams
  chains/       # rayon over chains; diagnostics
  query/        # conditionals; co-assignment; pi
  serde_state/  # reproducible dumps
py/             # PyO3 thin API
oracle/         # container + parity harness (NOT linked into lib)
```

**Parallelism:** `rayon` over independent chains and, carefully, over views when sufficient statistics allow. Prefer cache-friendly contingency updates over clever locks.

**SIMD:** only after profiles show hot numeric kernels (Normal–Gamma updates, dense binary Beta–Bernoulli counts).

**What we will not build in-tree yet:** BQL parser, SQLite GPM shell, MML, streaming, neural proposals.

---

## 6. Crate shortlist (Faraday cards)

Pinned question for crates: *Which Rust libraries give solid CRP/DP/Gibbs primitives we can **borrow math from**, without vendoring abandoned trees or hiding NNs?*

### Card L1 — `lace` / `lace_cc` / `rv` (promised-ai)

- **Identity:** Probabilistic cross-categorization engine in Rust (+ Python); [github.com/promised-ai/lace](https://github.com/promised-ai/lace); docs at lace.dev. Authors include Baxter Eaves (also on BayesDB paper).
- **Verified crates.io 2026-10-26:** `lace` **0.9.1**, MIT (LICENSE), ~17.8k downloads, recent ~82, updated **2026-03-06**; GitHub **168★**, `pushed_at` **2026-10-23Z**. `lace_cc` **0.7.0** (2024-06-25). `rv` **0.20.1**, MIT, ~1.62M downloads, actively developed (DPMM example = Neal collapsed Gibbs).
- **What it implements:** PCC = CrossCat-style views×categories; algorithms documented in `lace_cc::alg`: **Gibbs** (Neal), **SAMS** (Jain & Neal), **FiniteCpu** (approx, not valid MCMC), **Slice**.
- **Math claim vs code:** Honest about FiniteCpu underfitting risk; Slice needs care with \(\alpha\) prior — good Faraday signal.
- **Fit:** **Foil + algorithm borrow.** Strongest existing Rust CrossCat. TRACE rebuild must either beat it on TRACE-scoped metrics or justify owning a thinner core. **Do not** add as TRACE dependency without explicit ask.
- **Risk:** Product surface broader than TRACE wants; depending on it skips “we own the oracle-matched core.”

### Card L2 — `oxicuda-stats` (DP module)

- **Identity:** [docs.rs/oxicuda-stats](https://docs.rs/oxicuda-stats) nonparametric::dirichlet_process — CRP simulate, stick-breaking, 1-D Normal DP mixture Neal Alg. 3.
- **Verified crates.io:** `oxicuda-stats` **0.5.5**, Apache-2.0, ~162 downloads, recent ~54, updated **2026-08-13**; repo cool-japan/oxicuda.
- **Fit:** **Partial** — borrow 1-D collapsed Gibbs notes; too narrow / young for TRACE core dependency.
- **Risk:** Low download count; CUDA-oriented umbrella; prefer reimplement from Neal (2000) with tests vs oracle.

### Card L3 — `sklears-clustering` DPMM

- **Identity:** CRP DPMM via **variational** inference (docs.rs).
- **Verified:** `sklears-clustering` **0.2.0**, ~1.3k downloads, updated **2026-07-14**.
- **Fit:** **Foil only** (VB speed). Not default for glass-box MCMC parity with CrossCat.
- **Risk:** Different inference class → not an oracle substitute.

### Card L4 — foundation crates

| Crate | Verified | Role |
|-------|----------|------|
| `rayon` 1.12.0 | huge downloads; 2026-04-14 | chain/view parallelism |
| `ndarray` 0.17.2 | 2026-01-10 | dense tables / suffstats (or custom SoA) |
| `rand` 0.10.3 | 2026-09-20 | seeded RNG for parity |
| `pyo3` 0.30.1 | 2026-10-14 | thin Python bindings later |

**Shortlist decision:** Implement TRACE kernels against Neal/Jain–Neal math; use `rv` **ideas** (and Lace alg notes) as references; treat Lace as the modern speed/quality foil; pin CrossCat as oracle. Prefer **borrow math over vendoring**.

---

## 7. Phased roadmap (detail)

### P0 — math/spec freeze

- Write invariants + parity tolerances as executable claims (even if tests are `#[ignore]` stubs).
- Oracle image builds `1f2ac5a` (+ optional bayeslite `211e5eb`).
- **Accept:** spec review; oracle smoke.  
- **Kill:** oracle unbuildable.

### P1 — sampler parity + speed

- Collapsed Gibbs in Rust on synthetics; optional SAMS.
- Parity suite green.
- Speed: target **≥5×** wall-clock vs CrossCat on agreed sizes; report memory; report Lace foil.
- **Kill:** parity fail; or no credible path to large speedup vs CrossCat **and** no advantage vs Lace for TRACE APIs.

### P2 — TRACE pre-flight APIs

- Export views, \(\pi\), co-assignment for composition-vs-response diagnostics.
- **Accept:** synthetic independent blocks → high Pr[different views]; Python API without BQL.
- **Kill:** no decision-quality gain vs correlation baseline.

### P3 — predictive / impute (optional)

- Only if P1–P2 clear.
- Still no BayesDB product.

**Explicit non-coupling:** P0 ≠ V5 \(Q_{\mathrm{mix}}\). Faraday RC2 / BOARD_V5 mixing numerics stay separate; CrossCat answers *which features share a mixture*, not *how to form \(Q_{\mathrm{mix}}\)*.

---

## 8. TRACE scoping

**In:** structure discovery foil/diagnostic for mixture/composition attribution.  
**Out:** BQL product, industry datasets, NN amortizers, silent Q_mix suite.

Executive recommendation lives on the teaching page and in the PR report: **yes to a phased Rust rebuild with oracle parity**, with Lace as foil and hard kill on parity failure.

---

## 9. Citations

1. Mansinghka et al. (2016). CrossCat. *JMLR* 17(138). https://jmlr.org/papers/v17/11-392.html · arXiv:1512.01272  
2. Mansinghka et al. (2015). BayesDB. arXiv:1512.05006  
3. Neal (2000). MCMC for DP mixtures. *JCGS* 9(2).  
4. Jain & Neal (2004). Split-merge MCMC for DP mixtures. *JCGS* 13(1).  
5. Kurihara, Welling & Teh (2007). Collapsed variational DP mixtures. IJCAI.  
6. Random partition survey: arXiv:2303.17182 (2023).  
7. Vicentini (2023). Transcoding sampler. arXiv:2304.02563  
8. Oracles: crosscat@`1f2ac5a`; bayeslite@`211e5eb`  
9. Lace: https://github.com/promised-ai/lace · crates.io `lace` 0.9.1  

## 10. Verification log

| Claim | Source |
|-------|--------|
| CrossCat tip SHA/date/stars | `gh api repos/probcomp/crosscat` + `/commits/master` 2026-10-26 |
| bayeslite tip SHA/date/stars | `gh api` 2026-10-26 |
| bayeslite security/shifting warnings | raw README |
| CrossCat Travis / install bitrot | raw README via GitHub API |
| JMLR complexity / mixing warning / 10M cells | arXiv HTML 1512.01272 |
| lace/lace_cc/rv/oxicuda/sklears/rayon/ndarray/pyo3 versions | crates.io API User-Agent fetch 2026-10-26 |
| lace MIT + alg docs | GitHub LICENSE + docs.rs `lace_cc::alg` |
| No banks/industry wording | Manual pass |

**Stop rules:** No Rust implementation in this PR. No V5 \(Q_{\mathrm{mix}}\) suite started here.
