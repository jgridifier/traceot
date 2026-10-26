# BayesDB + CrossCat — Faraday-style cards for TRACE

**Date:** 2026-10-26 (America/New_York)  
**Pinned question:** What glass-box tabular **structure / mixture composition** ideas can TRACE borrow from CrossCat (and the BayesDB query layer) **without neural nets** — and without adopting a Bayesian database wholesale?  
**Constraints:** no banks/industry mentions in repo artifacts; no neural nets as the method; prefer algorithm borrow notes over new dependencies; verify stars / licenses / dates from sources (never invent).  
**Method:** Faraday pass on MIT ProbComp tabular NPBayes stack. Teaching page: [bayesdb_crosscat.html](../learn/bayesdb_crosscat.html). Follow-on plan: [CROSSCAT_MODERNIZATION.md](CROSSCAT_MODERNIZATION.md) · [crosscat_modernization.html](../learn/crosscat_modernization.html).

Related: [FARADAY_MIXTURE_CRATES.md](FARADAY_MIXTURE_CRATES.md) (RC2 numerics: \(Q_{\mathrm{mix}}\) vs barycenter), [BOARD_V5.md](BOARD_V5.md), [BOARD_V4.md](BOARD_V4.md) (RC2), [LANDSCAPE.md](LANDSCAPE.md).

---

## 1. One-sentence diagnosis

TRACE’s composition failure (RC2) is a **mixture geometry** problem. CrossCat’s complementary lesson is a **structure** problem: a single row-clustering for all columns underfits when different feature blocks need different groupings. The outer column partition + per-view row mixtures is the glass-box story worth stealing in words — not the BQL product.

---

## 2. Shortlist hooks (what to try later)

| # | Hook | Plain ask | Do **not** |
|---|------|-----------|------------|
| H1 | **Feature partition ≈ composition vs response** | Separate “who is here” features from “how \(Y\) responds” features so mix shifts and structural shifts are not forced through one latent | Port CrossCat Gibbs into Rust core |
| H2 | **Per-view category masses as explicit \(\pi\)** | Treat view-local cluster weights as the composition object TRACE’s mixing term should see when only \(\pi\) moves | Claim CrossCat fixes H7 without a TRACE check |
| H3 | **Column co-assignment as a diagnostic** | Posterior \(Pr[\text{same view}]\) as a pre-flight “shared signal?” map before trusting one additive decomposition | Build BayesDB / BQL as TRACE’s UI |

---

## 3. Faraday cards

### Card A — CrossCat (`probcomp/crosscat`)

- **Identity:** Domain-general Bayesian method for high-dimensional tables; Python + C++/Cython; [github.com/probcomp/crosscat](https://github.com/probcomp/crosscat); homepage linked in README as `http://probcomp.csail.mit.edu/crosscat/`.
- **Stars / license / activity (verified `gh api` 2026-10-26 ET):** **328★**; **Apache-2.0**; default branch `master`; tip commit visible in API **2019-11-19**; GitHub `pushed_at` **2024-02-22 UTC** (2024-02-21 evening America/New_York) — may not be a substantive code change on master (no post-2019 commits returned for `since=2020-01-01` on default branch). Created 2013-09-12 UTC. Open issues ~27.
- **What it implements:** Hierarchical NPBayes model — outer DP/CRP over **columns** (views), inner DP/CRP over **rows** per view, type-specific conjugate components (Beta–Bernoulli, Normal–Gamma, Dirichlet–Discrete). Gibbs with auxiliaries for view/category moves; `./check.sh` smoke path; `src/tests` present; examples (e.g. `dha_example.py`).
- **Math claim vs code:** JMLR/arXiv claim mixture-of-mixtures joint with explicit independencies across views; README matches (“nonparametric mixture modeling + Bayesian network structure learning strengths”). Predictive API = conditional simulate / density via posterior samples.
- **Quality signals:** Real tests + check script historically; Travis badge still in README (Travis largely sunset — treat CI as bitrot). Packaging aimed at ProbComp machines / conda era. Not abandoned in the sense of deleted, but **not actively maintained** as a modern library.
- **Fit to pinned question:** **Help** — feature partition × row clusters is the clearest glass-box composition/structure split in this stack.
- **Risk:** Bitrot; MCMC cost; hard views (non-overlapping column partitions) may be too coarse for TRACE’s overlapping feature channels; do not vendor C++/Cython into TRACE.

### Card B — bayeslite (`probcomp/bayeslite`)

- **Identity:** “BayesDB on SQLite” — BQL interpretation + storage; [github.com/probcomp/bayeslite](https://github.com/probcomp/bayeslite).
- **Stars / license / activity:** **942★**; **Apache-2.0**; tip commits ~**2020-03 / 2020-04**; `pushed_at` **2023-11-19 UTC**; open issues ~193. README: expect **rapidly shifting** code; **do not expect secure data handling**; Anaconda/conda install path; `./check.sh` smoke tests; Sphinx docs historically at ProbComp.
- **What it implements:** Query layer (BQL) over generative population models; default path historically tied to CrossCat-style baselines (see BayesDB paper).
- **Math claim vs code:** Paper defines BQL via model averaging over GPMs; repo is the engineering shell, not a new density model.
- **Quality signals:** Tests exist; maintenance sparse; issue count high relative to recent commits.
- **Fit:** **Partial** — UX/query inspiration only.
- **Risk:** Dependency hell / conda-era; security disclaimer; not a TRACE interface candidate.

### Card C — BayesDB legacy (`probcomp/BayesDB`)

- **Identity:** Landing / previous implementation; README redirects to bayeslite (commit message 2015-09-18).
- **Stars / license / activity:** **888★**; **Apache-2.0**; last push **2015-09-24**.
- **What it implements:** Nothing to run today — documentation pointer + license.
- **Fit:** **None** as code; keep the URL because that is what was saved for later.
- **Risk:** Confusion with bayeslite; cite bayeslite for implementation facts.

---

## 4. Papers (verified)

1. **Mansinghka, V., Shafto, P., Jonas, E., Petschulat, C., Gasner, M., & Tenenbaum, J. B. (2016).** CrossCat: A Fully Bayesian Nonparametric Method for Analyzing Heterogeneous, High Dimensional Data. *Journal of Machine Learning Research*, 17(138), 1–49.  
   - arXiv: [1512.01272](https://arxiv.org/abs/1512.01272) (submitted 2015-12-03)  
   - JMLR: [https://jmlr.org/papers/v17/11-392.html](https://jmlr.org/papers/v17/11-392.html)

2. **Mansinghka, V., Tibbetts, R., Baxter, J., Shafto, P., & Eaves, B. (2015).** BayesDB: A probabilistic programming system for querying the probable implications of data.  
   - arXiv: [1512.05006](https://arxiv.org/abs/1512.05006) (submitted 2015-12-15)  
   - Authors verified via arXiv Atom API on 2026-10-26.

---

## 5. Generative model (compact)

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

Predictive queries: sample category assignments for a new row given observed cells, then simulate or score targets; average over posterior samples (JMLR §2.5).

---

## 6. TRACE borrow vs ignore

| Borrow | Ignore / refuse |
|--------|-----------------|
| Column partition as composition vs response scaffold | Whole BayesDB product / BQL surface |
| Explicit per-view mixture weights \(\pi\) | Vendoring crosscat/bayeslite |
| Dependence \(z\)-matrix as diagnostic | Replacing OT channels with CRP Gibbs |
| Paper’s distractor-column lesson (one global mixture underfits) | Claiming production readiness of 2015–2020 code |

**Alignment with Faraday RC2 work:** Faraday R1–R3 fix *how* to form \(Q_{\mathrm{mix}}\) and weight \(\pi\). CrossCat answers a prior question: *which features share a mixture at all*. Use both lenses; do not collapse them into one experiment tonight.

---

## 7. Falsifiable checks (high level)

1. Synthetic independence of composition-block vs response-block columns → high posterior probability of different views.  
2. Pure \(\pi\)-shift between periods → TRACE mixing / composition term moves; if flat while CFM moves, fail.  
3. Any prototype stays out-of-tree until an explicit merge ask.

---

## 8. Verification log

| Claim | Source |
|-------|--------|
| CrossCat 328★, Apache-2.0, pushed_at 2024-02-22Z | `gh api repos/probcomp/crosscat` 2026-10-26 |
| Master tip commit 2019-11-19 | `gh api repos/probcomp/crosscat/commits/master` |
| bayeslite 942★, Apache-2.0, pushed_at 2023-11-19Z | `gh api repos/probcomp/bayeslite` |
| BayesDB 888★, redirect README, push 2015-09-24 | `gh api` + raw README |
| JMLR / arXiv metadata | arXiv API `id_list=1512.01272,1512.05006` + JMLR page |
| No banks/industry wording in this file | Manual pass |

**Stop rules:** No V5 scored suite and no Eulerian \(Q_{\mathrm{mix}}\) experiment started from this note.
