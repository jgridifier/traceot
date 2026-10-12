# Referee report: uniqueness of the TRACE shift budget

Independent adversarial review, written to a mathematics-journal standard.
Reviewed: `docs/theory/UNIQUENESS.md`, `docs/theory/verify/uniqueness_checks.py`,
`docs/benchmark/PREREG_TRACE_CLAIMS.md`. Context: `docs/theory/SHIFT_V2.md` on
`shift-v2` (`913f3e0`) and `docs/benchmark/PREREG_SHIFTBENCH.md`.
Independent numerical work is in `docs/theory/verify/referee_checks.py`
(permutation-average Shapley, Hosking’s explicit `P*_k`, a uniform open grid;
does not import or copy `uniqueness_checks.py`).

This file is a draft skeleton committed so work is not lost; the numbered
verdicts and computed constants are filled after the referee script runs.

## 0. Overall verdict

Theorems 1 and 2 are **correct as uniqueness theorems relative to the
factorisation axiom F**. They do **not** establish uniqueness among exact,
symmetric, efficient allocations of `W2²`. The switch-game Shapley value is a
genuine alternative, not a curiosity. Lemma 6’s identity `λ₂`-gap `= ED/8` is
correct for equal two-component mixtures and is **not** the barycenter-versus-
marginal `W2²` gap. The pre-registration bars H5–H8 are mostly coherent and not
rigged in favour of TRACE; several are tautological, ambiguous, or slightly
incoherent as calibration targets.

Suggested headline statement: *unique among two-stage rules that factor through
coefficient-scale Shapley pieces and then apply Shapley to the induced
quadratic game on those pieces*.

## 1. Verdict per claim

| claim | verdict | note |
|---|---|---|
| Lemma 1 (Parseval) | correct | |
| Lemma 2 (Shapley uniqueness) | correct | |
| Lemma 3 (stage-1 formula) | correct | |
| Lemma 4 (quadratic Hilbert game) | correct | |
| Corollary 4′ (channel separation) | correct | derived from orthogonality, as claimed |
| Lemma 5(a)(b) period / feature labels | correct | |
| Lemma 5(c) `y ↦ σy+m` | correct | translation invariance; `σ²` equivariance |
| Lemma 5(d) feature reparametrisation | correct with fixes | algebraic in `G`; not an estimator claim; joint maps are out of scope |
| Lemma 6 `λ₂` gap `= ED/8` | correct | 50-50 mixtures only; `≈0.47` is the gap, not `ED` |
| “barycenter vs marginal gap `= ED/8`” | wrong | if that is the intended reading of §6.2; Lemma 6 does not say this |
| Theorem 1 uniqueness | correct with fixes | uniqueness is relative to F |
| Theorem 1 formula and budget | correct | |
| Theorem 1 invariance list | correct with fixes | centring is a hidden parameter of per-feature *structure* |
| Theorem 2 | correct with fixes | same F caveat; residual optional; no per-feature claim |
| Axiom independence table | correct | S1 “equivalently” sentence is false as logic |
| F is a primitive / natural axiom | correct with fixes | F is architectural, reverse-engineered from SHIFT_V2 |
| Switch game undermines “the unique budget” | correct | it undermines the slogan, not the theorem-as-stated |
| Centring dependence | correctly disclosed | it still weakens per-feature uniqueness |
| H5 bars | correct with fixes | software exactness test; item 1 is tautological |
| H6 bars | correct | conservative, not rigged |
| H7 bars | correct with fixes | planted-channel list and primary rival metric are impure |
| H8 bars | correct with fixes | union-FPR vs 95% bands; CP convention unspecified |

## 2. Severity-ranked issues

To be expanded with numerical values after `referee_checks.py` completes.

1. **High — uniqueness is relative to F; the switch game is a live alternative.**
2. **High — `ED/8` is not the barycenter–marginal `W2²` gap.**
3. **Medium — per-feature structure is unique only after a centring convention.**
4. **Medium — H5.1 tests an algebraic identity, not an independent reconstruction of barycenter `W2²`.**
5. **Medium — H8(a) union FPR of two 95% bands, plus an unspecified Clopper–Pearson convention.**
6. **Medium — H7 “shape” scenarios move `λ₂` even at unit variance; planted `k` list is ambiguous.**
7. **Low — S1 “equivalently” wording; joint feature maps vs per-coordinate bijections; Holm family mixes test types.**

## 3. Cleaner theorem

See the closing section of the completed report.

## 4. Numerical status

Script not yet run in this draft.
