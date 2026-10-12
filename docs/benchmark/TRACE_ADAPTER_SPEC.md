# What the TRACE build must expose for ShiftBench and H5–H8

Status: specification, 2026-10-12. Nothing here exists yet: `traceot.shift_decompose` is not in
any branch (the shift v2 implementation, PR #8, has no Python entry point of that name).
The frozen adapter `benchmarks/shiftbench/sb_adapter.py::TraceMethod` already calls
`traceot.shift_decompose(XA, yA, XB, yB, seed=seed)` and reads `split`, `feat_comp`,
`feat_struct`, `intercept`. That file is frozen and is not edited; the extra outputs below are
read by a new subclass (`sb_adapter_trace_claims.py`, to be added when the binding exists) that
calls the same function and keeps the whole result in `Decomposition.extra`.

Notation: K = 4 channels (k = 0..3, shifted Legendre `P*_k`), p features, roles r ∈ {comp, struct},
player j = 0 is the intercept, j = 1..p are the input columns in input order. Hybrids:
A = (cov A, struct A), B = (cov B, struct B), C = (cov B, struct A), Cp = (cov A, struct B).
All values are plain floats / float64 numpy arrays; no NaN unless documented.

## 1. Entry point

```python
traceot.shift_decompose(
    XA, yA, XB, yB, *,              # float64, shapes (nA,p), (nA,), (nB,p), (nB,); features already centred by the harness
    seed: int = 0,                  # the only source of randomness
    K: int = 4,                     # reported channels (fit may use more; must say so in "config")
    folds: int = 5,                 # cross-fitting folds for the period-odds classifier
    trim: float = 0.99,             # weight trimming quantile
    return_model: bool = False,     # if True, result["model"] is a ShiftModel (section 3)
) -> dict
```

Errors: `ValueError` on non-finite input, p mismatch, or n < 2·folds. No silent imputation.

## 2. Returned mapping (all keys required unless marked optional)

| key | type | meaning | used by |
|---|---|---|---|
| `split` | `{stat: (comp, struct)}` for `mean, sd, q10, q50, q90` | order-free Shapley split of each frozen statistic computed on TRACE's four marginal laws (A, B, C, Cp), i.e. `comp = ½[(θ_B−θ_Cp)+(θ_C−θ_A)]` | frozen tracks (metrics 1, 3, 4, 5) |
| `split_aref` | same shape | A-reference split `(θ_C−θ_A, θ_B−θ_C)` | reported beside `split` (frozen truth is A-reference) |
| `feat_comp`, `feat_struct` | `(p,)` | mean-channel (k = 0) barycenter per-feature Shapley pieces on the λ₁ scale, pooled-mean centring | metric 2, H6 |
| `intercept` | float | k = 0 intercept structure piece | metric 2 |
| `config` | dict | K fitted, K reported, folds, trim, centring = "pooled_mean", version/commit of the core | provenance |
| `lmom` | `{"A","B","C","Cp": (K,)}` | marginal-law L-moments λ₁..λ_K of the four laws | H7, H8(b) |
| `lsplit` | `{"comp": (K,), "struct": (K,)}` | order-free channel split on the L-moment scale (SHIFT_V2 §4) | H7 |
| `lsplit_aref` | same | A-reference channel split | H7 (reported) |
| `bary_a` | `(K, p+1, 2)` | coefficient-scale pieces `a_{kjr}` (raw `c` scale) | H5, H6 |
| `bary_dc` | `(K,)` | `Δc̄_k` | H5 |
| `bary_phi` | `(K, p+1, 2)` | W2² pieces `φ_{kjr} = a_{kjr} Δc̄_k/(2k+1)` | H5, H6 |
| `marg_phi` | `{"comp": (K,), "struct": (K,)}` | channel W2² pieces for the marginal law | H5.2 |
| `resid_phi` | `{"comp": float, "struct": float}` | residual-channel pieces `⟨R_r, ΔR⟩` (Theorem 2) | H5.2 |
| `w2` | `{"marginal": float, "bary": float, "gap": float}` | W2²(F̂_A,F̂_B) from empirical quantile functions; Σ_k (Δc̄_k)²/(2k+1); and their difference | H5 |
| `trim_mass` | float | share of weight mass removed by trimming | reported |
| `interaction` | `(K, p+1)` | `I_{kj}` | reported (Theorem 3 relevance) |

Identities the harness checks (H5): `Σ bary_a[k] = bary_dc[k]`; `Σ bary_phi = w2["bary"]`;
`Σ marg_phi + Σ resid_phi = w2["marginal"]`; `lsplit.comp + lsplit.struct = lmom.B − lmom.A`;
each to `1e-9·max(1, |total|)`.

## 3. ShiftModel (for H5.1b, `return_model=True`)

```python
m.predict_quantile(X, u, period)  # X (n,p) float64, u (m,) in (0,1), period "A"|"B" -> (n, m): fitted Q_t(u|x), after any monotone rearrangement used in prediction
m.coef(X, period)                 # (n, K_fit) coefficients c_{t,k}(x) before rearrangement
m.rearranged                      # bool: whether predict_quantile rearranges
```

H5.1b averages `predict_quantile` over each period's own rows on a 256-node Gauss–Legendre grid and
compares `∫(Q̄_B − Q̄_A)²` with `Σ bary_phi`.

## 4. Determinism and invariance (needed for H5.4–H5.5)

* Same inputs and seed ⇒ bitwise-identical output, for any thread count.
* Row-order invariance: cross-fitting fold ids must not depend on row position. Required rule:
  fold id = `hash(seed, row bytes of (y, x)) mod folds` (or any function of seed and row content).
  Ties between identical rows are harmless (identical rows give identical contributions).
* Column permutation: permuting columns permutes `feat_*`, `bary_*[:, 1:]` and `interaction[:, 1:]`
  and nothing else.
* Period swap (`shift_decompose(XB, yB, XA, yA)`): `lsplit`, `bary_a` change sign, `bary_phi`,
  `marg_phi`, `resid_phi` unchanged up to 1e-9 (given the same folds, which the content-hash rule gives).

## 5. Resources

Accuracy grid up to n = 200,000 per period, p = 20; scale grid up to n = 10⁶, p = 100 within the
frozen 1,200 s / 8 GiB caps (4 threads). Bootstrap bands are produced by the harness calling
`shift_decompose` on resamples (B = 30 frozen band design; B = 100 for H8), so a single call at
n = 2,000, p = 5 should take well under 1 s for the H8 run (200 × 100 calls) to be practical.

## 6. Acceptance before any benchmark run

1. `pytest` for the binding: the identities of §2 on random data; determinism; the invariances of §4.
2. `python docs/theory/verify/uniqueness_checks.py` and `shift_v2_checks` pass.
3. A smoke call on `sb_dgp.make("null")` at n = 2,000 that is *not inspected for accuracy*
   (as in the frozen pre-registration), only for keys, shapes and run time.
