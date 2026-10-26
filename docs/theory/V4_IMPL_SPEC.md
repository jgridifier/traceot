# TRACE v4 implementation spec (hand-off for a cloud agent)

Status 2026-10-14 11:5x ET: written because the shared box was unresponsive (every Shell/Read call timed out or reported "still starting up" for ~50 min). **Nothing of v4 has been run or committed yet**: no verify scripts, no V4.md, no prereg, no code. Basis: `docs/research/BOARD_V4.md` (M1-M5, M7, M8) and Jared's direction of 2026-10-14. Rules: no neural networks; Rust core, thin Python bindings; never mention banks or any industry; commit + push after every step; long jobs via `nohup ... && ./finish.sh &` writing to `benchmarks/results/v4_*` so an interruption loses nothing.

Success standard (frozen in step 3): on accuracy, TRACE v4 is BEST, or statistically NON-INFERIOR to the best rival and >= 5x faster than CFM.

## Step 1. Verify the four sketched derivations first (`docs/theory/verify/v4_derivations.py`)
Pure numpy/scipy, seeded, prints PASS/FAIL per check, writes `verify/V4_DERIVATIONS.txt`. Drop or fix any that fail, and log that in V4.md.

- **D1 mixing-term expansion.** Y = mu(X) + sigma*eps. Claim: Q_M(u) = Qbar(u) - (v/(2 sigma)) f'(q(u))/f(q(u)) + O(E|delta|^3/sigma^2), v = Var mu(X). Check with eps in {N(0,1), skew-normal a=4, Gumbel}, delta ~ N(0, s^2), s/sigma in {0.05,0.1,0.2,0.4}. Exact Q_M by root-finding on F_M(y)=E F((y-mu)/sigma) (Gauss-Hermite 200 nodes) on u in [0.02,0.98]. PASS if sup-error of the residual after the correction scales like (s/sigma)^3 (log-log slope in [2.5,3.5]) and the correction removes >= 90% of the gap at s/sigma=0.1. Also check the Legendre-channel version (k=0..5) of the same.
- **D2 bin-marginal sufficiency.** Random additive piecewise-constant f_{kj} on bins; weights from raking (IPF) on period-t data to period-s bin marginals; check sum_i w_i f_kj(x_ij) == E_s f_kj to 1e-10 for all k,j and 100 random f. Negative control: logistic-odds weights must fail.
- **D3 half-sample unbiasedness.** D_hat^(1), D_hat^(2) from independent halves; MC over 20000 reps: mean <D1,D2> within 3 MC-SE of ||D||^2, while ||D_hat||^2 is biased by tr Var. Also check the symmetrised player form sums exactly (<= 1e-12) to <D1,D2>.
- **D4 path-attribution equivalence.** Along Q_tau=(1-tau)Q_A + tau Q_B in G-table space, compute Aumann-Shapley (integrated gradients, 2000-pt Gauss-Legendre in tau) of W(tau-path) = ||Qbar(G(tau)) - Qbar_A||^2 w.r.t. the 2pK player coordinates; compare to Phi of Theorem 4. Expected: EQUAL only when interactions I_kj = 0 (the game V_sym is linear, but the AS path splits interaction terms differently). PASS = equality to 1e-10 at I=0 AND an explicit report of the gap at I != 0. If the gap is nonzero, rewrite the claim as "AS along the geodesic equals Phi iff all interactions vanish" (do not claim general equivalence).

## Step 2. Theory: `docs/theory/V4.md`
The v4 object is the **model mixture law** per hybrid (s,t): F_[s,t](y) = E_{x~F_s} F_t(y|x), Q_[s,t] = F_[s,t]^{-1}, reported in the shifted-Legendre/L-moment channels. Sections:
1. Mixture target and channel Parseval (W2^2 = sum_k w_k dc_k^2 for marginals).
2. **Mixing-player game (M1).** Q_[s,t] = Qbar[s,t] + M[s,t]. Reference-symmetric switch game on marginal quantile functions; players = comp/struct per (channel, feature) + comp-mix/struct-mix. Theorem: efficiency to the model-marginal W2^2, period-swap antisymmetry, order invariance (proof extends Theorem 4 because V_sym(S)=<D_M, Q_S-Q_A> is linear). Per-feature split of mixing via v = sum_j Cov(f_0j, mu) with the D1 expansion; residual beyond 2nd order reported, not hidden.
3. **Bin-histogram balancing (M4)**: Proposition = D2. Entropy balancing (min KL s.t. bin marginals) for the barycenter; for the mixture (needs joint law) Riesz regression with dictionary = bin indicators + pairwise products of top-m features (m=10), closed-form ridge least squares.
4. **Fit what we report (M2)**: channel 0 by squared-loss boosting; channels k>=1 by least squares on PIT pseudo-outcomes (2k+1) Y P*_k(U_hat), U_hat from cross-fitted F_hat(Y|x) (2 iterations); ridge lambda*k^(2s) across k.
5. **Refit-free orthogonal inference (M3)**: split-product unbiased W2^2 pieces (D3), bin-level influence functions, multiplier bootstrap (B=2000, Rademacher), studentized label-shuffle test.
6. **Entry/exit with growth-death (M7, biology)**: when IDs observed, exact mass weights; when not, unbalanced OT (Hellinger-Kantorovich, Chizat et al. arXiv 1508.05216) on the 1-D outcome with rho stated; growth/death mass reported as its own player.
7. **Time-continuous decomposition (M8)**: only what D4 supports.
8. Novelty statement vs prior art (all verified in REFS_V4.md): CFM (Chernozhukov-Fernandez-Val-Melly arXiv 0904.0951) gives the counterfactual mixture but no per-channel/per-feature order-free exact budget; DFL/RIF give one functional at a time; Budhathoki et al. arXiv 2102.13384 give mechanism Shapley without a W2/L-moment channel budget; Waddington-OT/moscot do unbalanced transport without additive per-feature attribution. Claim to defend: first additive glass-box conditional-distribution model whose shift decomposition is an exact, order-free Shapley budget of the *marginal* W2^2 per L-moment channel and feature, including a mixing component, with refit-free bands. Label as [C] anything not proven.

## Step 3. ONE prereg, frozen before any result: `docs/benchmark/PREREG_V4.md`
Commit + push it alone, record the sha in the file header of the results doc.
- Scenarios: the 5 H7 shape scenarios (skew_struct, tail_struct, skew_comp, tail_comp, shape_mixed) at n=2000 and 20000; H8b coverage cells (same generators as TRACE_CLAIMS); ShiftBench main synthetic accuracy track (n=20000, p=20). Generators unchanged from `benchmarks/` (H7/H8 runner and ShiftBench).
- **Fresh seeds**: 9100-9149 (n=2000), 9200-9219 (n=20000), coverage 9300-9379, ShiftBench 9400-9429. Rivals CFM and DFL re-run on the SAME seeds; paired differences.
- Accuracy metric: existing graded-channel error (L-moment scale) for H7; mean-channel and per-feature error for ShiftBench.
- Bars (Holm across the family, alpha 0.05): per cell, v4 is BEST if paired mean err diff vs best rival has upper 95% CI < 0; NON-INFERIOR if upper 95% CI < margin delta = 0.10 * planted magnitude (H7) or 0.002 sd(y) (ShiftBench). Overall accuracy PASS = every cell best or non-inferior. Coverage PASS = pooled >= 0.90 and every cell >= 0.85. Speed: wall time of explain+bands vs CFM (same machine, 8 threads, CFM with its default threshold grid) at n=1e5 and 1e6, p=20; if any accuracy cell is only non-inferior, require speed ratio >= 5x at both sizes.
- Budget: whole run <= 2 h on 8 cores (B=2000 multiplier, no refits; CFM bands with its own bootstrap B=100 for timing only on 3 seeds). Early-stop rule: if stopped, cells not completed are INCOMPLETE, never imputed.

## Step 4. Rust implementation (crate: the existing TRACE core; add module `v4/`)
- `mixture.rs`: given fitted Q_t(u|x) on GL nodes (monotone after PAV), evaluate F_[s,t] on a y-grid (G=256, range = pooled 0.1%-99.9%) by averaging per-row CDFs via interpolation of the monotone Q (subsample cap 2e5 rows, deterministic chunked parallel sum), invert to Q_[s,t] on GL nodes, project to K channels. Tests: Gaussian location mixture equals sqrt(sigma^2+v) closed form to 1e-3; identity when X degenerate.
- `balance.rs`: raking/entropy balancing on bin indicators (Newton on the dual, fall back to IPF), Riesz ridge LS. Tests: D2 exactness; overlap diagnostics (max/median weight).
- `game.rs`: mixing-player game; budget sums <= 1e-9; swap antisymmetry; order invariance; per-feature mix split by covariance weights.
- `fit_l2.rs`: squared-loss channel 0 + PIT pseudo-outcome channels using the existing binned booster with an L2 gradient; Sobolev ridge across k.
- `infer.rs`: half-sample split-product estimator, bin-level influence functions, multiplier bootstrap (rayon, deterministic seeds), studentized permutation test.
- Python: expose `trace.explain_v4(xA, yA, xB, yB, config)` returning pieces, bands, timings; add a ShiftBench adapter `TRACE_V4` (see TRACE_ADAPTER_SPEC.md).
- `cargo test` + `pytest` green; ruff clean for docs/benchmarks.

## Step 5. Run, score, report
`benchmarks/run_v4.py` (resumable per-seed jsonl cache), `summarize_v4.py`, `render_v4.py` -> `docs/benchmark/V4_RESULTS.md` with verdict table (TRACE v4 vs CFM vs DFL, paired CIs, coverage, speed ratios), failures kept. PR to main; merge if CI green. Update `docs/learn/research_board.html` with a plain-language v4 section (what the mixture law is, the mixing player, results honestly).
