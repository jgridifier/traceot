# OT and distributional-regression landscape for improving TRACE

Scope: classical statistics and ML only (no deep nets, no pretrained models). Interpretability must survive. Every citation is in `REFS.md`, with the URL where it was verified; inline references use arXiv ids or DOIs.

**Evidence labels used throughout**
- **[P] Proven**: a theorem in the cited paper, or a standard identity (re-derived in TRACE `THEORY.md` where noted).
- **[E] Empirical**: reported in the cited paper's experiments. I give the qualitative finding only. I did not transcribe numbers unless a location is given.
- **[C] Conjecture**: my own expectation for TRACE. It is untested and has to be checked with the experiments in §4.
- **[T] TRACE repo**: measured in `docs/TRACE_SIMS.md` / `docs/theory/THEORY.md` on main (read 2026-10-12).

---

## 0. Diagnosis: what limits TRACE now (from the repo)

1. **The mean is a by-product.** TRACE gets the mean as ∫Q(u|x)du from pinball-fitted channels. EBM fits the mean directly with squared loss. Pinball loss weights level u by the density at the quantile (THEORY §1.5, Finding 1). So in the tails the fit is low-information and noisy, and those tails feed the mean. With skew (vendor-spend draw) this costs MSE. [T]+[C]
2. **The optimiser is not the target projection.** With early-stopped first-order boosting under the pinball metric, a pure scale effect leaks into He3 (peak 0.408 for scale truth 1.2; zero at convergence or with Newton steps, `v1:1.15`). T3/T6 still fail after the v1 round. The local curvature is H_jk = ∫e_j e_k f(Q(u))du, so the right step is a Newton/Fisher step in that metric. [T][P]
3. **Tail channels are noisy and get screened out.** In S1 the x_shape feature is dropped by the gain screen (shape gain −0.000044). The theory doc gives permutation power of 0.28 for the planted tail at n=300 per period. So this is a variance and identification problem, not only a bias problem. [T]
4. **Calibration trades off CRPS.** Calibrated TRACE is worse than raw on CRPS in 2 of 3 cases, and raw coverage is too wide (0.952 nominal-80% on n=2e4). So the raw distribution is miscalibrated in scale, and global conformal shifting is too blunt. [T]
5. **The shift decomposition has a large cross term** (composition 0.0850 + structure 0.00136 + cross −0.0191 for total 0.0672). The split is order-dependent and not a Shapley split. [T]

---

## 1. Map of the landscape

### 1.1 Distributional regression and scoring rules

**CRPS-direct fitting / scoring-rule theory.** *Problem:* choose a proper loss for the full predictive distribution. *Core idea:* CRPS is strictly proper [P] (Gneiting & Raftery 2007, doi 10.1198/016214506000001437). CRPS(F,y) = 2∫₀¹ ρ_u(y − Q_F(u)) du (the quantile/pinball representation; Laio & Tamea 2007, doi 10.5194/hess-11-1267-2007; Gneiting & Ranjan 2011). For a point mass at y this is 2·∫|Q_F − y|-type transport to δ_y. A weight w(u) gives the *quantile-weighted CRPS*, which is still proper and lets you emphasise tails [P] (Gneiting & Ranjan 2011, doi 10.1198/jbes.2010.08110). Minimum-CRPS estimation of parametric predictive laws is classical [E] (EMOS, Gneiting et al. 2005, doi 10.1175/MWR2904.1).
*TRACE plug-in:* TRACE already minimises a midpoint-rule CRPS. Upgrades: (a) evaluate the integral on Gauss–Legendre nodes with weights, so the loss is a proper quadrature of the CRPS and not a midpoint approximation; (b) add an optional tail weight w(u) for the *shape-channel* updates only, to raise tail signal-to-noise. *Effect:* small CRPS gain from (a) [C]; (b) helps tail identification at some cost to central CRPS [C]. *Cost:* trivial (weights vector). *Risk:* a tail-weighted loss changes the target of evaluation, so always report unweighted CRPS.

**Energy / kernel scores, MMD.** *Problem:* proper scores for multivariate outcomes. *Core idea:* the energy score generalises CRPS to ℝ^d and is strictly proper [P] (Gneiting & Raftery 2007). Energy distance and MMD are the associated two-sample discrepancies (Székely & Rizzo 2013; Gretton et al. 2012). *Plug-in:* the loss for a future multi-output TRACE, and a shift statistic that complements W2 (it is less tail-sensitive). *Cost:* O(n²) or subsampled; easy. *Risk:* not decomposable per Legendre channel the way W2² is (TRACE's Parseval property is lost).

**Distributional random forests (DRF).** Ćevid et al. 2022 (arXiv 2005.14458; JMLR 23). *Problem:* nonparametric conditional distribution estimation, including multivariate Y. *Core idea:* random forest splits chosen by an MMD two-sample test between child nodes. The output is a weighting of training points that estimates the whole conditional law. [P] consistency results; [E] competitive in their experiments. *Plug-in:* not as a component (black-box weights). Use it as a **strong distributional baseline** next to EBM+conformal, and as a *screening* oracle: if DRF beats TRACE by a lot on CRPS, the gap is interactions/non-additivity, not the channel design. *Cost:* none in Rust (benchmark in R/Python). *Interpretability:* low.

**Quantile regression forests (QRF).** Meinshausen 2006 (JMLR 7). Same role as DRF: a baseline. It is consistent for conditional quantiles [P].

**Distributional regression forests / GAMLSS / boosted GAMLSS.** Rigby & Stasinopoulos 2005; Mayr et al. 2012 (doi 10.1111/j.1467-9876.2011.01033.x); Schlosser et al. (arXiv 1804.02921). *Core idea:* model the location, scale and shape parameters of a parametric family, each as an additive model, fitted by component-wise boosting. **This is the closest classical relative of TRACE.** TRACE is effectively a nonparametric-shape GAMLSS in quantile space. *Plug-in:* gamboostLSS's practice of a separate step size and stopping per parameter maps directly onto per-channel learning rates and per-channel early stopping in TRACE [C]. *Cost:* low.

**Additive quantile regression with calibrated smoothing (qgam).** Fasiolo et al. (arXiv 1707.03307; JASA). *Core idea:* smooth the pinball loss (extended log-F) so that Newton/REML smoothing selection works. [E] good calibration of additive quantile models. *Plug-in:* a smoothed pinball gives a positive-definite Hessian, which makes the Newton steps in §1.2 well defined without density estimation [C]. *Cost:* moderate.

**Transformation models / isotonic distributional regression.** Hothorn, Kneib & Bühlmann (arXiv 1201.5786); Hothorn, Möst & Bühlmann (arXiv 1508.06749); Henzi, Ziegel & Gneiting (arXiv 1909.03725). *Core idea:* model the conditional CDF as a monotone transformation (Bernstein basis), fitted by likelihood; IDR fits conditional CDFs under stochastic-order constraints, with no tuning. *Plug-in:* IDR on TRACE's *mean-score index* gives a calibration layer that is monotone and tuning-free. It could replace global split-conformal shifting [C]. Bernstein bases are already mentioned in T7. *Cost:* IDR is PAV-like, O(n log n). *Risk:* IDR needs an ordering covariate.

**Wasserstein / OT distributional regression.** Fréchet regression (Petersen & Müller, arXiv 1608.03012; AoS 2019, doi 10.1214/17-AOS1624); Wasserstein regression via tangent spaces (Chen, Lin & Müller, arXiv 2006.09660; JASA); distribution-on-distribution regression via OT maps (Ghodrati & Panaretos, arXiv 2104.09418); ADOPT (Song & Müller 2026, PMLR 300). *Core idea:* for distribution-valued responses in 1-D, regress quantile functions in L² (the W2 geometry) with a projection back onto monotone functions; or log-map to a tangent space at the Fréchet mean, run linear regression there, and exp-map back. [P] rates in each paper. *Plug-in:* TRACE already has Fréchet fits for distribution-valued rows. The tangent-space idea matters for scalar Y too: fit channels as *displacements of a reference quantile function* (TRACE's Q0), which is what TRACE does, so the main lesson is the **projection-after-additive-fit** being the W2 metric projection (PAV in L²(du)) — already in place. *Accuracy effect:* small for scalar Y [C].

**Gradient boosting with distributional objectives (NGBoost).** Duan et al. (arXiv 1910.03225). *Core idea:* boost the parameters of a parametric predictive law. Precondition each gradient by the inverse Fisher information of the scoring rule (the *natural gradient*; for CRPS, the Riemannian metric induced by CRPS). [E] fixes training dynamics in multi-parameter boosting. *Plug-in:* TRACE's channels play the role of the parameters. The CRPS metric tensor is exactly H_jk = ∫e_j e_k f(Q)du (THEORY §1.5). **A per-bin natural-gradient step is the Newton step THEORY recommends**, so this is shortlist item 2. XGBoostLSS (arXiv 1907.03178) is a second-order (Newton) distributional booster. *Cost:* K×K solves per bin (K ≤ 8): negligible.

### 1.2 Fixes for the mean-accuracy gap

**Location-anchored hybrid.** *Problem:* the mean is fitted through pinball losses in the tails. *Core idea:* fit channel 0 (the mean, or equivalently the first L-moment λ₁ = ∫Q du) by **squared loss** boosting, as in EBM, and fit the centred residual quantile function (channels k ≥ 1 in the orthonormal Legendre basis, all of which have zero mean in u) by CRPS on the residual y − m̂(x). Because ⟨e_k, 1⟩ = 0 for k ≥ 1, the mean of the predictive law equals m̂(x) exactly, whatever the higher channels do [P, by orthogonality]. The W2² Parseval split is unchanged: channel 0 is still the λ₁ term. *Precedents:* GAMLSS location–scale separation; split-conformal residual quantiles around a mean model (Lei et al., arXiv 1604.04173), which is exactly the baseline that wins now. *Effect:* matches EBM on MSE by construction (same objective, same shape functions) [C, high confidence]. CRPS should improve on skewed data, where the mean is badly estimated by pinball [C]. *Risk:* PAV after the fit can move the mean. Use the mean-preserving variant (project only the centred part, then re-add m̂), or check the mean after PAV. Under heavy tails, squared loss is less robust than pinball. *Cost:* low; reuses the booster.

**Newton / natural-gradient steps in W2 geometry.** Li & Montúfar (arXiv 1803.07033) define natural gradients via the OT (Wasserstein) metric; Arbel et al. (arXiv 1910.09652) give a kernelised estimator. For 1-D quantile-parameterised models the W2 metric on channel coefficients is the identity in the orthonormal basis (Parseval), so a "W2-natural gradient" is plain gradient in a_k coordinates [P, follows from THEORY Thm 1.1]. The *Fisher/CRPS* metric (NGBoost) is the one that adds information, because it is not the identity. *Recommendation:* use the CRPS-Hessian Newton step for fitting and W2 coordinates for reporting (this is what THEORY recommends). *Risk:* f(Q(u)) has to be estimated. Use a per-bin difference quotient of the current Q, or a smoothed pinball (qgam).

**Shrinkage across quantile levels / channels.** Composite quantile regression (Zou & Yuan 2008, doi 10.1214/07-AOS507), interquantile shrinkage (Jiang, Wang & Bondell 2013, doi 10.1080/10618600.2012.707454), non-crossing constraints (Bondell, Reich & Wang 2010, doi 10.1093/biomet/asq048) and rearrangement (Chernozhukov, Fernández-Val & Galichon, arXiv 0704.3649; Econometrica 2010). *Core idea:* borrow strength across u. Rearrangement only improves estimation error in L^p norms [P, Chernozhukov et al. 2010], and TRACE's PAV is the L² analogue. *Plug-in:* in the Legendre basis, shrinking across u is **a ridge penalty that grows with channel index**, λ_k ∝ (2k+1)^s or k^{2s}. This is equivalent to a Sobolev smoothness prior on Q(·|x) − Q0 [P for the equivalence in the Legendre/Sobolev sense, standard]. Fit K_fit = 8 with this penalty and report K = 3–4. This removes the misspecification bias THEORY found (0.284 → 0.057 at K_fit = 8, `v1:1.17`) without the extra variance [C]. *Cost:* trivial (diagonal ridge inside the Newton step).

### 1.3 Shift explanation and attribution

**DFL reweighting and the distribution-regression decomposition.** DiNardo, Fortin & Lemieux 1996 (doi 10.2307/2171954); Machado & Mata 2005 (doi 10.1002/jae.788); Chernozhukov, Fernández-Val & Melly (arXiv 0904.0951; Econometrica 2013, doi 10.3982/ECTA10582); RIF / unconditional quantile regression (Firpo et al. 2009, doi 10.3982/ECTA6822). *Core idea:* the counterfactual F^{(A,P_B)} = ∫F_A(·|x)dP_B(x) is "A's structure with B's composition". It can be built by integrating a conditional model (CFM) or by reweighting A's rows by dP_B/dP_A (DFL). CFM prove functional CLTs and bootstrap validity for counterfactual distribution and quantile processes [P]. *Plug-in:* TRACE already computes mixture quantiles Q^{(m,P)}. Add (i) **DFL weights from a period classifier** (a logistic model or EBM on x predicting the period), giving a model-light second estimate of the composition term; (ii) the doubly robust combination of the two (cf. the CFM framework) for the counterfactual quantile function [C]; (iii) uniform confidence bands for Q^{(A,P_B)} − Q^{(A,P_A)} from CFM's bootstrap. *Effect:* better shift explanation. If the two estimates disagree, that flags model misspecification [C]. *Cost:* moderate (a classifier plus weighted quantiles). *Risk:* poor overlap of P_A and P_B gives extreme weights. Trim and report overlap.

**Shapley for distributional differences.** Budhathoki et al. (arXiv 2102.13384) attribute a change in a joint distribution to changes in causal mechanisms (Shapley over factors). Zhang et al. (arXiv 2210.10769) attribute changes in model performance to shifts in conditional and marginal components via Shapley. SHAP: Lundberg & Lee (arXiv 1705.07874). *Plug-in:* TRACE's 2-player game {composition, structure} has a cross term (−0.0191 above). The **Shapley value of the 2-factor decomposition** splits it symmetrically: φ_comp = ½[(A,P_B)−(A,P_A)] + ½[(B,P_B)−(B,P_A)], in W2² of quantile functions. Extending the game to per-feature composition (shift P_j alone, holding the copula via rank-preserving reweighting) gives per-feature composition attributions [C]. Within structure, TRACE's additivity already makes per-feature, per-channel attribution exact (THEORY §1.3). *Cost:* 2-player is trivial; per-feature composition costs p reweightings. *Risk:* single-feature composition shifts are not unique (they depend on how the dependence is held fixed).

**Interpretable / feature-sparse transport maps for shift.** Kulinski & Inouye (arXiv 2210.10275; ICML 2023): explain P_A → P_B by an OT problem restricted to interpretable map classes (k-sparse mean shifts, cluster-wise shifts), with a "PercentExplained" metric. Cuturi, Klein & Ablin (arXiv 2302.04065): costs ½‖·‖² + τ with sparsity-inducing τ give entropic-type maps whose displacements T(x)−x are sparse. *Plug-in:* applied to the **covariate** laws P_A → P_B, this tells *which features* drive the composition term. It complements TRACE's structure attribution, which already says which features drive the conditional change. Concretely: compute a k-sparse displacement map on binned x and report per-feature displacement mass next to TRACE's per-feature composition Shapley. *Effect:* explanation only, no accuracy change. *Cost:* the Kulinski sparse mean-shift variant is closed-form-ish. The Cuturi et al. map needs Sinkhorn plus prox, which is moderate in Rust. *Risk:* on mixed categorical/ordinal bins the Euclidean cost is arbitrary. Use the cost on bin quantile ranks.

**Nonlinear DiD / changes-in-changes / distributional synthetic controls.** Athey & Imbens 2006 (doi 10.1111/j.1468-0262.2006.00668.x); Torous, Gunsilius & Rigollet (arXiv 2108.05858); Gunsilius (arXiv 2001.06118). *Core idea:* with few periods and a treated/control (or reference) cross-section, the counterfactual distribution is the reference period-to-period quantile map applied to the treated group's quantiles. In 1-D, CiC is a composition of quantile functions; Torous et al. generalise it with OT maps. Distributional synthetic controls take W2 barycentres of control-unit distributions. *Plug-in:* the natural extension of TRACE's period comparison to designs with a reference group, using **conditional** CiC per bin, which TRACE can evaluate because it has Q(u|x) in both periods [C]. *Relevance to small-T, large-N:* these are exactly the identification tools for two-to-few periods with a large cross-section.

### 1.4 OT map estimation with statistical rates (mostly for multi-dimensional pieces: covariate shift and multi-output)

1-D scalar outcomes need none of this: W2 is closed-form via quantiles [P]. It matters for (a) the covariate law P_A → P_B in ℝ^p and (b) multi-output Y.

- **Curse of dimensionality.** Plug-in empirical OT converges at ~n^{−1/d} [P] (stated in the abstract of Genevay et al., arXiv 1810.02733). Survey: Panaretos & Zemel (arXiv 1806.05500); computation: Peyré & Cuturi (arXiv 1803.00567).
- **Entropic OT / debiased Sinkhorn divergences.** Feydy et al. (arXiv 1810.08278): S_ε = OT_ε − ½OT_ε(α,α) − ½OT_ε(β,β) is a positive-definite, metrising divergence that interpolates between OT and MMD [P]. Genevay et al. (arXiv 1810.02733) and Mena & Niles-Weed (arXiv 1905.11882) give sample complexity and a CLT. The rate is parametric in n with constants that depend on ε and d [P]. *TRACE use:* the multivariate composition-shift statistic with a permutation null. *Cost:* Sinkhorn in Rust is about 150 lines; the log-domain version is needed for stability.
- **Smooth/entropic map estimators.** Hütter & Rigollet (arXiv 1905.05828): first minimax rates for smooth Brenier maps (wavelet semi-dual) [P]. Manole et al. (arXiv 2107.12364): plug-in estimators reach those rates [P]. Pooladian & Niles-Weed (arXiv 2109.12004): the barycentric projection of the entropic plan is cheap and has finite-sample guarantees "comparable" to the others under smoothness [P, per abstract]. *TRACE use:* the counterfactual covariate map x ↦ T(x) for "move a period-A unit to its period-B position", giving a per-unit counterfactual prediction Q_A(u|T(x)). The Pooladian estimator is the practical choice.
- **Displacement-sparse maps.** Cuturi et al. 2023 (above). Sparse displacements are interpretable and the method is classical (no nets needed in the entropic variant) [E].
- **Low-rank OT.** Scetbon, Cuturi & Peyré (arXiv 2103.04737): the coupling is factorised through r anchors, giving linear-time OT. The anchors act as interpretable "archetypes" of the shift [C]. Useful at n ~ 10⁶ rows.
- **Sliced / max-sliced W.** Nadjahi et al. (arXiv 2003.05783): statistical/topological properties, with dimension-free sample complexity of sliced divergences [P per paper]. Manole, Balakrishnan & Wasserman (arXiv 1909.07862): minimax confidence intervals for sliced W [P]. Deshpande et al. (arXiv 1904.05877): max-sliced W (the worst-case direction). *TRACE use:* **max-sliced W restricted to coordinate directions is literally per-feature 1-D W between P_A,j and P_B,j**, which TRACE can already compute on bins. A general max-sliced direction gives a single interpretable linear "shift index" [C].
- **Gaussian / Bures–Wasserstein.** Closed-form W2 between Gaussians (mean term plus Bures term on covariances; Bhatia, Jain & Lim, arXiv 1712.01504). *TRACE use:* a cheap decomposition of covariate shift into mean-shift versus covariance-shift parts, with an exact per-feature split of the mean part [P]. A good first-order composition explainer.
- **Linearised OT embeddings.** Moosmüller & Cloninger (arXiv 2008.09165): embed distributions via maps from a reference, after which linear methods apply. TRACE's quantile-coordinates-relative-to-Q0 are the 1-D instance of this.

### 1.5 Multivariate / conditional quantiles via OT (route to multi-output TRACE)

Chernozhukov, Galichon, Hallin & Henry (arXiv 1412.8434; AoS 2017, doi 10.1214/16-AOS1450) define Monge–Kantorovich depth, quantiles, ranks and signs via the OT map to a reference (uniform on the ball) [P]. Hallin, del Barrio, Cuesta-Albertos & Matrán (arXiv 1806.01238; AoS 2021, doi 10.1214/20-AOS1996) prove center-outward distribution and quantile functions with Glivenko–Cantelli and distribution-freeness of ranks [P]. Carlier, Chernozhukov & Galichon (arXiv 1406.4643) define vector quantile regression: a conditional OT map from uniform U ∈ [0,1]^d to Y given x, linear in x, solved as an LP [P]. *Plug-in:* multi-output TRACE = additive VQR, Q(u|x) = Q0(u) + Σ_j f_j(x_j, u) with u ∈ [0,1]^d. The tensor-Legendre basis in u preserves Parseval for the *L²(u)* norm, but **this L² is no longer W2** unless the map is a gradient of a convex function (cyclical monotonicity) [P]. So the PAV step must become a projection onto gradients of convex functions, which is much harder. *Cost:* high. *Risk:* loses the exact W2² additivity that makes TRACE's attribution exact. **Recommendation:** for multi-output, use per-output TRACE plus a copula/rank-dependence layer (center-outward ranks of the residual vector) before attempting full VQR [C].

### 1.6 Robustness and calibration

**Wasserstein DRO.** Mohajerin Esfahani & Kuhn (arXiv 1505.05116); Gao & Kleywegt (arXiv 1604.02199; MOR 2023, doi 10.1287/moor.2022.1275); Gao, Chen & Kleywegt (arXiv 1712.06050), who show W-DRO ≈ empirical risk + ε·(gradient/Lipschitz norm), i.e. variation regularisation [P]; review: Kuhn, Shafiee & Wiesemann (arXiv 2411.02549; Acta Numerica 2025). *TRACE status:* the bin-grid robust objective exists, but T13 lost 0/100 on period-B pinball while the certificate held 100/100 [T]. This matches theory: W-DRO buys a *certificate*, not better accuracy, when the shift is not adversarial. *Plug-in recommendation:* use the variation-regularisation equivalence. A total-variation penalty on adjacent-bin differences of f_j is exactly the "Lipschitz in x" regulariser, and it doubles as smoothing, which can *help* accuracy at small bins [C]. Keep the DRO certificate as a reporting tool.

**Conformal under shift.** Split conformal (Lei et al., arXiv 1604.04173); CQR (Romano, Patterson & Candès, arXiv 1905.03222); distributional conformal via PIT ranks (Chernozhukov, Wüthrich & Zhu, arXiv 1909.07889; PNAS 2021); weighted conformal under covariate shift (Tibshirani, Barber, Candès & Ramdas, arXiv 1904.06019) [P coverage with known/estimated likelihood ratio]; non-exchangeable conformal with a coverage gap bounded by TV-type deviations (Barber, Candès, Ramdas & Tibshirani, arXiv 2202.13415) [P]; adaptive conformal for sequential shift (Gibbs & Candès, arXiv 2106.00170) [P long-run coverage]. *Plug-in:* replace TRACE's single global conformal shift with **CQR-style per-level adjustment on the PIT scale** (DCP). This rescales the whole predictive law instead of shifting it, which addresses the over-wide raw coverage without the CRPS loss [C]. Use the period classifier's odds (the DFL weights) as the weights in weighted conformal when calibrating on period A and predicting period B [P under correct weights]. *Cost:* low.

**OT-based covariate-shift reweighting / domain adaptation.** Shimodaira 2000 (importance weighting, doi 10.1016/S0378-3758(00)00115-4); Courty et al. (arXiv 1507.00504), which transports source to target via regularised OT. *Plug-in:* as alternatives to classifier-based weights, entropic-OT barycentric maps of A-rows to B give "transported A rows" for training TRACE on period B. Risk: OT maps change x, so feature effects are evaluated at moved points. Prefer reweighting for interpretability.

### 1.7 Unbalanced and partial OT

Chizat, Peyré, Schmitzer & Vialard (arXiv 1607.05816), scaling algorithms; Liero, Mielke & Savaré (arXiv 1508.07941), Hellinger–Kantorovich theory; Séjourné, Peyré & Vialard (arXiv 2211.08775), survey; Chapel, Alaya & Gasso (arXiv 2002.08276), partial OT. *Problem:* periods with different total mass or units entering/exiting. *Core idea:* replace hard marginal constraints with KL (or TV) penalties, so mass can be created or destroyed at a price. The optimal plan separates "transported" mass from "created/destroyed" mass. *Plug-in:* with unit IDs, add a **third term to TRACE's shift decomposition, entry/exit**: structure + composition (stayers, transported) + entry − exit. With a TV penalty in 1-D, the partial-OT solution on sorted samples is computable by a modified quantile matching [C]. *Cost:* the 1-D unbalanced case is moderate; Sinkhorn-UOT is about 50 extra lines on a Sinkhorn core. *Risk:* the split depends on the penalty ρ, so report it across a grid of ρ values.

### 1.8 Small-T, large-cross-section setting

Thin literature specifically on OT. Relevant pieces: Fréchet change-point detection for distribution sequences (Dubey & Müller, arXiv 1911.11864) [P]; autoregressive OT models for distribution-valued time series (Zhu & Müller, arXiv 2105.05439) [P]; CiC / OT-DiD / distributional synthetic controls (§1.3); CFM's counterfactual inference with a functional bootstrap (§1.3). For T = 2–5 periods with n large, the binding constraint is **within-period dependence and the few periods**, not n. Use permutation (TRACE §2.5) and cross-fitting for squared statistics (TRACE §2.4). A W2 change-point or AR model needs more periods than are usually available. **Evidence is thin here** (see §5).

---

## 2. Comparison table

Evidence quality: **A** = theorem plus broad empirical support; **B** = theorem or strong experiments in the cited paper; **C** = sparse or only indirectly applicable; **Conj** = my conjecture for the TRACE effect.

| Method | Problem solved | Plugs into TRACE how | Accuracy impact expected | Interp. kept? | Rust cost | Evidence quality |
|---|---|---|---|---|---|---|
| GL-quadrature CRPS | midpoint CRPS approximation; channel correlation | weights on GL nodes in the loss | small CRPS ↑ (Conj) | yes | trivial | A (quadrature exactness), Conj effect |
| Quantile-weighted CRPS | weak tail signal | w(u) on shape-channel updates | tail ID ↑, central CRPS ≈/↓ (Conj) | yes | trivial | A (propriety), Conj |
| Location-anchored hybrid (mean by L2, residual channels by CRPS) | MSE gap vs EBM | channel 0 = squared-loss booster; k≥1 on residual | MSE ≈ EBM; CRPS ↑ on skewed (Conj, high conf.) | yes (better: exact mean channel) | low | B (GAMLSS, split-conformal precedent) |
| Newton / natural-gradient (CRPS Fisher) per bin | scale→shape leakage, slow convergence | K×K Hessian solve per bin | CRPS ↑, T3/T6 leakage ↓ (Conj; THEORY v1:1.15 local result) | yes | low | B (NGBoost, THEORY) |
| Channel ridge ∝ k^{2s}, K_fit>K_report | tail variance; misspecification bias | diagonal penalty in Newton step | CRPS ↑ small n; tail ID ↑ (Conj) | yes | trivial | B (composite/interquantile QR) |
| Smoothed pinball (qgam) | non-smooth Hessian | replace ρ_u with smooth surrogate | stabilises Newton (Conj) | yes | low | B |
| Per-channel step/stopping (gamboostLSS) | channels need different complexity | separate lr and patience per channel | CRPS ↑ (Conj) | yes | low | B |
| TV/fused penalty on bins (W-DRO ≈ variation reg.) | noisy bin tables; robustness | penalty on adjacent-bin diffs | small ↑ at small n (Conj) | yes (smoother shapes) | low | A (Gao–Chen–Kleywegt) |
| DCP / CQR calibration on PIT scale | calibrated is worse than raw on CRPS | replace global shift | CRPS of calibrated ↑, coverage kept (Conj) | yes | low | A (coverage), Conj CRPS |
| Weighted conformal (period odds) | calibration across periods | weights from period classifier | coverage in B (P if weights right) | yes | low | A |
| IDR calibration layer | calibration | monotone recalibration on mean index | CRPS ↑ (Conj) | yes | low | B |
| DFL / CFM counterfactual + doubly robust | composition term validity, bands | classifier weights + TRACE mixture | explanation; bands | yes | moderate | A |
| 2-factor Shapley of W2² decomposition | cross term | symmetric average of 2 orders | explanation | yes | trivial | A (Shapley axioms) |
| Per-feature composition Shapley | which x drives composition | p reweightings | explanation | yes | moderate | C |
| Kulinski–Inouye interpretable maps | explain covariate shift | k-sparse shift on binned x | explanation | yes | low–moderate | B (E) |
| Feature-sparse entropic maps (Cuturi et al.) | sparse displacement explanations | Sinkhorn + prox on x | explanation | yes | moderate | B |
| Debiased Sinkhorn divergence | multivariate shift test | statistic + permutation | testing | partly | moderate | A |
| Entropic map (Pooladian–Niles-Weed) | counterfactual unit map | Q_A(u \| T̂(x)) | counterfactual predictions | partly | moderate | A |
| Low-rank OT | scale, archetypes | anchors as shift summary | — | yes | moderate | B |
| Sliced / max-sliced W | high-dim shift | per-feature or 1 direction | testing/explanation | yes | low | A |
| Bures–Wasserstein | quick covariate-shift split | mean vs covariance terms | explanation | yes | trivial | A |
| Unbalanced / partial OT | entry/exit, mass change | extra decomposition term | explanation | yes | moderate | A (theory), C (this use) |
| CiC / OT-DiD / dist. synthetic controls | small-T counterfactuals | conditional CiC per bin | explanation/causal | yes | moderate | A/B |
| Fréchet / Wasserstein regression, ADOPT | distribution-valued responses | already present | — for scalar Y | yes | done | A |
| VQR / MK quantiles / center-outward | multi-output | additive VQR or rank layer | enables multi-output | partly | high | A (theory), C (additive) |
| DRF, QRF | strong nonparametric baselines | benchmark only | — | no | none | A |
| Energy score / MMD | multivariate scoring | multi-output loss, alt. shift stat | — | partly | low | A |
| W natural gradient (Li–Montúfar) | geometry-aware steps | ≡ plain gradient in orthonormal Legendre coords for 1-D | none beyond Parseval (P) | yes | — | A |

---

## 3. Proven vs. empirical vs. conjecture: summary

- **Proven (standard or cited):** CRPS propriety and the pinball-integral representation; quantile-weighted CRPS propriety; 1-D W2 = L² of quantile functions, Parseval in the orthonormal Legendre basis (THEORY Thm 1.1); orthogonality ⇒ mean of predictive law = channel 0 when k≥1 channels are u-centred; rearrangement/isotonisation never increases L^p error to a monotone target (Chernozhukov et al. 2010); W-DRO ≈ variation regularisation (Gao, Chen & Kleywegt); weighted-conformal coverage under covariate shift with known likelihood ratio (Tibshirani et al. 2019); CFM functional CLT and bootstrap for counterfactual distributions; n^{−1/d} plug-in OT rate versus parametric Sinkhorn-divergence rate; minimax smooth-map rates (Hütter & Rigollet); MK/center-outward quantile theory.
- **Empirical (from papers, qualitative):** NGBoost competitive with natural gradient; DRF competitive for multivariate conditional distributions; qgam calibrated additive QR; Kulinski–Inouye better detail/interpretability trade-off on their datasets; Cuturi et al. sparse displacements recovered in high dimension.
- **Conjectures (mine; marked [C] above):** all the TRACE-specific accuracy effects in the table.

---

## 4. Ranked shortlist: top 5 modifications, each with a falsifiable experiment

Common protocol: the three existing benchmark cases (S1 period 2, vendor spend, n=2e4 p=10), plus 20 fresh seeds for each. Paired comparison per seed. "Pass" requires the paired mean difference to be in the stated direction with a one-sided paired t (or Wilcoxon) p < 0.05 over seeds, and the stated magnitude bar. Criteria are fixed before running, in the repo's style.

### #1 Location-anchored hybrid (squared-loss mean channel + CRPS residual channels)
- **Rationale:** the MSE gap is structural, because TRACE never optimises squared error. Orthogonality of Legendre channels makes the predictive mean equal to the L2-fitted channel 0 exactly, so EBM's MSE can be matched without giving up the distributional channels. Residual channels fitted on y − m̂(x) have a better-conditioned target (the location noise is removed) [C].
- **Design:** channel 0 by cyclic squared-loss boosting with the same bins, rounds and early stopping as EBM, using TRACE's own Rust booster with an L2 gradient. Then channels 1..K−1 by pinball/CRPS on residuals; PAV on the centred part; re-add m̂.
- **Baseline:** EBM + split-conformal residuals (current winner) and TRACE raw.
- **Metrics:** MSE of mean; CRPS (GL-exact); 80/90% coverage.
- **Pass bar:** MSE ≤ 1.01× EBM's on all 3 cases, **and** CRPS ≤ EBM+conformal on ≥ 2 of 3 cases, **and** no case worse than TRACE raw's CRPS by > 1%.
- **Falsified if** MSE is still > 1.01× EBM (it would mean the booster or binning differs, not the objective) or CRPS gets worse than TRACE raw.

### #2 Per-bin Newton / natural-gradient steps on (smoothed) CRPS in the orthonormal Legendre basis on GL nodes
- **Rationale:** the curvature H = ∫e_j e_k f(Q)du is what couples channels. A Newton/Fisher step removes the scale→He3 leakage at convergence (THEORY `v1:1.15`) and converges in far fewer rounds than lr-damped subgradients, which reduces early-stopping bias [P locally / C globally]. NGBoost is the precedent.
- **Design:** f(Q(u)) from a smoothed pinball (qgam-style) or from per-bin finite differences of the current Q; damped Newton (Levenberg λ); K_fit = 6–8 on n_GL ≥ K_fit nodes.
- **Baseline:** current boosting (lr 1.0, 40 rounds).
- **Metrics:** CRPS; T3/T6 leakage ratio (He3 energy of pure scale feature divided by v0 He3 energy); rounds to convergence.
- **Pass bar:** leakage ratio ≤ 0.10 (v1 got 0.50, bar 0.05 not reached), CRPS not worse on any case by > 0.5%, and better on ≥ 2 of 3.
- Note the T3 dispute: score leakage against the *pinball-population* oracle, not the L2 oracle, or both, declared in advance.

### #3 Channel-indexed ridge (Sobolev shrinkage across u) with K_fit > K_report, plus quantile-weighted CRPS for the shape channel
- **Rationale:** interquantile shrinkage in Legendre coordinates is a diagonal ridge on higher channels. It cuts the variance that makes tail effects unidentifiable while K_fit = 8 removes the truncation bias THEORY measured. The tail weight raises shape-channel SNR. This targets "planted tail effects are hard to identify" [C].
- **Design:** penalty λ·Σ_k k^{2s} a_k² with s ∈ {1, 2} and λ by validation CRPS; tail weight w(u) ∝ 1 + γ(u−½)² on channels ≥ 2 only.
- **Baseline:** v0/v1 fit (no penalty, K = 3).
- **Metrics:** S1 x_shape retained by the screen (rate over 50 seeds); He3 energy ratio x_shape / next feature; CRPS on all cases.
- **Pass bar:** x_shape retained in ≥ 80% of seeds (currently dropped); energy ratio ≥ 2 (relaxed from 3 given the measured oracle 0.753–2.0 limits, declared up-front); CRPS not worse than baseline by > 0.5% anywhere.

### #4 Shift decomposition v2: 2-factor Shapley + DFL/doubly-robust counterfactual + uniform bands (+ optional entry/exit term via unbalanced OT)
- **Rationale:** the cross term (−0.0191 of 0.0672) makes the current split order-dependent. Shapley removes this by construction [P, axioms]. DFL weights give a second, model-light composition estimate, and disagreement between the two diagnoses misspecification. CFM gives valid bands [P].
- **Design:** planted simulations with known truth: (a) pure composition shift, (b) pure structure shift in one feature's scale channel, (c) both, (d) 10% entry/exit of units. Compute Shapley composition/structure; DFL vs model-based composition; per-feature structure attribution.
- **Baseline:** current mixture decomposition.
- **Metrics:** absolute error of each attributed term against the oracle (computed from the generator); bootstrap band coverage of the oracle counterfactual quantile function; false structure attribution in (a).
- **Pass bar:** in (a), structure share ≤ 5% of total W2² (permutation floor respected); in (b), composition share ≤ 5%; uniform 90% band coverage ≥ 0.85 over 200 reps; in (d), the unbalanced term recovers entry/exit mass within 20% at the declared ρ.

### #5 PIT-scale distributional conformal (DCP/CQR) with period-odds weights, replacing the global conformal shift
- **Rationale:** the raw model is over-dispersed (0.952 coverage at nominal 80% on n=2e4), so a location shift cannot fix it. Rank/PIT recalibration rescales all levels jointly and keeps marginal coverage [P]. Weighting by P_B/P_A odds extends validity to period B [P with correct weights]. Expected: calibrated CRPS ≤ raw CRPS, unlike now [C].
- **Design:** calibration set from period A; predict on period B in S1 (covariate-shifted and conditional-shifted variants).
- **Baseline:** current split-conformal calibrated TRACE; EBM + conformal.
- **Metrics:** CRPS; coverage at 80/90% in period B; interval width.
- **Pass bar:** calibrated CRPS ≤ raw CRPS on all 3 cases; coverage within ±2 pp of nominal on in-period cases; within ±4 pp in period B under pure covariate shift with weights (and documented failure under conditional shift, which weights cannot fix: an expected-negative control).

**Not in the top 5, but next:** Kulinski–Inouye / sparse-map explanation of covariate shift (explanation only); TV penalty on bins; Bures–Wasserstein covariate-shift split; per-feature composition Shapley; multi-output via center-outward rank layer.

---

## 5. Where the evidence is thin
- **No published work** combines EBM-style additive binned shape functions with Legendre/L-moment quantile channels; all TRACE-specific accuracy effects are conjectures.
- **OT for small-T / large-N period shifts:** CiC/OT-DiD and CFM are solid for two periods, but OT-specific inference with few periods and serially dependent cross-sections is sparse. Distribution-valued time-series models (Zhu & Müller; Dubey & Müller) assume more periods.
- **Unbalanced OT as an entry/exit decomposition** of a regression-based shift: theory is mature, but I found no paper using it as an attribution term. The ρ-dependence is a real interpretability issue.
- **Per-feature composition attribution** with dependent features has no canonical answer; Shapley versions depend on the reference/dependence handling.
- **Additive VQR (multi-output)** keeping exact W2 additivity: no known result. Cyclical monotonicity breaks the simple PAV step.
- **W-DRO improving accuracy** (as opposed to certifying it) under non-adversarial shifts: weak. TRACE's own T13 (0/100 wins) agrees.
