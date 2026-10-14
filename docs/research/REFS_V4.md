# References for BOARD_V4

Every entry was checked on 2026-10-14: arXiv entries by fetching the abstract page (title matched), journal entries by resolving the DOI through the Crossref API (title, venue and year matched). Nothing below is unverified. Numbers `[R#]` are those used in [BOARD_V4.md](BOARD_V4.md). Years 2020+ are marked ★.

## Physics, fluids, OT computation
- [R1] Jordan, Kinderlehrer, Otto (1998). The variational formulation of the Fokker–Planck equation. *SIAM J. Math. Anal.* https://doi.org/10.1137/S0036141096303359
- [R2] Wibisono (2018). Sampling as optimization in the space of measures: the Langevin dynamics as a composite optimization problem. https://arxiv.org/abs/1802.08089
- [R3] Cuturi (2013). Sinkhorn distances: lightspeed computation of optimal transport distances. https://arxiv.org/abs/1306.0895
- [R4] Feydy, Séjourné, Vialard, Amari, Trouvé, Peyré (2019). Interpolating between optimal transport and MMD using Sinkhorn divergences. https://arxiv.org/abs/1810.08278
- [R5] Genevay, Peyré, Cuturi (2018). Learning generative models with Sinkhorn divergences. https://arxiv.org/abs/1706.00292 (cited for the debiasing construction only)
- [R6] Benamou, Brenier (2000). A computational fluid mechanics solution to the Monge–Kantorovich mass transfer problem. *Numer. Math.* https://doi.org/10.1007/s002110050002
- [R7] Bhatia, Norgard, Pascucci, Bremer (2013). The Helmholtz–Hodge decomposition—a survey. *IEEE TVCG.* https://doi.org/10.1109/TVCG.2012.316
- [R22] Schmitzer (2019; arXiv 2016). Stabilized sparse scaling algorithms for entropy regularized transport problems. https://arxiv.org/abs/1610.06519
- [R24] Mena, Niles-Weed (2019). Statistical bounds for entropic optimal transport: sample complexity and the central limit theorem. https://arxiv.org/abs/1905.11882

## Computer vision (transport embeddings)
- [R8] Wang, Slepčev, Basu, Ozolek, Rohde (2013; online 2012). A linear optimal transportation framework for quantifying and visualizing variations in sets of images. *IJCV.* https://doi.org/10.1007/s11263-012-0566-z
- [R9] ★ Moosmüller, Cloninger (2020). Linear optimal transport embedding: provable Wasserstein classification for certain rigid transformations and perturbations. https://arxiv.org/abs/2008.09165
- [R10] Park, Kolouri, Kundu, Rohde (2018; arXiv 2015). The cumulative distribution transform and linear pattern classification. https://arxiv.org/abs/1507.05936
- [R11b] Kolouri, Park, Rohde (2016). The Radon cumulative distribution transform and its application to image classification. *IEEE TIP.* https://doi.org/10.1109/TIP.2015.2509419
- [R12] Deshpande et al. (2019). Max-sliced Wasserstein distance and its use for GANs. https://arxiv.org/abs/1904.05877 (cited for the distance, not the generative use)
- [R13] ★ Nadjahi et al. (2020). Statistical and topological properties of sliced probability divergences. https://arxiv.org/abs/2003.05783

## Biology (unbalanced OT, trajectories)
- [R14] Schiebinger et al. (2019). Optimal-transport analysis of single-cell gene expression identifies developmental trajectories in reprogramming. *Cell.* https://doi.org/10.1016/j.cell.2019.01.006
- [R15] ★ Lavenant, Zhang, Kim, Schiebinger (2021). Towards a mathematical theory of trajectory inference. https://arxiv.org/abs/2102.09204
- [R16] ★ Klein et al. (2025). Mapping cells through time and space with moscot. *Nature.* https://doi.org/10.1038/s41586-024-08453-2
- [R17] Chizat, Peyré, Schmitzer, Vialard (2015). Unbalanced optimal transport: dynamic and Kantorovich formulation. https://arxiv.org/abs/1508.05216
- [R18] Liero, Mielke, Savaré (2015). Optimal entropy-transport problems and a new Hellinger–Kantorovich distance between positive measures. https://arxiv.org/abs/1508.07941
- [R19] Séjourné, Feydy, Vialard, Trouvé, Peyré (2019). Sinkhorn divergences for unbalanced optimal transport. https://arxiv.org/abs/1910.12958

## Econometrics
- [R20] Chernozhukov, Fernández-Val, Melly (2013; arXiv 2009). Inference on counterfactual distributions. https://arxiv.org/abs/0904.0951
- [R21] DiNardo, Fortin, Lemieux (1996). Labor market institutions and the distribution of wages, 1973–1992: a semiparametric approach. *Econometrica.* https://doi.org/10.2307/2171954 (cited for the reweighting method only)
- [R23] Firpo, Fortin, Lemieux (2009). Unconditional quantile regressions. *Econometrica.* https://doi.org/10.3982/ECTA6822
- [R25] Chernozhukov et al. (2018; arXiv 2016). Double/debiased machine learning for treatment and causal parameters. https://arxiv.org/abs/1608.00060
- [R27] Chernozhukov, Newey, Singh (2022; arXiv 2018). Automatic debiased machine learning of causal and structural effects. https://arxiv.org/abs/1809.05224
- [R28] ★ Chernozhukov, Newey, Quintas-Martínez, Syrgkanis (2021). Automatic debiased machine learning via Riesz regression. https://arxiv.org/abs/2104.14737 (we use the linear-dictionary version only)
- [R29] ★ Kallus, Mao, Uehara (2019, rev. 2020+). Localized debiased machine learning: efficient inference on quantile treatment effects and beyond. https://arxiv.org/abs/1912.12945
- [R34] Zubizarreta (2015). Stable weights that balance covariates for estimation with incomplete outcome data. *JASA.* https://doi.org/10.1080/01621459.2015.1023805
- [R35] Hainmueller (2012). Entropy balancing for causal effects. *Political Analysis.* https://doi.org/10.1093/pan/mpr025
- [R36] Imai, Ratkovic (2014; online 2013). Covariate balancing propensity score. *JRSS-B.* https://doi.org/10.1111/rssb.12027

## Statistics
- [R11] Gneiting, Raftery (2007). Strictly proper scoring rules, prediction, and estimation. *JASA.* https://doi.org/10.1198/016214506000001437
- [R26] Lei, G'Sell, Rinaldo, Tibshirani, Wasserman (2018; arXiv 2016). Distribution-free predictive inference for regression. https://arxiv.org/abs/1604.04173
- [R30] Donoho, Johnstone (1994). Ideal spatial adaptation by wavelet shrinkage. *Biometrika.* https://doi.org/10.1093/biomet/81.3.425
- [R31] Yu (2013). Stability. *Bernoulli.* https://arxiv.org/abs/1310.0150
- [R33] ★ Lengerich, Tan, Chang, Hooker, Caruana (2020). Purifying interaction effects with the functional ANOVA. https://arxiv.org/abs/1911.04974
- [R37] Hosking (1990). L-moments: analysis and estimation of distributions using linear combinations of order statistics. *JRSS-B.* https://doi.org/10.1111/j.2517-6161.1990.tb01775.x
- [R38] Chung, Romano (2013). Exact and asymptotically robust permutation tests. *Ann. Statist.* https://doi.org/10.1214/13-AOS1090 (arXiv 1304.5939)
- [R39] Fang, Santos (2019; arXiv 2014). Inference on directionally differentiable functions. https://arxiv.org/abs/1404.3763
- [R40] Gneiting, Ranjan (2011). Comparing density forecasts using threshold- and quantile-weighted scoring rules. *JBES.* https://doi.org/10.1198/jbes.2010.08110
- [R41] ★ Henzi, Ziegel, Gneiting (2021). Isotonic distributional regression. https://arxiv.org/abs/1909.03725
- [R42] Gretton, Borgwardt, Rasch, Schölkopf, Smola (2008/2012). A kernel method for the two-sample problem. https://arxiv.org/abs/0805.2368
- [R43] Muandet, Fukumizu, Sriperumbudur, Schölkopf (2017; arXiv 2016). Kernel mean embedding of distributions: a review and beyond. https://arxiv.org/abs/1605.09522
- [R44] Owen (1988). Empirical likelihood ratio confidence intervals for a single functional. *Biometrika.* https://doi.org/10.1093/biomet/75.2.237
- [R45] Meinshausen, Bühlmann (2010; arXiv 2008). Stability selection. https://arxiv.org/abs/0809.2932
- [R46] Shah, Samworth (2013; arXiv 2011). Variable selection with error control: another look at stability selection. https://arxiv.org/abs/1105.5578
- [R47] Barber, Candès (2015; arXiv 2014). Controlling the false discovery rate via knockoffs. https://arxiv.org/abs/1404.5609
- [R48] Candès, Fan, Janson, Lv (2018; arXiv 2016). Panning for gold: model-X knockoffs. https://arxiv.org/abs/1610.02351
- [R49] Romano, Patterson, Candès (2019). Conformalized quantile regression. https://arxiv.org/abs/1905.03222
- [R50] ★ Chernozhukov, Wüthrich, Zhu (2021; arXiv 2019). Distributional conformal prediction. https://arxiv.org/abs/1909.07889
- [R51] Tibshirani, Barber, Candès, Ramdas (2019). Conformal prediction under covariate shift. https://arxiv.org/abs/1904.06019
- [R52] Panaretos, Zemel (2019; arXiv 2018). Statistical aspects of Wasserstein distances. https://arxiv.org/abs/1806.05500
- [R53] del Barrio, Loubes (2019; arXiv 2017). Central limit theorems for empirical transportation cost in general dimension. https://arxiv.org/abs/1705.01299
- [R54] Sommerfeld, Munk (2018; arXiv 2016). Inference for empirical Wasserstein distances on finite spaces. https://arxiv.org/abs/1610.03287
- [R55] Kleiner, Talwalkar, Sarkar, Jordan (2014; arXiv 2011). A scalable bootstrap for massive data. https://arxiv.org/abs/1112.5016
- [R56] ★ Lam (2022). A cheap bootstrap method for fast inference. https://arxiv.org/abs/2202.00090
- [R57] Chernozhukov, Chetverikov, Kato (2013; arXiv 2012). Gaussian approximations and multiplier bootstrap for maxima of sums of high-dimensional random vectors. https://arxiv.org/abs/1212.6906
- [R58] Nori, Jenkins, Koch, Caruana (2019). InterpretML: a unified framework for machine learning interpretability. https://arxiv.org/abs/1909.09223
- [R59] ★ Budhathoki, Janzing, Blöbaum, Ng (2021). Why did the distribution change? https://arxiv.org/abs/2102.13384
- [R60] ★ Kulinski, Inouye (2023; arXiv 2022). Towards explaining distribution shifts. https://arxiv.org/abs/2210.10275
- [R61] Petersen, Müller (2019; arXiv 2016). Fréchet regression for random objects with Euclidean predictors. https://arxiv.org/abs/1608.03012
- [R62] ★ Chen, Lin, Müller (2021; arXiv 2020). Wasserstein regression. https://arxiv.org/abs/2006.09660
- [R63] Chernozhukov, Fernández-Val, Galichon (2010; arXiv 2007). Quantile and probability curves without crossing. https://arxiv.org/abs/0704.3649
- [R64] Tibshirani (2014; arXiv 2013). Adaptive piecewise polynomial estimation via trend filtering. https://arxiv.org/abs/1304.2986
- [R65] ★ Fasiolo, Wood, Zaffran, Nedellec, Goude (2021; arXiv 2017). Fast calibrated additive quantile regression. https://arxiv.org/abs/1707.03307

## Notes on verification
- Publication years in parentheses for journal versions of arXiv papers are from memory and were not each re-checked; the arXiv ids and titles were.
- Derivations marked [P] in BOARD_V4 that are *ours* (mixing-term expansion, bin-marginal sufficiency, split-product unbiasedness, path-attribution equivalence) are sketches; they should get a `verify/` numerical check before being cited as results.
