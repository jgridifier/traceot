"""Rival shift-decomposition methods. Reference period is A throughout.

* OB    Oaxaca-Blinder (OLS), mean only, per feature.
* DFL   DiNardo-Fortin-Lemieux reweighting (logit on [X, X^2]), all stats, aggregate.
* CFM   Chernozhukov-Fernandez-Val-Melly distribution regression (logit on [X, X^2]
        at K thresholds) with counterfactual CDF, all stats, aggregate.
* RIF   Firpo-Fortin-Lemieux RIF regression on [X, X^2] + Oaxaca-Blinder, all stats,
        per feature.
* SHAP  Gradient-boosted model per period (LightGBM) + TreeSHAP drift attribution,
        mean only, per feature.
"""

from __future__ import annotations

import numpy as np
from sb_adapter import Decomposition, ShiftMethod
from sb_dgp import STATS, stat

_Q = {"q10": 0.10, "q50": 0.50, "q90": 0.90}


def _ols(X, y):
    Z = np.column_stack([np.ones(len(X)), X])
    coef, *_ = np.linalg.lstsq(Z, y, rcond=None)
    return coef[0], coef[1:]


def _ob(XA, yA, XB, yB):
    aA, bA = _ols(XA, yA)
    aB, bB = _ols(XB, yB)
    mA, mB = XA.mean(0), XB.mean(0)
    fc = (mB - mA) * bA
    fs = mB * (bB - bA)  # X is centred at the pooled mean
    return fc, fs, float(aB - aA)


def _flex(X):
    """[X, X^2] basis; per-feature results sum each feature's two columns."""
    return np.column_stack([X, X * X])


def _fold(v, p):
    return v[:p] + v[p:]


def _logit(Z, t, C=1e4, max_iter=300):
    import warnings

    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression

    m = LogisticRegression(C=C, max_iter=max_iter)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)  # iteration cap is in the spec
        m.fit(Z, t)
    return m


class OaxacaBlinder(ShiftMethod):
    name = "OB"
    stats = ("mean",)
    per_feature = True

    def decompose(self, XA, yA, XB, yB, seed=0):
        fc, fs, ic = _ob(XA, yA, XB, yB)
        return Decomposition({"mean": (float(fc.sum()), float(fs.sum() + ic))}, fc, fs, ic)


class DFL(ShiftMethod):
    name = "DFL"
    stats = STATS

    def decompose(self, XA, yA, XB, yB, seed=0):
        Z = np.vstack([XA, XB])
        Z = np.column_stack([Z, Z * Z])
        sd = Z.std(0)
        sd[sd == 0] = 1
        Z = Z / sd
        t = np.r_[np.zeros(len(XA)), np.ones(len(XB))]
        m = _logit(Z, t)
        pa = m.predict_proba(Z[: len(XA)])[:, 1]
        w = pa / (1 - pa)
        split = {}
        for s in STATS:
            tA, tB, tC = stat(yA, s), stat(yB, s), stat(yA, s, w)
            split[s] = (tC - tA, tB - tC)
        return Decomposition(split, extra={"max_w_share": float(w.max() / w.sum())})


class CFM(ShiftMethod):
    """Distribution regression. F_t(y|x) = Lambda(x'b_t(y)) at K pooled-quantile
    thresholds; F_c = mean over X_B of the period-A conditional CDF. Statistics are
    read from the (rearranged) CDF on the threshold grid, the same way for F_A, F_B
    and F_c, so grid error is common to all three."""

    name = "CFM"
    stats = STATS
    K = 39

    def decompose(self, XA, yA, XB, yB, seed=0):
        ypool = np.r_[yA, yB]
        taus = np.arange(1, self.K + 1) / (self.K + 1)
        thr = np.quantile(ypool, taus)
        FXA, FXB = _flex(XA), _flex(XB)
        sd = np.vstack([FXA, FXB]).std(0)
        sd[sd == 0] = 1
        ZA, ZB = FXA / sd, FXB / sd
        FA, FB, FC = (np.empty(self.K) for _ in range(3))
        for k, c in enumerate(thr):
            tA, tB = (yA <= c).astype(int), (yB <= c).astype(int)
            if 0 < tA.sum() < len(tA):
                mA = _logit(ZA, tA, max_iter=500)
                FA[k] = mA.predict_proba(ZA)[:, 1].mean()
                FC[k] = mA.predict_proba(ZB)[:, 1].mean()
            else:  # one class only: the conditional CDF is that constant
                FA[k] = FC[k] = float(tA[0])
            if 0 < tB.sum() < len(tB):
                FB[k] = _logit(ZB, tB, max_iter=500).predict_proba(ZB)[:, 1].mean()
            else:
                FB[k] = float(tB[0])
        # Atom values: conditional means of the pooled sample in each grid cell.
        idx = np.searchsorted(thr, ypool, side="left")
        vals = np.array([ypool[idx == i].mean() for i in range(self.K + 1)])

        def stats_of(F):
            F = np.clip(np.sort(F), 0, 1)  # monotone rearrangement
            pm = np.diff(np.r_[0.0, F, 1.0])
            mu = float(np.sum(pm * vals))
            out = {"mean": mu, "sd": float(np.sqrt(np.sum(pm * (vals - mu) ** 2)))}
            for s, q in _Q.items():
                out[s] = float(np.interp(q, F, thr))
            return out

        sA, sB, sC = stats_of(FA), stats_of(FB), stats_of(FC)
        return Decomposition({s: (sC[s] - sA[s], sB[s] - sC[s]) for s in STATS})


def _rif(y, s):
    if s == "mean":
        return y
    if s == "sd":
        m, v = y.mean(), y.var()
        sd = np.sqrt(v)
        return sd + ((y - m) ** 2 - v) / (2 * sd)
    q = _Q[s]
    qv = np.quantile(y, q)
    h = 1.06 * y.std() * len(y) ** (-0.2)
    f = np.mean(np.exp(-0.5 * ((y - qv) / h) ** 2)) / (h * np.sqrt(2 * np.pi))
    return qv + (q - (y <= qv)) / f


class RIF(ShiftMethod):
    name = "RIF"
    stats = STATS
    per_feature = True

    def decompose(self, XA, yA, XB, yB, seed=0):
        split = {}
        p = XA.shape[1]
        FA, FB = _flex(XA), _flex(XB)
        for s in STATS:
            fc, fs, ic = _ob(FA, _rif(yA, s), FB, _rif(yB, s))
            fc, fs = _fold(fc, p), _fold(fs, p)
            split[s] = (float(fc.sum()), float(fs.sum() + ic))
            if s == "mean":
                mfc, mfs, mic = fc, fs, ic
        return Decomposition(split, mfc, mfs, mic)


class ShapDrift(ShiftMethod):
    """f_t = LightGBM fit on period t; phi^f = TreeSHAP values (LightGBM
    pred_contrib, tree-path-dependent). With P the pooled sample:
        comp_j   = mean_B phi^{f_A}_j - mean_A phi^{f_A}_j
        struct_j = mean_B d_j - mean_P d_j,  d = phi^{f_B} - phi^{f_A}
        intercept = mean structure - sum_j struct_j
    Totals use full-sample predictions; the per-feature split uses n_explain rows per
    period. The pooled centring of struct_j is the convention used for the planted truth."""

    name = "SHAP"
    stats = ("mean",)
    per_feature = True
    n_jobs = 1
    n_explain = 5000  # rows per period explained for the per-feature split

    def decompose(self, XA, yA, XB, yB, seed=0):
        import lightgbm as lgb

        prm = dict(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=15,
            min_child_samples=20,
            subsample=0.8,
            subsample_freq=1,
            random_state=seed,
            verbose=-1,
            n_jobs=self.n_jobs,
            deterministic=True,
        )
        fA = lgb.LGBMRegressor(**prm).fit(XA, yA)
        fB = lgb.LGBMRegressor(**prm).fit(XB, yB)

        rng = np.random.default_rng(seed)
        sA = XA[rng.choice(len(XA), min(self.n_explain, len(XA)), replace=False)]
        sB = XB[rng.choice(len(XB), min(self.n_explain, len(XB)), replace=False)]

        def contrib(f, X):
            return f.predict(X, pred_contrib=True)  # last column is the base value

        cAA, cAB, cBB, cBA = contrib(fA, sA), contrib(fA, sB), contrib(fB, sB), contrib(fB, sA)
        fc = cAB[:, :-1].mean(0) - cAA[:, :-1].mean(0)
        dB = (cBB - cAB)[:, :-1].mean(0)
        dA = (cBA - cAA)[:, :-1].mean(0)
        fs = dB - 0.5 * (dA + dB)
        hAA, hAB, hBB = fA.predict(XA).mean(), fA.predict(XB).mean(), fB.predict(XB).mean()
        comp, struct = float(hAB - hAA), float(hBB - hAB)
        return Decomposition({"mean": (comp, struct)}, fc, fs, struct - float(fs.sum()))


RIVALS = [OaxacaBlinder(), DFL(), CFM(), RIF(), ShapDrift()]
