"""L-moment read-outs of the frozen DFL and CFM counterfactual laws (Amendment 1, A2).

The fitting code is copied from sb_methods.DFL / sb_methods.CFM unchanged (same features,
logits, thresholds, rearrangement); only the read-out differs: L-moments lambda_1..lambda_4
of the period-A, period-B and counterfactual laws, computed exactly for discrete laws by
integrating the step quantile function against shifted Legendre polynomials.
"""
from __future__ import annotations

import numpy as np
from numpy.polynomial import legendre as L

from sb_adapter import Decomposition, ShiftMethod
from sb_dgp import STATS, stat
from sb_methods import CFM, DFL, _flex, _logit

K_LM = 4


def _pstar_antideriv(k):
    c = np.zeros(k + 1)
    c[k] = 1.0
    integ = L.legint(c)  # antiderivative in t = 2u - 1
    return lambda u: 0.5 * L.legval(2 * np.asarray(u) - 1, integ)


_ANTI = [_pstar_antideriv(k) for k in range(K_LM)]


def lmom_discrete(values, weights):
    """lambda_{k+1} = int_0^1 Q(u) P*_k(u) du for the discrete law sum_i w_i delta_{v_i}."""
    v = np.asarray(values, float)
    w = np.asarray(weights, float)
    o = np.argsort(v, kind="stable")
    v, w = v[o], w[o] / w.sum()
    F = np.clip(np.cumsum(w), 0, 1)
    F0 = np.r_[0.0, F[:-1]]
    return np.array([np.sum(v * (_ANTI[k](F) - _ANTI[k](F0))) for k in range(K_LM)])


def _lsplit(lA, lB, lC):
    return {f"l{k + 1}": (float(lC[k] - lA[k]), float(lB[k] - lC[k])) for k in range(K_LM)}


class DFL_L(DFL):
    name = "DFL"

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
        one = np.ones(len(yA))
        lA, lB, lC = lmom_discrete(yA, one), lmom_discrete(yB, np.ones(len(yB))), lmom_discrete(yA, w)
        return Decomposition(split, extra={"max_w_share": float(w.max() / w.sum()), "lsplit": _lsplit(lA, lB, lC)})


class CFM_L(CFM):
    name = "CFM"

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
            else:
                FA[k] = FC[k] = float(tA[0])
            if 0 < tB.sum() < len(tB):
                FB[k] = _logit(ZB, tB, max_iter=500).predict_proba(ZB)[:, 1].mean()
            else:
                FB[k] = float(tB[0])
        idx = np.searchsorted(thr, ypool, side="left")
        vals = np.array([ypool[idx == i].mean() for i in range(self.K + 1)])
        from sb_methods import _Q

        def law(F):
            F = np.clip(np.sort(F), 0, 1)
            return np.diff(np.r_[0.0, F, 1.0])

        def stats_of(F):
            Fs = np.clip(np.sort(F), 0, 1)
            pm = law(F)
            mu = float(np.sum(pm * vals))
            out = {"mean": mu, "sd": float(np.sqrt(np.sum(pm * (vals - mu) ** 2)))}
            for s, q in _Q.items():
                out[s] = float(np.interp(q, Fs, thr))
            return out

        sA, sB, sC = stats_of(FA), stats_of(FB), stats_of(FC)
        lA, lB, lC = (lmom_discrete(vals, law(F)) for F in (FA, FB, FC))
        return Decomposition({s: (sC[s] - sA[s], sB[s] - sC[s]) for s in STATS}, extra={"lsplit": _lsplit(lA, lB, lC)})


SHAPE_RIVALS = [DFL_L(), CFM_L()]
