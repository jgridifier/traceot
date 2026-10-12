"""Pre-run checks of TRACE_ADAPTER_SPEC.md §6 on random (non-benchmark) data. Exit 1 on failure."""
import numpy as np
import traceot

from sb_adapter_trace_claims import CFG

rng = np.random.default_rng(424242)
fails = []


def ck(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info)
    if not ok:
        fails.append(name)


for trial in range(5):
    n, p = 1200, 4
    XA, XB = rng.normal(size=(n, p)), rng.normal(size=(n, p)) * 1.2 + 0.3
    yA = XA @ rng.normal(size=p) + rng.normal(size=n)
    yB = XB @ rng.normal(size=p) + rng.standard_exponential(n)
    m = 0.5 * (XA.mean(0) + XB.mean(0)); XA, XB = XA - m, XB - m
    r = traceot.shift_decompose(XA, yA, XB, yB, seed=7, **CFG)
    tol = 1e-9 * max(1, r["w2"]["marginal"])
    a, dc, phi = np.asarray(r["bary_a"]), np.asarray(r["bary_dc"]), np.asarray(r["bary_phi"])
    ck(f"[{trial}] sum a = dc", np.max(np.abs(a.sum((1, 2)) - dc)) < tol)
    ck(f"[{trial}] sum phi = w2.bary", abs(phi.sum() - r["w2"]["bary"]) < tol)
    ms = np.sum(r["marg_phi"]["comp"]) + np.sum(r["marg_phi"]["struct"]) + r["resid_phi"]["comp"] + r["resid_phi"]["struct"]
    ck(f"[{trial}] marg+resid = w2.marginal", abs(ms - r["w2"]["marginal"]) < tol, f"{abs(ms - r['w2']['marginal']):.2e}")
    dl = np.asarray(r["lmom"]["B"]) - np.asarray(r["lmom"]["A"])
    ck(f"[{trial}] lsplit sums to dlambda", np.max(np.abs(np.asarray(r["lsplit"]["comp"]) + np.asarray(r["lsplit"]["struct"]) - dl)) < tol)
    r2 = traceot.shift_decompose(XA, yA, XB, yB, seed=7, **CFG)
    ck(f"[{trial}] deterministic", all(np.array_equal(np.asarray(r[k]), np.asarray(r2[k])) for k in ("bary_phi", "feat_comp")))
    pr = rng.permutation(n); pc = rng.permutation(p)
    r3 = traceot.shift_decompose(XA[pr][:, pc], yA[pr], XB[pr][:, pc], yB[pr], seed=7, **CFG)
    inv = np.argsort(pc)
    ck(f"[{trial}] row/col permutation", np.max(np.abs(np.asarray(r3["feat_comp"])[inv] - r["feat_comp"])) < 1e-6,
       f"{np.max(np.abs(np.asarray(r3['feat_comp'])[inv] - r['feat_comp'])):.1e}")
    r4 = traceot.shift_decompose(XB, yB, XA, yA, seed=7, **CFG)
    ck(f"[{trial}] period swap: bary_phi unchanged", np.max(np.abs(np.asarray(r4["bary_phi"]) - phi)) < 1e-9 * max(1, phi.sum()),
       f"{np.max(np.abs(np.asarray(r4['bary_phi']) - phi)):.1e}")
print("ALL PASS" if not fails else f"FAILED {fails}")
raise SystemExit(1 if fails else 0)
