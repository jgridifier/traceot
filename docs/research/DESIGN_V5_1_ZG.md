# Design note — zero-gated composition piece for TRACE v5.1 (V5.1-ZG)

**Status: DESIGN NOTE · NOT a pre-registration · NOT frozen · nothing run · no
seed drawn.** Written 2026-11-17 (America/New_York) after the struct-cells park
([PARK_V5_1_STRUCT.md](PARK_V5_1_STRUCT.md)). This file proposes the next TRACE
idea and the shape of its pre-registration. The pre-registration itself
(`PREREG_V5_1_ZG.md`) is a separate, later commit; that commit is the freeze point.
Nothing here changes, amends, or re-scores PREREG_V5_1_STRUCT.md.

**What this note is allowed to use.** The park says a new hypothesis "must not be
chosen by looking at this run beyond what is written here". This note uses only
the summary numbers printed in [PARK_V5_1_STRUCT.md](PARK_V5_1_STRUCT.md) and the
six-cell tables of
[V5_1_STRUCT_GATE_RESULTS.md](../benchmark/V5_1_STRUCT_GATE_RESULTS.md). It did not
open the per-replication file `v5_1_struct_gate_shape.jsonl`. The confirmatory
test it proposes runs on fresh seeds, so the cells that motivated it cannot also
confirm it.

## 1. What the failure showed (from the published summaries only)

Three of four structure-only cells failed false-attribution: the one-sided upper
bound of mean \(|\hat c|\) (the composition piece when the planted composition
truth is ≈ 0) was not below δ = 0.10 × planted magnitude.

| cell | n | mean ĉ (signed) | mean \|ĉ\| | signed / abs | δ | FA |
|---|---:|---:|---:|---:|---:|---|
| skew_struct | 2000 | 0.0001158 | 0.006704 | 1.7% | 0.005072 | FAIL |
| tail_struct | 2000 | 0.0008566 | 0.003703 | 23% | 0.0009827 | FAIL |
| skew_struct | 20000 | −0.00006785 | 0.002096 | 3.2% | 0.005072 | PASS |
| tail_struct | 20000 | 0.0002137 | 0.001104 | 19% | 0.0009827 | FAIL |

Three readings follow, all descriptive:

1. **Mostly noise, not bias, on `skew_struct`.** The signed mean is 1.7% and 3.2%
   of the absolute mean. The composition piece scatters around zero; its average
   size, not its average sign, is what crosses δ.
2. **Noise shrinks at the ordinary rate.** From n = 2 000 to n = 20 000 (10× the
   rows) mean \(|\hat c|\) falls 3.20× on `skew_struct` and 3.35× on
   `tail_struct`; √10 ≈ 3.16. That is consistent with plain 1/√n sampling noise
   (p also changes, 5 → 20, so this is suggestive, not a test).
3. **`tail_struct` also carries a small positive shift.** Signed mean is about a
   fifth of the absolute mean at both sizes; at n = 2 000 the signed error
   (ĉ − truth c = 0.0007715) is about 0.79 δ. This is the open `tail_comp`
   composition under-estimate seen from the other side and is **not** something
   gating can remove.

Consequence. Any estimator whose composition piece is unbiased with 1/√n noise
will fail an absolute-value bar fixed at 0.1 × the structural size, at small n,
however good its accuracy is. The fix has to be in what TRACE *reports* when the
data cannot tell composition from zero, not in a better location model.

## 2. The idea: report composition only when its band excludes zero

V5.1 already has a validated band: `BAND_V5_1_PB90`, the 90% two-sample
percentile bootstrap with B = 500
([PREREG_V5_1_COVERAGE.md](PREREG_V5_1_COVERAGE.md) §2), which PASSED coverage
on `skew_comp` and `tail_comp` at n = 2 000
([V5_1_COVERAGE_RESULTS.md](../benchmark/V5_1_COVERAGE_RESULTS.md)).

**V5.1-ZG** (proposed, per graded channel k):

1. Run `explain_v5_1` unchanged → \(\hat c_k, \hat s_k\).
2. Compute `BAND_V5_1_PB90` on \(\hat c_k\) → \([lo_k, hi_k]\).
3. If \(lo_k \le 0 \le hi_k\): report \(\tilde c_k = 0\) and
   \(\tilde s_k = \hat s_k + \hat c_k\). Otherwise report \(\tilde c_k = \hat c_k,
   \tilde s_k = \hat s_k\).

Step 3 keeps the two pieces summing to the estimated total change exactly (the
Shapley efficiency property TRACE already promises), so the total is never
altered; only its split is. This is a pre-test (hard-threshold) estimator, a
standard idea in statistics: it trades a little bias when composition is small
but real for much less spurious composition when it is absent. No tunable
parameter is added: the level (90%) and B (500) are the ones already validated.
The gate lives in the Rust core next to the existing percentile-band code
(`crates/core/src/trace/shift_v2.rs`, `percentile_bands`), with a thin Python
binding; the confirmatory runner would first check it reproduces the harness
band exactly, as PREREG_V5_1_COVERAGE §2 requires of any library band.

## 3. What to expect, written down before any run

Idealised arithmetic, assuming the null composition piece is centred and roughly
normal with sd σ and the band is calibrated. These are expectations, not results:

* Ungated, \(E|\hat c| = \sqrt{2/\pi}\,σ ≈ 0.798σ\).
* Gated at 90%, \(E|\tilde c| ≈ 2\varphi(1.645)\,σ ≈ 0.206σ\), about 3.9× smaller.

Scaling the published mean \(|\hat c|\) by 0.206/0.798:

| cell | n | published mean \|ĉ\| | idealised gated | δ | margin |
|---|---:|---:|---:|---:|---|
| skew_struct | 2000 | 0.006704 | ≈ 0.0017 | 0.005072 | comfortable |
| tail_struct | 2000 | 0.003703 | ≈ 0.00096 | 0.0009827 | **borderline**; the 0.79 δ shift (§1.3) makes the band exclude 0 more often than 10%, so a FAIL here is a live outcome |
| tail_struct | 20000 | 0.001104 | ≈ 0.00029 | 0.0009827 | comfortable |

The scored bar is the one-sided upper bound, not the mean, so real margins are
thinner than this table. **`tail_struct` at n = 2 000 must stay in the suite.**
Dropping it because it looks hard would be choosing cells from this run.

Costs to state honestly: (a) on `shape_mixed` and the composition cells the
composition truth is large relative to noise (e.g. `shape_mixed` truth c =
0.03951 against mean \(|\hat c|\) ≈ 0.04), so the gate should rarely fire, but
any firing there is a direct accuracy loss and the suite must score it;
(b) the band's coverage has only been checked on composition cells at
n = 2 000, not on structure-only cells and not at n = 20 000. If the band is
too narrow there, the gate fires too often and fails false-attribution anyway.

## 4. Shape of the pre-registration (to be frozen in PREREG_V5_1_ZG.md)

* **H (draft).** On the frozen linear-mean H7 shape simulator, V5.1-ZG (a) passes
  the PREREG_V5_1_STRUCT §5 false-attribution bar, unchanged, on all four
  structure-only cells; (b) is BEST or NON-INFERIOR after Holm against DFL and
  CFM on all six struct/mixed cells, with the role check; and (c) stays BEST or
  NON-INFERIOR on the four composition cells (`skew_comp`, `tail_comp`, both
  sizes), so the gate does not cost the claim that already stands.
* **Bars.** Inherited from PREREG_V5_1_STRUCT §5 without loosening. In
  particular the false-attribution bar stays on mean \(|\tilde c|\) with the
  same δ; no switch to a rate-based bar.
* **Secondary, descriptive.** Gate firing rate per cell; band coverage of the
  composition truth on structure-only cells; ungated V5.1 on the same draws as a
  paired reference.
* **Seeds (proposed, re-check at freeze).** 15000–15049 (n = 2 000) and
  15100–15119 (n = 20 000). At this note, on `main` @ `edea650`,
  `git grep -nE '(^|[^0-9.])15(0[0-4][0-9]|1[01][0-9])([^0-9]|$)'` returns
  nothing, and no `rep` field in `benchmarks/results/*.jsonl` lies in
  15000–15119. `FORBIDDEN` must name 9600–9619 (sealed), 10400–10799,
  14000–14049 and 14100–14119 (spent).
* **Sizing (from per-task medians already on record, no new run).** One V5.1
  fit is about 0.0065 s at n = 2 000 and 0.068 s at n = 20 000; B = 500 refits
  make ≈ 3.3 s and ≈ 34 s per task. Ten cells (5 per size) ≈ 250 × 3.3 s +
  100 × 34 s ≈ 4 200 core-seconds ≈ 9 minutes on 8 workers, plus rivals.
  Proposed cap 45 minutes.
* **Harness.** Spawn-safe harness, pre-flight in a spawned child, rehearsal on
  throwaway seeds 97/98, 5-minute watchdog, as in PREREG_V5_2C §2–4.
* **Kill rule.** Any failed check rejects H; record and park; no re-runs.

## 5. Order of work

1. Rust: `gate_composition(point, band) -> (c̃, s̃, fired)` next to
   `percentile_bands`, with unit tests (band containing 0 → fires and moves the
   piece to structure; sum preserved bit-for-bit). Thin Python binding.
2. Harness parity test: library band equals `BAND_V5_1_PB90` on one throwaway
   draw (seed 97).
3. Freeze `PREREG_V5_1_ZG.md` from §4, then runner PR, then one pass.

Rust core, thin Python binding, no neural nets. Seeds 9600–9619 stay sealed.
