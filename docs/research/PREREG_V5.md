# Pre-registration: TRACE v5 — mixture-first (Direction A), first gate only

**Frozen:** 2026-10-28 (America/New_York), on branch `quant/v5-prereg`, **before any
v5 empirical result exists**. Board: [BOARD_V5.md](BOARD_V5.md). Prior freeze:
[PREREG_V4.md](../benchmark/PREREG_V4.md) @ `d50732f` (FAIL recorded in
[V4_RESULTS.md](../benchmark/V4_RESULTS.md)). Adjacent structure track (do **not**
fold into this gate): [CROSSCAT_MODERNIZATION.md](CROSSCAT_MODERNIZATION.md).

**State at freeze.** No `explain_v5` / `TRACE_V5` adapter output has been scored. No
file under `benchmarks/results/v5_*` exists at this commit. The commit SHA that adds
*this* file is the freeze point; record that SHA in the header of any later
`V5_RESULTS.md` / first-gate log when results are written. Any change to scenarios,
seeds, margins, or bars after that SHA is a deviation and must be labelled as such.

**Scope of this freeze.** **First gate only** — Eulerian \(Q_{\mathrm{mix}}\) /
pure-mix falsifier on existing H7 composition cells. This is **not** a full scored
suite. Do not start H8b coverage, ShiftBench accuracy, or speed tracks under this
document. A later dated amendment (new freeze SHA) is required before those tracks
may run.

Success standard (**first gate**): on the pure-mix falsifier cells below, the
mixture-first object clears the **same** BEST / NON-INFERIOR bars as
[PREREG_V4.md](../benchmark/PREREG_V4.md) §5 (Holm, \(\alpha=0.05\)), applied only to
the cells named here. If the falsifier fails, **kill** — do not expand to a full suite.

---

## 1. Claim / hypothesis

**Claim.** TRACE v4 failed because the primary object was wrong: the game was played
on a Lagrangian quantile-curve average (barycenter)
\(\bar Q_{[s,t]}(u)=\mathbb{E}_{x\sim F_s} Q_t(u\mid x)\), which is **not** the
Eulerian mixture marginal that composition-shaped H7 cells and CFM target. Board
diagnosis: [BOARD_V5.md](BOARD_V5.md) §3; greenhouse telltale from the frozen v4
`mixed` mean split (same source, do not invent): truth
\((\mathrm{mix},\mathrm{response})\approx(0.660,-0.181)\), CFM
\(\approx(0.646,-0.174)\), v4 \(\approx(0.238,0.238)\).

**Hypothesis (Direction A).** Define the **mixture counterfactual** by averaging
conditional CDFs then inverting (Chernozhukov–Fernández-Val–Melly counterfactual
distributions; CDT / 1-D OT maps as the linear space for shares). Run TRACE’s
order-free per-feature game on that marginal. Then, on **pure composition** H7 cells
(`skew_comp`, `tail_comp`), graded mass lands in the mix role (not the response
role), and mean graded error is BEST or NON-INFERIOR to CFM under the **unchanged**
PREREG_V4 \(\delta\) rule.

Lead citations (verified links in [REFS_V4.md](REFS_V4.md) / board):

* Chernozhukov, Fernández-Val, Melly (2013), *Inference on Counterfactual
  Distributions*, Econometrica — [doi:10.3982/ECTA10582](https://doi.org/10.3982/ECTA10582).
* Kolouri, Park, Rohde (2016), *The cumulative distribution transform and linear
  pattern classification*, arXiv:1511.03206 — [arXiv](https://arxiv.org/abs/1511.03206).

This is not “run CFM”: CFM stops at the distribution effect; TRACE’s differentiator
is **order-free per-feature shares** of that effect. V5 changes the object those
shares attribute.

---

## 2. Method under test (plain math; fixed now)

Constraints unchanged: **Rust core / thin Python**; **no neural nets**; no banks or
industry framing anywhere.

### 2.1 Mixture CDF and quantile (Eulerian object)

For periods \(s,t\in\{A,B\}\) and outcome \(y\),

\[
F_{[s,t]}(y)
  = \mathbb{E}_{x\sim F_s}\bigl[F_t(y\mid x)\bigr],
\qquad
Q_{[s,t]}(u)
  = F_{[s,t]}^{-1}(u).
\]

\(F_{[s,t]}\) is the **mixture counterfactual** CDF (covariates from \(s\),
conditionals from \(t\)). \(Q_{[s,t]}\) is its left-inverse quantile. Numerics: evaluate
fitted conditionals on a shared \(y\)-grid, average CDF values under \(F_s\) (or
TRACE bin masses), invert. Same construction as the Faraday R1 borrow
([FARADAY_MIXTURE_CRATES.md](FARADAY_MIXTURE_CRATES.md)).

### 2.2 Residual vs Lagrangian barycenter

The Lagrangian curve-average (v4 primary object) is

\[
\bar Q_{[s,t]}(u)
  = \mathbb{E}_{x\sim F_s} Q_t(u\mid x).
\]

The **mixing residual** (Lagrangian–Eulerian gap) is

\[
M_{[s,t]}(u)
  = Q_{[s,t]}(u) - \bar Q_{[s,t]}(u).
\]

On pure location-mix composition, \(M\) carries the shape change that barycenter
channels \(k\ge 1\) miss. V5’s primary object for the game is \(Q_{[s,t]}\), not
\(\bar Q_{[s,t]}\).

### 2.3 Feature game on the mixture counterfactual

Run the existing TRACE order-free switch / Shapley-style per-feature budget on
**1-D OT / CDT coordinates of mixture laws** \(Q_{[A,A]}\), \(Q_{[A,B]}\),
\(Q_{[B,A]}\), \(Q_{[B,B]}\) (composition vs structure contrasts as in the shift
decomposition), not on barycenters of conditional quantile curves. Mixing players
receive mass from \(\langle D_M,\Delta M\rangle\) when a residual view is reported;
the **scored** first-gate estimand is the graded-channel L-moment error of the
mixture-first attribution vs Shapley truth (same metric family as PREREG_V4 §4).

Adapter name (when implemented): `TRACE_V5` / `trace.explain_v5(...)`. This freeze
does **not** require the adapter to exist yet; the first-gate run must not start
until the object and falsifier are implemented and this SHA is recorded.

Rivals for the paired first-gate comparison: **CFM** (and DFL if present on the same
H7 cells), re-run on the **same** seeds as TRACE v5 for those cells. Paired per
replication.

Any change to the above after this freeze is a deviation (§7).

---

## 3. First gate only — scenarios, sizes, seeds

Generators unchanged from `benchmarks/` (H7 runner). **No new generators.**

| Track | Scenarios | Size | Notes |
|---|---|---|---|
| H7 pure-mix falsifier | `skew_comp`, `tail_comp` | \(n=2000\) and \(n=20000\) (p as in the frozen H7 grid: 5 at 2e3, 20 at 2e4) | Graded-channel L-moment error vs Shapley truth; role check below |

**Excluded from this freeze (do not run under this SHA):** `skew_struct`,
`tail_struct`, `shape_mixed`; all H8b coverage; all ShiftBench accuracy; all speed
tracks; any new cell.

### Fresh seeds (exclusive to v5 first gate)

Replication indices (never used by PR #12, FIXES2, v3, v4, or prior pilots):

| Track | Indices | Count per cell |
|---|---|---|
| H7 `skew_comp` / `tail_comp`, \(n=2000\) | **9500–9549** | 50 |
| H7 `skew_comp` / `tail_comp`, \(n=20000\) | **9600–9619** | 20 |

Data / method / bootstrap streams use the frozen `seed_of(...)` of `run_synthetic.py`
evaluated at these indices. CFM (and DFL if scored) are re-run on the **same**
indices; comparisons are **paired** per replication.

Oracle truth files unchanged. Estimand: H7 `lsplit` vs Shapley, as in
PREREG_V4 / Amendment 1 / v3.

---

## 4. Metrics

* **Primary (accuracy):** existing graded-channel error on the L-moment scale
  (`lsplit` vs Shapley truth), same definition as [PREREG_V4.md](../benchmark/PREREG_V4.md) §4.
* **Role / pure-mix falsifier (hard):** on each first-gate cell, report
  \(|\widehat{c}|\) (mix) and \(|\widehat{s}|\) (response) averaged over replications.
  Planted composition cells have \(|\mathrm{truth}_c|\) large and
  \(|\mathrm{truth}_s|\) near zero (see BOARD_V5 / V4_RESULTS for the measured pattern
  under v4). The falsifier requires that estimated mass is **not** swapped into the
  silent role the way v4 did on `skew_comp` (BOARD_V5 §2.1).
* **Descriptive only (not a gate here):** wall time vs CFM on the same first-gate
  cells; greenhouse-style mean split on a single frozen `mixed` draw may be plotted
  for teaching but is **not** scored under this freeze.

No coverage rate, no ShiftBench mean-channel, no ≥5× speed gate in this document —
those remain future amendments that must **cite** PREREG_V4 §5 thresholds rather than
invent new bars.

---

## 5. Bars (inherited from PREREG_V4 §5; first-gate cells only)

Per accuracy cell, compute the paired mean error difference
\(d = \mathrm{err}_{\mathrm{v5}} - \mathrm{err}_{\mathrm{best\,rival}}\) and its one-sided
95% interval over replications (t-interval and 10 000-draw percentile bootstrap of the
\(d_r\), seed 2026; both must agree for a claim) — **identical protocol** to
[PREREG_V4.md](../benchmark/PREREG_V4.md) §5.

* **BEST** in a cell: upper 95% CI of \(d\) \(< 0\).
* **NON-INFERIOR** in a cell: upper 95% CI of \(d\) \(< \delta\), where
  * H7 / shape cells: \(\delta = 0.10 \times\) (planted magnitude of the graded channel)
    — same formula as PREREG_V4 (e.g. frozen v4 table used \(\delta=0.004025\) on
    `skew_comp`, \(\delta=0.0004586\) on `tail_comp`; recompute from planted magnitude
    on the v5 seeds, do not hard-code those point values as new bars).
* **First-gate accuracy PASS:** every first-gate cell (`skew_comp` / `tail_comp` × both
  sizes) is BEST or NON-INFERIOR.
* **Pure-mix falsifier PASS:** for each first-gate cell, the silent role’s mean absolute
  graded piece is below the active role’s by enough that the cell’s accuracy bar is
  not rescued by swapping roles — operationally: if accuracy would PASS only because
  mass sits in the wrong role relative to Shapley truth, mark **FAIL** (falsifier).
  Board evidence that v4 failed this: on `skew_comp` \(n=20000\), planted
  \(|\mathrm{truth}_c|\approx 0.040\), \(|\mathrm{truth}_s|\approx 0.0001\), but v4 put
  \(|\widehat{c}|\approx 0.0027\), \(|\widehat{s}|\approx 0.013\) (BOARD_V5 §2.1).

Holm adjustment across the primary paired tests in the **first-gate** accuracy family
only; unadjusted results also reported.

**No new bars** are introduced in this freeze. Speed / coverage / ShiftBench bars from
PREREG_V4 remain on the shelf until a dated amendment cites them by section.

---

## 6. Kill criteria

Stop. Do **not** open a full scored suite, do **not** amend seeds to chase NI, do
**not** silently edit this file, if any of the following hold after the first-gate run:

1. **Any** first-gate accuracy cell is FAIL under §5 (not BEST and not NON-INFERIOR).
2. **Pure-mix falsifier FAIL** on any first-gate cell (role swap / silent-role mass as
   in the v4 `skew_comp` pattern).
3. Implementation still attributes on \(\bar Q\) (Lagrangian) rather than
   \(Q_{[s,t]}\) (Eulerian) — verified by a unit-level check that
   \(\|M_{[s,t]}\|\) is nonzero on a two-component location mixture while the
   barycenter shape channels stay flat (Faraday R1 falsifier sketch).

On kill: record FAIL in a first-gate results note, keep failures, return to board
(Direction B or a new Direction A amendment). Parking research is **not** an option
under the Chief-of-Staff fork that chose pre-register; kill means **stop expanding**,
not park the program.

---

## 7. Holdout / leakage rules

* **No peeking at v5 scores before freeze.** This file must land before any
  `benchmarks/results/v5_*` exists (same discipline as PREREG_V4 §1).
* **No reuse of v4 seeds** (9100–9429). First-gate indices are 9500–9549 / 9600–9619
  only.
* **No tuning on first-gate cells.** Hyperparameters, grid sizes, and channel bases are
  those already frozen for TRACE conditionals in V4_IMPL_SPEC / in-repo defaults at
  this SHA; changing them after seeing v5 errors is a deviation.
* **No silent edit of PREREG_V4.md.** V4 remains the failed freeze of record.
* **No industry / bank framing** in code, docs, commit messages, or learn pages.
* **Holdout for later suite:** `skew_struct`, `tail_struct`, `shape_mixed`, H8b,
  ShiftBench, and speed stay untouched until a new pre-reg amendment. Running them
  “informally” under this SHA is a protocol breach.

---

## 8. Budget and early stop (first gate)

* **Wall-clock target:** first-gate four cells + paired CFM ≤ **30 minutes** on 8 cores
  (subset of PREREG_V4’s 2-hour whole-run budget; not a new accuracy bar).
* Resumable per-seed jsonl under `benchmarks/results/v5_gate1_*.jsonl` when the run
  starts (future commit; not created by this freeze).
* **Early-stop:** if the wall cap hits, cells not completed are **INCOMPLETE**, never
  imputed. A cell bar is PASS/FAIL only if completed replications already decide it;
  otherwise INCOMPLETE. INCOMPLETE on a required first-gate cell blocks “PASS” and
  blocks suite expansion (treat like kill until completed or formally abandoned).

---

## 9. Non-goals

* **No CrossCat Rust crate** and no oracle-parity rebuild work under this PR / gate
  (that is [CROSSCAT_MODERNIZATION.md](CROSSCAT_MODERNIZATION.md) — link only).
* **No BayesDB product**, no BQL DBMS, no vendoring of CrossCat/bayeslite/Lace into
  TRACE core.
* **No neural nets.**
* **No full scored suite** (H8b / ShiftBench / speed / remaining H7 cells) under this
  freeze.
* **No Direction B (Hellinger–Kantorovich)** implementation in this gate.
* **No parks.** Pre-register was the chosen fork; this document is that pre-registration.

---

## 10. Deviations

Any post-freeze change to scenarios, seeds, margins, bands, rival set, or bars is a
deviation: name it in the first-gate results note, do not silently edit this file.
Amending this file requires a new dated subsection and a new freeze SHA.

---

## 11. Report

When the first gate runs (separate PR, after implementation):

* Header with **this** file’s freeze SHA.
* Verdict table: TRACE v5 vs CFM (and DFL if scored), paired CIs, role pieces, every
  INCOMPLETE cell; failures kept.
* Explicit line: full scored suite **not started**.
* No result numbers appear in *this* file.

Teaching / status page at freeze: [v5_prereg.html](../learn/v5_prereg.html) —
status **PRE-REGISTERED**, awaiting first-gate run.

---

## 12. Pointers

* Board: [BOARD_V5.md](BOARD_V5.md) (Direction A preferred)
* Failed prior: [PREREG_V4.md](../benchmark/PREREG_V4.md), [V4_RESULTS.md](../benchmark/V4_RESULTS.md)
* Mixture math borrow: [FARADAY_MIXTURE_CRATES.md](FARADAY_MIXTURE_CRATES.md) R1
* Adjacent track: [CROSSCAT_MODERNIZATION.md](CROSSCAT_MODERNIZATION.md)
* Learn: [v5_prereg.html](../learn/v5_prereg.html), [nightly.html](../learn/nightly.html),
  [research_board.html](../learn/research_board.html)
