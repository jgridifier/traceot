# HOLD — TRACE v5.1 coverage of a 90% bootstrap band

**Status: HELD · NOT APPROVED · NOT RUN.**
The pre-registration itself is unchanged and remains frozen:
[PREREG_V5_1_COVERAGE.md](PREREG_V5_1_COVERAGE.md) (traceot PR #46; freeze commit
`dfaaf72e775ad6065b88a1c7931033c8dedbdbd3`, merge
`7ff4f36508fd87421d8159ae9f9ab17e63564e04`). This note records a hold, not an
approval to run, and not a park under the pre-registration's kill rule (the study
has not been run).

## The owner's words

On 2026-11-11 at 7:31 AM (America/New_York), the owner answered a card asking
"Run the V5.1 coverage study?" with, quoted verbatim:

> Not yet, hold the coverage study.

That is a hold. It is **not** approval. The pre-registration header stays
**NOT APPROVED**. Indices 10400–10799 were not used. Indices 9600–9619 stay sealed.
No runner for this study has been written, merged, or executed.

## What unblocks it

The owner's own dated go, in his own words, quoted verbatim in a later commit to
the header of [PREREG_V5_1_COVERAGE.md](PREREG_V5_1_COVERAGE.md). A message from
another agent saying he agreed is not approval. Until then:

* no runner is written, merged, or executed for this study;
* no index in PREREG_V5_1_COVERAGE §4 is touched, including 10400–10799;
* indices 9600–9619 stay sealed;
* no result numbers for this study appear anywhere.

The order of work in PREREG_V5_1_COVERAGE §8 still applies once that approval is
in the header: runner (runs nothing) in its own PR, then one single pass.

Learn page: [v5_1_coverage_prereg.html](../learn/v5_1_coverage_prereg.html).

## Proposed alternative, if the coverage study is never approved

**Proposed only. Not pre-registered. Not run. No numbers from it appear in this
file.** If the owner never gives the go to run the held coverage study, a narrower
uncertainty follow-up that could be pre-registered later is:

**Empirical \(\sqrt{n}\) sampling-variability of the already-committed V5.1 gate
replications.**

* **Question.** On each of the four gated linear-mean pure-composition cells, does
  the empirical standard deviation of the graded-channel composition piece
  \(\hat c\), read from the already-committed records in
  `benchmarks/results/v5_1_gate_shape.jsonl` (50 replications at n = 2000, 20 at
  n = 20 000), shrink from n = 2000 to n = 20 000 at a rate consistent with
  \(1/\sqrt{n}\), inside a pre-set *F* / chi-square acceptance interval? The same
  check for \(\hat s\) and for the graded-channel error. The exact interval would
  be frozen in that later pre-registration, **before** any standard deviation is
  computed for the record.
* **Inputs.** Only the committed jsonl and
  `benchmarks/results/shiftbench_truth_shape.json`, scored with the existing
  `graded_err` of `summarize_v5_1_gate.py`. No new TRACE fit, no new seed stream,
  no index in 10400–10799, no sealed index in 9600–9619.
* **What it would and would not claim.** A statement about sampling variability
  and the \(\sqrt{n}\) rate on the four already-gated cells. It would **not**
  validate a confidence band, would not add a band API to the library, and would
  not replace the held coverage study. If the \(\sqrt{n}\) rate holds, a later,
  separate pre-registration on fresh seeds outside every spent range in
  PREREG_V5_1_COVERAGE §4 **and** outside 10400–10799 could test an analytic
  (delta-method) interval; that would itself be a new freeze. If the rate fails,
  that is a reason to return to the held bootstrap-coverage study rather than to
  invent a band.
* **Seeds reserved.** 10400–10799 stay reserved for the held coverage study, so
  they can still be used if the owner later approves it. The alternative uses no
  new indices at all.

Do not register or run this alternative unless the owner asks for that
pre-registration in his own words.
